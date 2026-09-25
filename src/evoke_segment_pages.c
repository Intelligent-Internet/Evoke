#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/generic_xlog.h"
#include "access/xlog.h"
#include "access/xloginsert.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/indexfsm.h"
#include "storage/lmgr.h"
#include "utils/backend_status.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_am_options.h"
#include "evoke_block_ranges.h"
#include "evoke_document_cow.h"
#include "evoke_document_tid_lookup.h"
#include "evoke_lexicon_cow.h"
#include "evoke_prefix_cow.h"
#include "evoke_segment_pages.h"
#include "evoke_semantic_bmp.h"
#include "evoke_segments.h"
#include "evoke_term_cow.h"

#define EVOKE_RECYCLABLE_PAGE_MAGIC UINT64_C(0x4949343246524545)
#define EVOKE_RECYCLABLE_PAGE_VERSION UINT32_C(1)
#define EVOKE_FSM_CLAIM_MAX_PROBES UINT32_C(256)
#define EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK_SHIFT \
    (EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT - \
     EVOKE_DEFAULT_POSTING_BLOCK_SHIFT)
#define EVOKE_BMP_BLOCKS_PER_QUERY_BLOCK_SHIFT \
    (EVOKE_DEFAULT_POSTING_BLOCK_SHIFT - \
     EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT)
#define EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK \
    (UINT32_C(1) << EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK_SHIFT)
#define EVOKE_SEMANTIC_FORWARD_SUBRANGE_SHIFT UINT32_C(6)

bool evoke_test_semantic_accelerator_dense_subrange_reads = false;
bool evoke_test_semantic_accelerator_sparse_block_oracle = false;

typedef struct evoke_recyclable_page_marker
{
    uint64 magic;
    uint64 owner_manifest_id;
    uint32 block_number;
    uint32 version;
    uint64 checksum;
} evoke_recyclable_page_marker;

StaticAssertDecl(
    sizeof(evoke_recyclable_page_marker) == 32,
    "evoke recyclable page marker must remain fixed-width"
);
StaticAssertDecl(
    EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS >=
        EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM +
        EVOKE_SEGMENT_QUERY_TERM_FOLD_MAX_RUNS,
    "query plans must hold every raw term extent and fold run"
);

StaticAssertDecl(
    EVOKE_INITIAL_FOLD_TARGET_BYTES < MaxAllocSize,
    "initial fold bundles must fit in one PostgreSQL allocation"
);
StaticAssertDecl(
    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT <=
        EVOKE_DEFAULT_POSTING_BLOCK_SHIFT &&
    EVOKE_DEFAULT_POSTING_BLOCK_SHIFT <=
        EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT,
    "packed BMP and query block geometry must remain nested"
);

static void evoke_segment_pages_report_codec_error(
    const char *object_name,
    evoke_status status
);

static void evoke_segment_pages_report_relation_codec_error(
    Relation index_relation,
    const char *object_name,
    evoke_status status
);

static uint32
evoke_segment_pages_read_u32_le(const uint8 *bytes)
{
    return (uint32) bytes[0] |
        ((uint32) bytes[1] << 8) |
        ((uint32) bytes[2] << 16) |
        ((uint32) bytes[3] << 24);
}

static uint16
evoke_segment_pages_read_u16_le(const uint8 *bytes)
{
    return (uint16) bytes[0] |
        ((uint16) bytes[1] << 8);
}

static uint64
evoke_segment_pages_read_u64_le(const uint8 *bytes)
{
    return (uint64) bytes[0] |
        ((uint64) bytes[1] << 8) |
        ((uint64) bytes[2] << 16) |
        ((uint64) bytes[3] << 24) |
        ((uint64) bytes[4] << 32) |
        ((uint64) bytes[5] << 40) |
        ((uint64) bytes[6] << 48) |
        ((uint64) bytes[7] << 56);
}

static Size
evoke_segment_page_content_bytes(void)
{
    return BLCKSZ - MAXALIGN(SizeOfPageHeaderData);
}

static Size
evoke_segment_page_payload_capacity(void)
{
    return evoke_segment_page_content_bytes() -
        EVOKE_SEGMENT_PAGE_HEADER_SIZE;
}

static Size
evoke_l0_page_payload_capacity(void)
{
    return evoke_segment_page_content_bytes() -
        EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE;
}

static void
evoke_segment_set_page_content_size(Page page, Size content_size)
{
    PageHeader page_header = (PageHeader) page;
    Size content_start = MAXALIGN(SizeOfPageHeaderData);
    Size content_end;

    if (content_size > BLCKSZ - content_start)
    {
        ereport(ERROR, (errmsg("evoke segment page content is too large")));
    }
    content_end = content_start + content_size;
    if (content_end > page_header->pd_upper)
    {
        ereport(ERROR, (errmsg("evoke segment page content overlaps storage")));
    }
    page_header->pd_lower = (LocationIndex) content_end;
}

static void
evoke_segment_mark_buffer_dirty_with_wal(
    Relation index_relation,
    Buffer buffer
)
{
    Assert(CritSectionCount > 0);
    MarkBufferDirty(buffer);
    if (RelationNeedsWAL(index_relation))
    {
        (void) log_newpage_buffer(buffer, false);
    }
}

static void
evoke_segment_recyclable_marker_init(
    evoke_recyclable_page_marker *marker,
    BlockNumber block_number,
    uint64 owner_manifest_id
)
{
    memset(marker, 0, sizeof(*marker));
    marker->magic = EVOKE_RECYCLABLE_PAGE_MAGIC;
    marker->owner_manifest_id = owner_manifest_id;
    marker->block_number = block_number;
    marker->version = EVOKE_RECYCLABLE_PAGE_VERSION;
    marker->checksum = evoke_segment_blob_checksum(
        (const uint8 *) marker,
        offsetof(evoke_recyclable_page_marker, checksum)
    );
}

static bool
evoke_segment_recyclable_marker_matches(
    Page page,
    BlockNumber block_number
)
{
    evoke_recyclable_page_marker marker;
    PageHeader page_header;
    Size content_end;

    if (page == NULL || PageIsNew(page))
    {
        return false;
    }
    page_header = (PageHeader) page;
    content_end = MAXALIGN(SizeOfPageHeaderData) + sizeof(marker);
    if (page_header->pd_lower != content_end)
    {
        return false;
    }
    memcpy(&marker, PageGetContents(page), sizeof(marker));
    return marker.magic == EVOKE_RECYCLABLE_PAGE_MAGIC &&
        marker.owner_manifest_id != 0 &&
        marker.block_number == block_number &&
        marker.version == EVOKE_RECYCLABLE_PAGE_VERSION &&
        marker.checksum == evoke_segment_blob_checksum(
            (const uint8 *) &marker,
            offsetof(evoke_recyclable_page_marker, checksum)
        );
}

static bool
evoke_segment_recyclable_block_matches(
    Relation index_relation,
    BlockNumber block_number
)
{
    Buffer buffer;
    bool matches;

    if (block_number == 0 ||
        block_number >= RelationGetNumberOfBlocks(index_relation))
    {
        return false;
    }
    buffer = ReadBufferExtended(
        index_relation,
        MAIN_FORKNUM,
        block_number,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(buffer, BUFFER_LOCK_SHARE);
    matches = evoke_segment_recyclable_marker_matches(
        BufferGetPage(buffer),
        block_number
    );
    UnlockReleaseBuffer(buffer);
    return matches;
}

/* Caller serializes every Evoke FSM claim with the extension lock. */
static bool
evoke_segment_claim_fsm_range(
    Relation index_relation,
    uint32 page_count,
    BlockNumber *start_block_out
)
{
    evoke_block_range skipped_ranges[EVOKE_FSM_CLAIM_MAX_PROBES];
    BlockNumber first_block;
    BlockNumber relation_blocks;
    uint32 skipped_range_count = 0;
    uint32 probe_count = 0;
    bool claimed = false;

    if (index_relation == NULL || page_count == 0 ||
        start_block_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke FSM range claim")));
    }
    memset(skipped_ranges, 0, sizeof(skipped_ranges));
    relation_blocks = RelationGetNumberOfBlocks(index_relation);
    PG_TRY();
    {
        while (probe_count < EVOKE_FSM_CLAIM_MAX_PROBES)
        {
            BlockNumber left_block;
            BlockNumber right_block;
            uint64 run_count;

            first_block = GetFreeIndexPage(index_relation);
            if (first_block == InvalidBlockNumber)
            {
                break;
            }
            probe_count++;
            if (first_block == 0 || first_block >= relation_blocks ||
                !evoke_segment_recyclable_block_matches(
                    index_relation,
                    first_block))
            {
                RecordUsedIndexPage(index_relation, first_block);
                continue;
            }

            left_block = first_block;
            while (left_block > 1 &&
                   (uint64) first_block - left_block + 1 < page_count &&
                   evoke_segment_recyclable_block_matches(
                       index_relation,
                       left_block - 1))
            {
                left_block--;
            }
            right_block = first_block;
            run_count = (uint64) right_block - left_block + 1;
            while (run_count < page_count &&
                   right_block + 1 < relation_blocks &&
                   evoke_segment_recyclable_block_matches(
                       index_relation,
                       right_block + 1))
            {
                right_block++;
                run_count++;
            }
            if (run_count >= page_count)
            {
                for (uint32 offset = 0; offset < page_count; offset++)
                {
                    RecordUsedIndexPage(
                        index_relation,
                        left_block + offset
                    );
                }
                *start_block_out = left_block;
                claimed = true;
                break;
            }

            skipped_ranges[skipped_range_count].start_block = left_block;
            skipped_ranges[skipped_range_count].block_count =
                (uint32) run_count;
            skipped_range_count++;
            for (uint32 offset = 0; offset < run_count; offset++)
            {
                RecordUsedIndexPage(index_relation, left_block + offset);
            }
        }
    }
    PG_FINALLY();
    {
        /* Restore fragments hidden only while probing for a larger run. */
        for (uint32 range_index = 0;
             range_index < skipped_range_count;
             range_index++)
        {
            const evoke_block_range *range =
                &skipped_ranges[range_index];

            for (uint32 offset = 0; offset < range->block_count; offset++)
            {
                RecordFreeIndexPage(
                    index_relation,
                    range->start_block + offset
                );
            }
        }
        if (skipped_range_count > 0)
        {
            /* Rebuild upper FSM summaries for later writes in this COW. */
            IndexFreeSpaceMapVacuum(index_relation);
        }
    }
    PG_END_TRY();
    return claimed;
}

void
evoke_segment_pages_publish_fsm_handoff(
    Relation index_relation,
    uint64 owner_manifest_id,
    const evoke_segment_cow_result *result
)
{
    BlockNumber relation_blocks;
    uint64 block_count = 0;
    volatile bool extension_locked = false;

    if (index_relation == NULL || owner_manifest_id == 0 || result == NULL ||
        (result->fsm_handoff_range_count == 0) !=
            (result->fsm_handoff_ranges == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke FSM handoff")));
    }
    if (result->fsm_handoff_range_count == 0)
    {
        if (result->fsm_handoff_block_count != 0)
        {
            ereport(ERROR, (errmsg("invalid empty evoke FSM handoff")));
        }
        return;
    }

    relation_blocks = RelationGetNumberOfBlocks(index_relation);
    for (uint32 range_index = 0;
         range_index < result->fsm_handoff_range_count;
         range_index++)
    {
        const evoke_block_range *range =
            &result->fsm_handoff_ranges[range_index];
        uint64 range_end =
            (uint64) range->start_block + range->block_count;
        uint64 manifest_end =
            (uint64) result->manifest.start_block +
                result->manifest.page_count;

        if (range->start_block == 0 || range->block_count == 0 ||
            range_end > relation_blocks ||
            (range_index > 0 &&
             (uint64) result->fsm_handoff_ranges[
                 range_index - 1
             ].start_block +
                 result->fsm_handoff_ranges[
                     range_index - 1
                 ].block_count >= range->start_block) ||
            (result->manifest.page_count > 0 &&
             range->start_block < manifest_end &&
             result->manifest.start_block < range_end) ||
            UINT64_MAX - block_count < range->block_count)
        {
            ereport(ERROR, (errmsg("invalid evoke FSM handoff range")));
        }
        block_count += range->block_count;
    }
    if (block_count != result->fsm_handoff_block_count)
    {
        ereport(ERROR, (errmsg("invalid evoke FSM handoff block count")));
    }

    PG_TRY();
    {
        LockRelationForExtension(index_relation, ExclusiveLock);
        extension_locked = true;
        for (uint32 range_index = 0;
             range_index < result->fsm_handoff_range_count;
             range_index++)
        {
            const evoke_block_range *range =
                &result->fsm_handoff_ranges[range_index];

            for (uint32 offset = 0; offset < range->block_count; offset++)
            {
                BlockNumber block_number = range->start_block + offset;
                evoke_recyclable_page_marker marker;
                Buffer buffer = ReadBufferExtended(
                    index_relation,
                    MAIN_FORKNUM,
                    block_number,
                    RBM_NORMAL,
                    NULL
                );
                GenericXLogState *xlog_state;
                Page page;

                LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
                xlog_state = GenericXLogStart(index_relation);
                page = GenericXLogRegisterBuffer(
                    xlog_state,
                    buffer,
                    GENERIC_XLOG_FULL_IMAGE
                );
                PageInit(page, BLCKSZ, 0);
                evoke_segment_recyclable_marker_init(
                    &marker,
                    block_number,
                    owner_manifest_id
                );
                memcpy(PageGetContents(page), &marker, sizeof(marker));
                evoke_segment_set_page_content_size(page, sizeof(marker));
                (void) GenericXLogFinish(xlog_state);
                UnlockReleaseBuffer(buffer);
            }
        }

        /* Expose no page until every marker is WAL-logged. */
        for (uint32 range_index = 0;
             range_index < result->fsm_handoff_range_count;
             range_index++)
        {
            const evoke_block_range *range =
                &result->fsm_handoff_ranges[range_index];

            for (uint32 offset = 0; offset < range->block_count; offset++)
            {
                RecordFreeIndexPage(
                    index_relation,
                    range->start_block + offset
                );
            }
        }
        IndexFreeSpaceMapVacuum(index_relation);
    }
    PG_FINALLY();
    {
        if (extension_locked)
        {
            UnlockRelationForExtension(index_relation, ExclusiveLock);
        }
    }
    PG_END_TRY();
}

uint64
evoke_segment_pages_count_recyclable_markers(
    Relation index_relation,
    const evoke_segment_reachability_inventory *inventory
)
{
    evoke_block_range_allocator unreachable;
    uint64 marker_count = 0;
    evoke_status status;

    if (index_relation == NULL || inventory == NULL ||
        !inventory->blocks.finalized)
    {
        ereport(ERROR, (errmsg("invalid evoke recyclable marker scan")));
    }
    evoke_block_range_allocator_init(&unreachable);
    status = evoke_block_range_allocator_build(
        &inventory->blocks,
        &unreachable
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "recyclable marker inventory",
            status
        );
    }
    PG_TRY();
    {
        for (size_t range_index = 0;
             range_index < unreachable.range_count;
             range_index++)
        {
            const evoke_block_range *range =
                &unreachable.ranges[range_index];

            CHECK_FOR_INTERRUPTS();
            for (uint32 offset = 0; offset < range->block_count; offset++)
            {
                if ((offset & 1023) == 0)
                {
                    CHECK_FOR_INTERRUPTS();
                }
                if (evoke_segment_recyclable_block_matches(
                        index_relation,
                        range->start_block + offset))
                {
                    marker_count++;
                }
            }
        }
    }
    PG_FINALLY();
    {
        evoke_block_range_allocator_free(&unreachable);
    }
    PG_END_TRY();
    return marker_count;
}

static void
evoke_segment_page_ref_validate(
    const evoke_segment_object_ref *ref,
    BlockNumber relation_blocks
)
{
    uint32 expected_pages = 0;
    evoke_status status;

    if (ref == NULL || ref->object_bytes > MaxAllocSize)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page reference")));
    }
    status = evoke_segment_object_ref_validate(ref, relation_blocks);
    if (status != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page reference")));
    }
    status = evoke_segment_page_count_required(
        (size_t) ref->object_bytes,
        evoke_segment_page_content_bytes(),
        &expected_pages
    );
    if (status != EVOKE_OK ||
        ref->page_count != expected_pages ||
        (uint64) ref->start_block + ref->page_count > relation_blocks)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page range")));
    }
}

void
evoke_segment_page_reuse_arena_init(
    evoke_segment_page_reuse_arena *arena
)
{
    if (arena != NULL)
    {
        memset(arena, 0, sizeof(*arena));
        evoke_block_range_allocator_init(&arena->allocator);
        evoke_block_range_inventory_init(&arena->staged_writes);
    }
}

void
evoke_segment_page_reuse_arena_free(
    evoke_segment_page_reuse_arena *arena
)
{
    if (arena == NULL)
    {
        return;
    }
    evoke_block_range_allocator_free(&arena->allocator);
    evoke_block_range_inventory_free(&arena->staged_writes);
    memset(arena, 0, sizeof(*arena));
}

void
evoke_segment_cow_result_init(
    evoke_segment_cow_result *result
)
{
    if (result != NULL)
    {
        memset(result, 0, sizeof(*result));
    }
}

void
evoke_segment_cow_result_free(
    evoke_segment_cow_result *result
)
{
    if (result == NULL)
    {
        return;
    }
    free(result->fsm_handoff_ranges);
    free(result->staged_write_ranges);
    memset(result, 0, sizeof(*result));
}

evoke_status
evoke_segment_page_reuse_arena_build(
    const evoke_block_range_inventory *inventory,
    evoke_segment_page_reuse_arena *arena
)
{
    evoke_status status;

    if (inventory == NULL || arena == NULL ||
        arena->reader_fenced || arena->allocator.initialized)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_block_range_allocator_build(
        inventory,
        &arena->allocator
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    arena->source_high_watermark =
        inventory->published_block_high_watermark;
    arena->reader_fenced = true;
    return EVOKE_OK;
}

evoke_status
evoke_segment_page_reuse_arena_build_retired(
    const evoke_block_range *ranges,
    size_t range_count,
    uint32_t published_block_high_watermark,
    evoke_segment_page_reuse_arena *arena
)
{
    evoke_status status;

    if (ranges == NULL || range_count == 0 || arena == NULL ||
        arena->reader_fenced || arena->allocator.initialized)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_block_range_allocator_build_free(
        ranges,
        range_count,
        published_block_high_watermark,
        &arena->allocator
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    arena->source_high_watermark = published_block_high_watermark;
    arena->reader_fenced = true;
    return EVOKE_OK;
}

evoke_status
evoke_segment_page_reuse_arena_build_empty_fenced(
    uint32_t published_block_high_watermark,
    evoke_segment_page_reuse_arena *arena
)
{
    if (published_block_high_watermark == 0 || arena == NULL ||
        arena->reader_fenced || arena->allocator.initialized)
    {
        return EVOKE_ERR_INVALID;
    }
    arena->allocator.initialized = true;
    arena->source_high_watermark = published_block_high_watermark;
    arena->reader_fenced = true;
    return EVOKE_OK;
}

static uint32
evoke_segment_page_reuse_arena_reused_blocks(
    const evoke_segment_page_reuse_arena *arena
)
{
    uint64 reused_blocks;

    if (arena == NULL)
    {
        return 0;
    }
    reused_blocks = arena->allocator.allocated_block_count +
        arena->fsm_reused_block_count;
    if (reused_blocks > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("evoke reused block count exceeds uint32")));
    }
    return (uint32) reused_blocks;
}

static void
evoke_segment_pages_write_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_object_kind object_kind,
    uint64 object_id,
    uint64 owner_manifest_id,
    const uint8 *object_bytes,
    Size object_size,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *ref_out
)
{
    evoke_segment_object_ref ref;
    evoke_segment_page_header first_header;
    uint8 first_header_bytes[EVOKE_SEGMENT_PAGE_HEADER_SIZE];
    uint32 page_count = 0;
    Size payload_capacity;
    Size consumed = 0;
    volatile bool extension_locked = false;
    bool reuse_range = false;
    bool arena_reuse_range = false;
    uint32 reuse_high_watermark = 0;
    evoke_status status;

    if (index_relation == NULL ||
        (fork_number != MAIN_FORKNUM &&
         fork_number != INIT_FORKNUM) ||
        object_bytes == NULL ||
        object_size == 0 ||
        ref_out == NULL ||
        (fork_number != MAIN_FORKNUM && reuse_arena != NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke segment page write")));
    }
    if (object_size > MaxAllocSize)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke segment page write"),
                errdetail(
                    "Object kind %u requires %zu bytes; the PostgreSQL "
                    "allocation limit is %zu bytes.",
                    (unsigned int) object_kind,
                    (size_t) object_size,
                    (size_t) MaxAllocSize
                )
            )
        );
    }
    status = evoke_segment_page_count_required(
        object_size,
        evoke_segment_page_content_bytes(),
        &page_count
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg(
                "failed to size evoke segment page chain: %s",
                evoke_strerror(status)
            ))
        );
    }

    memset(&ref, 0, sizeof(ref));
    ref.object_kind = object_kind;
    ref.page_count = page_count;
    ref.object_id = object_id;
    ref.owner_manifest_id = owner_manifest_id;
    ref.object_bytes = object_size;
    CHECK_FOR_INTERRUPTS();
    ref.object_checksum = evoke_segment_blob_checksum(
        object_bytes,
        object_size
    );
    payload_capacity = evoke_segment_page_payload_capacity();
    memset(&first_header, 0, sizeof(first_header));
    first_header.object_kind = object_kind;
    first_header.object_id = object_id;
    first_header.owner_manifest_id = owner_manifest_id;
    first_header.object_bytes = object_size;
    first_header.object_checksum = ref.object_checksum;
    first_header.page_count = page_count;
    first_header.used_bytes = (uint32) Min(
        object_size,
        payload_capacity
    );
    first_header.payload_checksum = evoke_segment_page_payload_checksum(
        object_bytes,
        first_header.used_bytes
    );
    status = evoke_segment_page_header_serialize(
        &first_header,
        evoke_segment_page_content_bytes(),
        first_header_bytes,
        sizeof(first_header_bytes)
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg(
                "invalid evoke segment page object: %s",
                evoke_strerror(status)
            ))
        );
    }

    PG_TRY();
    {
        if (reuse_arena != NULL && reuse_arena->reader_fenced)
        {
            reuse_range =
                evoke_block_range_allocator_allocate_best_fit(
                    &reuse_arena->allocator,
                    page_count,
                    &ref.start_block
                );
            arena_reuse_range = reuse_range;
            if (reuse_range)
            {
                reuse_high_watermark =
                    reuse_arena->source_high_watermark;
            }
        }
        if (!reuse_range)
        {
            LockRelationForExtension(index_relation, ExclusiveLock);
            extension_locked = true;
            reuse_range = fork_number == MAIN_FORKNUM &&
                evoke_segment_claim_fsm_range(
                    index_relation,
                    page_count,
                    &ref.start_block
                );
            if (reuse_range)
            {
                reuse_high_watermark =
                    RelationGetNumberOfBlocks(index_relation);
            }
            else
            {
                ref.start_block = RelationGetNumberOfBlocksInFork(
                    index_relation,
                    fork_number
                );
            }
        }
        if (ref.start_block == 0 ||
            (uint64) ref.start_block + page_count > InvalidBlockNumber ||
            (reuse_range &&
             ((uint64) ref.start_block + page_count >
                  reuse_high_watermark ||
              reuse_high_watermark >
                  RelationGetNumberOfBlocksInFork(
                      index_relation,
                      fork_number))))
        {
            ereport(ERROR, (errmsg("evoke segment page range is invalid")));
        }

        for (uint32 ordinal = 0; ordinal < page_count; ordinal++)
        {
            evoke_segment_page_header header;
            Buffer buffer;
            BlockNumber block_number;
            GenericXLogState *xlog_state = NULL;
            Page page;
            uint8 *page_content;
            Size remaining = object_size - consumed;
            Size used_bytes = Min(remaining, payload_capacity);

            if ((ordinal & UINT32_C(0x3f)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            if (reuse_range)
            {
                buffer = ReadBufferExtended(
                    index_relation,
                    fork_number,
                    ref.start_block + ordinal,
                    RBM_NORMAL,
                    NULL
                );
                LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
                xlog_state = GenericXLogStart(index_relation);
                page = GenericXLogRegisterBuffer(
                    xlog_state,
                    buffer,
                    GENERIC_XLOG_FULL_IMAGE
                );
            }
            else
            {
                buffer = ExtendBufferedRel(
                    BMR_REL(index_relation),
                    fork_number,
                    NULL,
                    EB_LOCK_FIRST | EB_SKIP_EXTENSION_LOCK
                );
                page = BufferGetPage(buffer);
            }
            block_number = BufferGetBlockNumber(buffer);
            if (block_number != ref.start_block + ordinal)
            {
                if (xlog_state != NULL)
                {
                    GenericXLogAbort(xlog_state);
                }
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (errmsg("evoke segment page chain is not contiguous"))
                );
            }

            PageInit(page, BLCKSZ, 0);
            page_content = (uint8 *) PageGetContents(page);
            memset(&header, 0, sizeof(header));
            header.object_kind = object_kind;
            header.object_id = object_id;
            header.owner_manifest_id = owner_manifest_id;
            header.object_bytes = object_size;
            header.object_checksum = ref.object_checksum;
            header.ordinal = ordinal;
            header.page_count = page_count;
            header.used_bytes = (uint32) used_bytes;
            header.payload_checksum = evoke_segment_page_payload_checksum(
                object_bytes + consumed,
                used_bytes
            );
            status = evoke_segment_page_header_serialize(
                &header,
                evoke_segment_page_content_bytes(),
                page_content,
                EVOKE_SEGMENT_PAGE_HEADER_SIZE
            );
            if (status != EVOKE_OK)
            {
                if (xlog_state != NULL)
                {
                    GenericXLogAbort(xlog_state);
                }
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (errmsg(
                        "failed to encode evoke segment page: %s",
                        evoke_strerror(status)
                    ))
                );
            }
            memcpy(
                page_content + EVOKE_SEGMENT_PAGE_HEADER_SIZE,
                object_bytes + consumed,
                used_bytes
            );
            evoke_segment_set_page_content_size(
                page,
                EVOKE_SEGMENT_PAGE_HEADER_SIZE + used_bytes
            );
            if (xlog_state != NULL)
            {
                (void) GenericXLogFinish(xlog_state);
            }
            else
            {
                START_CRIT_SECTION();
                MarkBufferDirty(buffer);
                if (fork_number == INIT_FORKNUM ||
                    RelationNeedsWAL(index_relation))
                {
                    (void) log_newpage_buffer(buffer, false);
                }
                END_CRIT_SECTION();
            }
            UnlockReleaseBuffer(buffer);
            consumed += used_bytes;
        }
        if (consumed != object_size)
        {
            ereport(ERROR, (errmsg("evoke segment page write was incomplete")));
        }
        if (reuse_arena != NULL)
        {
            if (reuse_range && !arena_reuse_range)
            {
                reuse_arena->fsm_reused_block_count += ref.page_count;
            }
            status = evoke_block_range_inventory_add(
                &reuse_arena->staged_writes,
                ref.start_block,
                ref.page_count
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "track staged COW pages",
                    status
                );
            }
        }
        *ref_out = ref;
    }
    PG_FINALLY();
    {
        if (extension_locked)
        {
            UnlockRelationForExtension(index_relation, ExclusiveLock);
        }
    }
    PG_END_TRY();
}

void
evoke_segment_pages_write(
    Relation index_relation,
    evoke_segment_object_kind object_kind,
    uint64 object_id,
    uint64 owner_manifest_id,
    const uint8 *object_bytes,
    Size object_size,
    evoke_segment_object_ref *ref_out
)
{
    evoke_segment_pages_write_internal(
        index_relation,
        MAIN_FORKNUM,
        object_kind,
        object_id,
        owner_manifest_id,
        object_bytes,
        object_size,
        NULL,
        ref_out
    );
}

void
evoke_segment_pages_write_l0_record_chain(
    Relation index_relation,
    uint64 segment_id,
    uint32 ordinal_base,
    uint64 sequence,
    const uint8 *record_bytes,
    Size record_size,
    evoke_l0_chain_write_result *result_out
)
{
    evoke_l0_chain_write_result result;
    Size fragment_capacity;
    uint64 record_checksum;
    uint64 page_count;
    volatile bool extension_locked = false;

    if (index_relation == NULL || segment_id == 0 || sequence == 0 ||
        record_bytes == NULL || record_size < EVOKE_L0_RECORD_HEADER_SIZE ||
        record_size > UINT32_MAX || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke L0 record chain write")));
    }
    fragment_capacity = evoke_l0_page_payload_capacity() -
        EVOKE_L0_FRAME_HEADER_SIZE;
    if (fragment_capacity == 0)
    {
        ereport(ERROR, (errmsg("evoke L0 page has no fragment capacity")));
    }
    page_count =
        1 + ((uint64) record_size - 1) / (uint64) fragment_capacity;
    if (page_count > EVOKE_ACTIVE_L0_MAX_PAGES ||
        (uint64) ordinal_base + page_count >
            EVOKE_ACTIVE_L0_MAX_PAGES)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("evoke active L0 page bound exceeded")
            )
        );
    }

    memset(&result, 0, sizeof(result));
    result.page_count = (uint32) page_count;
    record_checksum = evoke_segment_blob_checksum(
        record_bytes,
        record_size
    );

    PG_TRY();
    {
        BlockNumber next_block = EVOKE_ACTIVE_L0_NO_NEXT_BLOCK;

        LockRelationForExtension(index_relation, ExclusiveLock);
        extension_locked = true;

        /*
         * L0 pages are an explicit linked chain. Write from tail to head so
         * each page can consume one safe FSM page without requiring a large
         * contiguous extent. The root remains unpublished until the caller
         * links the completed head, so a partial write is only orphan space.
         */
        for (uint32 reverse_index = result.page_count;
             reverse_index > 0;
             reverse_index--)
        {
            uint32 page_index = reverse_index - 1;
            evoke_active_l0_page_header page_header;
            evoke_l0_frame_header frame_header;
            uint8 frame_bytes[EVOKE_L0_FRAME_HEADER_SIZE];
            Buffer buffer;
            BlockNumber block_number;
            GenericXLogState *xlog_state = NULL;
            Page page;
            uint8 *page_content;
            Size fragment_offset =
                (Size) page_index * fragment_capacity;
            Size fragment_size = Min(
                record_size - fragment_offset,
                fragment_capacity
            );
            Size used_bytes =
                EVOKE_L0_FRAME_HEADER_SIZE + fragment_size;
            bool reuse_page;
            evoke_status status;

            reuse_page = evoke_segment_claim_fsm_range(
                index_relation,
                1,
                &block_number
            );
            if (reuse_page)
            {
                buffer = ReadBufferExtended(
                    index_relation,
                    MAIN_FORKNUM,
                    block_number,
                    RBM_NORMAL,
                    NULL
                );
                LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
                xlog_state = GenericXLogStart(index_relation);
                page = GenericXLogRegisterBuffer(
                    xlog_state,
                    buffer,
                    GENERIC_XLOG_FULL_IMAGE
                );
            }
            else
            {
                block_number = RelationGetNumberOfBlocks(index_relation);
                buffer = ExtendBufferedRel(
                    BMR_REL(index_relation),
                    MAIN_FORKNUM,
                    NULL,
                    EB_LOCK_FIRST | EB_SKIP_EXTENSION_LOCK
                );
                page = BufferGetPage(buffer);
            }
            if (BufferGetBlockNumber(buffer) != block_number ||
                block_number == 0 ||
                block_number >= InvalidBlockNumber)
            {
                if (xlog_state != NULL)
                {
                    GenericXLogAbort(xlog_state);
                }
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (errmsg("evoke L0 page allocation is invalid"))
                );
            }
            if (page_index + 1 == result.page_count)
            {
                result.tail_block = block_number;
            }
            if (page_index == 0)
            {
                result.head_block = block_number;
            }

            memset(&frame_header, 0, sizeof(frame_header));
            frame_header.flags = 0;
            if (fragment_offset == 0)
            {
                frame_header.flags |= EVOKE_L0_FRAME_FLAG_START;
            }
            if (fragment_offset + fragment_size == record_size)
            {
                frame_header.flags |= EVOKE_L0_FRAME_FLAG_END;
            }
            frame_header.sequence = sequence;
            frame_header.record_checksum = record_checksum;
            frame_header.record_bytes = (uint32) record_size;
            frame_header.fragment_offset = (uint32) fragment_offset;
            frame_header.fragment_bytes = (uint32) fragment_size;
            status = evoke_l0_frame_header_serialize(
                &frame_header,
                frame_bytes,
                sizeof(frame_bytes)
            );
            if (status != EVOKE_OK)
            {
                if (xlog_state != NULL)
                {
                    GenericXLogAbort(xlog_state);
                }
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (
                        errmsg("failed to encode evoke L0 frame"),
                        errdetail(
                            "Validation failed: %s.",
                            evoke_strerror(status)
                        )
                    )
                );
            }

            PageInit(page, BLCKSZ, 0);
            page_content = (uint8 *) PageGetContents(page);
            memcpy(
                page_content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
                frame_bytes,
                sizeof(frame_bytes)
            );
            memcpy(
                page_content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE +
                    EVOKE_L0_FRAME_HEADER_SIZE,
                record_bytes + fragment_offset,
                fragment_size
            );

            memset(&page_header, 0, sizeof(page_header));
            page_header.segment_id = segment_id;
            page_header.min_sequence = sequence;
            page_header.max_sequence = sequence;
            page_header.payload_checksum = evoke_segment_blob_checksum(
                page_content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
                used_bytes
            );
            page_header.ordinal = ordinal_base + page_index;
            page_header.next_block = next_block;
            page_header.frame_count = 1;
            page_header.used_bytes = (uint32) used_bytes;
            status = evoke_active_l0_page_header_serialize(
                &page_header,
                evoke_segment_page_content_bytes(),
                page_content,
                EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE
            );
            if (status != EVOKE_OK)
            {
                if (xlog_state != NULL)
                {
                    GenericXLogAbort(xlog_state);
                }
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (
                        errmsg("failed to encode evoke L0 page"),
                        errdetail(
                            "Validation failed: %s.",
                            evoke_strerror(status)
                        )
                    )
                );
            }
            evoke_segment_set_page_content_size(
                page,
                EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE + used_bytes
            );
            if (xlog_state != NULL)
            {
                (void) GenericXLogFinish(xlog_state);
            }
            else
            {
                START_CRIT_SECTION();
                evoke_segment_mark_buffer_dirty_with_wal(
                    index_relation,
                    buffer
                );
                END_CRIT_SECTION();
            }
            UnlockReleaseBuffer(buffer);
            next_block = block_number;
        }
        if (result.head_block == 0 || result.tail_block == 0 ||
            next_block != result.head_block)
        {
            ereport(ERROR, (errmsg("evoke L0 record chain is incomplete")));
        }
        *result_out = result;
    }
    PG_FINALLY();
    {
        if (extension_locked)
        {
            UnlockRelationForExtension(index_relation, ExclusiveLock);
        }
    }
    PG_END_TRY();
}

static bool
evoke_segment_query_page_validation_cache_contains(
    const evoke_segment_query_page_validation_cache *cache,
    BlockNumber block_number
)
{
    uint32 word_index;
    uint64 mask;
    uint32 key;
    uint32 slot;

    if (cache == NULL || block_number == InvalidBlockNumber)
    {
        return false;
    }
    key = (uint32) block_number + UINT32_C(1);
    slot = (uint32) block_number &
        (EVOKE_SEGMENT_QUERY_PAGE_VALIDATION_CACHE_SIZE - UINT32_C(1));
    if (cache->block_keys[slot] == key)
    {
        return true;
    }
    word_index = (uint32) block_number >> 6;
    if (cache->shared_block_words == NULL ||
        word_index >= cache->shared_block_word_count)
    {
        return false;
    }
    mask = UINT64_C(1) << ((uint32) block_number & UINT32_C(63));
    return (pg_atomic_read_u64(
        &cache->shared_block_words[word_index]
    ) & mask) != 0;
}

static void
evoke_segment_query_page_validation_cache_record(
    evoke_segment_query_page_validation_cache *cache,
    BlockNumber block_number
)
{
    uint32 word_index;
    uint32 slot;

    if (cache == NULL || block_number == InvalidBlockNumber)
    {
        return;
    }
    slot = (uint32) block_number &
        (EVOKE_SEGMENT_QUERY_PAGE_VALIDATION_CACHE_SIZE - UINT32_C(1));
    cache->block_keys[slot] = (uint32) block_number + UINT32_C(1);
    word_index = (uint32) block_number >> 6;
    if (cache->shared_block_words != NULL &&
        word_index < cache->shared_block_word_count)
    {
        pg_atomic_fetch_or_u64(
            &cache->shared_block_words[word_index],
            UINT64_C(1) << ((uint32) block_number & UINT32_C(63))
        );
    }
}

static void
evoke_segment_pages_read_range_bounded(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    uint8 *bytes_out,
    BlockNumber block_limit,
    evoke_segment_query_page_validation_cache *validation_cache
)
{
    Size object_size;
    Size copied = 0;
    Size payload_capacity = evoke_segment_page_payload_capacity();
    uint32 first_ordinal;
    uint32 last_ordinal;

    if (index_relation == NULL || ref == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page range read")));
    }
    evoke_segment_page_ref_validate(ref, block_limit);
    if (ref->object_bytes > SIZE_MAX)
    {
        ereport(ERROR, (errmsg("evoke segment object is too large")));
    }
    object_size = (Size) ref->object_bytes;
    if (offset > object_size || length > object_size - offset ||
        (length > 0 && bytes_out == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke segment page range")));
    }
    if (length == 0)
    {
        return;
    }
    first_ordinal = (uint32) (offset / payload_capacity);
    last_ordinal = (uint32) (
        (offset + length - 1) / payload_capacity
    );

    for (uint32 ordinal = first_ordinal;
         ordinal <= last_ordinal;
         ordinal++)
    {
        evoke_segment_page_header header;
        BlockNumber block_number;
        Buffer buffer;
        Page page;
        PageHeader page_header;
        const uint8 *page_content;
        const uint8 *page_payload;
        Size page_object_offset;
        Size range_begin;
        Size range_end;
        Size copy_size;
        Size minimum_content;
        bool payload_validated;
        evoke_status status;

        CHECK_FOR_INTERRUPTS();
        block_number = ref->start_block + ordinal;
        payload_validated =
            evoke_segment_query_page_validation_cache_contains(
                validation_cache,
                block_number
            );
        buffer = ReadBufferExtended(
            index_relation,
            MAIN_FORKNUM,
            block_number,
            RBM_NORMAL,
            NULL
        );
        LockBuffer(buffer, BUFFER_LOCK_SHARE);
        page = BufferGetPage(buffer);
        page_header = (PageHeader) page;
        page_content = (const uint8 *) PageGetContents(page);
        status = evoke_segment_page_header_deserialize(
            page_content,
            EVOKE_SEGMENT_PAGE_HEADER_SIZE,
            evoke_segment_page_content_bytes(),
            &header
        );
        if (status != EVOKE_OK)
        {
            UnlockReleaseBuffer(buffer);
            ereport(
                ERROR,
                (errmsg(
                    "invalid evoke segment page header: %s",
                    evoke_strerror(status)
                ))
            );
        }
        minimum_content =
            MAXALIGN(SizeOfPageHeaderData) +
            EVOKE_SEGMENT_PAGE_HEADER_SIZE +
            header.used_bytes;
        page_payload = page_content + EVOKE_SEGMENT_PAGE_HEADER_SIZE;
        if (page_header->pd_lower < minimum_content ||
            header.object_kind != ref->object_kind ||
            header.object_id != ref->object_id ||
            header.owner_manifest_id != ref->owner_manifest_id ||
            header.object_bytes != ref->object_bytes ||
            header.object_checksum != ref->object_checksum ||
            header.ordinal != ordinal ||
            header.page_count != ref->page_count ||
            header.used_bytes > payload_capacity)
        {
            UnlockReleaseBuffer(buffer);
            ereport(ERROR, (errmsg("evoke segment page chain mismatch")));
        }
        if (!payload_validated && !DataChecksumsEnabled())
        {
            status = evoke_segment_page_payload_validate(
                &header,
                page_payload,
                header.used_bytes
            );
            if (status != EVOKE_OK)
            {
                UnlockReleaseBuffer(buffer);
                ereport(
                    ERROR,
                    (errmsg("evoke segment page payload checksum mismatch"))
                );
            }
        }
        if (!payload_validated)
        {
            /*
             * PostgreSQL already verified the complete page while reading it
             * when data checksums are enabled. The Evoke payload checksum is
             * retained for clusters without that protection and the shared
             * generation bitmap makes either successful validation reusable.
             */
            evoke_segment_query_page_validation_cache_record(
                validation_cache,
                block_number
            );
        }

        page_object_offset = (Size) ordinal * payload_capacity;
        range_begin = offset > page_object_offset
            ? offset
            : page_object_offset;
        range_end = offset + length;
        if (range_end > page_object_offset + header.used_bytes)
        {
            range_end = page_object_offset + header.used_bytes;
        }
        if (range_begin >= range_end)
        {
            UnlockReleaseBuffer(buffer);
            ereport(ERROR, (errmsg("evoke segment page range is incomplete")));
        }
        copy_size = range_end - range_begin;
        memcpy(
            bytes_out + copied,
            page_payload + (range_begin - page_object_offset),
            copy_size
        );
        copied += copy_size;
        UnlockReleaseBuffer(buffer);
    }
    if (copied != length)
    {
        ereport(ERROR, (errmsg("evoke segment page range was incomplete")));
    }
}

void
evoke_segment_pages_read_range(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    uint8 *bytes_out
)
{
    BlockNumber block_limit;

    if (index_relation == NULL || ref == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page range read")));
    }
    block_limit = RelationGetNumberOfBlocks(index_relation);
    evoke_segment_pages_read_range_bounded(
        index_relation,
        ref,
        offset,
        length,
        bytes_out,
        block_limit,
        NULL
    );
}

bool
evoke_segment_pages_prewarm_range(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    BlockNumber published_block_high_watermark,
    uint64 page_budget,
    uint64 *pages_warmed
)
{
    Size object_size;
    Size payload_capacity = evoke_segment_page_payload_capacity();
    uint32 first_ordinal;
    uint32 last_ordinal;
    uint64 available_pages;
    uint64 required_pages;

    if (index_relation == NULL || ref == NULL || pages_warmed == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page prewarm")));
    }
    evoke_segment_page_ref_validate(ref, published_block_high_watermark);
    if (ref->object_bytes > SIZE_MAX)
    {
        ereport(ERROR, (errmsg("evoke segment object is too large")));
    }
    object_size = (Size) ref->object_bytes;
    if (offset > object_size || length > object_size - offset)
    {
        ereport(ERROR, (errmsg("invalid evoke segment prewarm range")));
    }
    if (length == 0 || *pages_warmed >= page_budget)
    {
        return length == 0;
    }
    first_ordinal = (uint32) (offset / payload_capacity);
    last_ordinal = (uint32) ((offset + length - 1) / payload_capacity);
    available_pages = page_budget - *pages_warmed;
    required_pages = (uint64) last_ordinal - first_ordinal + 1;
    for (uint32 ordinal = first_ordinal;
         ordinal <= last_ordinal && *pages_warmed < page_budget;
         ordinal++)
    {
        Buffer buffer;

        CHECK_FOR_INTERRUPTS();
        buffer = ReadBufferExtended(
            index_relation,
            MAIN_FORKNUM,
            ref->start_block + ordinal,
            RBM_NORMAL,
            NULL
        );
        ReleaseBuffer(buffer);
        (*pages_warmed)++;
    }
    return available_pages >= required_pages;
}

static void
evoke_segment_pages_read_query_range(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    uint8 *bytes_out
)
{
    if (context == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke query segment range read")));
    }
    evoke_segment_pages_read_range_bounded(
        index_relation,
        ref,
        offset,
        length,
        bytes_out,
        context->root.published_block_high_watermark,
        context->page_validation_cache
    );
}

static void
evoke_segment_pages_prefetch_query_range(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    BlockNumber *last_prefetched,
    bool *initiated_io
)
{
    Size object_size;
    Size payload_capacity = evoke_segment_page_payload_capacity();
    uint32 first_ordinal;
    uint32 last_ordinal;

    if (index_relation == NULL || context == NULL || ref == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke query segment prefetch")));
    }
    evoke_segment_page_ref_validate(
        ref,
        context->root.published_block_high_watermark
    );
    if (ref->object_bytes > SIZE_MAX)
    {
        ereport(ERROR, (errmsg("evoke segment object is too large")));
    }
    object_size = (Size) ref->object_bytes;
    if (offset > object_size || length > object_size - offset)
    {
        ereport(ERROR, (errmsg("invalid evoke segment prefetch range")));
    }
    if (length == 0)
    {
        return;
    }
    first_ordinal = (uint32) (offset / payload_capacity);
    last_ordinal = (uint32) ((offset + length - 1) / payload_capacity);
    for (uint32 ordinal = first_ordinal;
         ordinal <= last_ordinal;
         ordinal++)
    {
        BlockNumber block_number = ref->start_block + ordinal;
        PrefetchBufferResult result;

        if (last_prefetched != NULL &&
            *last_prefetched == block_number)
        {
            continue;
        }
        result = PrefetchBuffer(
            index_relation,
            MAIN_FORKNUM,
            block_number
        );

        if (initiated_io != NULL && result.initiated_io)
        {
            *initiated_io = true;
        }
        if (last_prefetched != NULL)
        {
            *last_prefetched = block_number;
        }
    }
}

/* Maintenance validates each read; a pinned query validates each page once. */
static void
evoke_segment_pages_read_plan_range(
    Relation index_relation,
    const evoke_segment_query_context *query_context,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    uint8 *bytes_out
)
{
    if (query_context == NULL)
    {
        evoke_segment_pages_read_range(
            index_relation,
            ref,
            offset,
            length,
            bytes_out
        );
    }
    else
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            query_context,
            ref,
            offset,
            length,
            bytes_out
        );
    }
}

static uint8 *
evoke_segment_pages_read_bounded(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size *object_size_out,
    BlockNumber block_limit,
    evoke_segment_query_page_validation_cache *validation_cache
)
{
    uint8 *object_bytes;
    Size object_size;

    if (index_relation == NULL || ref == NULL || object_size_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page read")));
    }
    *object_size_out = 0;
    evoke_segment_page_ref_validate(ref, block_limit);
    if (ref->object_bytes > MaxAllocSize)
    {
        ereport(ERROR, (errmsg("evoke segment object is too large")));
    }
    object_size = (Size) ref->object_bytes;
    object_bytes = palloc(object_size);

    PG_TRY();
    {
        evoke_segment_pages_read_range_bounded(
            index_relation,
            ref,
            0,
            object_size,
            object_bytes,
            block_limit,
            validation_cache
        );
        if (evoke_segment_blob_checksum(object_bytes, object_size) !=
            ref->object_checksum)
        {
            ereport(ERROR, (errmsg("evoke segment object checksum mismatch")));
        }
        *object_size_out = object_size;
    }
    PG_CATCH();
    {
        pfree(object_bytes);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return object_bytes;
}

uint8 *
evoke_segment_pages_read(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size *object_size_out
)
{
    if (index_relation == NULL || ref == NULL || object_size_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke segment page read")));
    }
    return evoke_segment_pages_read_bounded(
        index_relation,
        ref,
        object_size_out,
        RelationGetNumberOfBlocks(index_relation),
        NULL
    );
}

void
evoke_segment_storage_snapshot_init(
    evoke_segment_storage_snapshot *snapshot
)
{
    if (snapshot == NULL)
    {
        return;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    evoke_segment_manifest_init(&snapshot->manifest);
    evoke_segment_query_contract_init(&snapshot->query_contract);
    evoke_term_directory_init(&snapshot->term_directory);
    evoke_index_init(&snapshot->index_metadata);
    evoke_segment_read_view_init(&snapshot->read_view);
    evoke_l0_storage_snapshot_init(&snapshot->l0_snapshot);
}

void
evoke_segment_query_context_init(evoke_segment_query_context *context)
{
    if (context == NULL)
    {
        return;
    }
    memset(context, 0, sizeof(*context));
    evoke_segment_manifest_init(&context->manifest);
    evoke_segment_query_contract_init(&context->query_contract);
}

void
evoke_segment_query_context_free(evoke_segment_query_context *context)
{
    if (context == NULL)
    {
        return;
    }
    if (context->page_validation_cache != NULL)
    {
        pfree(context->page_validation_cache);
        context->page_validation_cache = NULL;
    }
    evoke_segment_query_contract_free(&context->query_contract);
    evoke_segment_manifest_free(&context->manifest);
    memset(context, 0, sizeof(*context));
}

static void
evoke_segment_storage_snapshot_free_term_folds(
    evoke_segment_storage_snapshot *snapshot
)
{
    free(snapshot->term_fold_states);
    free(snapshot->term_fold_plans);
    if (snapshot->term_fold_bundles != NULL)
    {
        for (uint32 bundle_index = 0;
             bundle_index < snapshot->term_fold_bundle_count;
             bundle_index++)
        {
            evoke_term_fold_bundle_free(
                &snapshot->term_fold_bundles[bundle_index]
            );
        }
        free(snapshot->term_fold_bundles);
    }
    snapshot->term_fold_states = NULL;
    snapshot->term_fold_plans = NULL;
    snapshot->term_fold_bundles = NULL;
    snapshot->term_fold_bundle_count = 0;
}

void
evoke_segment_storage_snapshot_free(
    evoke_segment_storage_snapshot *snapshot
)
{
    uint32 segment_index;

    if (snapshot == NULL)
    {
        return;
    }
    free(snapshot->l0_indices);
    free(snapshot->l0_values);
    if (snapshot->l0_block_allocations != NULL)
    {
        for (size_t allocation_index = 0;
             allocation_index < snapshot->l0_block_allocation_count;
             allocation_index++)
        {
            evoke_posting_block_records_free(
                snapshot->l0_block_allocations[allocation_index]
            );
        }
        free(snapshot->l0_block_allocations);
    }
    free(snapshot->retired_document_ids);
    free(snapshot->materialized_doc_frequencies);
    free(snapshot->lexical_catalog_refs);
    evoke_document_cow_block_extrema_free(
        snapshot->document_block_extrema
    );
    evoke_segment_storage_snapshot_free_term_folds(snapshot);
    evoke_l0_storage_snapshot_free(&snapshot->l0_snapshot);
    evoke_segment_read_view_free(&snapshot->read_view);
    evoke_index_free(&snapshot->index_metadata);
    if (snapshot->doc_tids != NULL)
    {
        pfree(snapshot->doc_tids);
    }
    if (snapshot->document_tie_break_keys != NULL)
    {
        pfree(snapshot->document_tie_break_keys);
    }
    if (snapshot->document_tie_break_order != NULL)
    {
        pfree(snapshot->document_tie_break_order);
    }
    if (snapshot->payloads != NULL)
    {
        for (segment_index = 0;
             segment_index < snapshot->manifest.segment_count;
             segment_index++)
        {
            evoke_segment_payload_free(
                &snapshot->payloads[segment_index]
            );
        }
    }
    if (snapshot->payload_views != NULL)
    {
        pfree(snapshot->payload_views);
    }
    if (snapshot->payloads != NULL)
    {
        pfree(snapshot->payloads);
    }
    evoke_term_directory_free(&snapshot->term_directory);
    evoke_segment_query_contract_free(&snapshot->query_contract);
    evoke_segment_manifest_free(&snapshot->manifest);
    memset(snapshot, 0, sizeof(*snapshot));
}

static void
evoke_segment_pages_report_codec_error(
    const char *object_name,
    evoke_status status
);

static bool evoke_segment_object_ref_is_zero(
    const evoke_segment_object_ref *ref
);

static bool evoke_segment_object_refs_equal(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
);

static void evoke_segment_pages_attach_query_vocabulary(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *catalog_refs,
    evoke_segment_query_contract *contract
);

static void evoke_segment_pages_load_term_folds(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_storage_snapshot *snapshot
);

typedef struct evoke_segment_document_metadata_build
{
    evoke_segment_storage_snapshot *snapshot;
    uint8 *available_document_slots;
    uint64 next_document_slot;
    uint64 visible_document_count;
    uint64 total_document_length;
    size_t retired_capacity;
    evoke_status status;
} evoke_segment_document_metadata_build;

static void
evoke_segment_pages_collect_document_metadata(
    void *context,
    const evoke_document_cow_record *record
)
{
    evoke_segment_document_metadata_build *build = context;
    evoke_segment_storage_snapshot *snapshot;
    const evoke_document_version_record *version;
    uint32 document_id;
    bool aborted;
    bool retired;

    if (build == NULL || record == NULL || build->status != EVOKE_OK)
    {
        return;
    }
    snapshot = build->snapshot;
    version = &record->version;
    if (snapshot == NULL ||
        version->document_slot != build->next_document_slot ||
        version->document_slot >= snapshot->index_metadata.num_docs)
    {
        build->status = EVOKE_ERR_FORMAT;
        return;
    }
    document_id = (uint32) version->document_slot;
    build->next_document_slot++;
    if (version->born_sequence == 0)
    {
        build->status = EVOKE_ERR_FORMAT;
        return;
    }
    snapshot->document_tie_break_keys[document_id] =
        version->born_sequence;
    aborted = (version->flags &
               EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0;
    retired = record->retirement.retirement_sequence != 0;
    snapshot->index_metadata.doc_lengths[document_id] =
        version->document_length;
    if (aborted)
    {
        build->available_document_slots[document_id] = 2;
        return;
    }
    build->available_document_slots[document_id] = 1;
    if (retired)
    {
        uint32 *resized;
        size_t next_capacity;

        if (snapshot->retired_document_count ==
            build->retired_capacity)
        {
            next_capacity = build->retired_capacity == 0
                ? 16
                : build->retired_capacity * 2;
            if (next_capacity < build->retired_capacity ||
                next_capacity > snapshot->index_metadata.num_docs)
            {
                next_capacity = snapshot->index_metadata.num_docs;
            }
            if (next_capacity <= build->retired_capacity ||
                next_capacity > SIZE_MAX / sizeof(*resized))
            {
                build->status = EVOKE_ERR_RANGE;
                return;
            }
            resized = realloc(
                snapshot->retired_document_ids,
                next_capacity * sizeof(*resized)
            );
            if (resized == NULL)
            {
                build->status = EVOKE_ERR_NOMEM;
                return;
            }
            snapshot->retired_document_ids = resized;
            build->retired_capacity = next_capacity;
        }
        snapshot->retired_document_ids[
            snapshot->retired_document_count++
        ] = document_id;
        return;
    }
    if (UINT64_MAX - build->total_document_length <
        version->document_length)
    {
        build->status = EVOKE_ERR_RANGE;
        return;
    }
    ItemPointerSet(
        &snapshot->doc_tids[document_id],
        version->heap_block,
        version->heap_offset
    );
    build->visible_document_count++;
    build->total_document_length += version->document_length;
}

static void
evoke_segment_pages_build_document_metadata(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_storage_snapshot *snapshot
)
{
    evoke_segment_document_metadata_build build;

    memset(&build, 0, sizeof(build));
    build.snapshot = snapshot;
    build.status = EVOKE_OK;
    if (snapshot->index_metadata.num_docs > 0)
    {
        snapshot->doc_tids = palloc0(
            (Size) snapshot->index_metadata.num_docs *
            sizeof(*snapshot->doc_tids)
        );
        snapshot->document_tie_break_keys = palloc0(
            (Size) snapshot->index_metadata.num_docs *
            sizeof(*snapshot->document_tie_break_keys)
        );
        build.available_document_slots = palloc0(
            (Size) snapshot->index_metadata.num_docs *
            sizeof(*build.available_document_slots)
        );
        evoke_segment_pages_visit_document_records(
            index_relation,
            root,
            &snapshot->manifest,
            evoke_segment_pages_collect_document_metadata,
            &build
        );
    }
    if (build.status == EVOKE_OK &&
        (build.next_document_slot !=
             snapshot->index_metadata.num_docs ||
         build.visible_document_count !=
             snapshot->manifest.visible_document_count ||
         build.total_document_length !=
             snapshot->manifest.total_document_length))
    {
        build.status = EVOKE_ERR_FORMAT;
    }
    if (build.status == EVOKE_OK)
    {
        build.status =
            evoke_segment_payload_document_references_validate(
                &snapshot->manifest,
                snapshot->payloads,
                snapshot->manifest.segment_count,
                build.available_document_slots,
                snapshot->index_metadata.num_docs
            );
    }
    if (build.available_document_slots != NULL)
    {
        pfree(build.available_document_slots);
    }
    if (build.status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document metadata",
            build.status
        );
    }
    if (snapshot->retired_document_count == 0)
    {
        free(snapshot->retired_document_ids);
        snapshot->retired_document_ids = NULL;
    }
    else if (snapshot->retired_document_count <
             build.retired_capacity)
    {
        uint32 *resized = realloc(
            snapshot->retired_document_ids,
            (size_t) snapshot->retired_document_count *
                sizeof(*resized)
        );

        if (resized != NULL)
        {
            snapshot->retired_document_ids = resized;
        }
    }
}

static void
evoke_segment_pages_report_codec_error(
    const char *object_name,
    evoke_status status
)
{
    ereport(
        ERROR,
        (errmsg(
            "invalid evoke %s object: %s",
            object_name,
            evoke_strerror(status)
        ))
    );
}

static void
evoke_segment_pages_report_relation_codec_error(
    Relation index_relation,
    const char *object_name,
    evoke_status status
)
{
    const char *namespace_name = NULL;
    const char *relation_name = NULL;

    if (index_relation != NULL)
    {
        namespace_name = get_namespace_name(
            RelationGetNamespace(index_relation)
        );
        relation_name = RelationGetRelationName(index_relation);
    }
    if (namespace_name != NULL && relation_name != NULL)
    {
        ereport(
            ERROR,
            (errmsg(
                "invalid evoke %s object for index %s.%s: %s",
                object_name,
                namespace_name,
                relation_name,
                evoke_strerror(status)
            ))
        );
    }
    evoke_segment_pages_report_codec_error(object_name, status);
}

static void
evoke_segment_pages_add_retired_ranges(
    evoke_block_range_inventory *inventory,
    const evoke_block_range *ranges,
    size_t range_count,
    evoke_status *status_out
)
{
    if (*status_out != EVOKE_OK)
    {
        return;
    }
    if ((range_count == 0) != (ranges == NULL))
    {
        *status_out = EVOKE_ERR_FORMAT;
        return;
    }
    for (size_t index = 0; index < range_count; index++)
    {
        *status_out = evoke_block_range_inventory_add(
            inventory,
            ranges[index].start_block,
            ranges[index].block_count
        );
        if (*status_out != EVOKE_OK)
        {
            return;
        }
    }
}

static void
evoke_segment_pages_set_fsm_handoff(
    const evoke_block_range_inventory *inventory,
    evoke_segment_cow_result *result
)
{
    uint64 block_count = 0;

    if (inventory == NULL || result == NULL || !inventory->finalized ||
        result->fsm_handoff_ranges != NULL ||
        result->fsm_handoff_range_count != 0 ||
        result->fsm_handoff_block_count != 0 ||
        inventory->range_count > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("invalid evoke FSM handoff inventory")));
    }
    for (size_t index = 0; index < inventory->range_count; index++)
    {
        if (block_count > UINT32_MAX -
                inventory->ranges[index].block_count)
        {
            ereport(ERROR, (errmsg("evoke FSM handoff is too large")));
        }
        block_count += inventory->ranges[index].block_count;
    }
    if (inventory->range_count > 0)
    {
        result->fsm_handoff_ranges = calloc(
            inventory->range_count,
            sizeof(*result->fsm_handoff_ranges)
        );
        if (result->fsm_handoff_ranges == NULL)
        {
            ereport(ERROR, (errmsg("out of memory")));
        }
        memcpy(
            result->fsm_handoff_ranges,
            inventory->ranges,
            inventory->range_count * sizeof(*result->fsm_handoff_ranges)
        );
    }
    result->fsm_handoff_range_count = (uint32) inventory->range_count;
    result->fsm_handoff_block_count = (uint32) block_count;
}

static void
evoke_segment_pages_set_staged_writes(
    evoke_segment_page_reuse_arena *arena,
    uint32 published_block_high_watermark,
    evoke_segment_cow_result *result
)
{
    evoke_status status;

    if (arena == NULL || result == NULL ||
        result->staged_write_ranges != NULL ||
        result->staged_write_range_count != 0 ||
        result->staged_write_block_count != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke staged COW inventory")));
    }
    if (arena->staged_writes.range_count == 0)
    {
        return;
    }
    status = evoke_block_range_inventory_finalize(
        &arena->staged_writes,
        published_block_high_watermark
    );
    if (status != EVOKE_OK ||
        arena->staged_writes.range_count > UINT32_MAX ||
        arena->staged_writes.reachable_block_count > UINT32_MAX)
    {
        evoke_segment_pages_report_codec_error(
            "finalize staged COW inventory",
            status == EVOKE_OK ? EVOKE_ERR_RANGE : status
        );
    }
    result->staged_write_ranges = calloc(
        arena->staged_writes.range_count,
        sizeof(*result->staged_write_ranges)
    );
    if (result->staged_write_ranges == NULL)
    {
        ereport(ERROR, (errmsg("out of memory")));
    }
    memcpy(
        result->staged_write_ranges,
        arena->staged_writes.ranges,
        arena->staged_writes.range_count *
            sizeof(*result->staged_write_ranges)
    );
    result->staged_write_range_count =
        (uint32) arena->staged_writes.range_count;
    result->staged_write_block_count =
        (uint32) arena->staged_writes.reachable_block_count;
}

void
evoke_segment_pages_abandon_staged_write(
    Relation index_relation,
    uint64 owner_manifest_id,
    const evoke_segment_cow_result *result
)
{
    evoke_segment_cow_result abandoned;

    if (index_relation == NULL || owner_manifest_id == 0 || result == NULL ||
        (result->staged_write_range_count == 0) !=
            (result->staged_write_ranges == NULL) ||
        (result->staged_write_range_count == 0) !=
            (result->staged_write_block_count == 0))
    {
        ereport(ERROR, (errmsg("invalid abandoned evoke COW write")));
    }
    if (result->staged_write_range_count == 0)
    {
        return;
    }
    evoke_segment_cow_result_init(&abandoned);
    /* Borrow the staging inventory; the original result retains ownership. */
    abandoned.fsm_handoff_ranges = result->staged_write_ranges;
    abandoned.fsm_handoff_range_count =
        result->staged_write_range_count;
    abandoned.fsm_handoff_block_count =
        result->staged_write_block_count;
    evoke_segment_pages_publish_fsm_handoff(
        index_relation,
        owner_manifest_id,
        &abandoned
    );
}

static bool
evoke_segment_pages_prepare_retired_ranges(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    uint32 first_retired_segment,
    uint32 retired_segment_count,
    const evoke_term_cow_tree *term_patch,
    const evoke_document_cow_tree *document_patch,
    const evoke_lexicon_cow_tree *lexicon_patch,
    const evoke_prefix_cow_tree *prefix_patch,
    const evoke_l0_storage_snapshot *retired_l0,
    const evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result,
    evoke_segment_manifest *next_manifest
)
{
    evoke_semantic_accelerator_directory accelerator_directory;
    evoke_block_range_inventory inventory;
    evoke_block_range_inventory handoff;
    const evoke_block_range *carried_ranges;
    size_t carried_range_count;
    uint32 handoff_high_watermark;
    bool reader_fenced;
    evoke_status status;

    if (index_relation == NULL || build_root == NULL ||
        old_manifest == NULL || result == NULL ||
        next_manifest == NULL ||
        first_retired_segment > old_manifest->segment_count ||
        retired_segment_count >
            old_manifest->segment_count - first_retired_segment ||
        result->fsm_handoff_ranges != NULL ||
        result->fsm_handoff_range_count != 0 ||
        result->fsm_handoff_block_count != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke retirement transition")));
    }
    free(next_manifest->retired_ranges);
    next_manifest->retired_ranges = NULL;
    next_manifest->retired_range_count = 0;
    reader_fenced = reuse_arena != NULL && reuse_arena->reader_fenced &&
        reuse_arena->allocator.initialized;
    if (reuse_arena != NULL && !reader_fenced)
    {
        ereport(ERROR, (errmsg("invalid evoke retirement reader fence")));
    }
    if (reader_fenced)
    {
        carried_ranges = NULL;
        carried_range_count = 0;
    }
    else
    {
        carried_ranges = old_manifest->retired_ranges;
        carried_range_count = old_manifest->retired_range_count;
    }

    evoke_block_range_inventory_init(&inventory);
    for (size_t index = 0; index < carried_range_count; index++)
    {
        status = evoke_block_range_inventory_add(
            &inventory,
            carried_ranges[index].start_block,
            carried_ranges[index].block_count
        );
        if (status != EVOKE_OK)
        {
            goto fail;
        }
    }
    status = evoke_block_range_inventory_add(
        &inventory,
        build_root->manifest.start_block,
        build_root->manifest.page_count
    );
    if (status != EVOKE_OK)
    {
        goto fail;
    }
    if ((old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0 &&
        (((next_manifest->flags &
           EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) == 0) ||
         !evoke_segment_object_refs_equal(
             &old_manifest->semantic_accelerator_directory,
             &next_manifest->semantic_accelerator_directory)))
    {
        evoke_semantic_accelerator_directory_init(
            &accelerator_directory
        );
        PG_TRY();
        {
            evoke_segment_pages_load_retired_semantic_accelerator_directory(
                index_relation,
                build_root,
                old_manifest,
                &accelerator_directory
            );
            status = evoke_block_range_inventory_add(
                &inventory,
                old_manifest->semantic_accelerator_directory.start_block,
                old_manifest->semantic_accelerator_directory.page_count
            );
            for (uint32 term_index = 0;
                 status == EVOKE_OK &&
                 term_index < accelerator_directory.term_count;
                 term_index++)
            {
                const evoke_segment_object_ref *term_ref =
                    &accelerator_directory.terms[
                        term_index
                    ].term_object;

                status = evoke_block_range_inventory_add(
                    &inventory,
                    term_ref->start_block,
                    term_ref->page_count
                );
            }
            for (uint32 forward_index = 0;
                 status == EVOKE_OK &&
                 forward_index <
                    accelerator_directory.forward_chunk_count;
                 forward_index++)
            {
                const evoke_segment_object_ref *forward_ref =
                    &accelerator_directory.forward_chunks[
                        forward_index
                    ].forward_object;

                status = evoke_block_range_inventory_add(
                    &inventory,
                    forward_ref->start_block,
                    forward_ref->page_count
                );
            }
            for (uint32 bound_index = 0;
                 status == EVOKE_OK &&
                 bound_index <
                    accelerator_directory.forward_bound_shard_count;
                 bound_index++)
            {
                const evoke_segment_object_ref *bound_ref =
                    &accelerator_directory.forward_bound_shards[bound_index];

                status = evoke_block_range_inventory_add(
                    &inventory,
                    bound_ref->start_block,
                    bound_ref->page_count
                );
            }
            if (status == EVOKE_OK &&
                evoke_semantic_accelerator_directory_has_scope(
                    &accelerator_directory))
            {
                status = evoke_block_range_inventory_add(
                    &inventory,
                    accelerator_directory.scope_object.start_block,
                    accelerator_directory.scope_object.page_count
                );
            }
            if (status == EVOKE_OK &&
                evoke_semantic_accelerator_directory_has_tid_lookup(
                    &accelerator_directory))
            {
                status = evoke_block_range_inventory_add(
                    &inventory,
                    accelerator_directory.tid_lookup_object.start_block,
                    accelerator_directory.tid_lookup_object.page_count
                );
            }
        }
        PG_FINALLY();
        {
            evoke_semantic_accelerator_directory_free(
                &accelerator_directory
            );
        }
        PG_END_TRY();
        if (status != EVOKE_OK)
        {
            goto fail;
        }
    }
    for (uint32 offset = 0; offset < retired_segment_count; offset++)
    {
        const evoke_segment_descriptor *segment =
            &old_manifest->segments[first_retired_segment + offset];

        status = evoke_block_range_inventory_add(
            &inventory,
            segment->start_block,
            segment->block_count
        );
        if (status != EVOKE_OK)
        {
            goto fail;
        }
    }
    if (term_patch != NULL)
    {
        evoke_segment_pages_add_retired_ranges(
            &inventory,
            term_patch->retired_ranges,
            term_patch->retired_range_count,
            &status
        );
    }
    if (document_patch != NULL)
    {
        evoke_segment_pages_add_retired_ranges(
            &inventory,
            document_patch->retired_ranges,
            document_patch->retired_range_count,
            &status
        );
    }
    if (lexicon_patch != NULL)
    {
        evoke_segment_pages_add_retired_ranges(
            &inventory,
            lexicon_patch->retired_ranges,
            lexicon_patch->retired_range_count,
            &status
        );
    }
    if (prefix_patch != NULL)
    {
        evoke_segment_pages_add_retired_ranges(
            &inventory,
            prefix_patch->retired_ranges,
            prefix_patch->retired_range_count,
            &status
        );
    }
    if (retired_l0 != NULL)
    {
        if ((retired_l0->page_count == 0) !=
                (retired_l0->page_blocks == NULL))
        {
            status = EVOKE_ERR_FORMAT;
        }
        for (uint32 index = 0;
             status == EVOKE_OK && index < retired_l0->page_count;
             index++)
        {
            status = evoke_block_range_inventory_add(
                &inventory,
                retired_l0->page_blocks[index],
                1
            );
        }
    }
    if (status != EVOKE_OK)
    {
        goto fail;
    }
    status = evoke_block_range_inventory_finalize(
        &inventory,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        goto fail;
    }
    if (reader_fenced)
    {
        evoke_block_range_inventory_init(&handoff);
        status = EVOKE_OK;
        if (reuse_arena->allocator.range_count > 0)
        {
            evoke_segment_pages_add_retired_ranges(
                &handoff,
                reuse_arena->allocator.ranges,
                reuse_arena->allocator.range_count,
                &status
            );
        }
        evoke_segment_pages_add_retired_ranges(
            &handoff,
            inventory.ranges,
            inventory.range_count,
            &status
        );
        handoff_high_watermark = Max(
            build_root->published_block_high_watermark,
            reuse_arena->source_high_watermark
        );
        if (status == EVOKE_OK)
        {
            status = evoke_block_range_inventory_finalize(
                &handoff,
                handoff_high_watermark
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_block_range_inventory_free(&handoff);
            goto fail;
        }
        evoke_segment_pages_set_fsm_handoff(&handoff, result);
        evoke_block_range_inventory_free(&handoff);
        evoke_block_range_inventory_free(&inventory);
        return true;
    }
    if (inventory.range_count >
        EVOKE_SEGMENT_MANIFEST_MAX_RETIRED_RANGES)
    {
        evoke_block_range_inventory_free(&inventory);
        return false;
    }
    next_manifest->retired_ranges = calloc(
        inventory.range_count,
        sizeof(*next_manifest->retired_ranges)
    );
    if (next_manifest->retired_ranges == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto fail;
    }
    memcpy(
        next_manifest->retired_ranges,
        inventory.ranges,
        inventory.range_count *
            sizeof(*next_manifest->retired_ranges)
    );
    next_manifest->retired_range_count =
        (uint32) inventory.range_count;
    evoke_block_range_inventory_free(&inventory);
    return true;

fail:
    evoke_block_range_inventory_free(&inventory);
    evoke_segment_pages_report_codec_error(
        "retirement transition",
        status
    );
    return false;
}

typedef struct evoke_term_cow_page_loader_context
{
    Relation index_relation;
    evoke_term_cow_object root_object;
    uint32 published_block_high_watermark;
    uint64 max_owner_manifest_id;
    bool root_available;
} evoke_term_cow_page_loader_context;

typedef evoke_segment_query_document_reader
    evoke_document_cow_page_loader_context;

typedef struct evoke_lexicon_cow_page_loader_context
{
    Relation index_relation;
    evoke_lexicon_cow_object root_object;
    uint32 published_block_high_watermark;
    uint64 max_owner_manifest_id;
    bool root_available;
} evoke_lexicon_cow_page_loader_context;

typedef struct evoke_prefix_cow_page_loader_context
{
    Relation index_relation;
    evoke_prefix_cow_object root_object;
    uint32 published_block_high_watermark;
    uint64 max_owner_manifest_id;
    bool root_available;
} evoke_prefix_cow_page_loader_context;

static bool
evoke_segment_pages_lexicon_cow_refs_equal(
    const evoke_lexicon_cow_ref *left,
    const evoke_lexicon_cow_ref *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->blob_checksum == right->blob_checksum;
}

static evoke_status
evoke_segment_pages_load_lexicon_cow_object(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    evoke_lexicon_cow_object *object_out
)
{
    evoke_lexicon_cow_page_loader_context *loader = context;
    evoke_segment_object_ref storage_ref;
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (loader == NULL || loader->index_relation == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (loader->root_available &&
        evoke_segment_pages_lexicon_cow_refs_equal(
            &loader->root_object.ref,
            ref
        ))
    {
        *object_out = loader->root_object;
        loader->root_available = false;
        return EVOKE_OK;
    }

    status = evoke_lexicon_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK ||
        storage_ref.owner_manifest_id >
            loader->max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            &storage_ref,
            loader->published_block_high_watermark) != EVOKE_OK)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    bytes = evoke_segment_pages_read(
        loader->index_relation,
        &storage_ref,
        &size
    );
    status = evoke_lexicon_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_lexicon_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
}

static evoke_status
evoke_segment_pages_load_serialized_lexicon_cow_object(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_lexicon_cow_page_loader_context *loader = context;
    evoke_segment_object_ref storage_ref;
    Size size = 0;
    evoke_status status;

    if (loader == NULL || loader->index_relation == NULL || ref == NULL ||
        bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_lexicon_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK ||
        storage_ref.owner_manifest_id > loader->max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            &storage_ref,
            loader->published_block_high_watermark) != EVOKE_OK)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *bytes_out = evoke_segment_pages_read(
        loader->index_relation,
        &storage_ref,
        &size
    );
    *size_out = size;
    return EVOKE_OK;
}

static void
evoke_segment_pages_release_serialized_lexicon_cow_object(
    void *context,
    uint8_t *bytes
)
{
    (void) context;
    pfree(bytes);
}

static evoke_status
evoke_segment_pages_open_lexicon_cow(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 published_block_high_watermark,
    uint64 max_owner_manifest_id,
    evoke_lexicon_cow_page_loader_context *loader
)
{
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (index_relation == NULL || root_ref == NULL || loader == NULL ||
        root_ref->object_kind != EVOKE_SEGMENT_OBJECT_LEXICON_LOOKUP ||
        root_ref->owner_manifest_id > max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            root_ref,
            published_block_high_watermark) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(loader, 0, sizeof(*loader));
    loader->index_relation = index_relation;
    loader->published_block_high_watermark =
        published_block_high_watermark;
    loader->max_owner_manifest_id = max_owner_manifest_id;
    bytes = evoke_segment_pages_read(index_relation, root_ref, &size);
    status = evoke_lexicon_cow_object_deserialize(
        bytes,
        size,
        &loader->root_object
    );
    pfree(bytes);
    if (status == EVOKE_OK)
    {
        status = evoke_lexicon_cow_object_bind_storage(
            &loader->root_object,
            root_ref
        );
    }
    if (status == EVOKE_OK)
    {
        loader->root_available = true;
    }
    return status;
}

static bool
evoke_segment_pages_prefix_cow_refs_equal(
    const evoke_prefix_cow_ref *left,
    const evoke_prefix_cow_ref *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->blob_checksum == right->blob_checksum &&
        left->term_count == right->term_count &&
        left->reserved2 == right->reserved2;
}

static evoke_status
evoke_segment_pages_load_prefix_cow_object(
    void *context,
    const evoke_prefix_cow_ref *ref,
    evoke_prefix_cow_object *object_out
)
{
    evoke_prefix_cow_page_loader_context *loader = context;
    evoke_segment_object_ref storage_ref;
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (loader == NULL || loader->index_relation == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (loader->root_available &&
        evoke_segment_pages_prefix_cow_refs_equal(
            &loader->root_object.ref,
            ref
        ))
    {
        *object_out = loader->root_object;
        loader->root_available = false;
        return EVOKE_OK;
    }
    status = evoke_prefix_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK ||
        storage_ref.owner_manifest_id >
            loader->max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            &storage_ref,
            loader->published_block_high_watermark) != EVOKE_OK)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    bytes = evoke_segment_pages_read(
        loader->index_relation,
        &storage_ref,
        &size
    );
    status = evoke_prefix_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_prefix_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
}

static evoke_status
evoke_segment_pages_open_prefix_cow(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 published_block_high_watermark,
    uint64 max_owner_manifest_id,
    evoke_prefix_cow_page_loader_context *loader
)
{
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (index_relation == NULL || root_ref == NULL || loader == NULL ||
        root_ref->object_kind != EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP ||
        root_ref->owner_manifest_id > max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            root_ref,
            published_block_high_watermark) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(loader, 0, sizeof(*loader));
    loader->index_relation = index_relation;
    loader->published_block_high_watermark =
        published_block_high_watermark;
    loader->max_owner_manifest_id = max_owner_manifest_id;
    bytes = evoke_segment_pages_read(index_relation, root_ref, &size);
    status = evoke_prefix_cow_object_deserialize(
        bytes,
        size,
        &loader->root_object
    );
    pfree(bytes);
    if (status == EVOKE_OK)
    {
        status = evoke_prefix_cow_object_bind_storage(
            &loader->root_object,
            root_ref
        );
    }
    if (status == EVOKE_OK)
    {
        loader->root_available = true;
    }
    return status;
}

static bool
evoke_segment_pages_document_cow_refs_equal(
    const evoke_document_cow_ref *left,
    const evoke_document_cow_ref *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
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
        left->semantic_pending_count ==
            right->semantic_pending_count &&
        left->earliest_retry_after == right->earliest_retry_after &&
        left->bounded_document_count == right->bounded_document_count &&
        left->min_document_length == right->min_document_length &&
        left->max_document_length == right->max_document_length &&
        left->reusable_document_count ==
            right->reusable_document_count &&
        left->first_reusable_document_slot ==
            right->first_reusable_document_slot &&
        left->min_live_born_sequence ==
            right->min_live_born_sequence;
}

static evoke_status
evoke_segment_pages_load_document_cow_object(
    void *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
)
{
    evoke_document_cow_page_loader_context *loader = context;
    evoke_segment_object_ref storage_ref;
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (loader == NULL || loader->index_relation == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (loader->initialized &&
        evoke_segment_pages_document_cow_refs_equal(
            &loader->root_object.ref,
            ref
        ))
    {
        *object_out = loader->root_object;
        return EVOKE_OK;
    }

    if (ref->kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        for (uint32 cache_index = 0;
             cache_index < EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE;
             cache_index++)
        {
            if (!loader->node_cache_valid[cache_index] ||
                !evoke_segment_pages_document_cow_refs_equal(
                    &loader->node_cache[cache_index].ref,
                    ref
                ))
            {
                continue;
            }
            loader->access_clock++;
            loader->node_cache_age[cache_index] = loader->access_clock;
            *object_out = loader->node_cache[cache_index];
            return EVOKE_OK;
        }
    }

    status = evoke_document_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK ||
        storage_ref.owner_manifest_id >
            loader->max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            &storage_ref,
            loader->published_block_high_watermark) != EVOKE_OK)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    bytes = evoke_segment_pages_read_bounded(
        loader->index_relation,
        &storage_ref,
        &size,
        loader->published_block_high_watermark,
        loader->page_validation_cache
    );
    status = evoke_document_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_document_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
    if (status == EVOKE_OK &&
        object_out->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        uint32 target_index = 0;

        for (uint32 cache_index = 0;
             cache_index < EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE;
             cache_index++)
        {
            if (!loader->node_cache_valid[cache_index])
            {
                target_index = cache_index;
                break;
            }
            if (loader->node_cache_age[cache_index] <
                loader->node_cache_age[target_index])
            {
                target_index = cache_index;
            }
        }
        loader->access_clock++;
        loader->node_cache[target_index] = *object_out;
        loader->node_cache_age[target_index] = loader->access_clock;
        loader->node_cache_valid[target_index] = true;
    }
    return status;
}

static evoke_status
evoke_segment_pages_open_document_cow(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 published_block_high_watermark,
    uint64 max_owner_manifest_id,
    evoke_document_cow_page_loader_context *loader
)
{
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (index_relation == NULL || root_ref == NULL || loader == NULL ||
        root_ref->object_kind !=
            EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY ||
        root_ref->owner_manifest_id > max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            root_ref,
            published_block_high_watermark) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(loader, 0, sizeof(*loader));
    loader->index_relation = index_relation;
    loader->published_block_high_watermark =
        published_block_high_watermark;
    loader->max_owner_manifest_id = max_owner_manifest_id;
    bytes = evoke_segment_pages_read_bounded(
        index_relation,
        root_ref,
        &size,
        published_block_high_watermark,
        NULL
    );
    status = evoke_document_cow_object_deserialize(
        bytes,
        size,
        &loader->root_object
    );
    pfree(bytes);
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_object_bind_storage(
            &loader->root_object,
            root_ref
        );
    }
    if (status == EVOKE_OK)
    {
        loader->initialized = true;
    }
    return status;
}

static bool
evoke_segment_pages_term_cow_refs_equal(
    const evoke_term_cow_ref *left,
    const evoke_term_cow_ref *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->blob_checksum == right->blob_checksum;
}

static evoke_status
evoke_segment_pages_load_term_cow_object(
    void *context,
    const evoke_term_cow_ref *ref,
    evoke_term_cow_object *object_out
)
{
    evoke_term_cow_page_loader_context *loader = context;
    evoke_segment_object_ref storage_ref;
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (loader == NULL || loader->index_relation == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (loader->root_available &&
        evoke_segment_pages_term_cow_refs_equal(
            &loader->root_object.ref,
            ref
        ))
    {
        *object_out = loader->root_object;
        loader->root_available = false;
        return EVOKE_OK;
    }

    status = evoke_term_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK ||
        storage_ref.owner_manifest_id >
            loader->max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            &storage_ref,
            loader->published_block_high_watermark) != EVOKE_OK)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    bytes = evoke_segment_pages_read(
        loader->index_relation,
        &storage_ref,
        &size
    );
    status = evoke_term_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_term_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
}

static evoke_status
evoke_segment_pages_open_term_cow(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 published_block_high_watermark,
    uint64 max_owner_manifest_id,
    evoke_term_cow_page_loader_context *loader
)
{
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (index_relation == NULL || root_ref == NULL || loader == NULL ||
        root_ref->owner_manifest_id > max_owner_manifest_id ||
        evoke_segment_object_ref_validate(
            root_ref,
            published_block_high_watermark) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(loader, 0, sizeof(*loader));
    loader->index_relation = index_relation;
    loader->published_block_high_watermark =
        published_block_high_watermark;
    loader->max_owner_manifest_id = max_owner_manifest_id;
    bytes = evoke_segment_pages_read(
        index_relation,
        root_ref,
        &size
    );
    status = evoke_term_cow_object_deserialize(
        bytes,
        size,
        &loader->root_object
    );
    pfree(bytes);
    if (status == EVOKE_OK)
    {
        status = evoke_term_cow_object_bind_storage(
            &loader->root_object,
            root_ref
        );
    }
    if (status == EVOKE_OK)
    {
        loader->root_available = true;
    }
    return status;
}

static void
evoke_segment_pages_write_term_cow_objects_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_term_cow_tree *tree,
    uint64 first_object_id,
    uint64 owner_manifest_id,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *root_out
)
{
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (index_relation == NULL || tree == NULL ||
        first_object_id == 0 ||
        first_object_id > tree->object_count ||
        owner_manifest_id == 0 || root_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW term-object write")));
    }
    memset(root_out, 0, sizeof(*root_out));

    PG_TRY();
    {
        for (uint64 object_id = first_object_id;
             object_id <= tree->object_count;
             object_id++)
        {
            evoke_segment_object_ref storage_ref;

            if ((object_id & UINT64_C(0x3ff)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            status = evoke_term_cow_tree_prepare_object_for_storage(
                tree,
                object_id,
                &bytes,
                &size
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW term object",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
                object_id,
                owner_manifest_id,
                bytes,
                size,
                reuse_arena,
                &storage_ref
            );
            free(bytes);
            bytes = NULL;
            size = 0;
            status = evoke_term_cow_tree_bind_object_storage(
                tree,
                object_id,
                &storage_ref
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW term-object page binding",
                    status
                );
            }
        }
        status = evoke_term_cow_ref_as_segment_object_ref(
            &tree->root,
            root_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term root",
                status
            );
        }
    }
    PG_FINALLY();
    {
        free(bytes);
    }
    PG_END_TRY();
}

void
evoke_segment_pages_write_term_cow_objects(
    Relation index_relation,
    evoke_term_cow_tree *tree,
    uint64 first_object_id,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
)
{
    evoke_segment_pages_write_term_cow_objects_internal(
        index_relation,
        MAIN_FORKNUM,
        tree,
        first_object_id,
        owner_manifest_id,
        NULL,
        root_out
    );
}

static void
evoke_segment_pages_write_lexicon_cow_objects_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_lexicon_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *root_out
)
{
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (index_relation == NULL || tree == NULL ||
        owner_manifest_id == 0 || root_out == NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW lexical-lookup write"))
        );
    }
    memset(root_out, 0, sizeof(*root_out));

    PG_TRY();
    {
        for (uint64 object_id = 1;
             object_id <= tree->object_count;
             object_id++)
        {
            evoke_segment_object_ref storage_ref;

            if ((object_id & UINT64_C(0x3ff)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            status =
                evoke_lexicon_cow_tree_prepare_object_for_storage(
                    tree,
                    object_id,
                    &bytes,
                    &size
                );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW lexical-lookup object",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_LEXICON_LOOKUP,
                object_id,
                owner_manifest_id,
                bytes,
                size,
                reuse_arena,
                &storage_ref
            );
            free(bytes);
            bytes = NULL;
            size = 0;
            status = evoke_lexicon_cow_tree_bind_object_storage(
                tree,
                object_id,
                &storage_ref
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW lexical-lookup page binding",
                    status
                );
            }
        }
        status = evoke_lexicon_cow_ref_as_segment_object_ref(
            &tree->root,
            root_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW lexical-lookup root",
                status
            );
        }
    }
    PG_FINALLY();
    {
        free(bytes);
    }
    PG_END_TRY();
}

void
evoke_segment_pages_write_lexicon_cow_objects(
    Relation index_relation,
    evoke_lexicon_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
)
{
    evoke_segment_pages_write_lexicon_cow_objects_internal(
        index_relation,
        MAIN_FORKNUM,
        tree,
        owner_manifest_id,
        NULL,
        root_out
    );
}

static void
evoke_segment_pages_write_prefix_cow_objects_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_prefix_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *root_out
)
{
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (index_relation == NULL || tree == NULL ||
        owner_manifest_id == 0 || root_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW prefix-lookup write")));
    }
    memset(root_out, 0, sizeof(*root_out));
    PG_TRY();
    {
        for (uint64 object_id = 1;
             object_id <= tree->object_count;
             object_id++)
        {
            evoke_segment_object_ref storage_ref;

            if ((object_id & UINT64_C(0x3ff)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            status = evoke_prefix_cow_tree_prepare_object_for_storage(
                tree,
                object_id,
                &bytes,
                &size
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW prefix-lookup object",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP,
                object_id,
                owner_manifest_id,
                bytes,
                size,
                reuse_arena,
                &storage_ref
            );
            free(bytes);
            bytes = NULL;
            size = 0;
            status = evoke_prefix_cow_tree_bind_object_storage(
                tree,
                object_id,
                &storage_ref
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW prefix-lookup page binding",
                    status
                );
            }
        }
        status = evoke_prefix_cow_ref_as_segment_object_ref(
            &tree->root,
            root_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW prefix-lookup root",
                status
            );
        }
    }
    PG_FINALLY();
    {
        free(bytes);
    }
    PG_END_TRY();
}

bool
evoke_segment_pages_lookup_lexicon(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const uint8 *bytes,
    Size bytes_len,
    uint32 *term_id_out
)
{
    evoke_lexicon_cow_page_loader_context loader;
    bool found = false;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        bytes == NULL || bytes_len == 0 || term_id_out == NULL ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW lexical lookup")));
    }
    status = evoke_segment_pages_open_lexicon_cow(
        index_relation,
        &manifest->lexicon_lookup,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_lexicon_cow_lookup_serialized_external(
            &loader.root_object,
            manifest->lexicon_hash_seed,
            bytes,
            bytes_len,
            evoke_segment_pages_load_serialized_lexicon_cow_object,
            evoke_segment_pages_release_serialized_lexicon_cow_object,
            &loader,
            term_id_out,
            &found
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW lexical lookup",
            status
        );
    }
    return found;
}

typedef struct evoke_segment_lexicon_prefix_context
{
    const uint8 *prefix;
    Size prefix_len;
    Size max_matches;
    Size max_token_bytes;
    Size match_count;
    Size match_capacity;
    Size token_bytes;
    evoke_segment_lexicon_prefix_match *matches;
    bool limit_exceeded;
} evoke_segment_lexicon_prefix_context;

static evoke_status
evoke_segment_pages_collect_lexicon_prefix_entry(
    void *context,
    const evoke_prefix_cow_entry *entry
)
{
    evoke_segment_lexicon_prefix_context *prefix_context = context;
    evoke_segment_lexicon_prefix_match *match;
    Size next_capacity;
    Size next_token_bytes;

    if (prefix_context == NULL || entry == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (entry->bytes_len < prefix_context->prefix_len ||
        memcmp(
            entry->bytes,
            prefix_context->prefix,
            prefix_context->prefix_len
        ) != 0)
    {
        return EVOKE_OK;
    }
    if (prefix_context->match_count >= prefix_context->max_matches ||
        prefix_context->token_bytes >= prefix_context->max_token_bytes ||
        (Size) entry->bytes_len >
            prefix_context->max_token_bytes -
                prefix_context->token_bytes - 1)
    {
        prefix_context->limit_exceeded = true;
        return EVOKE_ERR_RANGE;
    }
    next_token_bytes = prefix_context->token_bytes + entry->bytes_len + 1;
    if (next_token_bytes > prefix_context->max_token_bytes)
    {
        prefix_context->limit_exceeded = true;
        return EVOKE_ERR_RANGE;
    }
    if (prefix_context->match_count == prefix_context->match_capacity)
    {
        next_capacity = prefix_context->match_capacity == 0
            ? Min(prefix_context->max_matches, (Size) 16)
            : Min(
                prefix_context->max_matches,
                prefix_context->match_capacity * 2
            );
        if (next_capacity <= prefix_context->match_capacity ||
            next_capacity >
                MaxAllocSize / sizeof(*prefix_context->matches))
        {
            prefix_context->limit_exceeded = true;
            return EVOKE_ERR_RANGE;
        }
        prefix_context->matches = prefix_context->matches == NULL
            ? palloc(next_capacity * sizeof(*prefix_context->matches))
            : repalloc(
                prefix_context->matches,
                next_capacity * sizeof(*prefix_context->matches)
            );
        prefix_context->match_capacity = next_capacity;
    }
    match = &prefix_context->matches[prefix_context->match_count++];
    memset(match, 0, sizeof(*match));
    match->term_id = entry->term_id;
    match->token_len = entry->bytes_len;
    match->token = palloc((Size) entry->bytes_len + 1);
    memcpy(match->token, entry->bytes, entry->bytes_len);
    match->token[entry->bytes_len] = '\0';
    prefix_context->token_bytes = next_token_bytes;
    return EVOKE_OK;
}

static int
evoke_segment_pages_compare_lexicon_prefix_match(
    const void *left_pointer,
    const void *right_pointer
)
{
    const evoke_segment_lexicon_prefix_match *left = left_pointer;
    const evoke_segment_lexicon_prefix_match *right = right_pointer;
    Size common = Min((Size) left->token_len, (Size) right->token_len);
    int compared = memcmp(left->token, right->token, common);

    if (compared != 0)
    {
        return compared;
    }
    if (left->token_len != right->token_len)
    {
        return left->token_len < right->token_len ? -1 : 1;
    }
    if (left->term_id == right->term_id)
    {
        return 0;
    }
    return left->term_id < right->term_id ? -1 : 1;
}

void
evoke_segment_pages_free_lexicon_prefix_matches(
    evoke_segment_lexicon_prefix_match *matches,
    Size match_count
)
{
    if (matches == NULL)
    {
        return;
    }
    for (Size index = 0; index < match_count; index++)
    {
        if (matches[index].token != NULL)
        {
            pfree(matches[index].token);
        }
    }
    pfree(matches);
}

void
evoke_segment_pages_collect_lexicon_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const uint8 *prefix,
    Size prefix_len,
    Size max_matches,
    Size max_token_bytes,
    evoke_segment_lexicon_prefix_match **matches_out,
    Size *match_count_out
)
{
    evoke_prefix_cow_page_loader_context loader;
    evoke_segment_lexicon_prefix_context context = {0};
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        prefix == NULL || prefix_len == 0 || max_matches == 0 ||
        max_token_bytes == 0 || matches_out == NULL ||
        match_count_out == NULL ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW lexical prefix lookup")));
    }
    *matches_out = NULL;
    *match_count_out = 0;
    context.prefix = prefix;
    context.prefix_len = prefix_len;
    context.max_matches = max_matches;
    context.max_token_bytes = max_token_bytes;
    status = evoke_segment_pages_open_prefix_cow(
        index_relation,
        &manifest->prefix_lookup,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_prefix_cow_scan_prefix_external(
            &loader.root_object.ref,
            manifest->vocab_size,
            prefix,
            prefix_len,
            evoke_segment_pages_load_prefix_cow_object,
            &loader,
            evoke_segment_pages_collect_lexicon_prefix_entry,
            &context
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_free_lexicon_prefix_matches(
            context.matches,
            context.match_count
        );
        if (context.limit_exceeded)
        {
            ereport(
                ERROR,
                (
                    errmsg("evoke prefix expansion exceeds its query budget"),
                    errdetail(
                        "The prefix matched more than %zu terms or %zu bytes.",
                        max_matches,
                        max_token_bytes
                    ),
                    errhint("Use a longer prefix or an exact term query.")
                )
            );
        }
        evoke_segment_pages_report_codec_error(
            "COW lexical prefix lookup",
            status
        );
    }
    if (context.match_count > 1)
    {
        qsort(
            context.matches,
            context.match_count,
            sizeof(*context.matches),
            evoke_segment_pages_compare_lexicon_prefix_match
        );
    }
    *matches_out = context.matches;
    *match_count_out = context.match_count;
}

void
evoke_segment_pages_load_term_cow_record(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 vocab_size,
    uint32 term_id,
    evoke_term_cow_record *record_out
)
{
    evoke_term_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root_ref == NULL ||
        record_out == NULL || vocab_size == 0 ||
        term_id >= vocab_size)
    {
        ereport(ERROR, (errmsg("invalid evoke COW term lookup")));
    }
    status = evoke_segment_pages_open_term_cow(
        index_relation,
        root_ref,
        UINT32_MAX,
        root_ref->owner_manifest_id,
        &loader
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW term root",
            status
        );
    }
    status = evoke_term_cow_lookup_external(
        &loader.root_object.ref,
        vocab_size,
        term_id,
        evoke_segment_pages_load_term_cow_object,
        &loader,
        record_out
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW term lookup",
            status
        );
    }
}

void
evoke_segment_query_term_init(evoke_segment_query_term *term)
{
    if (term != NULL)
    {
        memset(term, 0, sizeof(*term));
    }
}

void
evoke_segment_query_term_free(evoke_segment_query_term *term)
{
    if (term == NULL)
    {
        return;
    }
    for (uint32 extent_index = 0;
         extent_index < term->extent_count;
         extent_index++)
    {
        evoke_segment_query_extent *extent =
            &term->extents[extent_index];

        if (extent->document_slots != NULL)
        {
            pfree(extent->document_slots);
        }
        if (extent->values != NULL)
        {
            pfree(extent->values);
        }
        if (extent->blocks != NULL)
        {
            pfree(extent->blocks);
        }
    }
    if (term->extents != NULL)
    {
        pfree(term->extents);
    }
    memset(term, 0, sizeof(*term));
}

static evoke_segment_query_extent *
evoke_segment_query_term_add_extent(evoke_segment_query_term *term)
{
    evoke_segment_query_extent *extent;

    if (term->extent_count == term->extent_capacity)
    {
        uint32 next_capacity = term->extent_capacity == 0
            ? 4
            : term->extent_capacity * 2;
        Size allocation_size;

        if (next_capacity < term->extent_capacity ||
            next_capacity >
                MaxAllocSize / sizeof(*term->extents))
        {
            ereport(ERROR, (errmsg("evoke query term has too many extents")));
        }
        allocation_size =
            (Size) next_capacity * sizeof(*term->extents);
        if (term->extents == NULL)
        {
            term->extents = palloc0(allocation_size);
        }
        else
        {
            term->extents = repalloc(term->extents, allocation_size);
            memset(
                &term->extents[term->extent_capacity],
                0,
                (Size) (next_capacity - term->extent_capacity) *
                    sizeof(*term->extents)
            );
        }
        term->extent_capacity = next_capacity;
    }
    extent = &term->extents[term->extent_count++];
    memset(extent, 0, sizeof(*extent));
    return extent;
}

static const evoke_segment_descriptor *
evoke_segment_pages_find_descriptor(
    const evoke_segment_manifest *manifest,
    uint64 segment_id
)
{
    for (uint32 segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (manifest->segments[segment_index].segment_id == segment_id)
        {
            return &manifest->segments[segment_index];
        }
    }
    return NULL;
}

static evoke_status
evoke_segment_pages_read_payload_header(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *descriptor,
    evoke_segment_object_ref *ref_out,
    evoke_segment_payload_disk_header *header_out,
    const evoke_segment_query_context *query_context
)
{
    uint8 bytes[EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE];
    evoke_segment_object_ref ref;
    evoke_status status;

    status = evoke_segment_descriptor_payload_ref(
        manifest,
        descriptor,
        &ref
    );
    if (status != EVOKE_OK ||
        ref.owner_manifest_id > manifest->manifest_id ||
        evoke_segment_object_ref_validate(
            &ref,
            root->published_block_high_watermark
        ) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_segment_pages_read_plan_range(
        index_relation,
        query_context,
        &ref,
        0,
        sizeof(bytes),
        bytes
    );
    status = evoke_segment_payload_disk_header_decode(
        bytes,
        sizeof(bytes),
        manifest,
        descriptor,
        header_out
    );
    if (status != EVOKE_OK || header_out->total_size != ref.object_bytes)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_find_payload_run(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    const evoke_segment_payload_disk_header *header,
    uint32 term_id,
    evoke_posting_extent_kind kind,
    evoke_segment_term_run *run_out,
    const evoke_segment_query_context *query_context
)
{
    uint32 low = 0;
    uint32 high = header->run_count;
    uint8 bytes[EVOKE_SEGMENT_TERM_RUN_SIZE];

    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;
        evoke_segment_term_run run;
        evoke_status status;

        evoke_segment_pages_read_plan_range(
            index_relation,
            query_context,
            ref,
            (Size) header->runs_offset +
                (Size) middle * EVOKE_SEGMENT_TERM_RUN_SIZE,
            sizeof(bytes),
            bytes
        );
        status = evoke_segment_term_run_decode(
            bytes,
            sizeof(bytes),
            &run
        );
        if (status != EVOKE_OK ||
            run.term_id >= header->vocab_size ||
            run.posting_offset > header->posting_count ||
            run.posting_count >
                header->posting_count - run.posting_offset ||
            run.block_offset > header->block_count ||
            run.block_count > header->block_count - run.block_offset)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        if (run.term_id < term_id ||
            (run.term_id == term_id && run.kind < kind))
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= header->run_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_segment_pages_read_plan_range(
        index_relation,
        query_context,
        ref,
        (Size) header->runs_offset +
            (Size) low * EVOKE_SEGMENT_TERM_RUN_SIZE,
        sizeof(bytes),
        bytes
    );
    if (evoke_segment_term_run_decode(
            bytes,
            sizeof(bytes),
            run_out
        ) != EVOKE_OK ||
        run_out->term_id != term_id || run_out->kind != kind ||
        run_out->posting_offset > header->posting_count ||
        run_out->posting_count >
            header->posting_count - run_out->posting_offset ||
        run_out->block_offset > header->block_count ||
        run_out->block_count >
            header->block_count - run_out->block_offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_read_extent_blocks(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    uint64 blocks_offset,
    uint64 first_block,
    uint32 block_count,
    evoke_segment_query_extent *extent
)
{
    uint8 *bytes;
    Size byte_count;

    if (block_count == 0 ||
        block_count > MaxAllocSize / EVOKE_POSTING_BLOCK_RECORD_SIZE)
    {
        return EVOKE_ERR_RANGE;
    }
    byte_count =
        (Size) block_count * EVOKE_POSTING_BLOCK_RECORD_SIZE;
    if (first_block > (UINT64_MAX - blocks_offset) /
            EVOKE_POSTING_BLOCK_RECORD_SIZE)
    {
        return EVOKE_ERR_RANGE;
    }
    extent->blocks = palloc0(
        (Size) block_count * sizeof(*extent->blocks)
    );
    bytes = palloc(byte_count);
    evoke_segment_pages_read_range(
        index_relation,
        ref,
        (Size) (blocks_offset +
            first_block * EVOKE_POSTING_BLOCK_RECORD_SIZE),
        byte_count,
        bytes
    );
    for (uint32 block_index = 0;
         block_index < block_count;
         block_index++)
    {
        evoke_status status = evoke_posting_block_record_decode(
            bytes +
                (Size) block_index *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
            EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &extent->blocks[block_index]
        );

        if (status != EVOKE_OK)
        {
            pfree(bytes);
            return status;
        }
    }
    pfree(bytes);
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_globalize_payload_slots(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    const evoke_segment_payload_disk_header *header,
    uint32 document_slot_count,
    uint32 *document_slots,
    uint64 posting_count,
    BlockNumber block_limit,
    evoke_segment_query_page_validation_cache *validation_cache
)
{
    uint32 prior_document_slot = 0;

    for (uint64 posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        uint32 local_document_id = document_slots[posting_index];

        if (local_document_id >= header->local_document_count ||
            (posting_index > 0 &&
             document_slots[posting_index - 1] >= local_document_id))
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    if ((header->flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) == 0)
    {
        for (uint64 posting_index = 0;
             posting_index < posting_count;
             posting_index++)
        {
            uint64 global_document_id =
                (uint64) header->document_id_base +
                document_slots[posting_index];

            if (global_document_id >= document_slot_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            document_slots[posting_index] =
                (uint32) global_document_id;
        }
    }
    else
    {
        uint64 posting_index = 0;
        Size page_capacity = evoke_segment_page_payload_capacity();

        if (header->document_map_offset == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        while (posting_index < posting_count)
        {
            uint64 group_end = posting_index + 1;
            uint32 first_local = document_slots[posting_index];
            uint32 last_local = first_local;
            uint64 first_offset = header->document_map_offset +
                (uint64) first_local * sizeof(uint32);
            uint64 page_ordinal = first_offset / page_capacity;
            uint8 *map_bytes;
            Size map_byte_count;

            while (group_end < posting_count)
            {
                uint32 next_local = document_slots[group_end];
                uint64 next_offset = header->document_map_offset +
                    (uint64) next_local * sizeof(uint32);

                if (next_offset / page_capacity != page_ordinal)
                {
                    break;
                }
                last_local = next_local;
                group_end++;
            }
            map_byte_count =
                (Size) (last_local - first_local + 1) *
                sizeof(uint32);
            map_bytes = palloc(map_byte_count);
            evoke_segment_pages_read_range_bounded(
                index_relation,
                ref,
                (Size) first_offset,
                map_byte_count,
                map_bytes,
                block_limit,
                validation_cache
            );
            for (uint64 group_index = posting_index;
                 group_index < group_end;
                 group_index++)
            {
                uint32 local_document_id =
                    document_slots[group_index];
                uint32 global_document_id =
                    evoke_segment_pages_read_u32_le(
                        map_bytes +
                            (Size) (local_document_id - first_local) *
                                sizeof(uint32)
                    );

                if (global_document_id >= document_slot_count ||
                    (group_index > 0 &&
                     prior_document_slot >= global_document_id))
                {
                    pfree(map_bytes);
                    return EVOKE_ERR_FORMAT;
                }
                document_slots[group_index] = global_document_id;
                prior_document_slot = global_document_id;
            }
            pfree(map_bytes);
            posting_index = group_end;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_load_payload_extent(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    const evoke_segment_payload_disk_header *header,
    const evoke_segment_term_run *run,
    uint32 document_slot_count,
    evoke_segment_query_term *term
)
{
    evoke_segment_query_extent *extent;
    evoke_index index_shape;
    Size posting_bytes;
    evoke_status status;

    if (run->posting_count > MaxAllocSize / sizeof(uint32) ||
        run->posting_count > MaxAllocSize /
            sizeof(evoke_posting_value) ||
        document_slot_count == 0)
    {
        return EVOKE_ERR_RANGE;
    }
    posting_bytes = (Size) run->posting_count * sizeof(uint32);
    extent = evoke_segment_query_term_add_extent(term);
    extent->document_slots = palloc(posting_bytes);
    extent->values = palloc(posting_bytes);
    evoke_segment_pages_read_range(
        index_relation,
        ref,
        (Size) (header->indices_offset +
            run->posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) extent->document_slots
    );
    evoke_segment_pages_read_range(
        index_relation,
        ref,
        (Size) (header->values_offset +
            run->posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) extent->values
    );
    for (uint64 posting_index = 0;
         posting_index < run->posting_count;
         posting_index++)
    {
        uint8 *value_bytes =
            (uint8 *) extent->values +
            (Size) posting_index * sizeof(uint32);
        uint32 value_bits;

        extent->document_slots[posting_index] =
            evoke_segment_pages_read_u32_le(
                (uint8 *) extent->document_slots +
                    (Size) posting_index * sizeof(uint32)
            );
        value_bits = evoke_segment_pages_read_u32_le(value_bytes);
        if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            extent->values[posting_index].term_frequency = value_bits;
            if (value_bits == 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        else
        {
            memcpy(
                &extent->values[posting_index].impact,
                &value_bits,
                sizeof(value_bits)
            );
            if (!isfinite(extent->values[posting_index].impact))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    status = evoke_segment_pages_globalize_payload_slots(
        index_relation,
        ref,
        header,
        document_slot_count,
        extent->document_slots,
        run->posting_count,
        RelationGetNumberOfBlocks(index_relation),
        NULL
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_pages_read_extent_blocks(
        index_relation,
        ref,
        header->blocks_offset,
        run->block_offset,
        run->block_count,
        extent
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    extent->view.indices = extent->document_slots;
    extent->view.values = extent->values;
    extent->view.blocks = extent->blocks;
    extent->view.len = run->posting_count;
    extent->view.local_document_count = document_slot_count;
    extent->view.block_count = run->block_count;
    extent->view.block_shift = header->block_shift;
    extent->view.kind = run->kind;
    memset(&index_shape, 0, sizeof(index_shape));
    index_shape.num_docs = document_slot_count;
    return evoke_posting_extent_validate_layout(
        &index_shape,
        &extent->view
    );
}

static evoke_status
evoke_segment_pages_read_fold_header(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_segment_object_kind expected_kind,
    evoke_term_fold_disk_header *header_out,
    const evoke_segment_query_context *query_context
)
{
    uint8 bytes[EVOKE_TERM_FOLD_HEADER_SIZE];
    evoke_status status;

    if (ref->object_kind != expected_kind ||
        ref->owner_manifest_id > manifest->manifest_id ||
        evoke_segment_object_ref_validate(
            ref,
            root->published_block_high_watermark
        ) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_segment_pages_read_plan_range(
        index_relation,
        query_context,
        ref,
        0,
        sizeof(bytes),
        bytes
    );
    status = evoke_term_fold_disk_header_decode(
        bytes,
        sizeof(bytes),
        header_out
    );
    if (status != EVOKE_OK ||
        header_out->object_kind != expected_kind ||
        header_out->owner_manifest_id != ref->owner_manifest_id ||
        header_out->total_size != ref->object_bytes)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_pages_query_authority_format_status(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    evoke_term_fold_disk_header fold_header;
    evoke_term_cow_record term_record;
    evoke_segment_payload_disk_header payload_header;
    evoke_segment_object_ref payload_ref;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_read_root_validate(root);
    if (status == EVOKE_OK)
    {
        status = evoke_segment_manifest_validate_published(
            manifest,
            &root->manifest,
            root->published_block_high_watermark
        );
    }
    if (status != EVOKE_OK)
    {
        return status;
    }
    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD) != 0)
    {
        status = evoke_segment_pages_read_fold_header(
            index_relation,
            root,
            manifest,
            &manifest->neutral_fold,
            EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
            &fold_header,
            NULL
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    if ((manifest->flags & EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD) != 0)
    {
        status = evoke_segment_pages_read_fold_header(
            index_relation,
            root,
            manifest,
            &manifest->impact_fold,
            EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
            &fold_header,
            NULL
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0)
    {
        for (uint32 term_id = 0; term_id < manifest->vocab_size; term_id++)
        {
            memset(&term_record, 0, sizeof(term_record));
            evoke_segment_pages_load_term_cow_record(
                index_relation,
                &manifest->term_directory,
                manifest->vocab_size,
                term_id,
                &term_record
            );
            if (term_record.neutral_fold_coverage == 0 &&
                term_record.neutral_minor_fold_coverage == 0)
            {
                continue;
            }
            status = evoke_segment_pages_read_fold_header(
                index_relation,
                root,
                manifest,
                term_record.neutral_minor_fold_coverage != 0
                    ? &term_record.neutral_minor_fold
                    : &term_record.neutral_fold,
                EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                &fold_header,
                NULL
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            if (term_record.impact_fold_coverage != 0)
            {
                status = evoke_segment_pages_read_fold_header(
                    index_relation,
                    root,
                    manifest,
                    &term_record.impact_fold,
                    EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
                    &fold_header,
                    NULL
                );
                if (status != EVOKE_OK)
                {
                    return status;
                }
            }
            break;
        }
    }
    for (uint32 segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        status = evoke_segment_pages_read_payload_header(
            index_relation,
            root,
            manifest,
            &manifest->segments[segment_index],
            &payload_ref,
            &payload_header,
            NULL
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_find_fold_runs(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    const evoke_term_fold_disk_header *header,
    uint32 term_id,
    uint64 coverage_sequence,
    evoke_term_fold_run *runs_out,
    uint32 run_capacity,
    uint32 *run_count_out,
    const evoke_segment_query_context *query_context
)
{
    uint32 low = 0;
    uint32 high = header->run_count;
    uint8 bytes[EVOKE_TERM_FOLD_RUN_SIZE];

    *run_count_out = 0;
    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;
        evoke_term_fold_run run;
        evoke_status status;

        evoke_segment_pages_read_plan_range(
            index_relation,
            query_context,
            ref,
            (Size) header->runs_offset +
                (Size) middle * EVOKE_TERM_FOLD_RUN_SIZE,
            sizeof(bytes),
            bytes
        );
        status = evoke_term_fold_run_decode(
            bytes,
            sizeof(bytes),
            &run
        );
        if (status != EVOKE_OK ||
            run.posting_offset > header->posting_count ||
            run.posting_count >
                header->posting_count - run.posting_offset ||
            run.block_offset > header->block_count ||
            run.block_count > header->block_count - run.block_offset)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        if (run.term_id < term_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    while (low < header->run_count)
    {
        evoke_term_fold_run run;
        evoke_status status;

        evoke_segment_pages_read_plan_range(
            index_relation,
            query_context,
            ref,
            (Size) header->runs_offset +
                (Size) low * EVOKE_TERM_FOLD_RUN_SIZE,
            sizeof(bytes),
            bytes
        );
        status = evoke_term_fold_run_decode(
            bytes,
            sizeof(bytes),
            &run
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (run.term_id != term_id)
        {
            break;
        }
        if (run.coverage_sequence != coverage_sequence ||
            run.posting_offset > header->posting_count ||
            run.posting_count >
                header->posting_count - run.posting_offset ||
            run.block_offset > header->block_count ||
            run.block_count > header->block_count - run.block_offset ||
            *run_count_out >= run_capacity)
        {
            return EVOKE_ERR_FORMAT;
        }
        runs_out[(*run_count_out)++] = run;
        low++;
    }
    return *run_count_out == 0 ? EVOKE_ERR_FORMAT : EVOKE_OK;
}

static evoke_status
evoke_segment_pages_load_fold_extent(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    const evoke_term_fold_disk_header *header,
    const evoke_term_fold_run *run,
    uint32 document_slot_count,
    evoke_segment_query_term *term
)
{
    evoke_segment_query_extent *extent;
    evoke_index index_shape;
    Size posting_bytes;
    uint32 prior_document_slot = 0;
    evoke_status status;

    if (run->posting_count > MaxAllocSize / sizeof(uint32) ||
        run->posting_count > MaxAllocSize /
            sizeof(evoke_posting_value) ||
        document_slot_count == 0)
    {
        return EVOKE_ERR_RANGE;
    }
    posting_bytes = (Size) run->posting_count * sizeof(uint32);
    extent = evoke_segment_query_term_add_extent(term);
    extent->document_slots = palloc(posting_bytes);
    extent->values = palloc(posting_bytes);
    evoke_segment_pages_read_range(
        index_relation,
        ref,
        (Size) (header->document_slots_offset +
            run->posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) extent->document_slots
    );
    evoke_segment_pages_read_range(
        index_relation,
        ref,
        (Size) (header->values_offset +
            run->posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) extent->values
    );
    for (uint64 posting_index = 0;
         posting_index < run->posting_count;
         posting_index++)
    {
        uint8 *value_bytes =
            (uint8 *) extent->values +
            (Size) posting_index * sizeof(uint32);
        uint32 document_slot = evoke_segment_pages_read_u32_le(
            (uint8 *) extent->document_slots +
                (Size) posting_index * sizeof(uint32)
        );
        uint32 value_bits =
            evoke_segment_pages_read_u32_le(value_bytes);

        if (document_slot >= document_slot_count ||
            (posting_index > 0 &&
             prior_document_slot >= document_slot))
        {
            return EVOKE_ERR_FORMAT;
        }
        extent->document_slots[posting_index] = document_slot;
        prior_document_slot = document_slot;
        if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            extent->values[posting_index].term_frequency = value_bits;
            if (value_bits == 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        else
        {
            memcpy(
                &extent->values[posting_index].impact,
                &value_bits,
                sizeof(value_bits)
            );
            if (!isfinite(extent->values[posting_index].impact))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    status = evoke_segment_pages_read_extent_blocks(
        index_relation,
        ref,
        header->blocks_offset,
        run->block_offset,
        run->block_count,
        extent
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    extent->view.indices = extent->document_slots;
    extent->view.values = extent->values;
    extent->view.blocks = extent->blocks;
    extent->view.len = run->posting_count;
    extent->view.local_document_count = document_slot_count;
    extent->view.block_count = run->block_count;
    extent->view.block_shift = header->block_shift;
    extent->view.kind = run->kind;
    memset(&index_shape, 0, sizeof(index_shape));
    index_shape.num_docs = document_slot_count;
    return evoke_posting_extent_validate_layout(
        &index_shape,
        &extent->view
    );
}

void
evoke_segment_pages_load_query_term(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 term_id,
    evoke_segment_query_term *term_out
)
{
    evoke_segment_query_term term;
    evoke_term_cow_record record;
    uint64 effective_coverage;
    bool use_impact;
    bool lexical_neutral_seen = false;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        term_out == NULL || term_id >= manifest->vocab_size ||
        manifest->document_slot_count > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term load")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status == EVOKE_OK)
    {
        status = evoke_segment_manifest_validate_published(
            manifest,
            &root->manifest,
            root->published_block_high_watermark
        );
    }
    if (status != EVOKE_OK ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0)
    {
        evoke_segment_pages_report_codec_error(
            "query-term root",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }

    evoke_segment_query_term_init(&term);
    memset(&record, 0, sizeof(record));
    PG_TRY();
    {
        evoke_segment_pages_load_term_cow_record(
            index_relation,
            &manifest->term_directory,
            manifest->vocab_size,
            term_id,
            &record
        );
        if (record.term_id != term_id)
        {
            evoke_segment_pages_report_codec_error(
                "query-term record",
                EVOKE_ERR_FORMAT
            );
        }
        term.term_id = term_id;
        term.raw_document_frequency = record.raw_document_frequency;
        effective_coverage = record.neutral_minor_fold_coverage != 0
            ? record.neutral_minor_fold_coverage
            : record.neutral_fold_coverage;
        use_impact = root->active_l0.record_count == 0 &&
            root->pending_l0.record_count == 0 &&
            effective_coverage != 0 &&
            record.impact_fold_coverage == effective_coverage &&
            record.impact_statistics_epoch ==
                manifest->statistics_epoch;

        for (uint32 fold_level = 0; fold_level < 2; fold_level++)
        {
            const evoke_segment_object_ref *ref = fold_level == 0
                ? &record.neutral_fold
                : &record.neutral_minor_fold;
            uint64 coverage = fold_level == 0
                ? record.neutral_fold_coverage
                : record.neutral_minor_fold_coverage;
            evoke_term_fold_disk_header header;
            evoke_term_fold_run runs[2];
            uint32 run_count = 0;

            if (coverage == 0)
            {
                continue;
            }
            status = evoke_segment_pages_read_fold_header(
                index_relation,
                root,
                manifest,
                ref,
                EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                &header,
                NULL
            );
            if (status == EVOKE_OK)
            {
                status = evoke_segment_pages_find_fold_runs(
                    index_relation,
                    ref,
                    &header,
                    term_id,
                    coverage,
                    runs,
                    lengthof(runs),
                    &run_count,
                    NULL
                );
            }
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term neutral fold",
                    status
                );
            }
            for (uint32 run_index = 0;
                 run_index < run_count;
                 run_index++)
            {
                const evoke_term_fold_run *run = &runs[run_index];

                if (run->kind ==
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    lexical_neutral_seen = true;
                    if (use_impact)
                    {
                        continue;
                    }
                }
                else if (run->kind !=
                         EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
                {
                    evoke_segment_pages_report_codec_error(
                        "query-term neutral fold kind",
                        EVOKE_ERR_FORMAT
                    );
                }
                status = evoke_segment_pages_load_fold_extent(
                    index_relation,
                    ref,
                    &header,
                    run,
                    (uint32) manifest->document_slot_count,
                    &term
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "query-term neutral postings",
                        status
                    );
                }
            }
        }

        if (use_impact)
        {
            evoke_term_fold_disk_header header;
            evoke_term_fold_run run;
            uint32 run_count = 0;

            if (!lexical_neutral_seen)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term impact replacement",
                    EVOKE_ERR_FORMAT
                );
            }
            status = evoke_segment_pages_read_fold_header(
                index_relation,
                root,
                manifest,
                &record.impact_fold,
                EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
                &header,
                NULL
            );
            if (status == EVOKE_OK &&
                header.statistics_epoch !=
                    record.impact_statistics_epoch)
            {
                status = EVOKE_ERR_FORMAT;
            }
            if (status == EVOKE_OK)
            {
                status = evoke_segment_pages_find_fold_runs(
                    index_relation,
                    &record.impact_fold,
                    &header,
                    term_id,
                    record.impact_fold_coverage,
                    &run,
                    1,
                    &run_count,
                    NULL
                );
            }
            if (status != EVOKE_OK || run_count != 1 ||
                run.kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term impact fold",
                    status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
                );
            }
            status = evoke_segment_pages_load_fold_extent(
                index_relation,
                &record.impact_fold,
                &header,
                &run,
                (uint32) manifest->document_slot_count,
                &term
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term impact postings",
                    status
                );
            }
        }

        for (uint32 extent_index = 0;
             extent_index < record.extent_count;
             extent_index++)
        {
            const evoke_stable_term_extent *stable_extent =
                &record.extents[extent_index];
            const evoke_segment_descriptor *descriptor =
                evoke_segment_pages_find_descriptor(
                    manifest,
                    stable_extent->segment_id
                );
            evoke_segment_object_ref ref;
            evoke_segment_payload_disk_header header;
            evoke_segment_term_run run;

            if (descriptor == NULL ||
                (effective_coverage != 0 &&
                 descriptor->min_sequence <= effective_coverage))
            {
                evoke_segment_pages_report_codec_error(
                    "query-term tail extent",
                    EVOKE_ERR_FORMAT
                );
            }
            status = evoke_segment_pages_read_payload_header(
                index_relation,
                root,
                manifest,
                descriptor,
                &ref,
                &header,
                NULL
            );
            if (status == EVOKE_OK)
            {
                status = evoke_segment_pages_find_payload_run(
                    index_relation,
                    &ref,
                    &header,
                    term_id,
                    stable_extent->kind,
                    &run,
                    NULL
                );
            }
            if (status != EVOKE_OK ||
                run.posting_count != stable_extent->posting_count)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term payload run",
                    status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
                );
            }
            status = evoke_segment_pages_load_payload_extent(
                index_relation,
                &ref,
                &header,
                &run,
                (uint32) manifest->document_slot_count,
                &term
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term payload postings",
                    status
                );
            }
        }

        evoke_segment_query_term_free(term_out);
        *term_out = term;
        evoke_segment_query_term_init(&term);
    }
    PG_CATCH();
    {
        evoke_segment_query_term_free(&term);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

static evoke_status
evoke_segment_pages_load_query_bmp_term(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *ref,
    uint64 section_offset,
    uint64 section_size,
    uint32 term_id,
    evoke_segment_query_bmp_term *term_out
)
{
    evoke_semantic_bmp_packed_disk_header header;
    uint8 header_bytes[EVOKE_SEMANTIC_BMP_HEADER_SIZE];
    uint8 term_bytes[EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE];
    uint32 low;
    uint32 high;
    evoke_status status;

    if (index_relation == NULL || context == NULL || ref == NULL ||
        term_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(term_out, 0, sizeof(*term_out));
    if (section_size == 0)
    {
        return EVOKE_OK;
    }
    if (section_offset > ref->object_bytes ||
        section_size > ref->object_bytes - section_offset ||
        section_size < sizeof(header_bytes) ||
        section_offset > SIZE_MAX || section_size > SIZE_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_segment_pages_read_plan_range(
        index_relation,
        context,
        ref,
        (Size) section_offset,
        sizeof(header_bytes),
        header_bytes
    );
    status = evoke_semantic_bmp_packed_disk_header_decode(
        header_bytes,
        sizeof(header_bytes),
        &header
    );
    if (status != EVOKE_OK || header.total_size != section_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    low = 0;
    high = header.term_count;
    while (low < high)
    {
        evoke_semantic_bmp_packed_term term;
        uint32 middle = low + (high - low) / 2;
        uint64 relative_offset = header.terms_offset +
            (uint64) middle * EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE;

        if (relative_offset > section_size ||
            sizeof(term_bytes) > section_size - relative_offset ||
            section_offset > SIZE_MAX - relative_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        evoke_segment_pages_read_plan_range(
            index_relation,
            context,
            ref,
            (Size) (section_offset + relative_offset),
            sizeof(term_bytes),
            term_bytes
        );
        status = evoke_semantic_bmp_packed_term_decode(
            term_bytes,
            sizeof(term_bytes),
            &term
        );
        if (status != EVOKE_OK || term.first_ref > header.ref_count ||
            term.ref_count > header.ref_count - term.first_ref ||
            term.first_super_ref > header.super_ref_count ||
            term.super_ref_count >
                header.super_ref_count - term.first_super_ref ||
            term.first_block_membership_byte >
                header.block_membership_bytes ||
            term.block_membership_bytes >
                header.block_membership_bytes -
                    term.first_block_membership_byte)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        if (term.term_id < term_id)
        {
            low = middle + 1;
        }
        else if (term.term_id > term_id)
        {
            high = middle;
        }
        else
        {
            term_out->section_offset = section_offset;
            term_out->section_size = section_size;
            term_out->super_refs_offset =
                section_offset + header.super_refs_offset;
            term_out->refs_offset = section_offset + header.refs_offset;
            term_out->block_membership_offset =
                section_offset + header.block_membership_offset;
            term_out->doc_deltas_offset =
                section_offset + header.doc_deltas_offset;
            term_out->impacts_offset = section_offset + header.impacts_offset;
            term_out->document_count = header.document_count;
            term_out->block_count = header.block_count;
            term_out->superblock_count = header.superblock_count;
            term_out->record_count = header.record_count;
            term_out->posting_count = term.posting_count;
            term_out->first_ref = term.first_ref;
            term_out->ref_count = term.ref_count;
            term_out->first_super_ref = term.first_super_ref;
            term_out->super_ref_count = term.super_ref_count;
            term_out->first_block_membership_byte =
                term.first_block_membership_byte;
            term_out->block_membership_bytes =
                term.block_membership_bytes;
            term_out->first_doc_byte = term.first_doc_byte;
            term_out->first_document = term.first_document;
            term_out->doc_delta_width = term.doc_delta_width;
            term_out->min_impact = term.min_impact;
            term_out->max_impact = term.max_impact;
            term_out->impact_precision = header.impact_precision;
            {
                uint8 super_bytes[
                    EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE
                ];
                evoke_semantic_bmp_packed_super_ref super_ref;

                evoke_segment_pages_read_plan_range(
                    index_relation,
                    context,
                    ref,
                    (Size) (term_out->super_refs_offset +
                        (uint64) term.first_super_ref *
                            EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE),
                    sizeof(super_bytes),
                    super_bytes
                );
                status = evoke_semantic_bmp_packed_super_ref_decode(
                    super_bytes,
                    sizeof(super_bytes),
                    term.min_impact,
                    term.max_impact,
                    &super_ref
                );
                if (status != EVOKE_OK ||
                    super_ref.first_ref != term.first_ref ||
                    super_ref.first_impact > header.posting_count ||
                    term.posting_count >
                        header.posting_count - super_ref.first_impact)
                {
                    return status == EVOKE_OK
                        ? EVOKE_ERR_FORMAT
                        : status;
                }
                term_out->first_impact = super_ref.first_impact;
            }
            term_out->available = true;
            return EVOKE_OK;
        }
    }
    return EVOKE_ERR_FORMAT;
}

static evoke_status
evoke_segment_query_term_plan_add_run(
    evoke_segment_query_term_plan *plan,
    const evoke_segment_object_ref *ref,
    evoke_segment_query_run_source source,
    evoke_posting_extent_kind kind,
    uint64 stable_posting_offset,
    uint64 source_posting_offset,
    uint64 posting_count,
    uint64 block_offset,
    uint32 block_count,
    uint64 indices_offset,
    uint64 values_offset,
    uint64 blocks_offset,
    uint64 document_map_offset,
    uint32 block_shift,
    uint32 payload_flags,
    uint32 document_id_base,
    uint32 local_document_count,
    const evoke_segment_query_bmp_term *semantic_bmp
)
{
    evoke_segment_query_run *run;
    uint32 semantic_query_blocks_per_superblock;

    if (plan == NULL || ref == NULL ||
        plan->run_count >= EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS ||
        (source != EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD &&
         source != EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD) ||
        kind < EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
        kind > EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
        posting_count == 0 ||
        stable_posting_offset > UINT64_MAX - posting_count ||
        source_posting_offset > UINT64_MAX - posting_count ||
        (kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
         block_count == 0) ||
        (kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
         (semantic_bmp == NULL ||
          semantic_bmp->posting_count != posting_count)) ||
        block_shift != EVOKE_DEFAULT_POSTING_BLOCK_SHIFT ||
        local_document_count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    run = &plan->runs[plan->run_count++];
    memset(run, 0, sizeof(*run));
    run->ref = *ref;
    run->stable_posting_offset = stable_posting_offset;
    run->source_posting_offset = source_posting_offset;
    run->posting_offset = source_posting_offset;
    run->posting_count = posting_count;
    run->block_offset = block_offset;
    run->indices_offset = indices_offset;
    run->values_offset = values_offset;
    run->blocks_offset = blocks_offset;
    run->document_map_offset = document_map_offset;
    run->block_count = block_count;
    run->block_shift = block_shift;
    run->payload_flags = payload_flags;
    run->document_id_base = document_id_base;
    run->local_document_count = local_document_count;
    run->kind = kind;
    run->source = source;
    if (semantic_bmp != NULL)
    {
        if (EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT < block_shift)
        {
            return EVOKE_ERR_FORMAT;
        }
        semantic_query_blocks_per_superblock = UINT32_C(1) <<
            (EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT - block_shift);
        if (semantic_bmp->super_ref_count >
            UINT32_MAX / semantic_query_blocks_per_superblock)
        {
            return EVOKE_ERR_RANGE;
        }
        run->semantic_bmp = *semantic_bmp;
        run->posting_offset = 0;
        run->block_offset = 0;
        run->block_count = semantic_bmp->super_ref_count *
            semantic_query_blocks_per_superblock;
    }
    return EVOKE_OK;
}

static void
evoke_segment_pages_load_query_term_plan_internal(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 term_id,
    uint64 minimum_sequence,
    evoke_segment_query_term_plan *plan_out
)
{
    evoke_segment_query_term_plan plan;
    evoke_term_cow_record record;
    uint64 effective_coverage;
    bool use_impact;
    bool lexical_neutral_seen = false;
    evoke_status status;

    if (index_relation == NULL || context == NULL || plan_out == NULL ||
        term_id >= context->manifest.vocab_size ||
        context->manifest.document_slot_count > UINT32_MAX ||
        context->query_contract.block_shift !=
            EVOKE_DEFAULT_POSTING_BLOCK_SHIFT ||
        context->query_contract.vocab != NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term plan load")));
    }
    status = evoke_segment_read_root_validate(&context->root);
    if (status == EVOKE_OK)
    {
        status = evoke_segment_manifest_validate_published(
            &context->manifest,
            &context->root.manifest,
            context->root.published_block_high_watermark
        );
    }
    if (status != EVOKE_OK ||
        (context->manifest.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0)
    {
        evoke_segment_pages_report_codec_error(
            "query-term plan root",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }

    memset(&plan, 0, sizeof(plan));
    memset(&record, 0, sizeof(record));
    plan.term_id = term_id;
    evoke_segment_pages_load_term_cow_record(
        index_relation,
        &context->manifest.term_directory,
        context->manifest.vocab_size,
        term_id,
        &record
    );
    if (record.term_id != term_id)
    {
        evoke_segment_pages_report_codec_error(
            "query-term plan record",
            EVOKE_ERR_FORMAT
        );
    }
    plan.raw_document_frequency = record.raw_document_frequency;
    effective_coverage = record.neutral_minor_fold_coverage != 0
        ? record.neutral_minor_fold_coverage
        : record.neutral_fold_coverage;
    use_impact = minimum_sequence == 0 &&
        context->root.active_l0.record_count == 0 &&
        context->root.pending_l0.record_count == 0 &&
        effective_coverage != 0 &&
        record.impact_fold_coverage == effective_coverage &&
        record.impact_statistics_epoch ==
            context->manifest.statistics_epoch;

    for (uint32 fold_level = 0; fold_level < 2; fold_level++)
    {
        const evoke_segment_object_ref *ref = fold_level == 0
            ? &record.neutral_fold
            : &record.neutral_minor_fold;
        uint64 coverage = fold_level == 0
            ? record.neutral_fold_coverage
            : record.neutral_minor_fold_coverage;
        evoke_term_fold_disk_header header;
        evoke_term_fold_run runs[2];
        uint32 run_count = 0;

        if (coverage == 0 || coverage <= minimum_sequence)
        {
            continue;
        }
        status = evoke_segment_pages_read_fold_header(
            index_relation,
            &context->root,
                &context->manifest,
                ref,
                EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                &header,
                context
        );
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_find_fold_runs(
                index_relation,
                ref,
                &header,
                term_id,
                coverage,
                    runs,
                    lengthof(runs),
                    &run_count,
                    context
            );
        }
        if (status != EVOKE_OK ||
            header.block_shift != context->query_contract.block_shift)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan neutral fold",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        for (uint32 run_index = 0;
             run_index < run_count;
             run_index++)
        {
            const evoke_term_fold_run *run = &runs[run_index];
            evoke_segment_query_bmp_term semantic_bmp;

            memset(&semantic_bmp, 0, sizeof(semantic_bmp));

            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                lexical_neutral_seen = true;
                if (use_impact)
                {
                    continue;
                }
            }
            else if (run->kind !=
                     EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term plan neutral fold kind",
                    EVOKE_ERR_FORMAT
                );
            }
            else if (header.semantic_bmp_size > 0)
            {
                status = evoke_segment_pages_load_query_bmp_term(
                    index_relation,
                    context,
                    ref,
                    header.semantic_bmp_offset,
                    header.semantic_bmp_size,
                    term_id,
                    &semantic_bmp
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "query-term plan neutral BMP",
                        status
                    );
                }
            }
            else
            {
                evoke_segment_pages_report_codec_error(
                    "query-term plan missing neutral BMP",
                    EVOKE_ERR_FORMAT
                );
            }
            status = evoke_segment_query_term_plan_add_run(
                &plan,
                ref,
                EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD,
                run->kind,
                run->posting_offset,
                run->posting_offset,
                run->posting_count,
                run->block_offset,
                run->block_count,
                header.document_slots_offset,
                header.values_offset,
                header.blocks_offset,
                0,
                header.block_shift,
                0,
                0,
                (uint32) context->manifest.document_slot_count,
                semantic_bmp.available ? &semantic_bmp : NULL
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term plan neutral run",
                    status
                );
            }
        }
    }

    if (use_impact)
    {
        evoke_term_fold_disk_header header;
        evoke_term_fold_run run;
        uint32 run_count = 0;

        if (!lexical_neutral_seen)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan impact replacement",
                EVOKE_ERR_FORMAT
            );
        }
        status = evoke_segment_pages_read_fold_header(
            index_relation,
            &context->root,
            &context->manifest,
                &record.impact_fold,
                EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
                &header,
                context
        );
        if (status == EVOKE_OK &&
            header.statistics_epoch !=
                record.impact_statistics_epoch)
        {
            status = EVOKE_ERR_FORMAT;
        }
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_find_fold_runs(
                index_relation,
                &record.impact_fold,
                &header,
                term_id,
                record.impact_fold_coverage,
                    &run,
                    1,
                    &run_count,
                    context
            );
        }
        if (status != EVOKE_OK || run_count != 1 ||
            run.kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
            header.block_shift != context->query_contract.block_shift)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan impact fold",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        status = evoke_segment_query_term_plan_add_run(
            &plan,
            &record.impact_fold,
            EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD,
            run.kind,
            run.posting_offset,
            run.posting_offset,
            run.posting_count,
            run.block_offset,
            run.block_count,
            header.document_slots_offset,
            header.values_offset,
            header.blocks_offset,
            0,
            header.block_shift,
            0,
            0,
            (uint32) context->manifest.document_slot_count,
            NULL
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan impact run",
                status
            );
        }
    }

    for (uint32 extent_index = 0;
         extent_index < record.extent_count;
         extent_index++)
    {
        const evoke_stable_term_extent *stable_extent =
            &record.extents[extent_index];
        const evoke_segment_descriptor *descriptor =
            evoke_segment_pages_find_descriptor(
                &context->manifest,
                stable_extent->segment_id
            );
        evoke_segment_object_ref ref;
        evoke_segment_payload_disk_header header;
        evoke_segment_term_run run;
        evoke_segment_query_bmp_term semantic_bmp;

        memset(&semantic_bmp, 0, sizeof(semantic_bmp));

        if (descriptor == NULL ||
            (effective_coverage != 0 &&
             descriptor->min_sequence <= effective_coverage))
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan tail extent",
                EVOKE_ERR_FORMAT
            );
        }
        if (descriptor->max_sequence <= minimum_sequence)
        {
            continue;
        }
        status = evoke_segment_pages_read_payload_header(
            index_relation,
            &context->root,
            &context->manifest,
                descriptor,
                &ref,
                &header,
                context
        );
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_find_payload_run(
                index_relation,
                &ref,
                &header,
                    term_id,
                    stable_extent->kind,
                    &run,
                    context
            );
        }
        if (status != EVOKE_OK ||
            run.posting_count != stable_extent->posting_count ||
            header.block_shift != context->query_contract.block_shift)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan payload run",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        if (run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
            header.semantic_bmp_size > 0)
        {
            status = evoke_segment_pages_load_query_bmp_term(
                index_relation,
                context,
                &ref,
                header.semantic_bmp_offset,
                header.semantic_bmp_size,
                term_id,
                &semantic_bmp
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term plan payload BMP",
                    status
                );
            }
        }
        else if (run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan missing payload BMP",
                EVOKE_ERR_FORMAT
            );
        }
        status = evoke_segment_query_term_plan_add_run(
            &plan,
            &ref,
            EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD,
            run.kind,
            stable_extent->posting_offset,
            run.posting_offset,
            run.posting_count,
            run.block_offset,
            run.block_count,
            header.indices_offset,
            header.values_offset,
            header.blocks_offset,
            header.document_map_offset,
            header.block_shift,
            header.flags,
            header.document_id_base,
            header.local_document_count,
            semantic_bmp.available ? &semantic_bmp : NULL
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan payload extent",
                status
            );
        }
    }
    *plan_out = plan;
}

void
evoke_segment_pages_load_query_term_plan(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 term_id,
    evoke_segment_query_term_plan *plan_out
)
{
    evoke_segment_pages_load_query_term_plan_internal(
        index_relation,
        context,
        term_id,
        0,
        plan_out
    );
}

void
evoke_segment_pages_load_query_term_plan_after_sequence(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 term_id,
    uint64 minimum_sequence,
    evoke_segment_query_term_plan *plan_out
)
{
    evoke_segment_pages_load_query_term_plan_internal(
        index_relation,
        context,
        term_id,
        minimum_sequence,
        plan_out
    );
}

void
evoke_segment_pages_load_query_term_from_plan(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    evoke_segment_query_term *term_out
)
{
    evoke_segment_query_term term;
    evoke_status status = EVOKE_OK;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        term_out == NULL || plan->term_id >= context->manifest.vocab_size ||
        plan->run_count > EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS ||
        context->manifest.document_slot_count == 0 ||
        context->manifest.document_slot_count > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term plan materialization")));
    }

    evoke_segment_query_term_init(&term);
    term.term_id = plan->term_id;
    term.raw_document_frequency = plan->raw_document_frequency;
    PG_TRY();
    {
        for (uint32 run_index = 0;
             run_index < plan->run_count;
             run_index++)
        {
            const evoke_segment_query_run *source =
                &plan->runs[run_index];

            if (source->ref.owner_manifest_id >
                    context->manifest.manifest_id ||
                evoke_segment_object_ref_validate(
                    &source->ref,
                    context->root.published_block_high_watermark
                ) != EVOKE_OK ||
                source->block_shift !=
                    context->query_contract.block_shift)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            if (source->source == EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD)
            {
                evoke_term_fold_disk_header header;
                evoke_term_fold_run run;

                memset(&header, 0, sizeof(header));
                header.block_shift = source->block_shift;
                header.blocks_offset = source->blocks_offset;
                header.document_slots_offset = source->indices_offset;
                header.values_offset = source->values_offset;
                memset(&run, 0, sizeof(run));
                run.term_id = plan->term_id;
                run.kind = source->kind;
                run.posting_offset = source->source_posting_offset;
                run.posting_count = source->posting_count;
                run.block_offset = source->block_offset;
                run.block_count = source->block_count;
                status = evoke_segment_pages_load_fold_extent(
                    index_relation,
                    &source->ref,
                    &header,
                    &run,
                    (uint32) context->manifest.document_slot_count,
                    &term
                );
            }
            else if (source->source ==
                     EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD)
            {
                if (source->kind ==
                    EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
                {
                    evoke_segment_query_extent *extent;
                    evoke_segment_query_posting_cursor cursor;
                    evoke_index index_shape;
                    Size document_slot_bytes;
                    Size value_bytes;
                    uint64 posting_index = 0;

                    if (source->posting_count >
                            MaxAllocSize / sizeof(uint32) ||
                        source->posting_count >
                            MaxAllocSize / sizeof(evoke_posting_value))
                    {
                        status = EVOKE_ERR_RANGE;
                        break;
                    }
                    document_slot_bytes =
                        (Size) source->posting_count * sizeof(uint32);
                    value_bytes =
                        (Size) source->posting_count *
                            sizeof(evoke_posting_value);
                    extent = evoke_segment_query_term_add_extent(&term);
                    extent->document_slots = palloc(document_slot_bytes);
                    extent->values = palloc(value_bytes);
                    evoke_segment_query_posting_cursor_init(&cursor);
                    while (posting_index < source->posting_count)
                    {
                        uint32 loaded =
                            evoke_segment_pages_load_query_term_posting_stream_window(
                                index_relation,
                                context,
                                plan,
                                run_index,
                                &cursor,
                                &extent->document_slots[posting_index],
                                &extent->values[posting_index],
                                EVOKE_SEGMENT_QUERY_POSTING_WINDOW
                            );

                        if (loaded == 0 ||
                            loaded >
                                source->posting_count - posting_index)
                        {
                            status = EVOKE_ERR_FORMAT;
                            break;
                        }
                        posting_index += loaded;
                    }
                    if (status != EVOKE_OK)
                    {
                        break;
                    }
                    extent->view.indices = extent->document_slots;
                    extent->view.values = extent->values;
                    extent->view.len = source->posting_count;
                    extent->view.local_document_count =
                        (uint32) context->manifest.document_slot_count;
                    extent->view.kind = source->kind;
                    memset(&index_shape, 0, sizeof(index_shape));
                    index_shape.num_docs =
                        context->manifest.document_slot_count;
                    status = evoke_posting_extent_validate_layout(
                        &index_shape,
                        &extent->view
                    );
                }
                else
                {
                    evoke_segment_payload_disk_header header;
                    evoke_segment_term_run run;

                    memset(&header, 0, sizeof(header));
                    header.flags = source->payload_flags;
                    header.local_document_count =
                        source->local_document_count;
                    header.document_id_base = source->document_id_base;
                    header.block_shift = source->block_shift;
                    header.blocks_offset = source->blocks_offset;
                    header.indices_offset = source->indices_offset;
                    header.values_offset = source->values_offset;
                    header.document_map_offset =
                        source->document_map_offset;
                    memset(&run, 0, sizeof(run));
                    run.term_id = plan->term_id;
                    run.kind = source->kind;
                    run.posting_offset = source->source_posting_offset;
                    run.posting_count = source->posting_count;
                    run.block_offset = source->block_offset;
                    run.block_count = source->block_count;
                    status = evoke_segment_pages_load_payload_extent(
                        index_relation,
                        &source->ref,
                        &header,
                        &run,
                        (uint32) context->manifest.document_slot_count,
                        &term
                    );
                }
            }
            else
            {
                status = EVOKE_ERR_FORMAT;
            }
            if (status != EVOKE_OK)
            {
                break;
            }
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "query-term plan materialization",
                status
            );
        }
        evoke_segment_query_term_free(term_out);
        *term_out = term;
        evoke_segment_query_term_init(&term);
    }
    PG_CATCH();
    {
        evoke_segment_query_term_free(&term);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

void
evoke_segment_pages_load_query_term_block_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    evoke_posting_block_record *record_out
)
{
    uint32 record_count;

    record_count = evoke_segment_pages_load_query_term_block_records(
        index_relation,
        context,
        plan,
        run_index,
        block_index,
        record_out,
        1
    );
    if (record_count != 1)
    {
        ereport(ERROR, (errmsg("incomplete evoke query-term block record")));
    }
}

uint32
evoke_segment_pages_load_query_term_block_records(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_block_index,
    evoke_posting_block_record *records_out,
    uint32 record_capacity
)
{
    const evoke_segment_query_run *run;
    uint8 record_bytes[
        EVOKE_SEGMENT_QUERY_BLOCK_RECORD_WINDOW *
        EVOKE_POSTING_BLOCK_RECORD_SIZE
    ];
    uint32 record_count;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        records_out == NULL || record_capacity == 0 ||
        record_capacity > EVOKE_SEGMENT_QUERY_BLOCK_RECORD_WINDOW ||
        plan->term_id >= context->manifest.vocab_size ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term block record load")));
    }
    run = &plan->runs[run_index];
    if (first_block_index >= run->block_count ||
        run->block_shift != context->query_contract.block_shift ||
        run->block_shift != EVOKE_DEFAULT_POSTING_BLOCK_SHIFT)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term block index")));
    }
    record_count = Min(
        record_capacity,
        run->block_count - first_block_index
    );
    if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        for (uint32 record_index = 0;
             record_index < record_count;
             record_index++)
        {
            evoke_posting_block_record *record =
                &records_out[record_index];
            evoke_semantic_bmp_super_ref super_ref;
            evoke_semantic_bmp_ref refs[16];
            uint32 semantic_block_index =
                first_block_index + record_index;
            uint32 super_ref_index = semantic_block_index >>
                EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK_SHIFT;
            uint32 local_query_block = semantic_block_index &
                (EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK - UINT32_C(1));
            uint32 loaded;
            uint32 outer_block_id;
            uint64 posting_offset;
            bool found = false;

            if (evoke_segment_pages_load_query_bmp_super_ref_window(
                    index_relation,
                    context,
                    plan,
                    run_index,
                    super_ref_index,
                    &super_ref,
                    1) != 1 ||
                super_ref.first_ref < run->semantic_bmp.first_ref)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic query superblock",
                    EVOKE_ERR_FORMAT
                );
            }
            loaded = evoke_segment_pages_load_query_bmp_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                super_ref.first_ref - run->semantic_bmp.first_ref,
                refs,
                super_ref.ref_count
            );
            if (loaded != super_ref.ref_count)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic query refs",
                    EVOKE_ERR_FORMAT
                );
            }
            outer_block_id = (super_ref.superblock_id <<
                EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK_SHIFT) +
                local_query_block;
            posting_offset = super_ref.first_ref ==
                    run->semantic_bmp.first_ref
                ? 0
                : UINT64_MAX;
            memset(record, 0, sizeof(*record));
            record->block_id = outer_block_id;
            record->kind = run->kind;
            for (uint32 ref_index = 0; ref_index < loaded; ref_index++)
            {
                evoke_semantic_bmp_record packed_record;
                float impacts[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS];
                uint32 first_local = 0;
                uint32 last_local =
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS - 1U;
                uint64 mask;

                if (!evoke_segment_pages_load_query_bmp_record(
                        index_relation,
                        context,
                        plan,
                        run_index,
                        refs[ref_index].block_id,
                        &packed_record,
                        impacts))
                {
                    evoke_segment_pages_report_codec_error(
                        "packed semantic query record",
                        EVOKE_ERR_FORMAT
                    );
                }
                if (posting_offset == UINT64_MAX ||
                    packed_record.first_impact -
                        run->semantic_bmp.first_impact < posting_offset)
                {
                    posting_offset = packed_record.first_impact -
                        run->semantic_bmp.first_impact;
                }
                if ((refs[ref_index].block_id >>
                     EVOKE_BMP_BLOCKS_PER_QUERY_BLOCK_SHIFT) !=
                    outer_block_id)
                {
                    continue;
                }
                mask = packed_record.document_mask;
                while ((mask & UINT64_C(1)) == 0)
                {
                    first_local++;
                    mask >>= 1;
                }
                mask = packed_record.document_mask;
                while ((mask & (UINT64_C(1) <<
                    (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                     UINT32_C(1)))) == 0)
                {
                    last_local--;
                    mask <<= 1;
                }
                if (!found)
                {
                    record->posting_offset =
                        packed_record.first_impact -
                            run->semantic_bmp.first_impact;
                    record->first_document_id =
                        (refs[ref_index].block_id <<
                            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) +
                            first_local;
                    record->min_impact = refs[ref_index].min_impact;
                    record->max_impact = refs[ref_index].max_impact;
                    found = true;
                }
                else
                {
                    record->min_impact = Min(
                        record->min_impact,
                        refs[ref_index].min_impact
                    );
                    record->max_impact = Max(
                        record->max_impact,
                        refs[ref_index].max_impact
                    );
                }
                record->posting_count += packed_record.impact_count;
                record->last_document_id =
                    (refs[ref_index].block_id <<
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) + last_local;
            }
            if (!found)
            {
                record->posting_offset = posting_offset == UINT64_MAX
                    ? run->posting_count
                    : posting_offset;
            }
            else if (record->posting_offset > run->posting_count ||
                     record->posting_count >
                        run->posting_count - record->posting_offset ||
                     (record->first_document_id >> run->block_shift) !=
                        record->block_id ||
                     (record->last_document_id >> run->block_shift) !=
                        record->block_id)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic query block",
                    EVOKE_ERR_FORMAT
                );
            }
        }
        return record_count;
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (run->blocks_offset +
            (run->block_offset + first_block_index) *
                EVOKE_POSTING_BLOCK_RECORD_SIZE),
        (Size) record_count * EVOKE_POSTING_BLOCK_RECORD_SIZE,
        record_bytes
    );
    for (uint32 record_index = 0;
         record_index < record_count;
         record_index++)
    {
        evoke_posting_block_record *record = &records_out[record_index];
        evoke_status status = evoke_posting_block_record_decode(
            record_bytes +
                (Size) record_index * EVOKE_POSTING_BLOCK_RECORD_SIZE,
            EVOKE_POSTING_BLOCK_RECORD_SIZE,
            record
        );

        if (status != EVOKE_OK || record->kind != run->kind ||
            record->posting_count == 0 ||
            record->posting_count >
                EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS ||
            record->posting_offset > run->posting_count ||
            record->posting_count >
                run->posting_count - record->posting_offset ||
            record->first_document_id > record->last_document_id ||
            record->last_document_id >=
                context->manifest.document_slot_count ||
            (record->first_document_id >> run->block_shift) !=
                record->block_id ||
            (record->last_document_id >> run->block_shift) !=
                record->block_id)
        {
            evoke_segment_pages_report_codec_error(
                "query-term posting block",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        if (record_index > 0 &&
            records_out[record_index - 1].block_id >= record->block_id)
        {
            evoke_segment_pages_report_codec_error(
                "query-term posting block order",
                EVOKE_ERR_FORMAT
            );
        }
    }
    return record_count;
}

uint32
evoke_segment_pages_load_query_term_posting_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint64 first_posting_index,
    uint32 *document_slots_out,
    evoke_posting_value *values_out,
    uint32 posting_capacity
)
{
    const evoke_segment_query_run *run;
    uint64 absolute_posting_offset;
    uint64 posting_byte_offset;
    uint64 indices_byte_offset;
    uint64 values_byte_offset;
    uint32 posting_count;
    Size posting_bytes;
    evoke_status status = EVOKE_OK;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        document_slots_out == NULL || values_out == NULL ||
        posting_capacity == 0 ||
        posting_capacity > EVOKE_SEGMENT_QUERY_POSTING_WINDOW ||
        plan->term_id >= context->manifest.vocab_size ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query posting-window load")));
    }
    run = &plan->runs[run_index];
    if (first_posting_index >= run->posting_count ||
        run->posting_offset > UINT64_MAX - first_posting_index)
    {
        ereport(ERROR, (errmsg("invalid evoke query posting-window offset")));
    }
    posting_count = (uint32) Min(
        (uint64) posting_capacity,
        run->posting_count - first_posting_index
    );
    if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        uint32 low = 0;
        uint32 high = run->semantic_bmp.ref_count;
        uint32 output_count = 0;
        uint64 posting_limit = first_posting_index + posting_count;

        while (low < high)
        {
            evoke_semantic_bmp_ref ref;
            uint32 middle = low + (high - low) / 2;

            if (evoke_segment_pages_load_query_bmp_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                middle,
                &ref,
                1) != 1)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic posting ref",
                    EVOKE_ERR_FORMAT
                );
            }
            if (ref.record_index - run->semantic_bmp.first_impact <
                first_posting_index)
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }
        if (low > 0)
        {
            low--;
        }
        for (uint32 ref_index = low;
             ref_index < run->semantic_bmp.ref_count &&
                output_count < posting_count;
             ref_index++)
        {
            evoke_semantic_bmp_ref ref;
            evoke_semantic_bmp_record packed_record;
            float impacts[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS];
            uint64 mask;
            uint32 impact_index = 0;

            if (evoke_segment_pages_load_query_bmp_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                ref_index,
                &ref,
                1) != 1)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic posting ref",
                    EVOKE_ERR_FORMAT
                );
            }
            if (ref.record_index - run->semantic_bmp.first_impact >=
                posting_limit)
            {
                break;
            }
            if (!evoke_segment_pages_load_query_bmp_record(
                    index_relation,
                    context,
                    plan,
                    run_index,
                    ref.block_id,
                    &packed_record,
                    impacts) || packed_record.first_impact !=
                        ref.record_index)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic posting window",
                    EVOKE_ERR_FORMAT
                );
            }
            mask = packed_record.document_mask;
            for (uint32 local_document = 0;
                 mask != 0;
                 local_document++, mask >>= 1)
            {
                uint64 posting_index =
                    packed_record.first_impact -
                        run->semantic_bmp.first_impact + impact_index;

                if ((mask & UINT64_C(1)) == 0)
                {
                    continue;
                }
                if (posting_index >= first_posting_index &&
                    posting_index < posting_limit)
                {
                    document_slots_out[output_count] =
                        (ref.block_id <<
                            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) +
                            local_document;
                    values_out[output_count].impact =
                        impacts[impact_index];
                    output_count++;
                }
                impact_index++;
            }
        }
        if (output_count != posting_count)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic posting count",
                EVOKE_ERR_FORMAT
            );
        }
        return posting_count;
    }
    absolute_posting_offset = run->posting_offset + first_posting_index;
    if (absolute_posting_offset > SIZE_MAX / sizeof(uint32))
    {
        evoke_segment_pages_report_codec_error(
            "query posting-window range",
            EVOKE_ERR_RANGE
        );
    }
    posting_byte_offset = absolute_posting_offset * sizeof(uint32);
    if (run->indices_offset > SIZE_MAX - posting_byte_offset ||
        run->values_offset > SIZE_MAX - posting_byte_offset)
    {
        evoke_segment_pages_report_codec_error(
            "query posting-window offset",
            EVOKE_ERR_RANGE
        );
    }
    indices_byte_offset = run->indices_offset + posting_byte_offset;
    values_byte_offset = run->values_offset + posting_byte_offset;
    posting_bytes = (Size) posting_count * sizeof(uint32);
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) indices_byte_offset,
        posting_bytes,
        (uint8 *) document_slots_out
    );
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) values_byte_offset,
        posting_bytes,
        (uint8 *) values_out
    );
    for (uint32 posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        uint8 *slot_bytes = (uint8 *) document_slots_out +
            (Size) posting_index * sizeof(uint32);
        uint8 *value_bytes = (uint8 *) values_out +
            (Size) posting_index * sizeof(uint32);
        uint32 value_bits;

        document_slots_out[posting_index] =
            evoke_segment_pages_read_u32_le(slot_bytes);
        value_bits = evoke_segment_pages_read_u32_le(value_bytes);
        if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            values_out[posting_index].term_frequency = value_bits;
            if (value_bits == 0)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
        }
        else
        {
            memcpy(
                &values_out[posting_index].impact,
                &value_bits,
                sizeof(value_bits)
            );
            if (!isfinite(values_out[posting_index].impact))
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
        }
    }
    if (status == EVOKE_OK &&
        run->source == EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD)
    {
        evoke_segment_payload_disk_header header;

        memset(&header, 0, sizeof(header));
        header.flags = run->payload_flags;
        header.local_document_count = run->local_document_count;
        header.document_id_base = run->document_id_base;
        header.document_map_offset = run->document_map_offset;
        status = evoke_segment_pages_globalize_payload_slots(
            index_relation,
            &run->ref,
            &header,
            (uint32) context->manifest.document_slot_count,
            document_slots_out,
            posting_count,
            context->root.published_block_high_watermark,
            context->page_validation_cache
        );
    }
    else if (status == EVOKE_OK &&
             run->source == EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD)
    {
        for (uint32 posting_index = 0;
             posting_index < posting_count;
             posting_index++)
        {
            uint32 document_slot = document_slots_out[posting_index];

            if (document_slot >= context->manifest.document_slot_count ||
                (posting_index > 0 &&
                 document_slots_out[posting_index - 1] >= document_slot))
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
        }
    }
    else if (status == EVOKE_OK)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "query posting window",
            status
        );
    }
    return posting_count;
}

void
evoke_segment_query_posting_cursor_init(
    evoke_segment_query_posting_cursor *cursor
)
{
    if (cursor != NULL)
    {
        memset(cursor, 0, sizeof(*cursor));
    }
}

static uint32
evoke_segment_pages_read_packed_delta(
    const uint8 *bytes,
    uint32 width
)
{
    if (width == 1)
    {
        return bytes[0];
    }
    if (width == 2)
    {
        return evoke_segment_pages_read_u16_le(bytes);
    }
    if (width == 4)
    {
        return evoke_segment_pages_read_u32_le(bytes);
    }
    evoke_segment_pages_report_codec_error(
        "packed semantic document delta",
        EVOKE_ERR_FORMAT
    );
    return 0;
}

uint32
evoke_segment_pages_accumulate_query_semantic_stream_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    evoke_segment_query_posting_cursor *cursor,
    uint8 *delta_scratch,
    uint8 *impact_scratch,
    uint32 posting_capacity,
    float query_weight,
    float *scores,
    uint64 score_count,
    double absolute_impact_floor,
    uint64 *omitted_postings
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint64 first_posting_index;
    uint64 delta_count;
    uint64 first_delta_index;
    uint64 delta_byte_offset;
    uint64 delta_byte_count;
    uint64 first_impact;
    uint64 impact_byte_offset;
    size_t impact_width;
    uint32 document_slot;
    uint32 posting_count;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        cursor == NULL || delta_scratch == NULL || impact_scratch == NULL ||
        scores == NULL || omitted_postings == NULL ||
        posting_capacity == 0 ||
        posting_capacity > EVOKE_SEGMENT_QUERY_POSTING_WINDOW ||
        run_index >= plan->run_count || !isfinite(query_weight) ||
        !isfinite(absolute_impact_floor) || absolute_impact_floor < 0.0)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic score stream")));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    first_posting_index = cursor->next_posting_index;
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        first_posting_index >= run->posting_count ||
        !bmp->available || bmp->posting_count != run->posting_count ||
        bmp->document_count > context->manifest.document_slot_count ||
        bmp->document_count > score_count ||
        (bmp->posting_count == 1 && bmp->doc_delta_width != 0) ||
        (bmp->posting_count > 1 && bmp->doc_delta_width != 1 &&
         bmp->doc_delta_width != 2 && bmp->doc_delta_width != 4) ||
        (first_posting_index == 0 && cursor->have_previous_document) ||
        (first_posting_index > 0 && !cursor->have_previous_document))
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic score stream",
            EVOKE_ERR_FORMAT
        );
    }
    posting_count = (uint32) Min(
        (uint64) posting_capacity,
        run->posting_count - first_posting_index
    );
    delta_count = posting_count;
    if (first_posting_index == 0)
    {
        delta_count--;
        first_delta_index = 0;
        document_slot = bmp->first_document;
    }
    else
    {
        first_delta_index = first_posting_index - 1;
        document_slot = cursor->previous_document_slot;
    }
    if (delta_count > 0 &&
        (bmp->doc_delta_width == 0 ||
         delta_count > UINT64_MAX / bmp->doc_delta_width ||
         first_delta_index > UINT64_MAX / bmp->doc_delta_width))
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic score delta range",
            EVOKE_ERR_RANGE
        );
    }
    delta_byte_count = delta_count * bmp->doc_delta_width;
    delta_byte_offset = first_delta_index * bmp->doc_delta_width;
    if (bmp->doc_deltas_offset > SIZE_MAX - bmp->first_doc_byte ||
        bmp->doc_deltas_offset + bmp->first_doc_byte >
            SIZE_MAX - delta_byte_offset ||
        delta_byte_count >
            (uint64) posting_capacity * sizeof(uint32))
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic score delta range",
            EVOKE_ERR_RANGE
        );
    }
    if (delta_byte_count > 0)
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &run->ref,
            (Size) (bmp->doc_deltas_offset + bmp->first_doc_byte +
                delta_byte_offset),
            (Size) delta_byte_count,
            delta_scratch
        );
    }
    if (bmp->first_impact > UINT64_MAX - first_posting_index)
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic score impact range",
            EVOKE_ERR_RANGE
        );
    }
    first_impact = bmp->first_impact + first_posting_index;
    impact_width = evoke_semantic_bmp_impact_width(
        bmp->impact_precision
    );
    if (impact_width == 0 || first_impact > UINT64_MAX / impact_width ||
        bmp->impacts_offset > SIZE_MAX - first_impact * impact_width)
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic score impact range",
            EVOKE_ERR_RANGE
        );
    }
    impact_byte_offset = bmp->impacts_offset +
        first_impact * impact_width;
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) impact_byte_offset,
        (Size) posting_count * impact_width,
        impact_scratch
    );

    for (uint32 posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        float impact;

        if (!(first_posting_index == 0 && posting_index == 0))
        {
            uint64 delta_index = first_posting_index == 0
                ? posting_index - 1
                : posting_index;
            uint32 delta = evoke_segment_pages_read_packed_delta(
                delta_scratch + delta_index * bmp->doc_delta_width,
                bmp->doc_delta_width
            );

            if (delta == 0 || document_slot >= bmp->document_count ||
                delta > bmp->document_count - 1 - document_slot)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic score delta",
                    EVOKE_ERR_FORMAT
                );
            }
            document_slot += delta;
        }
        if (document_slot >= bmp->document_count ||
            (posting_index == 0 && cursor->have_previous_document &&
             document_slot <= cursor->previous_document_slot))
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic score document stream",
                EVOKE_ERR_FORMAT
            );
        }
        {
            evoke_status status = evoke_semantic_bmp_packed_impact_decode(
                impact_scratch + (Size) posting_index * impact_width,
                (Size) (posting_count - posting_index) * impact_width,
                bmp->impact_precision,
                bmp->min_impact,
                bmp->max_impact,
                &impact
            );

            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic score impact stream",
                    status
                );
            }
        }
        if (!isfinite(impact) || impact == 0.0f)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic score impact stream",
                EVOKE_ERR_FORMAT
            );
        }
        if (absolute_impact_floor > 0.0 &&
            fabs((double) impact) <= absolute_impact_floor)
        {
            if (*omitted_postings < UINT64_MAX)
            {
                (*omitted_postings)++;
            }
            continue;
        }
        scores[document_slot] += query_weight * impact;
    }
    cursor->previous_document_slot = document_slot;
    cursor->have_previous_document = true;
    cursor->next_posting_index += posting_count;
    return posting_count;
}

uint32
evoke_segment_pages_load_query_term_posting_stream_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    evoke_segment_query_posting_cursor *cursor,
    uint32 *document_slots_out,
    evoke_posting_value *values_out,
    uint32 posting_capacity
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint64 first_posting_index;
    uint32 posting_count;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        cursor == NULL || document_slots_out == NULL || values_out == NULL ||
        posting_capacity == 0 ||
        posting_capacity > EVOKE_SEGMENT_QUERY_POSTING_WINDOW ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query posting stream load")));
    }
    run = &plan->runs[run_index];
    first_posting_index = cursor->next_posting_index;
    if (first_posting_index >= run->posting_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query posting stream offset")));
    }
    posting_count = (uint32) Min(
        (uint64) posting_capacity,
        run->posting_count - first_posting_index
    );
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        uint32 loaded = evoke_segment_pages_load_query_term_posting_window(
            index_relation,
            context,
            plan,
            run_index,
            first_posting_index,
            document_slots_out,
            values_out,
            posting_capacity
        );

        if (loaded == 0 || loaded > posting_count)
        {
            evoke_segment_pages_report_codec_error(
                "query posting stream window",
                EVOKE_ERR_FORMAT
            );
        }
        cursor->previous_document_slot = document_slots_out[loaded - 1];
        cursor->have_previous_document = true;
        cursor->next_posting_index += loaded;
        return loaded;
    }

    bmp = &run->semantic_bmp;
    if (!bmp->available || bmp->posting_count != run->posting_count ||
        bmp->document_count > context->manifest.document_slot_count ||
        (bmp->posting_count == 1 && bmp->doc_delta_width != 0) ||
        (bmp->posting_count > 1 && bmp->doc_delta_width != 1 &&
         bmp->doc_delta_width != 2 && bmp->doc_delta_width != 4) ||
        (first_posting_index == 0 && cursor->have_previous_document) ||
        (first_posting_index > 0 && !cursor->have_previous_document))
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic posting stream",
            EVOKE_ERR_FORMAT
        );
    }
    {
        uint64 delta_count = posting_count;
        uint64 first_delta_index;
        uint64 delta_byte_offset;
        uint64 delta_byte_count;
        uint32 document_slot;
        uint8 *delta_bytes = (uint8 *) values_out;

        if (first_posting_index == 0)
        {
            delta_count--;
            first_delta_index = 0;
            document_slot = bmp->first_document;
        }
        else
        {
            first_delta_index = first_posting_index - 1;
            document_slot = cursor->previous_document_slot;
        }
        if (delta_count > 0 &&
            (bmp->doc_delta_width == 0 ||
             delta_count > UINT64_MAX / bmp->doc_delta_width ||
             first_delta_index > UINT64_MAX / bmp->doc_delta_width))
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic document delta range",
                EVOKE_ERR_RANGE
            );
        }
        delta_byte_count = delta_count * bmp->doc_delta_width;
        delta_byte_offset = first_delta_index * bmp->doc_delta_width;
        if (bmp->doc_deltas_offset > SIZE_MAX - bmp->first_doc_byte ||
            bmp->doc_deltas_offset + bmp->first_doc_byte >
                SIZE_MAX - delta_byte_offset ||
            delta_byte_count > MaxAllocSize)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic document delta range",
                EVOKE_ERR_RANGE
            );
        }
        if (delta_byte_count > 0)
        {
            evoke_segment_pages_read_query_range(
                index_relation,
                context,
                &run->ref,
                (Size) (bmp->doc_deltas_offset + bmp->first_doc_byte +
                    delta_byte_offset),
                (Size) delta_byte_count,
                delta_bytes
            );
        }
        for (uint32 posting_index = 0;
             posting_index < posting_count;
             posting_index++)
        {
            if (!(first_posting_index == 0 && posting_index == 0))
            {
                uint64 delta_index = first_posting_index == 0
                    ? posting_index - 1
                    : posting_index;
                uint32 delta = evoke_segment_pages_read_packed_delta(
                    delta_bytes + delta_index * bmp->doc_delta_width,
                    bmp->doc_delta_width
                );
                if (delta == 0 || document_slot >= bmp->document_count ||
                    delta > bmp->document_count - 1 - document_slot)
                {
                    evoke_segment_pages_report_codec_error(
                        "packed semantic document delta",
                        EVOKE_ERR_FORMAT
                    );
                }
                document_slot += delta;
            }
            if (document_slot >= bmp->document_count ||
                (posting_index > 0 &&
                 document_slots_out[posting_index - 1] >= document_slot))
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic document stream",
                    EVOKE_ERR_FORMAT
                );
            }
            document_slots_out[posting_index] = document_slot;
        }
    }
    {
        uint64 first_impact;
        uint64 impact_byte_offset;
        size_t impact_width = evoke_semantic_bmp_impact_width(
            bmp->impact_precision
        );
        Size impact_bytes;

        if (impact_width == 0 || posting_count > SIZE_MAX / impact_width)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic impact stream precision",
                EVOKE_ERR_FORMAT
            );
        }
        impact_bytes = (Size) posting_count * impact_width;

        if (bmp->first_impact > UINT64_MAX - first_posting_index)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic impact stream range",
                EVOKE_ERR_RANGE
            );
        }
        first_impact = bmp->first_impact + first_posting_index;
        if (first_impact > UINT64_MAX / impact_width ||
            bmp->impacts_offset > SIZE_MAX -
                first_impact * impact_width)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic impact stream range",
                EVOKE_ERR_RANGE
            );
        }
        impact_byte_offset = bmp->impacts_offset +
            first_impact * impact_width;
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &run->ref,
            (Size) impact_byte_offset,
            impact_bytes,
            (uint8 *) values_out
        );
        if (bmp->impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32)
        {
            for (uint32 posting_index = 0;
                 posting_index < posting_count;
                 posting_index++)
            {
                uint8 *impact_bytes_at = (uint8 *) values_out +
                    (Size) posting_index * impact_width;
                uint32 impact_bits =
                    evoke_segment_pages_read_u32_le(impact_bytes_at);

                memcpy(
                    &values_out[posting_index].impact,
                    &impact_bits,
                    sizeof(impact_bits)
                );
                if (!isfinite(values_out[posting_index].impact) ||
                    values_out[posting_index].impact == 0.0f)
                {
                    evoke_segment_pages_report_codec_error(
                        "packed semantic impact stream",
                        EVOKE_ERR_FORMAT
                    );
                }
            }
        }
        else
        {
            uint8 *encoded = (uint8 *) values_out;

            for (uint32 posting_index = posting_count;
                 posting_index > 0;
                 posting_index--)
            {
                uint32 position = posting_index - 1;
                evoke_status status =
                    evoke_semantic_bmp_packed_impact_decode(
                        encoded + (Size) position * impact_width,
                        impact_bytes - (Size) position * impact_width,
                        bmp->impact_precision,
                        bmp->min_impact,
                        bmp->max_impact,
                        &values_out[position].impact
                    );

                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "packed semantic impact stream",
                        status
                    );
                }
            }
        }
    }
    cursor->previous_document_slot =
        document_slots_out[posting_count - 1];
    cursor->have_previous_document = true;
    cursor->next_posting_index += posting_count;
    return posting_count;
}

uint32
evoke_segment_pages_load_query_bmp_cached_super_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_segment_query_bmp_cached_super_ref *refs_out,
    uint32 ref_capacity
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint8 *bytes;
    uint32 ref_count;
    Size byte_count;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        refs_out == NULL || ref_capacity == 0 ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP super ref load")));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || first_ref_index >= bmp->super_ref_count)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP super ref range")));
    }
    ref_count = Min(
        ref_capacity,
        bmp->super_ref_count - first_ref_index
    );
    if (ref_count >
            MaxAllocSize / EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        bmp->first_super_ref > UINT32_MAX - first_ref_index ||
        bmp->super_refs_offset > SIZE_MAX -
            (uint64) (bmp->first_super_ref + first_ref_index) *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE)
    {
        evoke_segment_pages_report_codec_error(
            "semantic BMP super ref range",
            EVOKE_ERR_RANGE
        );
    }
    byte_count = (Size) ref_count *
        EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
    bytes = palloc(byte_count);
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (bmp->super_refs_offset +
            (uint64) (bmp->first_super_ref + first_ref_index) *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE),
        byte_count,
        bytes
    );
    for (uint32 ref_index = 0; ref_index < ref_count; ref_index++)
    {
        evoke_semantic_bmp_packed_super_ref packed;
        evoke_status status = evoke_semantic_bmp_packed_super_ref_decode(
            bytes + (Size) ref_index *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE,
            EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE,
            bmp->min_impact,
            bmp->max_impact,
            &packed
        );

        if (status != EVOKE_OK ||
            packed.superblock_id >= bmp->superblock_count ||
            packed.first_ref < bmp->first_ref ||
            packed.ref_count > bmp->ref_count -
                (packed.first_ref - bmp->first_ref) ||
            packed.first_impact < bmp->first_impact ||
            packed.first_impact - bmp->first_impact >
                bmp->posting_count ||
            (ref_index > 0 &&
             refs_out[ref_index - 1].superblock_id >=
                packed.superblock_id))
        {
            pfree(bytes);
            evoke_segment_pages_report_codec_error(
                "semantic BMP super ref",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        refs_out[ref_index].superblock_id = packed.superblock_id;
        refs_out[ref_index].first_ref = packed.first_ref;
        refs_out[ref_index].first_impact = packed.first_impact;
        refs_out[ref_index].ref_count = packed.ref_count;
        refs_out[ref_index].reserved = 0;
        refs_out[ref_index].min_impact = packed.min_impact;
        refs_out[ref_index].max_impact = packed.max_impact;
    }
    pfree(bytes);
    return ref_count;
}

uint32
evoke_segment_pages_load_query_bmp_super_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_semantic_bmp_super_ref *refs_out,
    uint32 ref_capacity
)
{
    evoke_segment_query_bmp_cached_super_ref *cached_refs;
    uint32 ref_count;

    if (refs_out == NULL || ref_capacity == 0 ||
        ref_capacity > MaxAllocSize / sizeof(*cached_refs))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP super ref load")));
    }
    cached_refs = palloc((Size) ref_capacity * sizeof(*cached_refs));
    ref_count = evoke_segment_pages_load_query_bmp_cached_super_ref_window(
        index_relation,
        context,
        plan,
        run_index,
        first_ref_index,
        cached_refs,
        ref_capacity
    );
    for (uint32 ref_index = 0; ref_index < ref_count; ref_index++)
    {
        refs_out[ref_index].superblock_id =
            cached_refs[ref_index].superblock_id;
        refs_out[ref_index].first_ref = cached_refs[ref_index].first_ref;
        refs_out[ref_index].ref_count = cached_refs[ref_index].ref_count;
        refs_out[ref_index].reserved = 0;
        refs_out[ref_index].min_impact =
            cached_refs[ref_index].min_impact;
        refs_out[ref_index].max_impact =
            cached_refs[ref_index].max_impact;
    }
    pfree(cached_refs);
    return ref_count;
}

static uint32
evoke_segment_pages_mask_count(uint64 mask)
{
    return (uint32) __builtin_popcountll((unsigned long long) mask);
}

static void
evoke_segment_pages_load_query_bmp_packed_super_ref(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_run *run,
    uint32 relative_super_ref,
    evoke_semantic_bmp_packed_super_ref *ref_out
)
{
    const evoke_segment_query_bmp_term *bmp = &run->semantic_bmp;
    uint8 bytes[EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE];
    evoke_status status;

    if (!bmp->available || relative_super_ref >= bmp->super_ref_count)
    {
        ereport(ERROR, (errmsg("invalid evoke packed super ref range")));
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (bmp->super_refs_offset +
            (uint64) (bmp->first_super_ref + relative_super_ref) *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE),
        sizeof(bytes),
        bytes
    );
    status = evoke_semantic_bmp_packed_super_ref_decode(
        bytes,
        sizeof(bytes),
        bmp->min_impact,
        bmp->max_impact,
        ref_out
    );
    if (status != EVOKE_OK || ref_out->first_ref < bmp->first_ref ||
        ref_out->ref_count > bmp->ref_count -
            (ref_out->first_ref - bmp->first_ref) ||
        ref_out->first_impact < bmp->first_impact ||
        ref_out->first_impact - bmp->first_impact > bmp->posting_count)
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic super ref",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }
}

uint32
evoke_segment_pages_load_query_bmp_block_membership(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint8 *membership_out,
    uint32 membership_capacity
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        membership_out == NULL || run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg(
            "invalid evoke semantic BMP membership load"
        )));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || bmp->block_membership_bytes == 0 ||
        membership_capacity < bmp->block_membership_bytes ||
        bmp->block_membership_offset > SIZE_MAX -
            bmp->first_block_membership_byte)
    {
        ereport(ERROR, (errmsg(
            "invalid evoke semantic BMP membership range"
        )));
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (bmp->block_membership_offset +
            bmp->first_block_membership_byte),
        (Size) bmp->block_membership_bytes,
        membership_out
    );
    return bmp->block_membership_bytes;
}

uint32
evoke_segment_pages_load_query_bmp_packed_ref_page(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 relative_ref_index,
    uint8 *refs_out,
    uint32 ref_capacity
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint64 absolute_ref;
    uint64 byte_offset;
    Size payload_capacity;
    Size page_remaining;
    uint32 page_ref_count;
    uint32 ref_count;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        refs_out == NULL || ref_capacity == 0 ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg(
            "invalid evoke semantic BMP packed ref page load"
        )));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || relative_ref_index >= bmp->ref_count ||
        bmp->first_ref > UINT32_MAX - relative_ref_index)
    {
        ereport(ERROR, (errmsg(
            "invalid evoke semantic BMP packed ref page range"
        )));
    }
    absolute_ref = (uint64) bmp->first_ref + relative_ref_index;
    if (absolute_ref > UINT64_MAX /
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE ||
        bmp->refs_offset > SIZE_MAX - absolute_ref *
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE)
    {
        evoke_segment_pages_report_codec_error(
            "semantic BMP packed ref page range",
            EVOKE_ERR_RANGE
        );
    }
    byte_offset = bmp->refs_offset + absolute_ref *
        EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
    payload_capacity = evoke_segment_page_payload_capacity();
    page_remaining = payload_capacity -
        (Size) (byte_offset % payload_capacity);
    page_ref_count = (uint32) (
        page_remaining / EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE
    );
    if (page_ref_count == 0)
    {
        /* One packed ref may straddle two immutable relation pages. */
        page_ref_count = 1;
    }
    ref_count = Min(
        ref_capacity,
        Min(bmp->ref_count - relative_ref_index, page_ref_count)
    );
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) byte_offset,
        (Size) ref_count * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
        refs_out
    );
    return ref_count;
}

uint32
evoke_segment_pages_load_query_bmp_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_semantic_bmp_ref *refs_out,
    uint32 ref_capacity
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint32 ref_count;
    uint32 output_count = 0;
    uint32 absolute_ref;
    uint32 super_low;
    uint32 super_high;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        refs_out == NULL || ref_capacity == 0 ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP ref load")));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || first_ref_index >= bmp->ref_count)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP ref range")));
    }
    ref_count = Min(ref_capacity, bmp->ref_count - first_ref_index);
    absolute_ref = bmp->first_ref + first_ref_index;
    super_low = 0;
    super_high = bmp->super_ref_count;
    while (super_low < super_high)
    {
        evoke_semantic_bmp_packed_super_ref super_ref;
        uint32 middle = super_low + (super_high - super_low) / 2;

        evoke_segment_pages_load_query_bmp_packed_super_ref(
            index_relation,
            context,
            run,
            middle,
            &super_ref
        );
        if (super_ref.first_ref + super_ref.ref_count <= absolute_ref)
        {
            super_low = middle + 1;
        }
        else
        {
            super_high = middle;
        }
    }
    while (output_count < ref_count && super_low < bmp->super_ref_count)
    {
        evoke_semantic_bmp_packed_super_ref super_ref;
        uint8 ref_bytes[16 * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE];
        uint32 impact_index;

        evoke_segment_pages_load_query_bmp_packed_super_ref(
            index_relation,
            context,
            run,
            super_low++,
            &super_ref
        );
        if (absolute_ref < super_ref.first_ref ||
            absolute_ref >= super_ref.first_ref + super_ref.ref_count)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic ref ownership",
                EVOKE_ERR_FORMAT
            );
        }
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &run->ref,
            (Size) (bmp->refs_offset +
                (uint64) super_ref.first_ref *
                    EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE),
            (Size) super_ref.ref_count *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
            ref_bytes
        );
        impact_index = super_ref.first_impact;
        for (uint32 child = 0;
             child < super_ref.ref_count && output_count < ref_count;
             child++)
        {
            evoke_semantic_bmp_packed_ref packed;
            uint32 child_absolute_ref = super_ref.first_ref + child;
            evoke_status status = evoke_semantic_bmp_packed_ref_decode(
                ref_bytes +
                    (Size) child * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
                bmp->min_impact,
                bmp->max_impact,
                &packed
            );
            uint32 child_posting_count;

            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic ref",
                    status
                );
            }
            child_posting_count =
                evoke_segment_pages_mask_count(packed.document_mask);
            if (
                impact_index < bmp->first_impact ||
                child_posting_count > bmp->posting_count -
                    (impact_index - bmp->first_impact))
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic ref",
                    status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
                );
            }
            if (child_absolute_ref >= absolute_ref)
            {
                evoke_semantic_bmp_ref *target =
                    &refs_out[output_count++];

                target->block_id =
                    (super_ref.superblock_id << UINT32_C(4)) +
                        packed.local_block_id;
                target->record_index = impact_index;
                target->min_impact = packed.min_impact;
                target->max_impact = packed.max_impact;
                absolute_ref++;
            }
            impact_index += child_posting_count;
        }
    }
    if (output_count != ref_count)
    {
        evoke_segment_pages_report_codec_error(
            "packed semantic ref window",
            EVOKE_ERR_FORMAT
        );
    }
    return output_count;
}

bool
evoke_segment_pages_load_query_bmp_cached_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    const evoke_segment_query_bmp_cached_super_ref *super_ref,
    uint32 block_id,
    evoke_semantic_bmp_record *record_out,
    float impacts_out[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS]
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    uint8 ref_bytes[16 * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE];
    uint8 impact_bytes[
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS * sizeof(uint32)
    ];
    size_t impact_width;
    evoke_status status;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        super_ref == NULL || record_out == NULL || impacts_out == NULL ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke cached semantic BMP record")));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    impact_width = evoke_semantic_bmp_impact_width(
        bmp->impact_precision
    );
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || block_id >= bmp->block_count ||
        impact_width == 0 ||
        super_ref->superblock_id != (block_id >> UINT32_C(4)) ||
        super_ref->first_ref < bmp->first_ref ||
        super_ref->ref_count > bmp->ref_count -
            (super_ref->first_ref - bmp->first_ref) ||
        super_ref->first_impact < bmp->first_impact ||
        super_ref->first_impact - bmp->first_impact > bmp->posting_count)
    {
        return false;
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (bmp->refs_offset +
            (uint64) super_ref->first_ref *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE),
        (Size) super_ref->ref_count *
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
        ref_bytes
    );
    record_out->first_impact = super_ref->first_impact;
    for (uint32 child = 0; child < super_ref->ref_count; child++)
    {
        evoke_semantic_bmp_packed_ref packed;
        uint32 impact_count;

        status = evoke_semantic_bmp_packed_ref_decode(
            ref_bytes + (Size) child *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
            bmp->min_impact,
            bmp->max_impact,
            &packed
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "cached semantic BMP record",
                status
            );
        }
        impact_count =
            evoke_segment_pages_mask_count(packed.document_mask);
        if (packed.local_block_id != (block_id & UINT32_C(15)))
        {
            record_out->first_impact += impact_count;
            continue;
        }
        if (record_out->first_impact < bmp->first_impact ||
            impact_count > bmp->posting_count -
                (record_out->first_impact - bmp->first_impact))
        {
            evoke_segment_pages_report_codec_error(
                "cached semantic BMP impact range",
                EVOKE_ERR_FORMAT
            );
        }
        record_out->term_id = plan->term_id;
        record_out->document_mask = packed.document_mask;
        record_out->impact_count = (uint16) impact_count;
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &run->ref,
            (Size) (bmp->impacts_offset +
                (uint64) record_out->first_impact * impact_width),
            (Size) impact_count * impact_width,
            impact_bytes
        );
        status = evoke_semantic_bmp_packed_impacts_decode(
            impact_bytes,
            (Size) impact_count * impact_width,
            impact_count,
            bmp->impact_precision,
            bmp->min_impact,
            bmp->max_impact,
            impacts_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "cached semantic BMP impacts",
                status
            );
        }
        return true;
    }
    return false;
}

bool
evoke_segment_pages_load_query_bmp_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_id,
    evoke_semantic_bmp_record *record_out,
    float impacts_out[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS]
)
{
    const evoke_segment_query_run *run;
    const evoke_segment_query_bmp_term *bmp;
    evoke_semantic_bmp_packed_super_ref super_ref;
    uint8 ref_bytes[16 * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE];
    uint8 impact_bytes[
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS * sizeof(uint32)
    ];
    uint32 low;
    uint32 high;
    size_t impact_width;
    evoke_status status;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        record_out == NULL || impacts_out == NULL ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic BMP record load")));
    }
    run = &plan->runs[run_index];
    bmp = &run->semantic_bmp;
    impact_width = evoke_semantic_bmp_impact_width(
        bmp->impact_precision
    );
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !bmp->available || block_id >= bmp->block_count ||
        impact_width == 0)
    {
        return false;
    }
    low = 0;
    high = bmp->super_ref_count;
    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;

        evoke_segment_pages_load_query_bmp_packed_super_ref(
            index_relation,
            context,
            run,
            middle,
            &super_ref
        );
        if (super_ref.superblock_id <
            (block_id >> UINT32_C(4)))
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= bmp->super_ref_count)
    {
        return false;
    }
    evoke_segment_pages_load_query_bmp_packed_super_ref(
        index_relation,
        context,
        run,
        low,
        &super_ref
    );
    if (super_ref.superblock_id !=
        (block_id >> UINT32_C(4)))
    {
        return false;
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (bmp->refs_offset +
            (uint64) super_ref.first_ref *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE),
        (Size) super_ref.ref_count *
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
        ref_bytes
    );
    record_out->first_impact = super_ref.first_impact;
    for (uint32 child = 0; child < super_ref.ref_count; child++)
    {
        evoke_semantic_bmp_packed_ref packed;
        uint32 impact_count;

        status = evoke_semantic_bmp_packed_ref_decode(
            ref_bytes +
                (Size) child * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
            bmp->min_impact,
            bmp->max_impact,
            &packed
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic record",
                status
            );
        }
        impact_count =
            evoke_segment_pages_mask_count(packed.document_mask);
        if (packed.local_block_id !=
            (block_id & UINT32_C(15)))
        {
            record_out->first_impact += impact_count;
            continue;
        }
        if (record_out->first_impact < bmp->first_impact ||
            impact_count > bmp->posting_count -
                (record_out->first_impact - bmp->first_impact))
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic impact range",
                EVOKE_ERR_FORMAT
            );
        }
        record_out->term_id = plan->term_id;
        record_out->document_mask = packed.document_mask;
        record_out->impact_count = (uint16) impact_count;
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &run->ref,
            (Size) (bmp->impacts_offset +
                (uint64) record_out->first_impact * impact_width),
            (Size) impact_count * impact_width,
            impact_bytes
        );
        status = evoke_semantic_bmp_packed_impacts_decode(
            impact_bytes,
            (Size) impact_count * impact_width,
            impact_count,
            bmp->impact_precision,
            bmp->min_impact,
            bmp->max_impact,
            impacts_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic impacts",
                status
            );
        }
        return true;
    }
    return false;
}

void
evoke_segment_pages_load_query_term_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    evoke_segment_query_block *block_out
)
{
    evoke_posting_block_record source_record;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        block_out == NULL || plan->term_id >= context->manifest.vocab_size ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term block load")));
    }
    evoke_segment_pages_load_query_term_block_record(
        index_relation,
        context,
        plan,
        run_index,
        block_index,
        &source_record
    );
    evoke_segment_pages_load_query_term_block_from_record(
        index_relation,
        context,
        plan,
        run_index,
        block_index,
        &source_record,
        block_out
    );
}

void
evoke_segment_pages_load_query_term_block_from_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    const evoke_posting_block_record *source_record,
    evoke_segment_query_block *block_out
)
{
    const evoke_segment_query_run *run;
    evoke_segment_query_block block;
    uint64 absolute_posting_offset;
    Size posting_bytes;
    evoke_status status;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        source_record == NULL || block_out == NULL ||
        plan->term_id >= context->manifest.vocab_size ||
        run_index >= plan->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query-term block load")));
    }
    run = &plan->runs[run_index];
    if (block_index >= run->block_count ||
        source_record->kind != run->kind ||
        source_record->posting_count == 0 ||
        source_record->posting_count >
            EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS ||
        source_record->posting_offset > run->posting_count ||
        source_record->posting_count >
            run->posting_count - source_record->posting_offset ||
        source_record->first_document_id >
            source_record->last_document_id ||
        source_record->last_document_id >=
            context->manifest.document_slot_count ||
        (source_record->first_document_id >> run->block_shift) !=
            source_record->block_id ||
        (source_record->last_document_id >> run->block_shift) !=
            source_record->block_id)
    {
        evoke_segment_pages_report_codec_error(
            "query-term posting block",
            EVOKE_ERR_FORMAT
        );
    }
    if (run->posting_offset >
            UINT64_MAX - source_record->posting_offset)
    {
        evoke_segment_pages_report_codec_error(
            "query-term posting offset",
            EVOKE_ERR_RANGE
        );
    }
    absolute_posting_offset =
        run->posting_offset + source_record->posting_offset;
    posting_bytes =
        (Size) source_record->posting_count * sizeof(uint32);
    memset(&block, 0, sizeof(block));
    block.source_posting_offset = source_record->posting_offset;
    block.record = *source_record;
    block.record.posting_offset = 0;
    if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        evoke_semantic_bmp_super_ref super_ref;
        evoke_semantic_bmp_ref refs[16];
        uint32 super_ref_index = block_index >>
            EVOKE_QUERY_BLOCKS_PER_BMP_SUPERBLOCK_SHIFT;
        uint32 loaded;
        uint32 posting_index = 0;

        if (evoke_segment_pages_load_query_bmp_super_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                super_ref_index,
                &super_ref,
                1) != 1 ||
            super_ref.first_ref < run->semantic_bmp.first_ref)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic block super ref",
                EVOKE_ERR_FORMAT
            );
        }
        loaded = evoke_segment_pages_load_query_bmp_ref_window(
            index_relation,
            context,
            plan,
            run_index,
            super_ref.first_ref - run->semantic_bmp.first_ref,
            refs,
            super_ref.ref_count
        );
        if (loaded != super_ref.ref_count)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic block refs",
                EVOKE_ERR_FORMAT
            );
        }
        for (uint32 ref_index = 0; ref_index < loaded; ref_index++)
        {
            evoke_semantic_bmp_record packed_record;
            float impacts[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS];
            uint64 mask;
            uint32 impact_index = 0;

            if ((refs[ref_index].block_id >>
                 EVOKE_BMP_BLOCKS_PER_QUERY_BLOCK_SHIFT) !=
                source_record->block_id)
            {
                continue;
            }
            if (!evoke_segment_pages_load_query_bmp_record(
                    index_relation,
                    context,
                    plan,
                    run_index,
                    refs[ref_index].block_id,
                    &packed_record,
                    impacts))
            {
                evoke_segment_pages_report_codec_error(
                    "packed semantic block record",
                    EVOKE_ERR_FORMAT
                );
            }
            mask = packed_record.document_mask;
            for (uint32 local_document = 0;
                 mask != 0;
                 local_document++, mask >>= 1)
            {
                if ((mask & UINT64_C(1)) == 0)
                {
                    continue;
                }
                if (posting_index >=
                    EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS)
                {
                    evoke_segment_pages_report_codec_error(
                        "packed semantic block capacity",
                        EVOKE_ERR_RANGE
                    );
                }
                block.document_slots[posting_index] =
                    (refs[ref_index].block_id <<
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) +
                        local_document;
                block.values[posting_index].impact = impacts[impact_index++];
                if (posting_index == 0)
                {
                    block.record.min_impact =
                        block.values[posting_index].impact;
                    block.record.max_impact =
                        block.values[posting_index].impact;
                }
                else
                {
                    block.record.min_impact = Min(
                        block.record.min_impact,
                        block.values[posting_index].impact
                    );
                    block.record.max_impact = Max(
                        block.record.max_impact,
                        block.values[posting_index].impact
                    );
                }
                posting_index++;
            }
        }
        if (posting_index != source_record->posting_count ||
            source_record->posting_offset > run->posting_count ||
            source_record->posting_count >
                run->posting_count - source_record->posting_offset)
        {
            evoke_segment_pages_report_codec_error(
                "packed semantic block mask",
                EVOKE_ERR_FORMAT
            );
        }
        goto validate_block;
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (run->indices_offset +
            absolute_posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) block.document_slots
    );
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &run->ref,
        (Size) (run->values_offset +
            absolute_posting_offset * sizeof(uint32)),
        posting_bytes,
        (uint8 *) block.values
    );
    for (uint32 posting_index = 0;
         posting_index < source_record->posting_count;
         posting_index++)
    {
        uint8 *slot_bytes = (uint8 *) block.document_slots +
            (Size) posting_index * sizeof(uint32);
        uint8 *value_bytes = (uint8 *) block.values +
            (Size) posting_index * sizeof(uint32);
        uint32 value_bits;

        block.document_slots[posting_index] =
            evoke_segment_pages_read_u32_le(slot_bytes);
        value_bits = evoke_segment_pages_read_u32_le(value_bytes);
        if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            block.values[posting_index].term_frequency = value_bits;
            if (value_bits == 0)
            {
                evoke_segment_pages_report_codec_error(
                    "query-term lexical posting",
                    EVOKE_ERR_FORMAT
                );
            }
        }
        else
        {
            memcpy(
                &block.values[posting_index].impact,
                &value_bits,
                sizeof(value_bits)
            );
            if (!isfinite(block.values[posting_index].impact))
            {
                evoke_segment_pages_report_codec_error(
                    "query-term impact posting",
                    EVOKE_ERR_FORMAT
                );
            }
        }
    }
    if (run->source == EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD)
    {
        evoke_segment_payload_disk_header header;

        memset(&header, 0, sizeof(header));
        header.flags = run->payload_flags;
        header.local_document_count = run->local_document_count;
        header.document_id_base = run->document_id_base;
        header.document_map_offset = run->document_map_offset;
        status = evoke_segment_pages_globalize_payload_slots(
            index_relation,
            &run->ref,
            &header,
            (uint32) context->manifest.document_slot_count,
            block.document_slots,
            source_record->posting_count,
            context->root.published_block_high_watermark,
            context->page_validation_cache
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "query-term payload block mapping",
                status
            );
        }
    }
    else if (run->source == EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD)
    {
        for (uint32 posting_index = 0;
             posting_index < source_record->posting_count;
             posting_index++)
        {
            uint32 document_slot = block.document_slots[posting_index];

            if (document_slot >= context->manifest.document_slot_count ||
                (posting_index > 0 &&
                 block.document_slots[posting_index - 1] >=
                    document_slot))
            {
                evoke_segment_pages_report_codec_error(
                    "query-term fold block mapping",
                    EVOKE_ERR_FORMAT
                );
            }
        }
    }
    else
    {
        evoke_segment_pages_report_codec_error(
            "query-term block source",
            EVOKE_ERR_FORMAT
        );
    }
validate_block:
    if (block.document_slots[0] != source_record->first_document_id ||
        block.document_slots[source_record->posting_count - 1] !=
            source_record->last_document_id)
    {
        evoke_segment_pages_report_codec_error(
            "query-term block bounds",
            EVOKE_ERR_FORMAT
        );
    }
    for (uint32 posting_index = 0;
         posting_index < source_record->posting_count;
         posting_index++)
    {
        if ((block.document_slots[posting_index] >> run->block_shift) !=
            source_record->block_id)
        {
            evoke_segment_pages_report_codec_error(
                "query-term block identity",
                EVOKE_ERR_FORMAT
            );
        }
    }

    block.view.indices = block.document_slots;
    block.view.values = block.values;
    block.view.blocks = &block.record;
    block.view.len = source_record->posting_count;
    block.view.local_document_count =
        (uint32) context->manifest.document_slot_count;
    block.view.block_count = 1;
    block.view.block_shift = run->block_shift;
    block.view.kind = run->kind;
    {
        evoke_index index_shape;

        memset(&index_shape, 0, sizeof(index_shape));
        index_shape.num_docs =
            (uint32) context->manifest.document_slot_count;
        status = evoke_posting_extent_validate_layout(
            &index_shape,
            &block.view
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "query-term block layout",
            status
        );
    }
    *block_out = block;
    block_out->view.indices = block_out->document_slots;
    block_out->view.values = block_out->values;
    block_out->view.blocks = &block_out->record;
}

static void
evoke_segment_pages_write_document_cow_objects_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_document_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *root_out
)
{
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (index_relation == NULL || tree == NULL ||
        tree->object_count == 0 ||
        owner_manifest_id == 0 || root_out == NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW document-object write"))
        );
    }
    memset(root_out, 0, sizeof(*root_out));

    PG_TRY();
    {
        for (uint64 object_id = 1;
             object_id <= tree->object_count;
             object_id++)
        {
            evoke_segment_object_ref storage_ref;

            if ((object_id & UINT64_C(0x3ff)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            status =
                evoke_document_cow_tree_prepare_object_for_storage(
                    tree,
                    object_id,
                    &bytes,
                    &size
                );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW document object",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY,
                object_id,
                owner_manifest_id,
                bytes,
                size,
                reuse_arena,
                &storage_ref
            );
            free(bytes);
            bytes = NULL;
            size = 0;
            status = evoke_document_cow_tree_bind_object_storage(
                tree,
                object_id,
                &storage_ref
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW document-object page binding",
                    status
                );
            }
        }
        status = evoke_document_cow_ref_as_segment_object_ref(
            &tree->root,
            root_out
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW document root",
                status
            );
        }
    }
    PG_FINALLY();
    {
        free(bytes);
    }
    PG_END_TRY();
}

void
evoke_segment_pages_write_document_cow_objects(
    Relation index_relation,
    evoke_document_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
)
{
    evoke_segment_pages_write_document_cow_objects_internal(
        index_relation,
        MAIN_FORKNUM,
        tree,
        owner_manifest_id,
        NULL,
        root_out
    );
}

void
evoke_segment_pages_load_document_record(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 document_slot,
    evoke_document_cow_record *record_out
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        record_out == NULL ||
        document_slot >= manifest->document_slot_count)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document lookup")));
    }
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_lookup_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            document_slot,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            record_out
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document lookup",
            status
        );
    }
}

static evoke_status
evoke_segment_pages_prepare_query_document_reader(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_query_document_reader *reader
)
{
    evoke_status status;

    if (index_relation == NULL || context == NULL || reader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (!reader->initialized)
    {
        status = evoke_segment_pages_open_document_cow(
            index_relation,
            &context->manifest.document_directory,
            context->root.published_block_high_watermark,
            context->manifest.manifest_id,
            reader
        );
        if (status == EVOKE_OK)
        {
            reader->page_validation_cache =
                context->page_validation_cache;
        }
    }
    else if (reader->index_relation != index_relation ||
             reader->published_block_high_watermark !=
                 context->root.published_block_high_watermark ||
             reader->max_owner_manifest_id !=
                 context->manifest.manifest_id ||
             reader->page_validation_cache !=
                 context->page_validation_cache)
    {
        status = EVOKE_ERR_FORMAT;
    }
    else
    {
        status = EVOKE_OK;
    }
    return status;
}

void
evoke_segment_pages_load_query_document_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_query_document_reader *reader,
    uint32 block_id,
    evoke_segment_query_document_block *block_out
)
{
    evoke_segment_query_document_block block;
    uint64 first_document_slot;
    uint64 remaining;
    evoke_status status;

    if (index_relation == NULL || context == NULL || reader == NULL ||
        block_out == NULL ||
        context->query_contract.block_shift !=
            EVOKE_DEFAULT_POSTING_BLOCK_SHIFT ||
        context->manifest.document_slot_count == 0 ||
        (context->manifest.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY) == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke query document block")));
    }
    first_document_slot = (uint64) block_id <<
        context->query_contract.block_shift;
    if (first_document_slot >= context->manifest.document_slot_count ||
        first_document_slot > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("invalid evoke document block index")));
    }
    remaining = context->manifest.document_slot_count -
        first_document_slot;
    memset(&block, 0, sizeof(block));
    block.block_id = block_id;
    block.first_document_slot = (uint32) first_document_slot;
    block.record_count = remaining >
        EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        ? EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        : (uint32) remaining;

    status = evoke_segment_pages_prepare_query_document_reader(
        index_relation,
        context,
        reader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_read_range_external(
            &reader->root_object.ref,
            context->manifest.document_slot_count,
            first_document_slot,
            block.record_count,
            evoke_segment_pages_load_document_cow_object,
            reader,
            block.records
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW query document block",
            status
        );
    }
    *block_out = block;
}

typedef struct evoke_segment_pages_document_record_visit
{
    evoke_segment_document_record_visitor visitor;
    void *visitor_context;
} evoke_segment_pages_document_record_visit;

static evoke_status
evoke_segment_pages_visit_document_object(
    void *context,
    const evoke_document_cow_object *object
)
{
    evoke_segment_pages_document_record_visit *visit = context;

    if (visit == NULL || visit->visitor == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (object->ref.kind != EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        return EVOKE_OK;
    }
    for (uint32 record_index = 0;
         record_index < object->value.leaf.record_count;
         record_index++)
    {
        visit->visitor(
            visit->visitor_context,
            &object->value.leaf.records[record_index]
        );
    }
    return EVOKE_OK;
}

void
evoke_segment_pages_visit_document_records(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_document_record_visitor visitor,
    void *visitor_context
)
{
    evoke_segment_pages_document_record_visit visit;
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        visitor == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document traversal")));
    }
    if (manifest->document_slot_count == 0)
    {
        return;
    }
    visit.visitor = visitor;
    visit.visitor_context = visitor_context;
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_visit_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            evoke_segment_pages_visit_document_object,
            &visit
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document traversal",
            status
        );
    }
}

void
evoke_segment_pages_visit_query_document_records(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_query_document_reader *reader,
    evoke_segment_document_record_visitor visitor,
    void *visitor_context
)
{
    evoke_segment_pages_document_record_visit visit;
    evoke_status status;

    if (index_relation == NULL || context == NULL || reader == NULL ||
        visitor == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke query document traversal")));
    }
    if (context->manifest.document_slot_count == 0)
    {
        return;
    }
    visit.visitor = visitor;
    visit.visitor_context = visitor_context;
    status = evoke_segment_pages_prepare_query_document_reader(
        index_relation,
        context,
        reader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_visit_external(
            &reader->root_object.ref,
            context->manifest.document_slot_count,
            evoke_segment_pages_load_document_cow_object,
            reader,
            evoke_segment_pages_visit_document_object,
            &visit
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW query document traversal",
            status
        );
    }
}

#define EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE UINT64_C(48)
#define EVOKE_SEGMENT_QUERY_VERSION_CHUNK_RECORDS UINT64_C(16384)

bool
evoke_segment_pages_load_query_document_lengths(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 *document_lengths
)
{
    uint8 *seen = NULL;
    uint8 *bytes = NULL;
    uint64 expected_version_count = 0;
    uint64 observed_version_count = 0;
    uint64 total_document_length = 0;
    Size seen_bytes;
    Size chunk_bytes;
    bool complete = false;

    if (index_relation == NULL || context == NULL ||
        document_lengths == NULL ||
        context->manifest.document_slot_count == 0 ||
        context->manifest.visible_document_count !=
            context->manifest.document_slot_count ||
        context->manifest.document_slot_count > UINT32_MAX)
    {
        return false;
    }
    if (context->resident_document_lengths != NULL &&
        context->resident_document_length_count ==
            context->manifest.document_slot_count)
    {
        memcpy(
            document_lengths,
            context->resident_document_lengths,
            (Size) context->manifest.document_slot_count * sizeof(uint32)
        );
        return true;
    }
    for (uint32 segment_index = 0;
         segment_index < context->manifest.segment_count;
         segment_index++)
    {
        const evoke_segment_descriptor *descriptor =
            &context->manifest.segments[segment_index];

        if (descriptor->retirement_count != 0 ||
            UINT64_MAX - expected_version_count <
                descriptor->document_count)
        {
            return false;
        }
        expected_version_count += descriptor->document_count;
    }
    if (expected_version_count != context->manifest.document_slot_count)
    {
        return false;
    }

    seen_bytes = (Size) ((context->manifest.document_slot_count + 7) / 8);
    chunk_bytes = (Size) (
        EVOKE_SEGMENT_QUERY_VERSION_CHUNK_RECORDS *
        EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE
    );
    seen = palloc0(seen_bytes);
    bytes = palloc(chunk_bytes);

    for (uint32 segment_index = 0;
         segment_index < context->manifest.segment_count;
         segment_index++)
    {
        const evoke_segment_descriptor *descriptor =
            &context->manifest.segments[segment_index];
        evoke_segment_object_ref ref;
        evoke_segment_payload_disk_header header;
        uint64 version_index = 0;
        evoke_status status;

        status = evoke_segment_pages_read_payload_header(
            index_relation,
            &context->root,
            &context->manifest,
            descriptor,
            &ref,
            &header,
            context
        );
        if (status != EVOKE_OK ||
            header.version_count != descriptor->document_count ||
            header.retirement_count != 0 ||
            header.versions_offset > header.total_size ||
            (uint64) header.version_count >
                (header.total_size - header.versions_offset) /
                    EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE)
        {
            goto done;
        }

        while (version_index < header.version_count)
        {
            uint64 chunk_record_count =
                header.version_count - version_index;
            Size read_bytes;

            if (chunk_record_count >
                EVOKE_SEGMENT_QUERY_VERSION_CHUNK_RECORDS)
            {
                chunk_record_count =
                    EVOKE_SEGMENT_QUERY_VERSION_CHUNK_RECORDS;
            }
            read_bytes = (Size) (
                chunk_record_count *
                EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE
            );
            evoke_segment_pages_read_plan_range(
                index_relation,
                context,
                &ref,
                (Size) (
                    header.versions_offset +
                    version_index *
                        EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE
                ),
                read_bytes,
                bytes
            );
            for (uint64 chunk_index = 0;
                 chunk_index < chunk_record_count;
                 chunk_index++)
            {
                const uint8 *record = bytes +
                    (Size) (
                        chunk_index *
                        EVOKE_SEGMENT_QUERY_VERSION_RECORD_SIZE
                    );
                uint64 document_slot =
                    evoke_segment_pages_read_u64_le(record + 0);
                uint64 born_sequence =
                    evoke_segment_pages_read_u64_le(record + 8);
                uint32 record_xid =
                    evoke_segment_pages_read_u32_le(record + 16);
                uint32 document_length =
                    evoke_segment_pages_read_u32_le(record + 24);
                uint16 heap_offset =
                    evoke_segment_pages_read_u16_le(record + 28);
                uint16 flags =
                    evoke_segment_pages_read_u16_le(record + 30);
                uint8 mask;

                if (document_slot >=
                        context->manifest.document_slot_count ||
                    born_sequence < descriptor->min_sequence ||
                    born_sequence > descriptor->max_sequence ||
                    heap_offset == 0 ||
                    (flags &
                     EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0 ||
                    (((flags &
                       EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID) != 0) !=
                     (record_xid == 0)))
                {
                    goto done;
                }
                mask = (uint8) (UINT8_C(1) << (document_slot & 7));
                if ((seen[document_slot >> 3] & mask) != 0 ||
                    UINT64_MAX - total_document_length < document_length)
                {
                    goto done;
                }
                seen[document_slot >> 3] |= mask;
                document_lengths[document_slot] = document_length;
                total_document_length += document_length;
                observed_version_count++;
            }
            version_index += chunk_record_count;
        }
    }
    complete =
        observed_version_count == context->manifest.document_slot_count &&
        total_document_length == context->manifest.total_document_length;

done:
    pfree(bytes);
    pfree(seen);
    return complete;
}

void
evoke_segment_pages_load_matching_live_born_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    size_t limit,
    evoke_document_cow_record_predicate predicate,
    void *predicate_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        record_count_out == NULL || stats_out == NULL ||
        limit > record_capacity ||
        (record_capacity > 0 && records_out == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke live born-prefix request")));
    }
    *record_count_out = 0;
    memset(stats_out, 0, sizeof(*stats_out));
    if (limit == 0 || manifest->document_slot_count == 0 ||
        manifest->visible_document_count == 0)
    {
        return;
    }
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_collect_live_born_prefix_matching_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            limit,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            predicate,
            predicate_context,
            records_out,
            record_capacity,
            record_count_out,
            stats_out
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW live born-prefix query",
            status
        );
    }
}

void
evoke_segment_pages_load_live_born_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    size_t limit,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
)
{
    evoke_segment_pages_load_matching_live_born_prefix(
        index_relation,
        root,
        manifest,
        limit,
        NULL,
        NULL,
        records_out,
        record_capacity,
        record_count_out,
        stats_out
    );
}

void
evoke_segment_pages_load_document_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 *live_document_count_out,
    uint64 *semantic_pending_count_out,
    int64 *earliest_retry_after_out
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        live_document_count_out == NULL ||
        semantic_pending_count_out == NULL ||
        earliest_retry_after_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document summary")));
    }
    *live_document_count_out = 0;
    *semantic_pending_count_out = 0;
    *earliest_retry_after_out = INT64_MAX;
    if (manifest->document_slot_count == 0)
    {
        return;
    }

    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document summary",
            status
        );
    }
    *live_document_count_out =
        loader.root_object.ref.live_document_count;
    *semantic_pending_count_out =
        loader.root_object.ref.semantic_pending_count;
    *earliest_retry_after_out =
        loader.root_object.ref.earliest_retry_after;
}

void
evoke_segment_pages_load_document_length_extrema(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_document_cow_length_extrema *extrema_out
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        extrema_out == NULL || manifest->document_slot_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document extrema")));
    }
    memset(extrema_out, 0, sizeof(*extrema_out));
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_length_extrema_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            0,
            manifest->document_slot_count,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            extrema_out
        );
    }
    if (status != EVOKE_OK || extrema_out->document_count == 0)
    {
        evoke_segment_pages_report_codec_error(
            "COW document length extrema",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }
}

void
evoke_segment_pages_load_document_block_extrema(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    uint32 block_id,
    evoke_document_cow_length_extrema *extrema_out
)
{
    evoke_document_cow_page_loader_context loader;
    uint64 block_size;
    uint64 first_document_slot;
    uint64 range_document_slot_count;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        query_contract == NULL || extrema_out == NULL ||
        evoke_segment_query_contract_validate(
            query_contract,
            manifest) != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW document-block lookup"))
        );
    }
    memset(extrema_out, 0, sizeof(*extrema_out));
    block_size = UINT64_C(1) << query_contract->block_shift;
    first_document_slot =
        (uint64) block_id << query_contract->block_shift;
    if (first_document_slot >= manifest->document_slot_count)
    {
        ereport(
            ERROR,
            (errmsg("evoke document block is outside the manifest"))
        );
    }
    range_document_slot_count =
        manifest->document_slot_count - first_document_slot;
    if (range_document_slot_count > block_size)
    {
        range_document_slot_count = block_size;
    }

    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_length_extrema_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            first_document_slot,
            range_document_slot_count,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            extrema_out
        );
    }
    if (status != EVOKE_OK || extrema_out->document_count == 0)
    {
        evoke_segment_pages_report_codec_error(
            "COW document-block lookup",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }
}

static void
evoke_segment_pages_load_document_block_extrema_all(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        query_contract == NULL || extrema_out == NULL ||
        extrema_count_out == NULL ||
        evoke_segment_query_contract_validate(
            query_contract,
            manifest) != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW document-block materialization"))
        );
    }
    *extrema_out = NULL;
    *extrema_count_out = 0;
    if (manifest->document_slot_count == 0)
    {
        return;
    }

    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_block_extrema_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            query_contract->block_shift,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            extrema_out,
            extrema_count_out
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_document_cow_block_extrema_free(*extrema_out);
        *extrema_out = NULL;
        *extrema_count_out = 0;
        evoke_segment_pages_report_codec_error(
            "COW document-block materialization",
            status
        );
    }
}

bool
evoke_segment_pages_find_actionable_document(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    int64 now,
    evoke_document_cow_record *record_out
)
{
    return evoke_segment_pages_find_actionable_document_from(
        index_relation,
        root,
        manifest,
        0,
        now,
        record_out
    );
}

bool
evoke_segment_pages_find_actionable_document_from(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 first_document_slot,
    int64 now,
    evoke_document_cow_record *record_out
)
{
    evoke_document_cow_page_loader_context loader;
    bool found = false;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        record_out == NULL ||
        first_document_slot >= manifest->document_slot_count)
    {
        return false;
    }
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_find_actionable_external_from(
            &loader.root_object.ref,
            manifest->document_slot_count,
            first_document_slot,
            now,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            record_out,
            &found
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW actionable document lookup",
            status
        );
    }
    return found;
}

bool
evoke_segment_pages_find_reusable_document_from(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 first_document_slot,
    evoke_document_cow_record *record_out
)
{
    evoke_document_cow_page_loader_context loader;
    bool found = false;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        record_out == NULL ||
        first_document_slot >= manifest->document_slot_count)
    {
        return false;
    }
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_find_reusable_external_from(
            &loader.root_object.ref,
            manifest->document_slot_count,
            first_document_slot,
            evoke_segment_pages_load_document_cow_object,
            &loader,
            record_out,
            &found
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW reusable-document lookup",
            status
        );
    }
    return found;
}

static void
evoke_segment_pages_open_maintenance_term_cow(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_term_cow_page_loader_context *loader
)
{
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        loader == NULL || root->root_id != manifest->manifest_id ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke maintenance COW root")));
    }
    status = evoke_segment_pages_open_term_cow(
        index_relation,
        &manifest->term_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        loader
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "maintenance COW term root",
            status
        );
    }
}

bool
evoke_segment_pages_term_cow_append_has_capacity(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *new_payload
)
{
    evoke_term_cow_page_loader_context loader;
    evoke_segment_payload_view payload_view;
    bool has_capacity = false;
    evoke_status status;

    if (new_payload == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW append capacity check")));
    }
    evoke_segment_pages_open_maintenance_term_cow(
        index_relation,
        root,
        manifest,
        &loader
    );
    evoke_segment_payload_as_view(new_payload, &payload_view);
    status = evoke_term_cow_append_has_capacity_external(
        &loader.root_object.ref,
        manifest->vocab_size,
        &payload_view,
        evoke_segment_pages_load_term_cow_object,
        &loader,
        &has_capacity
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW append capacity",
            status
        );
    }
    return has_capacity;
}

bool
evoke_segment_pages_find_term_extent_pressure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 threshold,
    evoke_term_cow_record *record_out
)
{
    evoke_term_cow_page_loader_context loader;
    bool found = false;
    evoke_status status;

    if (record_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW pressure lookup")));
    }
    evoke_segment_pages_open_maintenance_term_cow(
        index_relation,
        root,
        manifest,
        &loader
    );
    status = evoke_term_cow_find_extent_pressure_external(
        &loader.root_object.ref,
        manifest->vocab_size,
        threshold,
        evoke_segment_pages_load_term_cow_object,
        &loader,
        record_out,
        &found
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW extent-pressure lookup",
            status
        );
    }
    return found;
}

bool
evoke_segment_pages_cow_replace_fold_safe(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 first_segment_index,
    const evoke_segment_payload *payloads,
    uint32 payload_count,
    uint32 *conflict_term_id_out
)
{
    evoke_term_cow_page_loader_context loader;
    evoke_segment_payload_view *payload_views;
    bool safe = false;
    evoke_status status;

    if (payloads == NULL || payload_count < 2 ||
        conflict_term_id_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW fold preflight")));
    }
    *conflict_term_id_out = UINT32_MAX;
    evoke_segment_pages_open_maintenance_term_cow(
        index_relation,
        root,
        manifest,
        &loader
    );
    payload_views = palloc0(
        (Size) payload_count * sizeof(*payload_views)
    );
    for (uint32 payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        evoke_segment_payload_as_view(
            &payloads[payload_index],
            &payload_views[payload_index]
        );
    }
    status = evoke_term_cow_replace_fold_preflight_external(
        &loader.root_object.ref,
        manifest,
        first_segment_index,
        payload_views,
        payload_count,
        evoke_segment_pages_load_term_cow_object,
        &loader,
        &safe,
        conflict_term_id_out
    );
    pfree(payload_views);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW replacement fold preflight",
            status
        );
    }
    return safe;
}

void
evoke_l0_storage_snapshot_init(evoke_l0_storage_snapshot *snapshot)
{
    if (snapshot != NULL)
    {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}

void
evoke_l0_storage_snapshot_free(evoke_l0_storage_snapshot *snapshot)
{
    uint32 record_index;

    if (snapshot == NULL)
    {
        return;
    }
    for (record_index = 0;
         record_index < snapshot->record_count;
         record_index++)
    {
        if (snapshot->records[record_index].bytes != NULL)
        {
            pfree(snapshot->records[record_index].bytes);
        }
    }
    if (snapshot->records != NULL)
    {
        pfree(snapshot->records);
    }
    if (snapshot->page_blocks != NULL)
    {
        pfree(snapshot->page_blocks);
    }
    memset(snapshot, 0, sizeof(*snapshot));
}

typedef void (*evoke_l0_record_emitter)(
    void *context,
    uint8 **record_bytes,
    Size record_size,
    const evoke_l0_record_view *record
);

typedef struct evoke_l0_stream_state
{
    uint32 *page_blocks;
    uint32 record_capacity;
    uint32 page_capacity;
    evoke_l0_visit_stats stats;
    evoke_l0_record_emitter emit;
    void *emit_context;
} evoke_l0_stream_state;

static void
evoke_segment_pages_stream_l0_frontier(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_active_l0_frontier *frontier,
    evoke_l0_stream_state *state
)
{
    uint8 *record_bytes = NULL;
    uint32 start_record_count;
    uint32 start_page_count;
    uint64 start_payload_bytes;
    uint64 expected_sequence;
    uint64 suffix_expected_sequence;
    uint64 previous_page_max = 0;
    uint64 record_checksum = 0;
    uint32 record_bytes_expected = 0;
    uint32 record_bytes_used = 0;
    uint32 current_block;
    volatile Buffer active_buffer = InvalidBuffer;

    if (frontier->record_count == 0)
    {
        return;
    }
    if (state == NULL || state->emit == NULL ||
        state->stats.record_count > state->record_capacity ||
        frontier->record_count >
            state->record_capacity - state->stats.record_count ||
        state->stats.page_count > state->page_capacity ||
        frontier->page_count >
            state->page_capacity - state->stats.page_count ||
        state->page_blocks == NULL ||
        frontier->payload_bytes >
            (uint64) frontier->page_count * BLCKSZ)
    {
        ereport(ERROR, (errmsg("evoke linked L0 frontier is too large")));
    }

    start_record_count = state->stats.record_count;
    start_page_count = state->stats.page_count;
    start_payload_bytes = state->stats.payload_bytes;
    expected_sequence = frontier->min_sequence;
    suffix_expected_sequence = frontier->max_sequence == UINT64_MAX
        ? 0
        : frontier->max_sequence + 1;
    current_block = frontier->head_block;

    PG_TRY();
    {
        for (uint32 page_index = 0;
             page_index < frontier->page_count;
             page_index++)
        {
            Page page;
            PageHeader postgres_header;
            const uint8 *content;
            const uint8 *payload;
            evoke_active_l0_page_header page_header;
            uint64 observed_min_sequence = 0;
            uint64 observed_max_sequence = 0;
            Size minimum_content;
            Size offset = 0;
            bool frontier_tail_page =
                page_index + 1 == frontier->page_count;
            evoke_status status;

            if (current_block == 0 ||
                current_block == EVOKE_ACTIVE_L0_NO_NEXT_BLOCK ||
                current_block >= root->published_block_high_watermark)
            {
                ereport(ERROR, (errmsg("invalid evoke linked L0 block")));
            }
            for (uint32 prior_index = 0;
                 prior_index < start_page_count + page_index;
                 prior_index++)
            {
                if (state->page_blocks[prior_index] == current_block)
                {
                    if (prior_index < start_page_count)
                    {
                        ereport(
                            ERROR,
                            (errmsg("overlapping evoke linked L0 frontiers"))
                        );
                    }
                    ereport(ERROR, (errmsg("cyclic evoke linked L0 chain")));
                }
            }
            state->page_blocks[
                start_page_count + page_index
            ] = current_block;
            active_buffer = ReadBufferExtended(
                index_relation,
                MAIN_FORKNUM,
                current_block,
                RBM_NORMAL,
                NULL
            );
            LockBuffer(active_buffer, BUFFER_LOCK_SHARE);
            page = BufferGetPage(active_buffer);
            if (PageIsNew(page))
            {
                ereport(ERROR, (errmsg("empty evoke linked L0 page")));
            }
            postgres_header = (PageHeader) page;
            content = (const uint8 *) PageGetContents(page);
            status = evoke_active_l0_page_header_deserialize(
                content,
                EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
                evoke_segment_page_content_bytes(),
                &page_header
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "linked L0 page header",
                    status
                );
            }
            minimum_content =
                MAXALIGN(SizeOfPageHeaderData) +
                EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE +
                page_header.used_bytes;
            if (postgres_header->pd_lower < minimum_content ||
                page_header.segment_id != frontier->segment_id ||
                page_header.ordinal != page_index ||
                page_header.min_sequence < frontier->min_sequence ||
                page_header.min_sequence > frontier->max_sequence ||
                (!frontier_tail_page &&
                 page_header.max_sequence > frontier->max_sequence) ||
                (page_index == 0 &&
                 page_header.min_sequence != frontier->min_sequence) ||
                (page_index > 0 &&
                 (page_header.min_sequence < previous_page_max ||
                  page_header.min_sequence > previous_page_max + 1)))
            {
                ereport(ERROR, (errmsg("invalid evoke linked L0 page order")));
            }
            payload = content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE;
            status = evoke_active_l0_page_payload_validate(
                &page_header,
                payload,
                page_header.used_bytes
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "linked L0 page payload",
                    status
                );
            }

            for (uint32 frame_index = 0;
                 frame_index < page_header.frame_count;
                 frame_index++)
            {
                evoke_l0_frame_header frame;
                bool is_start;
                bool is_end;
                bool root_visible;

                if (offset > page_header.used_bytes ||
                    page_header.used_bytes - offset <
                        EVOKE_L0_FRAME_HEADER_SIZE)
                {
                    ereport(ERROR, (errmsg("truncated evoke L0 frame")));
                }
                status = evoke_l0_frame_header_deserialize(
                    payload + offset,
                    EVOKE_L0_FRAME_HEADER_SIZE,
                    &frame
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "linked L0 frame",
                        status
                    );
                }
                offset += EVOKE_L0_FRAME_HEADER_SIZE;
                if (frame.fragment_bytes >
                    page_header.used_bytes - offset)
                {
                    ereport(ERROR, (errmsg("truncated evoke L0 fragment")));
                }
                if (frame.sequence < page_header.min_sequence ||
                    frame.sequence > page_header.max_sequence)
                {
                    ereport(ERROR, (errmsg("invalid evoke L0 frame sequence")));
                }
                if (observed_min_sequence == 0)
                {
                    observed_min_sequence = frame.sequence;
                }
                if (observed_max_sequence > frame.sequence)
                {
                    ereport(ERROR, (errmsg("unordered evoke L0 frames")));
                }
                observed_max_sequence = frame.sequence;
                is_start =
                    (frame.flags & EVOKE_L0_FRAME_FLAG_START) != 0;
                is_end =
                    (frame.flags & EVOKE_L0_FRAME_FLAG_END) != 0;
                root_visible = frame.sequence <= frontier->max_sequence;

                /*
                 * A page is append-only, but its header and payload can have
                 * advanced after this reader copied an older metapage root.
                 * The old root still owns an immutable logical prefix. Verify
                 * complete newer inline records, then exclude that suffix
                 * from the root-relative snapshot.
                 */
                if (!root_visible)
                {
                    if (!frontier_tail_page || record_bytes != NULL ||
                        suffix_expected_sequence == 0 ||
                        frame.sequence != suffix_expected_sequence ||
                        !is_start || !is_end ||
                        frame.fragment_offset != 0 ||
                        frame.fragment_bytes != frame.record_bytes ||
                        evoke_segment_blob_checksum(
                            payload + offset,
                            frame.fragment_bytes
                        ) != frame.record_checksum)
                    {
                        ereport(
                            ERROR,
                            (errmsg("invalid evoke L0 page suffix"))
                        );
                    }
                    offset += frame.fragment_bytes;
                    suffix_expected_sequence =
                        suffix_expected_sequence == UINT64_MAX
                            ? 0
                            : suffix_expected_sequence + 1;
                    continue;
                }

                if (is_start)
                {
                    if (record_bytes != NULL ||
                        frame.sequence != expected_sequence ||
                        frame.record_bytes > frontier->payload_bytes ||
                        frame.record_bytes >
                            (uint64) frontier->page_count * BLCKSZ)
                    {
                        ereport(
                            ERROR,
                            (errmsg("invalid evoke L0 record start"))
                        );
                    }
                    record_bytes = palloc(frame.record_bytes);
                    record_checksum = frame.record_checksum;
                    record_bytes_expected = frame.record_bytes;
                    record_bytes_used = 0;
                }
                if (record_bytes == NULL ||
                    frame.sequence != expected_sequence ||
                    frame.record_checksum != record_checksum ||
                    frame.record_bytes != record_bytes_expected ||
                    frame.fragment_offset != record_bytes_used)
                {
                    ereport(ERROR, (errmsg("invalid evoke L0 fragment chain")));
                }
                memcpy(
                    record_bytes + record_bytes_used,
                    payload + offset,
                    frame.fragment_bytes
                );
                record_bytes_used += frame.fragment_bytes;
                offset += frame.fragment_bytes;

                if (is_end)
                {
                    evoke_l0_record_view record;

                    if (record_bytes_used != record_bytes_expected ||
                        evoke_segment_blob_checksum(
                            record_bytes,
                            record_bytes_used
                        ) != record_checksum ||
                        state->stats.record_count >=
                            state->record_capacity)
                    {
                        ereport(ERROR, (errmsg("invalid evoke L0 record end")));
                    }
                    status = evoke_l0_record_view_parse(
                        record_bytes,
                        record_bytes_used,
                        &record
                    );
                    if (status != EVOKE_OK ||
                        record.sequence != expected_sequence ||
                        record.document_slot >=
                            root->next_document_slot)
                    {
                        evoke_segment_pages_report_codec_error(
                            "linked L0 logical record",
                            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
                        );
                    }
                    state->emit(
                        state->emit_context,
                        &record_bytes,
                        record_bytes_used,
                        &record
                    );
                    state->stats.record_count++;
                    state->stats.payload_bytes += record_bytes_used;
                    if (record_bytes != NULL)
                    {
                        pfree(record_bytes);
                        record_bytes = NULL;
                    }
                    record_checksum = 0;
                    record_bytes_expected = 0;
                    record_bytes_used = 0;
                    expected_sequence++;
                }
            }
            if (offset != page_header.used_bytes ||
                observed_min_sequence != page_header.min_sequence ||
                observed_max_sequence != page_header.max_sequence)
            {
                ereport(ERROR, (errmsg("invalid evoke L0 page closure")));
            }
            previous_page_max = Min(
                page_header.max_sequence,
                frontier->max_sequence
            );
            if (frontier_tail_page)
            {
                if (current_block != frontier->tail_block)
                {
                    ereport(ERROR, (errmsg("invalid evoke L0 chain tail")));
                }
            }
            else if (page_header.next_block ==
                         EVOKE_ACTIVE_L0_NO_NEXT_BLOCK)
            {
                ereport(ERROR, (errmsg("truncated evoke L0 page chain")));
            }
            current_block = page_header.next_block;
            UnlockReleaseBuffer(active_buffer);
            active_buffer = InvalidBuffer;
        }

        if (record_bytes != NULL ||
            state->stats.record_count - start_record_count !=
                frontier->record_count ||
            state->stats.payload_bytes - start_payload_bytes !=
                frontier->payload_bytes ||
            expected_sequence != frontier->max_sequence + 1)
        {
            ereport(ERROR, (errmsg("invalid evoke linked L0 frontier")));
        }
        state->stats.page_count += frontier->page_count;
    }
    PG_FINALLY();
    {
        if (BufferIsValid(active_buffer))
        {
            UnlockReleaseBuffer(active_buffer);
        }
        if (record_bytes != NULL)
        {
            pfree(record_bytes);
        }
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_collect_l0_record(
    void *context,
    uint8 **record_bytes,
    Size record_size,
    const evoke_l0_record_view *record
)
{
    evoke_l0_storage_snapshot *snapshot = context;
    evoke_l0_stored_record *stored;

    if (snapshot == NULL || record_bytes == NULL ||
        *record_bytes == NULL || record == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke L0 record collection")));
    }
    stored = &snapshot->records[snapshot->record_count];
    stored->bytes = *record_bytes;
    stored->size = record_size;
    stored->view = *record;
    snapshot->record_count++;
    *record_bytes = NULL;
}

static void
evoke_segment_pages_load_l0_frontiers(
    Relation index_relation,
    const evoke_segment_read_root *root,
    bool include_pending,
    bool include_active,
    evoke_l0_storage_snapshot *snapshot_out
)
{
    evoke_l0_storage_snapshot snapshot;
    evoke_l0_stream_state stream;
    uint64 total_records;
    uint64 expected_payload_bytes;
    uint32 expected_page_count;
    evoke_status status;

    if (index_relation == NULL || root == NULL || snapshot_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke linked L0 snapshot load")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    total_records =
        (include_pending
            ? (uint64) root->pending_l0.record_count
            : 0) +
        (include_active
            ? (uint64) root->active_l0.record_count
            : 0);
    expected_page_count =
        (include_pending ? root->pending_l0.page_count : 0) +
        (include_active ? root->active_l0.page_count : 0);
    expected_payload_bytes =
        (include_pending ? root->pending_l0.payload_bytes : 0) +
        (include_active ? root->active_l0.payload_bytes : 0);
    if (total_records >
        (uint64) EVOKE_ACTIVE_L0_MAX_RECORDS * 2)
    {
        ereport(ERROR, (errmsg("evoke linked L0 record bound exceeded")));
    }

    evoke_l0_storage_snapshot_init(&snapshot);
    memset(&stream, 0, sizeof(stream));
    if (total_records > 0)
    {
        snapshot.records = palloc0(
            sizeof(*snapshot.records) * (Size) total_records
        );
    }
    if (expected_page_count > 0)
    {
        snapshot.page_blocks = palloc0(
            sizeof(*snapshot.page_blocks) * (Size) expected_page_count
        );
    }
    stream.page_blocks = snapshot.page_blocks;
    stream.record_capacity = (uint32) total_records;
    stream.page_capacity = expected_page_count;
    stream.emit = evoke_segment_pages_collect_l0_record;
    stream.emit_context = &snapshot;
    PG_TRY();
    {
        if (include_pending)
        {
            evoke_segment_pages_stream_l0_frontier(
                index_relation,
                root,
                &root->pending_l0,
                &stream
            );
        }
        if (include_active)
        {
            evoke_segment_pages_stream_l0_frontier(
                index_relation,
                root,
                &root->active_l0,
                &stream
            );
        }
        snapshot.page_count = stream.stats.page_count;
        snapshot.payload_bytes = stream.stats.payload_bytes;
        if (snapshot.record_count != stream.stats.record_count ||
            snapshot.record_count != total_records ||
            snapshot.page_count != expected_page_count ||
            snapshot.payload_bytes != expected_payload_bytes)
        {
            ereport(ERROR, (errmsg("invalid evoke linked L0 snapshot")));
        }
        evoke_l0_storage_snapshot_free(snapshot_out);
        *snapshot_out = snapshot;
        evoke_l0_storage_snapshot_init(&snapshot);
    }
    PG_FINALLY();
    {
        evoke_l0_storage_snapshot_free(&snapshot);
    }
    PG_END_TRY();
}

void
evoke_segment_pages_load_l0_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_l0_storage_snapshot *snapshot_out
)
{
    evoke_segment_pages_load_l0_frontiers(
        index_relation,
        root,
        true,
        true,
        snapshot_out
    );
}

void
evoke_segment_pages_load_pending_l0_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_l0_storage_snapshot *snapshot_out
)
{
    evoke_segment_pages_load_l0_frontiers(
        index_relation,
        root,
        true,
        false,
        snapshot_out
    );
}

typedef struct evoke_l0_visitor_emitter_context
{
    evoke_l0_record_visitor visitor;
    void *visitor_context;
} evoke_l0_visitor_emitter_context;

static void
evoke_segment_pages_emit_l0_record(
    void *context,
    uint8 **record_bytes,
    Size record_size,
    const evoke_l0_record_view *record
)
{
    evoke_l0_visitor_emitter_context *emitter = context;

    (void) record_bytes;
    (void) record_size;
    emitter->visitor(emitter->visitor_context, record);
}

void
evoke_segment_pages_visit_l0_records(
    Relation index_relation,
    const evoke_segment_read_root *root,
    bool include_pending,
    bool include_active,
    evoke_l0_record_visitor visitor,
    void *visitor_context,
    evoke_l0_visit_stats *stats_out
)
{
    evoke_l0_stream_state stream;
    evoke_l0_visitor_emitter_context emitter;
    uint32 *page_blocks = NULL;
    uint64 total_records;
    uint64 expected_payload_bytes;
    uint32 expected_page_count;
    evoke_status status;

    if (index_relation == NULL || root == NULL || visitor == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke linked L0 record visitor")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    total_records =
        (include_pending
            ? (uint64) root->pending_l0.record_count
            : 0) +
        (include_active
            ? (uint64) root->active_l0.record_count
            : 0);
    expected_page_count =
        (include_pending ? root->pending_l0.page_count : 0) +
        (include_active ? root->active_l0.page_count : 0);
    expected_payload_bytes =
        (include_pending ? root->pending_l0.payload_bytes : 0) +
        (include_active ? root->active_l0.payload_bytes : 0);
    if (total_records >
        (uint64) EVOKE_ACTIVE_L0_MAX_RECORDS * 2)
    {
        ereport(ERROR, (errmsg("evoke linked L0 record bound exceeded")));
    }

    memset(&stream, 0, sizeof(stream));
    emitter.visitor = visitor;
    emitter.visitor_context = visitor_context;
    if (expected_page_count > 0)
    {
        page_blocks = palloc(
            sizeof(*page_blocks) * (Size) expected_page_count
        );
    }
    stream.page_blocks = page_blocks;
    stream.record_capacity = (uint32) total_records;
    stream.page_capacity = expected_page_count;
    stream.emit = evoke_segment_pages_emit_l0_record;
    stream.emit_context = &emitter;

    PG_TRY();
    {
        if (include_pending)
        {
            evoke_segment_pages_stream_l0_frontier(
                index_relation,
                root,
                &root->pending_l0,
                &stream
            );
        }
        if (include_active)
        {
            evoke_segment_pages_stream_l0_frontier(
                index_relation,
                root,
                &root->active_l0,
                &stream
            );
        }
        if (stream.stats.record_count != total_records ||
            stream.stats.page_count != expected_page_count ||
            stream.stats.payload_bytes != expected_payload_bytes)
        {
            ereport(ERROR, (errmsg("invalid evoke linked L0 stream")));
        }
        if (stats_out != NULL)
        {
            *stats_out = stream.stats;
        }
    }
    PG_FINALLY();
    {
        if (page_blocks != NULL)
        {
            pfree(page_blocks);
        }
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_validate_term_cow_closure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    evoke_term_cow_page_loader_context loader;
    evoke_term_directory directory;
    evoke_segment_query_contract contract;
    uint32 *doc_frequencies = NULL;
    evoke_segment_object_ref *catalog_refs = NULL;
    evoke_term_cow_fold_state *fold_states = NULL;
    evoke_segment_storage_snapshot fold_snapshot;
    evoke_status status;

    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0)
    {
        return;
    }
    evoke_term_directory_init(&directory);
    evoke_segment_query_contract_init(&contract);
    memset(&fold_snapshot, 0, sizeof(fold_snapshot));
    PG_TRY();
    {
        CHECK_FOR_INTERRUPTS();
        status = evoke_segment_pages_open_term_cow(
            index_relation,
            &manifest->term_directory,
            root->published_block_high_watermark,
            manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            CHECK_FOR_INTERRUPTS();
            status = evoke_term_cow_validate_external(
                &loader.root_object.ref,
                manifest->vocab_size,
                evoke_segment_pages_load_term_cow_object,
                &loader
            );
        }
        if (status == EVOKE_OK)
        {
            CHECK_FOR_INTERRUPTS();
            status = evoke_term_cow_materialize_external(
                &loader.root_object.ref,
                manifest,
                evoke_segment_pages_load_term_cow_object,
                &loader,
                &directory,
                &doc_frequencies,
                &catalog_refs,
                &fold_states
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_relation_codec_error(
                index_relation,
                "COW term-directory closure",
                status
            );
        }
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_load_query_contract(
            index_relation,
            manifest,
            &contract
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_attach_query_vocabulary(
            index_relation,
            manifest,
            catalog_refs,
            &contract
        );
        fold_snapshot.root = *root;
        fold_snapshot.manifest = *manifest;
        fold_snapshot.term_fold_states = fold_states;
        fold_states = NULL;
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_load_term_folds(
            index_relation,
            root,
            &fold_snapshot
        );
    }
    PG_FINALLY();
    {
        free(fold_states);
        evoke_segment_storage_snapshot_free_term_folds(
            &fold_snapshot
        );
        free(catalog_refs);
        free(doc_frequencies);
        evoke_segment_query_contract_free(&contract);
        evoke_term_directory_free(&directory);
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_validate_document_cow_closure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    evoke_document_cow_page_loader_context loader;
    evoke_status status;

    if (manifest->document_slot_count == 0)
    {
        return;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_segment_pages_open_document_cow(
        index_relation,
        &manifest->document_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        CHECK_FOR_INTERRUPTS();
        status = evoke_document_cow_validate_external(
            &loader.root_object.ref,
            manifest->document_slot_count,
            evoke_segment_pages_load_document_cow_object,
            &loader
        );
    }
    if (status == EVOKE_OK &&
        loader.root_object.ref.live_document_count !=
            manifest->visible_document_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_relation_codec_error(
            index_relation,
            "COW document-directory closure",
            status
        );
    }
}

static void
evoke_segment_pages_validate_lexicon_cow_closure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    evoke_lexicon_cow_page_loader_context loader;
    evoke_status status;

    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) == 0)
    {
        return;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_segment_pages_open_lexicon_cow(
        index_relation,
        &manifest->lexicon_lookup,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        CHECK_FOR_INTERRUPTS();
        status = evoke_lexicon_cow_validate_external(
            &loader.root_object.ref,
            manifest->lexicon_hash_seed,
            manifest->vocab_size,
            evoke_segment_pages_load_lexicon_cow_object,
            &loader
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW lexical-lookup closure",
            status
        );
    }
}

static void
evoke_segment_pages_validate_prefix_cow_closure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    evoke_prefix_cow_page_loader_context loader;
    evoke_status status;

    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) == 0)
    {
        return;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_segment_pages_open_prefix_cow(
        index_relation,
        &manifest->prefix_lookup,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        CHECK_FOR_INTERRUPTS();
        status = evoke_prefix_cow_validate_external(
            &loader.root_object.ref,
            manifest->vocab_size,
            evoke_segment_pages_load_prefix_cow_object,
            &loader
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW prefix-lookup closure",
            status
        );
    }
}

typedef struct evoke_segment_reachability_context
{
    evoke_block_range_inventory *blocks;
    uint32 published_block_high_watermark;
} evoke_segment_reachability_context;

static bool
evoke_segment_pages_refs_overlap(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
)
{
    uint64 left_end;
    uint64 right_end;

    if (left == NULL || right == NULL || left->page_count == 0 ||
        right->page_count == 0)
    {
        return false;
    }
    left_end = (uint64) left->start_block + left->page_count;
    right_end = (uint64) right->start_block + right->page_count;
    return (uint64) left->start_block < right_end &&
        (uint64) right->start_block < left_end;
}

static evoke_status
evoke_segment_pages_validate_accelerator_child_ref(
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *child_ref,
    const evoke_segment_object_ref *const *authority_refs,
    size_t authority_ref_count
)
{
    evoke_status status = evoke_segment_object_ref_validate(
        child_ref,
        root->published_block_high_watermark
    );

    for (size_t ref_index = 0;
         status == EVOKE_OK && ref_index < authority_ref_count;
         ref_index++)
    {
        if (evoke_segment_pages_refs_overlap(
                child_ref,
                authority_refs[ref_index]))
        {
            status = EVOKE_ERR_FORMAT;
        }
    }
    for (uint32 segment_index = 0;
         status == EVOKE_OK && segment_index < manifest->segment_count;
         segment_index++)
    {
        evoke_segment_object_ref payload_ref;

        status = evoke_segment_descriptor_payload_ref(
            manifest,
            &manifest->segments[segment_index],
            &payload_ref
        );
        if (status == EVOKE_OK &&
            evoke_segment_pages_refs_overlap(child_ref, &payload_ref))
        {
            status = EVOKE_ERR_FORMAT;
        }
    }
    for (uint32 range_index = 0;
         status == EVOKE_OK && range_index < manifest->retired_range_count;
         range_index++)
    {
        const evoke_block_range *range =
            &manifest->retired_ranges[range_index];
        uint64 child_end =
            (uint64) child_ref->start_block + child_ref->page_count;
        uint64 range_end =
            (uint64) range->start_block + range->block_count;

        if ((uint64) child_ref->start_block < range_end &&
            (uint64) range->start_block < child_end)
        {
            status = EVOKE_ERR_FORMAT;
        }
    }
    return status;
}

static bool
evoke_segment_pages_accelerator_identity_matches(
    uint64 source_manifest_id,
    uint64 source_authority_checksum,
    uint64 owner_manifest_id,
    uint32 document_count,
    uint32 vocab_size,
    const evoke_segment_manifest *manifest,
    uint64 authority_checksum
)
{
    const evoke_segment_object_ref *directory_ref;
    uint64 baseline_sequence;
    bool exact_current;

    if (manifest == NULL)
    {
        return false;
    }
    directory_ref = &manifest->semantic_accelerator_directory;
    baseline_sequence =
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            manifest
        );
    exact_current = owner_manifest_id == manifest->manifest_id &&
        baseline_sequence == manifest->max_sequence &&
        document_count == manifest->document_slot_count;
    return source_manifest_id != 0 &&
        source_authority_checksum != 0 &&
        owner_manifest_id > source_manifest_id &&
        owner_manifest_id <= manifest->manifest_id &&
        directory_ref->object_id == owner_manifest_id &&
        directory_ref->owner_manifest_id == owner_manifest_id &&
        baseline_sequence != 0 &&
        (!exact_current ||
         source_authority_checksum == authority_checksum) &&
        document_count != 0 &&
        document_count <= manifest->document_slot_count &&
        vocab_size <= manifest->vocab_size;
}

static evoke_status
evoke_segment_pages_accelerator_artifact_authority(
    uint64 source_manifest_id,
    uint64 source_authority_checksum,
    uint64 owner_manifest_id,
    uint32 document_count,
    uint32 vocab_size,
    const evoke_segment_manifest *manifest,
    uint64 *authority_checksum_out
)
{
    uint64 manifest_authority_checksum = 0;
    evoke_status status;

    if (manifest == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_manifest_authority_checksum(
        manifest,
        &manifest_authority_checksum
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_segment_pages_accelerator_identity_matches(
            source_manifest_id,
            source_authority_checksum,
            owner_manifest_id,
            document_count,
            vocab_size,
            manifest,
            manifest_authority_checksum))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (authority_checksum_out != NULL)
    {
        *authority_checksum_out = source_authority_checksum;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_pages_try_load_semantic_accelerator_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
)
{
    uint8 bytes[EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE];
    Size header_size;
    evoke_semantic_accelerator_directory_summary summary;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        summary_out == NULL ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&summary, 0, sizeof(summary));
    if (manifest->semantic_accelerator_directory.object_bytes <
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_MIN_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    header_size = (Size) Min(
        manifest->semantic_accelerator_directory.object_bytes,
        (uint64) sizeof(bytes)
    );
    evoke_segment_pages_read_range(
        index_relation,
        &manifest->semantic_accelerator_directory,
        0,
        header_size,
        bytes
    );
    status = evoke_semantic_accelerator_directory_summary_deserialize(
        bytes,
        header_size,
        &summary
    );
    if (status == EVOKE_OK)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            summary.source_manifest_id,
            summary.source_authority_checksum,
            summary.owner_manifest_id,
            summary.document_count,
            summary.vocab_size,
            manifest,
            NULL
        );
    }
    if (status == EVOKE_OK &&
        summary.total_size !=
            manifest->semantic_accelerator_directory.object_bytes)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        *summary_out = summary;
    }
    return status;
}

void
evoke_segment_pages_load_semantic_accelerator_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
)
{
    evoke_status status =
        evoke_segment_pages_try_load_semantic_accelerator_summary(
            index_relation,
            root,
            manifest,
            summary_out
        );

    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator summary",
            status
        );
    }
}

bool
evoke_segment_pages_semantic_accelerator_compatible(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
)
{
    evoke_semantic_accelerator_directory_summary summary;
    evoke_scope_header scope_header;
    uint8 scope_bytes[EVOKE_SCOPE_HEADER_SIZE];
    evoke_status status;
    bool has_scope_columns;
    bool has_scope;

    if (index_relation == NULL || root == NULL || manifest == NULL)
    {
        return false;
    }
    memset(&summary, 0, sizeof(summary));
    status = evoke_segment_pages_try_load_semantic_accelerator_summary(
        index_relation,
        root,
        manifest,
        &summary
    );
    if (status != EVOKE_OK ||
        !evoke_semantic_accelerator_directory_summary_is_current(&summary) ||
        !evoke_semantic_accelerator_directory_summary_has_complete_forward(
            &summary) ||
        summary.document_count == 0 ||
        summary.document_count > manifest->document_slot_count ||
        summary.vocab_size > manifest->vocab_size)
    {
        return false;
    }
    has_scope_columns = index_relation->rd_index->indnatts >
        index_relation->rd_index->indnkeyatts;
    has_scope =
        evoke_semantic_accelerator_directory_summary_has_scope(&summary);
    if (has_scope_columns != has_scope)
    {
        return false;
    }
    if (has_scope)
    {
        if (summary.scope_object.object_bytes < sizeof(scope_bytes))
        {
            return false;
        }
        evoke_segment_pages_read_range(
            index_relation,
            &summary.scope_object,
            0,
            sizeof(scope_bytes),
            scope_bytes
        );
        status = evoke_scope_header_deserialize(
            scope_bytes,
            sizeof(scope_bytes),
            (Size) summary.scope_object.object_bytes,
            summary.source_authority_checksum,
            &scope_header
        );
        if (status != EVOKE_OK ||
            !evoke_scope_header_is_current(&scope_header) ||
            scope_header.document_count != summary.document_count)
        {
            return false;
        }
    }
    if (summary_out != NULL)
    {
        *summary_out = summary;
    }
    return true;
}

bool
evoke_segment_pages_open_semantic_accelerator_scope(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_object_ref *scope_ref_out,
    evoke_scope_header *scope_header_out
)
{
    const evoke_segment_object_ref *directory_ref;
    const evoke_segment_object_ref *authority_refs[9];
    evoke_semantic_accelerator_directory_summary summary;
    uint8 directory_bytes[EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE];
    Size directory_header_size;
    uint8 scope_bytes[EVOKE_SCOPE_HEADER_SIZE];
    uint64 authority_checksum = 0;
    evoke_status status;

    if (index_relation == NULL || context == NULL ||
        scope_ref_out == NULL || scope_header_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke scope-posting open")));
    }
    if (!context->semantic_accelerator_compatible ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest))
    {
        return false;
    }
    directory_ref = &context->manifest.semantic_accelerator_directory;
    if (directory_ref->object_bytes <
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_MIN_HEADER_SIZE)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator scope directory",
            EVOKE_ERR_FORMAT
        );
    }
    directory_header_size = (Size) Min(
        directory_ref->object_bytes,
        (uint64) sizeof(directory_bytes)
    );
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        directory_ref,
        0,
        directory_header_size,
        directory_bytes
    );
    status = evoke_semantic_accelerator_directory_summary_deserialize(
        directory_bytes,
        directory_header_size,
        &summary
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator scope directory",
            status
        );
    }
    if (!evoke_semantic_accelerator_directory_summary_is_current(&summary) ||
        !evoke_semantic_accelerator_directory_summary_has_scope(&summary))
    {
        return false;
    }
    status = evoke_segment_pages_accelerator_artifact_authority(
        summary.source_manifest_id,
        summary.source_authority_checksum,
        summary.owner_manifest_id,
        summary.document_count,
        summary.vocab_size,
        &context->manifest,
        &authority_checksum
    );
    if (status == EVOKE_OK &&
        summary.total_size != directory_ref->object_bytes)
    {
        status = EVOKE_ERR_FORMAT;
    }
    authority_refs[0] = &context->root.manifest;
    authority_refs[1] = &context->manifest.query_contract;
    authority_refs[2] = &context->manifest.term_directory;
    authority_refs[3] = &context->manifest.neutral_fold;
    authority_refs[4] = &context->manifest.impact_fold;
    authority_refs[5] = &context->manifest.document_directory;
    authority_refs[6] = &context->manifest.lexicon_lookup;
    authority_refs[7] = &context->manifest.prefix_lookup;
    authority_refs[8] = directory_ref;
    if (status == EVOKE_OK)
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            &context->root,
            &context->manifest,
            &summary.scope_object,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator scope authority",
            status
        );
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &summary.scope_object,
        0,
        sizeof(scope_bytes),
        scope_bytes
    );
    status = evoke_scope_header_deserialize(
        scope_bytes,
        sizeof(scope_bytes),
        summary.scope_object.object_bytes,
        authority_checksum,
        scope_header_out
    );
    if (status == EVOKE_OK &&
        scope_header_out->document_count != summary.document_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator scope header",
            status
        );
    }
    *scope_ref_out = summary.scope_object;
    return true;
}

bool
evoke_segment_pages_load_semantic_accelerator_tid_lookup(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint8 **bytes_out,
    Size *size_out,
    evoke_document_tid_lookup_view *view_out
)
{
    const evoke_segment_object_ref *directory_ref;
    const evoke_segment_object_ref *authority_refs[9];
    evoke_semantic_accelerator_directory_summary summary;
    uint8 directory_bytes[EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE];
    Size directory_header_size;
    uint8 *bytes = NULL;
    uint64 authority_checksum = 0;
    evoke_status status;

    if (index_relation == NULL || context == NULL || bytes_out == NULL ||
        size_out == NULL || view_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke TID lookup load")));
    }
    *bytes_out = NULL;
    *size_out = 0;
    memset(view_out, 0, sizeof(*view_out));
    if (!context->semantic_accelerator_compatible ||
        !evoke_segment_manifest_semantic_accelerator_published(
            &context->root,
            &context->manifest))
    {
        return false;
    }
    directory_ref = &context->manifest.semantic_accelerator_directory;
    if (directory_ref->object_bytes <
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_MIN_HEADER_SIZE)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator TID lookup directory",
            EVOKE_ERR_FORMAT
        );
    }
    directory_header_size = (Size) Min(
        directory_ref->object_bytes,
        (uint64) sizeof(directory_bytes)
    );
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        directory_ref,
        0,
        directory_header_size,
        directory_bytes
    );
    status = evoke_semantic_accelerator_directory_summary_deserialize(
        directory_bytes,
        directory_header_size,
        &summary
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator TID lookup directory",
            status
        );
    }
    if (!evoke_semantic_accelerator_directory_summary_is_current(&summary) ||
        !evoke_semantic_accelerator_directory_summary_has_tid_lookup(
            &summary))
    {
        return false;
    }
    status = evoke_segment_pages_accelerator_artifact_authority(
        summary.source_manifest_id,
        summary.source_authority_checksum,
        summary.owner_manifest_id,
        summary.document_count,
        summary.vocab_size,
        &context->manifest,
        &authority_checksum
    );
    if (status == EVOKE_OK &&
        (summary.total_size != directory_ref->object_bytes ||
         summary.tid_lookup_object.object_bytes == 0 ||
         summary.tid_lookup_object.object_bytes > MaxAllocSize ||
         summary.document_count == 0))
    {
        status = EVOKE_ERR_FORMAT;
    }
    authority_refs[0] = &context->root.manifest;
    authority_refs[1] = &context->manifest.query_contract;
    authority_refs[2] = &context->manifest.term_directory;
    authority_refs[3] = &context->manifest.neutral_fold;
    authority_refs[4] = &context->manifest.impact_fold;
    authority_refs[5] = &context->manifest.document_directory;
    authority_refs[6] = &context->manifest.lexicon_lookup;
    authority_refs[7] = &context->manifest.prefix_lookup;
    authority_refs[8] = directory_ref;
    if (status == EVOKE_OK)
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            &context->root,
            &context->manifest,
            &summary.tid_lookup_object,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    if (status == EVOKE_OK)
    {
        bytes = palloc((Size) summary.tid_lookup_object.object_bytes);
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &summary.tid_lookup_object,
            0,
            (Size) summary.tid_lookup_object.object_bytes,
            bytes
        );
        status = evoke_document_tid_lookup_open(
            bytes,
            (Size) summary.tid_lookup_object.object_bytes,
            authority_checksum,
            0,
            summary.document_count,
            view_out
        );
    }
    if (status != EVOKE_OK)
    {
        if (bytes != NULL)
        {
            pfree(bytes);
        }
        evoke_segment_pages_report_codec_error(
            "semantic accelerator TID lookup",
            status
        );
    }
    *bytes_out = bytes;
    *size_out = (Size) summary.tid_lookup_object.object_bytes;
    return true;
}

void
evoke_segment_pages_read_semantic_accelerator_scope_range(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *scope_ref,
    Size offset,
    Size length,
    uint8 *bytes_out
)
{
    if (index_relation == NULL || context == NULL || scope_ref == NULL ||
        scope_ref->object_kind !=
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_SCOPE ||
        scope_ref->owner_manifest_id !=
            context->manifest.semantic_accelerator_directory.
                owner_manifest_id)
    {
        ereport(ERROR, (errmsg("invalid evoke scope-posting range read")));
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        scope_ref,
        offset,
        length,
        bytes_out
    );
}

static void
evoke_segment_pages_load_semantic_accelerator_directory_internal(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory *directory_out,
    bool allow_retired_format
)
{
    evoke_semantic_accelerator_directory directory;
    const evoke_segment_object_ref *authority_refs[9];
    uint8 *bytes;
    Size size;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        directory_out == NULL ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke semantic accelerator load")));
    }
    evoke_semantic_accelerator_directory_init(&directory);
    bytes = evoke_segment_pages_read(
        index_relation,
        &manifest->semantic_accelerator_directory,
        &size
    );
    status = allow_retired_format
        ? evoke_semantic_accelerator_directory_deserialize_retired(
              bytes,
              size,
              &directory
          )
        : evoke_semantic_accelerator_directory_deserialize(
              bytes,
              size,
              &directory
          );
    pfree(bytes);
    if (status == EVOKE_OK)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory.source_manifest_id,
            directory.source_authority_checksum,
            directory.owner_manifest_id,
            directory.document_count,
            directory.vocab_size,
            manifest,
            NULL
        );
    }

    authority_refs[0] = &root->manifest;
    authority_refs[1] = &manifest->query_contract;
    authority_refs[2] = &manifest->term_directory;
    authority_refs[3] = &manifest->neutral_fold;
    authority_refs[4] = &manifest->impact_fold;
    authority_refs[5] = &manifest->document_directory;
    authority_refs[6] = &manifest->lexicon_lookup;
    authority_refs[7] = &manifest->prefix_lookup;
    authority_refs[8] = &manifest->semantic_accelerator_directory;
    for (uint32 term_index = 0;
         status == EVOKE_OK && term_index < directory.term_count;
         term_index++)
    {
        const evoke_segment_object_ref *term_ref =
            &directory.terms[term_index].term_object;

        status = evoke_segment_pages_validate_accelerator_child_ref(
            root,
            manifest,
            term_ref,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    for (uint32 forward_index = 0;
         status == EVOKE_OK &&
         forward_index < directory.forward_chunk_count;
         forward_index++)
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            root,
            manifest,
            &directory.forward_chunks[forward_index].forward_object,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    if (status == EVOKE_OK &&
        evoke_semantic_accelerator_directory_has_scope(&directory))
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            root,
            manifest,
            &directory.scope_object,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    if (status == EVOKE_OK &&
        evoke_semantic_accelerator_directory_has_tid_lookup(&directory))
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            root,
            manifest,
            &directory.tid_lookup_object,
            authority_refs,
            lengthof(authority_refs)
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_semantic_accelerator_directory_free(&directory);
        evoke_segment_pages_report_codec_error(
            "semantic accelerator directory",
            status
        );
    }
    evoke_semantic_accelerator_directory_free(directory_out);
    *directory_out = directory;
}

void
evoke_segment_pages_load_semantic_accelerator_directory(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory *directory_out
)
{
    evoke_segment_pages_load_semantic_accelerator_directory_internal(
        index_relation,
        root,
        manifest,
        directory_out,
        false
    );
}

void
evoke_segment_pages_load_retired_semantic_accelerator_directory(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory *directory_out
)
{
    evoke_segment_pages_load_semantic_accelerator_directory_internal(
        index_relation,
        root,
        manifest,
        directory_out,
        true
    );
}

void
evoke_segment_pages_load_semantic_accelerator_forward_directory(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_semantic_accelerator_directory *directory_out
)
{
    uint8 header_bytes[EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE];
    evoke_semantic_accelerator_directory_summary summary;
    evoke_semantic_accelerator_directory directory;
    const evoke_segment_object_ref *directory_ref;
    uint8 *forward_bytes = NULL;
    size_t forward_offset = 0;
    size_t forward_size = 0;
    evoke_status status;

    if (index_relation == NULL || context == NULL ||
        directory_out == NULL ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest))
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke semantic forward directory load"))
        );
    }
    directory_ref = &context->manifest.semantic_accelerator_directory;
    evoke_semantic_accelerator_directory_init(&directory);
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        directory_ref,
        0,
        sizeof(header_bytes),
        header_bytes
    );
    status = evoke_semantic_accelerator_directory_forward_slice(
        header_bytes,
        sizeof(header_bytes),
        &summary,
        &forward_offset,
        &forward_size
    );
    if (status == EVOKE_OK &&
        (summary.total_size != directory_ref->object_bytes ||
         forward_size == 0 || forward_size > MaxAllocSize))
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        forward_bytes = palloc(forward_size);
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            directory_ref,
            forward_offset,
            forward_size,
            forward_bytes
        );
        status =
            evoke_semantic_accelerator_forward_directory_deserialize(
                &summary,
                forward_bytes,
                forward_size,
                &directory
            );
    }
    if (forward_bytes != NULL)
    {
        pfree(forward_bytes);
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory.source_manifest_id,
            directory.source_authority_checksum,
            directory.owner_manifest_id,
            directory.document_count,
            directory.vocab_size,
            &context->manifest,
            NULL
        );
    }
    for (uint32 forward_index = 0;
         status == EVOKE_OK &&
         forward_index < directory.forward_chunk_count;
         forward_index++)
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            &context->root,
            &context->manifest,
            &directory.forward_chunks[forward_index].forward_object,
            &directory_ref,
            1
        );
    }
    for (uint32 bound_index = 0;
         status == EVOKE_OK &&
         bound_index < directory.forward_bound_shard_count;
         bound_index++)
    {
        status = evoke_segment_pages_validate_accelerator_child_ref(
            &context->root,
            &context->manifest,
            &directory.forward_bound_shards[bound_index],
            &directory_ref,
            1
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_semantic_accelerator_directory_free(&directory);
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward directory",
            status
        );
    }
    evoke_semantic_accelerator_directory_free(directory_out);
    *directory_out = directory;
}

void
evoke_segment_pages_load_semantic_accelerator_term(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_semantic_accelerator_directory *directory,
    uint32 term_id,
    evoke_semantic_accelerator_index *index_out
)
{
    const evoke_semantic_accelerator_directory_entry *entry;
    evoke_semantic_accelerator_index index;
    uint64 authority_checksum = 0;
    uint8 *bytes;
    Size size;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        directory == NULL || index_out == NULL ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            root,
            manifest))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic accelerator term")));
    }
    status = evoke_segment_pages_accelerator_artifact_authority(
        directory->source_manifest_id,
        directory->source_authority_checksum,
        directory->owner_manifest_id,
        directory->document_count,
        directory->vocab_size,
        manifest,
        &authority_checksum
    );
    entry = evoke_semantic_accelerator_directory_find(
        directory,
        term_id
    );
    if (status != EVOKE_OK || entry == NULL ||
        evoke_segment_object_ref_validate(
            &entry->term_object,
            root->published_block_high_watermark) != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator term reference",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }

    evoke_semantic_accelerator_index_init(&index);
    bytes = evoke_segment_pages_read(
        index_relation,
        &entry->term_object,
        &size
    );
    status = evoke_semantic_accelerator_term_deserialize(
        bytes,
        size,
        authority_checksum,
        term_id,
        &index
    );
    pfree(bytes);
    if (status == EVOKE_OK &&
        index.document_count != directory->document_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        evoke_semantic_accelerator_index_free(&index);
        evoke_segment_pages_report_codec_error(
            "semantic accelerator term",
            status
        );
    }
    evoke_semantic_accelerator_index_free(index_out);
    *index_out = index;
}

void
evoke_segment_pages_load_semantic_accelerator_forward(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_semantic_accelerator_directory *directory,
    uint32 document_id,
    evoke_semantic_forward_chunk *chunk_out
)
{
    const evoke_semantic_accelerator_forward_entry *entry;
    evoke_semantic_forward_chunk chunk;
    uint64 authority_checksum = 0;
    uint8 *bytes;
    Size size;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        directory == NULL || chunk_out == NULL ||
        !evoke_segment_manifest_semantic_accelerator_eligible(root, manifest))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic forward load")));
    }
    status = evoke_segment_pages_accelerator_artifact_authority(
        directory->source_manifest_id,
        directory->source_authority_checksum,
        directory->owner_manifest_id,
        directory->document_count,
        directory->vocab_size,
        manifest,
        &authority_checksum
    );
    entry = evoke_semantic_accelerator_directory_find_forward(
        directory,
        document_id
    );
    if (status != EVOKE_OK || entry == NULL ||
        evoke_segment_object_ref_validate(
            &entry->forward_object,
            root->published_block_high_watermark) != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward reference",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }

    evoke_semantic_forward_chunk_init(&chunk);
    bytes = evoke_segment_pages_read(
        index_relation,
        &entry->forward_object,
        &size
    );
    status = evoke_semantic_forward_chunk_deserialize(
        bytes,
        size,
        authority_checksum,
        &chunk
    );
    pfree(bytes);
    if (status == EVOKE_OK &&
        (chunk.first_document != entry->first_document ||
         chunk.document_count != entry->document_count))
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        evoke_semantic_forward_chunk_free(&chunk);
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward",
            status
        );
    }
    evoke_semantic_forward_chunk_free(chunk_out);
    *chunk_out = chunk;
}

void
evoke_segment_forward_bound_reader_init(
    evoke_segment_forward_bound_reader *reader
)
{
    if (reader != NULL)
    {
        memset(reader, 0, sizeof(*reader));
    }
}

void
evoke_segment_pages_visit_semantic_accelerator_forward_bound(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_bound_reader *reader,
    uint32 term_id,
    evoke_semantic_forward_bound_visitor visitor,
    void *visitor_context,
    uint32 *entry_count_out,
    uint64 *bytes_read_out
)
{
    const evoke_segment_object_ref *ref;
    uint8 header_bytes[EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE];
    uint8 offset_bytes[2U * sizeof(uint64)];
    uint8 *payload = NULL;
    size_t payload_offset;
    size_t payload_size;
    size_t offsets_size;
    uint32 local_term;
    uint32 entry_count = 0;
    uint64 bytes_read = 0;
    evoke_status status;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        reader == NULL || visitor == NULL || entry_count_out == NULL ||
        bytes_read_out == NULL || term_id >= directory->vocab_size ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic forward bound read")));
    }
    *entry_count_out = 0;
    *bytes_read_out = 0;
    if (!reader->authority_ready)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory->source_manifest_id,
            directory->source_authority_checksum,
            directory->owner_manifest_id,
            directory->document_count,
            directory->vocab_size,
            &context->manifest,
            &reader->authority_checksum
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward bound authority",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        reader->authority_ready = true;
    }
    ref = evoke_semantic_accelerator_directory_find_forward_bound(
        directory,
        term_id
    );
    if (ref == NULL ||
        evoke_segment_object_ref_validate(
            ref,
            context->root.published_block_high_watermark) != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward bound reference",
            EVOKE_ERR_FORMAT
        );
    }
    if (!reader->shard_ready ||
        reader->shard_ref.object_id != ref->object_id ||
        reader->shard_ref.object_checksum != ref->object_checksum)
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            ref,
            0,
            sizeof(header_bytes),
            header_bytes
        );
        status = evoke_semantic_forward_bound_header_deserialize(
            header_bytes,
            sizeof(header_bytes),
            ref->object_bytes,
            reader->authority_checksum,
            &reader->summary
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward bound header",
                status
            );
        }
        if (reader->summary.document_count != directory->document_count ||
            reader->summary.vocab_size != directory->vocab_size ||
            reader->summary.first_term !=
                (term_id /
                    EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD) *
                    EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD ||
            reader->summary.term_count != Min(
                EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD,
                directory->vocab_size - reader->summary.first_term
            ))
        {
            status = EVOKE_ERR_FORMAT;
        }
        offsets_size =
            ((size_t) reader->summary.term_count + 1U) * sizeof(uint64);
        if (status != EVOKE_OK || offsets_size > sizeof(reader->offsets))
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward bound header",
                status == EVOKE_OK ? EVOKE_ERR_RANGE : status
            );
        }
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            ref,
            reader->summary.offsets_offset,
            offsets_size,
            reader->offsets
        );
        reader->shard_ref = *ref;
        reader->shard_ready = true;
        bytes_read += sizeof(header_bytes) + offsets_size;
    }
    local_term = term_id - reader->summary.first_term;
    if (local_term >= reader->summary.term_count)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward bound term",
            EVOKE_ERR_FORMAT
        );
    }
    memcpy(
        offset_bytes,
        reader->offsets + (size_t) local_term * sizeof(uint64),
        sizeof(offset_bytes)
    );
    status = evoke_semantic_forward_bound_term_slice(
        &reader->summary,
        offset_bytes,
        sizeof(offset_bytes),
        term_id,
        &payload_offset,
        &payload_size
    );
    if (status != EVOKE_OK || payload_size > MaxAllocSize)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward bound slice",
            status == EVOKE_OK ? EVOKE_ERR_RANGE : status
        );
    }
    if (payload_size > 0)
    {
        payload = palloc(payload_size);
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            ref,
            payload_offset,
            payload_size,
            payload
        );
    }
    status = evoke_semantic_forward_bound_visit(
        payload,
        payload_size,
        (directory->document_count - 1U) /
            (UINT32_C(1) << EVOKE_SEMANTIC_FORWARD_BOUND_BLOCK_SHIFT) + 1U,
        visitor,
        visitor_context,
        &entry_count
    );
    if (payload != NULL)
    {
        pfree(payload);
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward bound payload",
            status
        );
    }
    *entry_count_out = entry_count;
    *bytes_read_out = bytes_read + payload_size;
}

static uint32
evoke_segment_pages_forward_read_u32(const uint8 *bytes)
{
    return (uint32) bytes[0] |
        ((uint32) bytes[1] << 8) |
        ((uint32) bytes[2] << 16) |
        ((uint32) bytes[3] << 24);
}

static uint16
evoke_segment_pages_forward_read_u16(const uint8 *bytes)
{
    return (uint16) bytes[0] | (uint16) ((uint16) bytes[1] << 8);
}

static float
evoke_segment_pages_forward_read_f32(const uint8 *bytes)
{
    uint32 bits = evoke_segment_pages_forward_read_u32(bytes);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32
evoke_segment_pages_forward_find_dense_term(
    const uint8 *serialized_term_ids,
    uint32 term_count,
    uint32 term_id
)
{
    uint32 low = 0;
    uint32 high = term_count;

    while (low < high)
    {
        uint32 middle = low + (high - low) / 2U;
        uint32 candidate = evoke_segment_pages_forward_read_u32(
            serialized_term_ids + (Size) middle * sizeof(uint32)
        );

        if (candidate < term_id)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    if (low < term_count &&
        evoke_segment_pages_forward_read_u32(
            serialized_term_ids + (Size) low * sizeof(uint32)) == term_id)
    {
        return low;
    }
    return UINT32_MAX;
}

static uint32
evoke_segment_pages_accumulate_dense_codes(
    const uint8 *codes,
    const uint8 *serialized_scales,
    uint32 first_document,
    uint32 document_count,
    const uint8 *allowed_document_bitmap,
    float query_weight,
    double *scores
)
{
    uint32 examined = 0;

    if (allowed_document_bitmap == NULL)
    {
        for (uint32 document = 0; document < document_count; document++)
        {
            int8 code = (int8) codes[document];
            float scale = evoke_segment_pages_forward_read_f32(
                serialized_scales + (Size) document * sizeof(float)
            );

            if (code == INT8_MIN || !isfinite(scale) || scale <= 0.0f)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator transpose dense lane",
                    EVOKE_ERR_FORMAT
                );
            }
            examined++;
            if (code == 0)
            {
                continue;
            }
            scores[document] += (double) code * scale * query_weight;
            if (!isfinite(scores[document]) || scores[document] > FLT_MAX ||
                scores[document] < -FLT_MAX)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator transpose dense score",
                    EVOKE_ERR_RANGE
                );
            }
        }
        return examined;
    }
    {
        uint64 first = first_document;
        uint64 end = first + document_count;
        uint64 first_byte = first >> 3;
        uint64 end_byte = (end + UINT64_C(7)) >> 3;

        for (uint64 byte_index = first_byte;
             byte_index < end_byte;
             byte_index++)
        {
            uint8 candidates = allowed_document_bitmap[byte_index];

            while (candidates != 0)
            {
                unsigned bit_index = (unsigned) __builtin_ctz(
                    (unsigned) candidates
                );
                uint64 global_document = byte_index * UINT64_C(8) + bit_index;
                uint32 document;
                int8 code;
                float scale;

                candidates &= (uint8) (candidates - UINT8_C(1));
                if (global_document < first || global_document >= end)
                {
                    continue;
                }
                document = (uint32) (global_document - first);
                code = (int8) codes[document];
                scale = evoke_segment_pages_forward_read_f32(
                    serialized_scales + (Size) document * sizeof(float)
                );
                if (code == INT8_MIN || !isfinite(scale) || scale <= 0.0f)
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator transpose dense lane",
                        EVOKE_ERR_FORMAT
                    );
                }
                examined++;
                if (code == 0)
                {
                    continue;
                }
                scores[document] += (double) code * scale * query_weight;
                if (!isfinite(scores[document]) ||
                    scores[document] > FLT_MAX ||
                    scores[document] < -FLT_MAX)
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator transpose dense score",
                        EVOKE_ERR_RANGE
                    );
                }
            }
        }
    }
    return examined;
}

#define EVOKE_SEGMENT_FORWARD_ROW_WINDOW_BYTES ((Size) BLCKSZ)

void
evoke_segment_forward_row_reader_init(
    evoke_segment_forward_row_reader *reader
)
{
    if (reader != NULL)
    {
        memset(reader, 0, sizeof(*reader));
    }
}

void
evoke_segment_forward_row_reader_reset(
    evoke_segment_forward_row_reader *reader
)
{
    if (reader == NULL)
    {
        return;
    }
    if (reader->row_bytes != NULL)
    {
        pfree(reader->row_bytes);
    }
    if (reader->row_offsets != NULL)
    {
        pfree(reader->row_offsets);
    }
    memset(reader, 0, sizeof(*reader));
}

void
evoke_segment_forward_row_reader_set_exact_reads(
    evoke_segment_forward_row_reader *reader,
    bool exact_row_reads
)
{
    if (reader != NULL && reader->exact_row_reads != exact_row_reads)
    {
        reader->exact_row_reads = exact_row_reads;
        reader->row_window_offset = 0;
        reader->row_window_size = 0;
    }
}

void
evoke_segment_forward_row_reader_set_exact_data_reads(
    evoke_segment_forward_row_reader *reader,
    bool exact_row_data_reads
)
{
    if (reader != NULL &&
        reader->exact_row_data_reads != exact_row_data_reads)
    {
        reader->exact_row_data_reads = exact_row_data_reads;
        reader->row_window_offset = 0;
        reader->row_window_size = 0;
    }
}

void
evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    const uint32 *document_ids,
    size_t document_count
)
{
    Size header_size = evoke_semantic_forward_header_size();
    uint8 *header_bytes = NULL;
    uint8 *serialized_offsets = NULL;
    Size serialized_offset_capacity = 0;
    uint64 authority_checksum;
    size_t document_index = 0;
    uint32 previous_document = 0;
    bool have_previous_document = false;
    evoke_status status;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        (document_count > 0 && document_ids == NULL) ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest
        ))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic forward prefetch")));
    }
    if (document_count == 0)
    {
        return;
    }
    status = evoke_segment_pages_accelerator_artifact_authority(
        directory->source_manifest_id,
        directory->source_authority_checksum,
        directory->owner_manifest_id,
        directory->document_count,
        directory->vocab_size,
        &context->manifest,
        &authority_checksum
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward prefetch authority",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }

    /* Submit metadata reads for the complete bounded batch first. */
    for (size_t index = 0; index < document_count; index++)
    {
        const evoke_semantic_accelerator_forward_entry *entry;
        uint32 document_id = document_ids[index];

        if (document_id >= directory->document_count ||
            (have_previous_document && document_id <= previous_document))
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward prefetch documents",
                EVOKE_ERR_FORMAT
            );
        }
        previous_document = document_id;
        have_previous_document = true;
        if (index > 0 &&
            (document_ids[index - 1] >> directory->forward_document_shift) ==
                (document_id >> directory->forward_document_shift))
        {
            continue;
        }
        entry = evoke_semantic_accelerator_directory_find_forward(
            directory,
            document_id
        );
        if (entry == NULL)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward prefetch reference",
                EVOKE_ERR_FORMAT
            );
        }
        evoke_segment_pages_prefetch_query_range(
            index_relation,
            context,
            &entry->forward_object,
            0,
            Min((Size) entry->forward_object.object_bytes, (Size) BLCKSZ),
            NULL,
            NULL
        );
    }

    /*
     * Resident chunk metadata does not imply resident row payloads.  Decode
     * the offsets and submit payload prefetches even when every header was
     * already cached by the background warm pass.
     */

    header_bytes = palloc(header_size);
    while (document_index < document_count)
    {
        const evoke_semantic_accelerator_forward_entry *entry;
        evoke_semantic_forward_header header;
        Size row_offset_bytes;
        size_t chunk_document_end = document_index + 1;
        BlockNumber last_prefetched = InvalidBlockNumber;

        entry = evoke_semantic_accelerator_directory_find_forward(
            directory,
            document_ids[document_index]
        );
        if (entry == NULL)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward prefetch reference",
                EVOKE_ERR_FORMAT
            );
        }
        while (chunk_document_end < document_count &&
               document_ids[chunk_document_end] >= entry->first_document &&
               document_ids[chunk_document_end] <
                   entry->first_document + entry->document_count)
        {
            chunk_document_end++;
        }
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            0,
            header_size,
            header_bytes
        );
        status = evoke_semantic_forward_header_deserialize(
            header_bytes,
            header_size,
            entry->forward_object.object_bytes,
            authority_checksum,
            &header
        );
        if (status != EVOKE_OK ||
            header.first_document != entry->first_document ||
            header.document_count != entry->document_count)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward prefetch header",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        row_offset_bytes =
            ((Size) header.document_count + 1U) * sizeof(uint32);
        if (row_offset_bytes >
                header.row_data_offset - header.row_offsets_offset)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward prefetch directory",
                EVOKE_ERR_FORMAT
            );
        }
        if (row_offset_bytes > serialized_offset_capacity)
        {
            serialized_offsets = serialized_offsets == NULL
                ? palloc(row_offset_bytes)
                : repalloc(serialized_offsets, row_offset_bytes);
            serialized_offset_capacity = row_offset_bytes;
        }
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            header.row_offsets_offset,
            row_offset_bytes,
            serialized_offsets
        );
        for (size_t index = document_index;
             index < chunk_document_end;
             index++)
        {
            uint32 row = document_ids[index] - entry->first_document;
            uint32 row_start = evoke_segment_pages_forward_read_u32(
                serialized_offsets + (Size) row * sizeof(uint32)
            );
            uint32 row_end = evoke_segment_pages_forward_read_u32(
                serialized_offsets + (Size) (row + 1U) * sizeof(uint32)
            );

            if (row_start > row_end ||
                row_end > header.row_data_end - header.row_data_offset)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator forward prefetch offsets",
                    EVOKE_ERR_FORMAT
                );
            }
            evoke_segment_pages_prefetch_query_range(
                index_relation,
                context,
                &entry->forward_object,
                header.row_data_offset + row_start,
                (Size) row_end - row_start,
                &last_prefetched,
                NULL
            );
        }
        document_index = chunk_document_end;
        CHECK_FOR_INTERRUPTS();
    }
    pfree(serialized_offsets);
    pfree(header_bytes);
}

static void
evoke_segment_pages_score_semantic_accelerator_forward_row_internal(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_row_reader *reader,
    uint32 document_id,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    bool query_is_sorted,
    float *score_out,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
)
{
    const evoke_semantic_accelerator_forward_entry *entry;
    const evoke_semantic_forward_header *header;
    uint8 *header_bytes = NULL;
    uint32 row;
    uint32 row_start;
    uint32 row_end;
    uint32 posting_count;
    Size header_size = evoke_semantic_forward_header_size();
    uint64 bytes_read;
    uint64 offset_bytes_read = 0;
    uint64 row_bytes_read = 0;
    const uint8 *row_view = NULL;
    bool header_loaded = false;
    evoke_status status;

    if (index_relation == NULL || context == NULL ||
        directory == NULL || reader == NULL ||
        query_ids == NULL || query_weights == NULL ||
        score_out == NULL || postings_examined_out == NULL ||
        bytes_read_out == NULL ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest
        ))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic forward row score")));
    }
    *postings_examined_out = 0;
    *bytes_read_out = 0;
    if (!reader->authority_ready)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory->source_manifest_id,
            directory->source_authority_checksum,
            directory->owner_manifest_id,
            directory->document_count,
            directory->vocab_size,
            &context->manifest,
            &reader->authority_checksum
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward row authority",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        reader->authority_ready = true;
    }
    entry = reader->entry;
    if (entry == NULL || document_id < entry->first_document ||
        document_id >= entry->first_document + entry->document_count)
    {
        entry = evoke_semantic_accelerator_directory_find_forward(
            directory,
            document_id
        );
        if (entry == NULL ||
            evoke_segment_object_ref_validate(
                &entry->forward_object,
                context->root.published_block_high_watermark) != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward row reference",
                EVOKE_ERR_FORMAT
            );
        }
        header_bytes = palloc(header_size);
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            0,
            header_size,
            header_bytes
        );
        status = evoke_semantic_forward_header_deserialize(
            header_bytes,
            header_size,
            entry->forward_object.object_bytes,
            reader->authority_checksum,
            &reader->header
        );
        pfree(header_bytes);
        header_bytes = NULL;
        if (status != EVOKE_OK ||
            reader->header.first_document != entry->first_document ||
            reader->header.document_count != entry->document_count)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward row header",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        if (!reader->exact_row_reads)
        {
            uint32 row_offset_count =
                reader->header.document_count + UINT32_C(1);
            Size row_offset_bytes =
                (Size) row_offset_count * sizeof(uint32);
            uint8 *serialized_offsets;

            if (row_offset_count == 0 ||
                row_offset_bytes > reader->header.row_data_offset -
                    reader->header.row_offsets_offset)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator forward row directory",
                    EVOKE_ERR_FORMAT
                );
            }
            if (row_offset_bytes > reader->row_offset_capacity)
            {
                reader->row_offsets = reader->row_offsets == NULL
                    ? palloc(row_offset_bytes)
                    : repalloc(reader->row_offsets, row_offset_bytes);
                reader->row_offset_capacity = row_offset_bytes;
            }
            serialized_offsets = palloc(row_offset_bytes);
            evoke_segment_pages_read_query_range(
                index_relation,
                context,
                &entry->forward_object,
                reader->header.row_offsets_offset,
                row_offset_bytes,
                serialized_offsets
            );
            for (uint32 offset_index = 0;
                 offset_index < row_offset_count;
                 offset_index++)
            {
                reader->row_offsets[offset_index] =
                    evoke_segment_pages_forward_read_u32(
                        serialized_offsets +
                            (Size) offset_index * sizeof(uint32)
                    );
                if ((offset_index > 0 &&
                     reader->row_offsets[offset_index] <
                         reader->row_offsets[offset_index - 1]) ||
                    reader->row_offsets[offset_index] >
                        reader->header.row_data_end -
                            reader->header.row_data_offset)
                {
                    pfree(serialized_offsets);
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator forward row directory",
                        EVOKE_ERR_FORMAT
                    );
                }
            }
            pfree(serialized_offsets);
            reader->row_offset_count = row_offset_count;
        }
        else
        {
            reader->row_offset_count = 0;
        }
        reader->entry = entry;
        reader->row_window_offset = 0;
        reader->row_window_size = 0;
        header_loaded = true;
    }
    header = &reader->header;
    row = document_id - header->first_document;
    if (row >= header->document_count)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward row directory",
            EVOKE_ERR_FORMAT
        );
    }
    if (reader->exact_row_reads)
    {
        uint8 row_offset_bytes[sizeof(uint32) * 2U];

        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            header->row_offsets_offset + (Size) row * sizeof(uint32),
            sizeof(row_offset_bytes),
            row_offset_bytes
        );
        row_start = evoke_segment_pages_forward_read_u32(row_offset_bytes);
        row_end = evoke_segment_pages_forward_read_u32(
            row_offset_bytes + sizeof(uint32)
        );
        offset_bytes_read = sizeof(row_offset_bytes);
    }
    else
    {
        if (row + UINT32_C(1) >= reader->row_offset_count)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward row directory",
                EVOKE_ERR_FORMAT
            );
        }
        row_start = reader->row_offsets[row];
        row_end = reader->row_offsets[row + UINT32_C(1)];
    }
    if (row_start > row_end ||
        row_end > header->row_data_end - header->row_data_offset)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward row offsets",
            EVOKE_ERR_FORMAT
        );
    }
    posting_count = 0;
    if (row_end > row_start)
    {
        Size row_data_size = header->row_data_end - header->row_data_offset;

        if (reader->row_window_size == 0 ||
            row_start < reader->row_window_offset ||
            row_end > reader->row_window_offset +
                reader->row_window_size)
        {
            Size window_offset = reader->exact_row_data_reads
                ? (Size) row_start
                : ((Size) row_start / BLCKSZ) * BLCKSZ;
            Size window_size = reader->exact_row_data_reads
                ? (Size) row_end - row_start
                : Min(
                    EVOKE_SEGMENT_FORWARD_ROW_WINDOW_BYTES,
                    row_data_size - window_offset
                );

            if (window_size < (Size) row_end - window_offset)
            {
                window_size = (Size) row_end - window_offset;
            }
            if (window_size > reader->row_capacity)
            {
                reader->row_bytes = reader->row_bytes == NULL
                    ? palloc(window_size)
                    : repalloc(reader->row_bytes, window_size);
                reader->row_capacity = window_size;
            }
            evoke_segment_pages_read_query_range(
                index_relation,
                context,
                &entry->forward_object,
                header->row_data_offset + window_offset,
                window_size,
                reader->row_bytes
            );
            reader->row_window_offset = window_offset;
            reader->row_window_size = window_size;
            row_bytes_read = window_size;
        }
        row_view = reader->row_bytes +
            ((Size) row_start - reader->row_window_offset);
    }
    if (query_is_sorted)
    {
        status = evoke_semantic_forward_score_serialized_row_sorted(
            row_view,
            (Size) row_end - row_start,
            query_ids,
            query_weights,
            query_count,
            score_out,
            &posting_count
        );
    }
    else
    {
        evoke_semantic_forward_chunk chunk;

        evoke_semantic_forward_chunk_init(&chunk);
        evoke_segment_pages_load_semantic_accelerator_forward(
            index_relation,
            &context->root,
            &context->manifest,
            directory,
            document_id,
            &chunk
        );
        status = evoke_semantic_forward_chunk_score(
            &chunk,
            document_id,
            query_ids,
            query_weights,
            query_count,
            score_out
        );
        evoke_semantic_forward_chunk_free(&chunk);
    }
    bytes_read = (header_loaded
        ? (uint64) header_size +
            (uint64) reader->row_offset_count * sizeof(uint32)
        : UINT64_C(0)) +
        offset_bytes_read +
        row_bytes_read;
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward row",
            status
        );
    }
    *postings_examined_out = posting_count;
    *bytes_read_out = bytes_read;
}

void
evoke_segment_pages_score_semantic_accelerator_forward_row(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_row_reader *reader,
    uint32 document_id,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    bool query_is_sorted,
    float *score_out,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
)
{
    evoke_segment_pages_score_semantic_accelerator_forward_row_internal(
        index_relation,
        context,
        directory,
        reader,
        document_id,
        query_ids,
        query_weights,
        query_count,
        query_is_sorted,
        score_out,
        postings_examined_out,
        bytes_read_out
    );
}

void
evoke_segment_pages_score_semantic_accelerator_forward_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_row_reader *reader,
    uint32 block_id,
    uint8 allowed_mask,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    float scores_out[8],
    uint32 *documents_scored_out,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
)
{
    const evoke_semantic_accelerator_forward_entry *entry;
    uint64 first_document_u64 = (uint64) block_id * UINT64_C(8);
    uint32 first_document;
    uint32 valid_document_count;
    uint32 first_bit = 0;
    uint32 last_bit = 0;
    uint32 chunk_index;
    uint64 offset_index;
    uint32 read_start;
    uint32 read_end;
    Size read_size;
    evoke_status status;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        reader == NULL || query_ids == NULL || query_weights == NULL ||
        scores_out == NULL || documents_scored_out == NULL ||
        postings_examined_out == NULL || bytes_read_out == NULL ||
        directory->forward_row_offsets == NULL ||
        first_document_u64 >= directory->document_count ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest
        ))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic forward block score")));
    }
    memset(scores_out, 0, sizeof(float) * 8U);
    *documents_scored_out = 0;
    *postings_examined_out = 0;
    *bytes_read_out = 0;
    valid_document_count = (uint32) Min(
        UINT64_C(8),
        directory->document_count - first_document_u64
    );
    if (valid_document_count < 8U)
    {
        allowed_mask &= (uint8) (
            (UINT16_C(1) << valid_document_count) - UINT16_C(1)
        );
    }
    if (allowed_mask == 0)
    {
        return;
    }
    if (!reader->authority_ready)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory->source_manifest_id,
            directory->source_authority_checksum,
            directory->owner_manifest_id,
            directory->document_count,
            directory->vocab_size,
            &context->manifest,
            &reader->authority_checksum
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward block authority",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        reader->authority_ready = true;
    }
    first_document = (uint32) first_document_u64;
    entry = evoke_semantic_accelerator_directory_find_forward(
        directory,
        first_document
    );
    if (entry == NULL || entry->row_data_offset == 0 ||
        evoke_segment_object_ref_validate(
            &entry->forward_object,
            context->root.published_block_high_watermark) != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward block reference",
            EVOKE_ERR_FORMAT
        );
    }
    chunk_index = (uint32) (entry - directory->forward_chunks);
    if (chunk_index >= directory->forward_chunk_count ||
        first_document < entry->first_document ||
        first_document - entry->first_document >= entry->document_count)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward block directory",
            EVOKE_ERR_FORMAT
        );
    }
    while ((allowed_mask & (uint8) (UINT8_C(1) << first_bit)) == 0)
    {
        first_bit++;
    }
    last_bit = 7U;
    while ((allowed_mask & (uint8) (UINT8_C(1) << last_bit)) == 0)
    {
        last_bit--;
    }
    if (first_document + last_bit >=
            entry->first_document + entry->document_count)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward block boundary",
            EVOKE_ERR_FORMAT
        );
    }
    offset_index = (uint64) entry->first_document + chunk_index +
        (first_document - entry->first_document);
    read_start = directory->forward_row_offsets[offset_index + first_bit];
    read_end = directory->forward_row_offsets[offset_index + last_bit + 1U];
    if (read_start > read_end ||
        (uint64) entry->row_data_offset + read_end >
            entry->forward_object.object_bytes)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator forward block offsets",
            EVOKE_ERR_FORMAT
        );
    }
    read_size = (Size) read_end - read_start;
    if (read_size > reader->row_capacity)
    {
        reader->row_bytes = reader->row_bytes == NULL
            ? palloc(read_size)
            : repalloc(reader->row_bytes, read_size);
        reader->row_capacity = read_size;
    }
    if (read_size > 0)
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            entry->row_data_offset + read_start,
            read_size,
            reader->row_bytes
        );
    }
    for (uint32 bit = first_bit; bit <= last_bit; bit++)
    {
        uint32 row_start;
        uint32 row_end;
        uint32 posting_count = 0;

        if ((allowed_mask & (uint8) (UINT8_C(1) << bit)) == 0)
        {
            continue;
        }
        row_start = directory->forward_row_offsets[offset_index + bit];
        row_end = directory->forward_row_offsets[offset_index + bit + 1U];
        if (row_start > row_end || row_start < read_start ||
            row_end > read_end)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward block row",
                EVOKE_ERR_FORMAT
            );
        }
        status = evoke_semantic_forward_score_serialized_row_sorted(
            row_end > row_start
                ? reader->row_bytes + ((Size) row_start - read_start)
                : NULL,
            (Size) row_end - row_start,
            query_ids,
            query_weights,
            query_count,
            &scores_out[bit],
            &posting_count
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator forward block row",
                status
            );
        }
        (*documents_scored_out)++;
        *postings_examined_out = evoke_u64_saturating_add(
            *postings_examined_out,
            posting_count
        );
    }
    *bytes_read_out = read_size;
}

void
evoke_segment_forward_transpose_reader_init(
    evoke_segment_forward_transpose_reader *reader
)
{
    if (reader != NULL)
    {
        memset(reader, 0, sizeof(*reader));
    }
}

void
evoke_segment_forward_transpose_reader_reset(
    evoke_segment_forward_transpose_reader *reader
)
{
    if (reader == NULL)
    {
        return;
    }
    if (reader->scores != NULL)
    {
        pfree(reader->scores);
    }
    if (reader->postings != NULL)
    {
        pfree(reader->postings);
    }
    if (reader->sparse_oracle_pages != NULL)
    {
        pfree(reader->sparse_oracle_pages);
    }
    if (reader->document_runs != NULL)
    {
        pfree(reader->document_runs);
    }
    if (reader->query_dense_indexes != NULL)
    {
        pfree(reader->query_dense_indexes);
    }
    if (reader->term_offsets != NULL)
    {
        pfree(reader->term_offsets);
    }
    if (reader->query_ranges != NULL)
    {
        pfree(reader->query_ranges);
    }
    if (reader->dense_codes != NULL)
    {
        pfree(reader->dense_codes);
    }
    if (reader->dense_term_ids != NULL)
    {
        pfree(reader->dense_term_ids);
    }
    if (reader->scales != NULL)
    {
        pfree(reader->scales);
    }
    memset(reader, 0, sizeof(*reader));
}

static bool
evoke_segment_pages_forward_block_allowed(
    const uint8 *allowed_document_bitmap,
    uint32 first_document,
    uint32 local_start,
    uint32 local_end
)
{
    for (uint32 document = local_start; document < local_end; document++)
    {
        uint32 global_document = first_document + document;

        if ((allowed_document_bitmap[global_document >> 3] &
             (uint8) (UINT8_C(1) << (global_document & 7U))) != 0)
        {
            return true;
        }
    }
    return false;
}

static uint64
evoke_segment_pages_forward_active_b64_mask(
    const uint8 *allowed_document_bitmap,
    uint32 first_document,
    uint32 document_count
)
{
    const uint32 documents_per_block =
        UINT32_C(1) << EVOKE_SEMANTIC_FORWARD_SUBRANGE_SHIFT;
    uint32 block_count =
        (document_count + documents_per_block - 1U) /
        documents_per_block;
    uint64 mask = 0;

    if (block_count > 64U)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator sparse block oracle",
            EVOKE_ERR_RANGE
        );
    }
    for (uint32 block = 0; block < block_count; block++)
    {
        uint32 start = block * documents_per_block;
        uint32 end = Min(start + documents_per_block, document_count);

        if (evoke_segment_pages_forward_block_allowed(
                allowed_document_bitmap,
                first_document,
                start,
                end))
        {
            mask |= UINT64_C(1) << block;
        }
    }
    return mask;
}

static uint32
evoke_segment_pages_forward_build_document_runs(
    evoke_segment_forward_transpose_reader *reader,
    const uint8 *allowed_document_bitmap,
    uint32 first_document,
    uint32 document_count,
    uint32 *selected_document_count_out
)
{
    const uint32 documents_per_block =
        UINT32_C(1) << EVOKE_SEMANTIC_FORWARD_SUBRANGE_SHIFT;
    uint32 block_count =
        (document_count + documents_per_block - 1U) /
        documents_per_block;
    Size runs_size = (Size) block_count * 2U * sizeof(uint32);
    uint32 run_count = 0;
    uint32 selected_document_count = 0;

    if (runs_size > MaxAllocSize)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose document runs",
            EVOKE_ERR_RANGE
        );
    }
    if (runs_size > reader->document_runs_capacity)
    {
        reader->document_runs = reader->document_runs == NULL
            ? palloc(runs_size)
            : repalloc(reader->document_runs, runs_size);
        reader->document_runs_capacity = runs_size;
    }
    for (uint32 block = 0; block < block_count; block++)
    {
        uint32 start = block * documents_per_block;
        uint32 end = Min(start + documents_per_block, document_count);

        if (!evoke_segment_pages_forward_block_allowed(
                allowed_document_bitmap,
                first_document,
                start,
                end))
        {
            continue;
        }
        selected_document_count += end - start;
        if (run_count > 0 &&
            reader->document_runs[(run_count - 1U) * 2U + 1U] == start)
        {
            reader->document_runs[(run_count - 1U) * 2U + 1U] = end;
            continue;
        }
        reader->document_runs[run_count * 2U] = start;
        reader->document_runs[run_count * 2U + 1U] = end;
        run_count++;
    }
    *selected_document_count_out = selected_document_count;
    return run_count;
}

static uint64
evoke_segment_pages_forward_read_document_runs(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *object,
    Size base_offset,
    Size element_size,
    uint8 *bytes_out,
    const uint32 *document_runs,
    uint32 run_count
)
{
    uint64 bytes_read = 0;

    for (uint32 run = 0; run < run_count; run++)
    {
        uint32 start = document_runs[run * 2U];
        uint32 end = document_runs[run * 2U + 1U];
        Size run_offset = (Size) start * element_size;
        Size run_bytes = (Size) (end - start) * element_size;

        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            object,
            base_offset + run_offset,
            run_bytes,
            bytes_out + run_offset
        );
        bytes_read += run_bytes;
    }
    return bytes_read;
}

void
evoke_segment_pages_score_semantic_accelerator_forward_transposed(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_transpose_reader *reader,
    uint32 forward_chunk_index,
    const uint8 *allowed_document_bitmap,
    Size allowed_document_bitmap_size,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    float *scores_out,
    Size score_count,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
)
{
    const evoke_semantic_accelerator_forward_entry *entry;
    evoke_semantic_forward_header header;
    uint8 header_bytes[EVOKE_SEMANTIC_FORWARD_HEADER_BYTES];
    Size header_size = evoke_semantic_forward_header_size();
    Size scales_size;
    Size dense_term_ids_size;
    Size query_ranges_size;
    Size query_dense_indexes_size;
    Size score_bytes;
    uint32 document_run_count = 0;
    uint32 selected_document_count = 0;
    uint64 active_b64_mask = 0;
    bool use_document_subranges = false;
    bool measure_sparse_block_oracle = false;
    uint64 postings_examined = 0;
    uint64 bytes_read;
    evoke_status status;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        reader == NULL || query_ids == NULL || query_weights == NULL ||
        scores_out == NULL || postings_examined_out == NULL ||
        bytes_read_out == NULL || header_size != sizeof(header_bytes) ||
        forward_chunk_index >= directory->forward_chunk_count ||
        query_count == 0 ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest
        ))
    {
        ereport(ERROR, (errmsg("invalid evoke semantic transpose score")));
    }
    *postings_examined_out = 0;
    *bytes_read_out = 0;
    entry = &directory->forward_chunks[forward_chunk_index];
    if (score_count < entry->document_count ||
        (allowed_document_bitmap != NULL &&
         allowed_document_bitmap_size <
            ((Size) directory->document_count + 7U) / 8U) ||
        evoke_segment_object_ref_validate(
            &entry->forward_object,
            context->root.published_block_high_watermark) != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose reference",
            EVOKE_ERR_FORMAT
        );
    }
    if (!reader->authority_ready)
    {
        status = evoke_segment_pages_accelerator_artifact_authority(
            directory->source_manifest_id,
            directory->source_authority_checksum,
            directory->owner_manifest_id,
            directory->document_count,
            directory->vocab_size,
            &context->manifest,
            &reader->authority_checksum
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator transpose authority",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        reader->authority_ready = true;
    }
    evoke_segment_pages_read_query_range(
        index_relation,
        context,
        &entry->forward_object,
        0,
        header_size,
        header_bytes
    );
    status = evoke_semantic_forward_header_deserialize(
        header_bytes,
        header_size,
        entry->forward_object.object_bytes,
        reader->authority_checksum,
        &header
    );
    if (status != EVOKE_OK ||
        header.first_document != entry->first_document ||
        header.document_count != entry->document_count)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose header",
            status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
        );
    }
    if (header.format_version != EVOKE_SEMANTIC_FORWARD_TRANSPOSE_VERSION)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose version",
            EVOKE_ERR_FORMAT
        );
    }
    if (header.vocab_size != directory->vocab_size ||
        header.transpose_entry_size !=
            EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose layout",
            EVOKE_ERR_FORMAT
        );
    }
    for (size_t query = 0; query < query_count; query++)
    {
        if (!isfinite(query_weights[query]) ||
            query_ids[query] >= header.vocab_size ||
            (query > 0 && query_ids[query - 1U] >= query_ids[query]))
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator transpose query",
                EVOKE_ERR_INVALID
            );
        }
    }
    scales_size = (Size) header.document_count * sizeof(float);
    if (query_count > MaxAllocSize / (2U * sizeof(uint32)))
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose query ranges",
            EVOKE_ERR_RANGE
        );
    }
    query_ranges_size = query_count * 2U * sizeof(uint32);
    query_dense_indexes_size = query_count * sizeof(uint32);
    dense_term_ids_size =
        (Size) header.transpose_dense_term_count * sizeof(uint32);
    score_bytes = (Size) header.document_count * sizeof(*reader->scores);
    if (scales_size > MaxAllocSize ||
        dense_term_ids_size > MaxAllocSize ||
        query_ranges_size > MaxAllocSize ||
        query_dense_indexes_size > MaxAllocSize ||
        score_bytes > MaxAllocSize)
    {
        evoke_segment_pages_report_codec_error(
            "semantic accelerator transpose allocation",
            EVOKE_ERR_RANGE
        );
    }
    if (scales_size > reader->scales_capacity)
    {
        reader->scales = reader->scales == NULL
            ? palloc(scales_size)
            : repalloc(reader->scales, scales_size);
        reader->scales_capacity = scales_size;
    }
    if (query_ranges_size > reader->query_ranges_capacity)
    {
        reader->query_ranges = reader->query_ranges == NULL
            ? palloc(query_ranges_size)
            : repalloc(reader->query_ranges, query_ranges_size);
        reader->query_ranges_capacity = query_ranges_size;
    }
    if (query_dense_indexes_size > reader->query_dense_indexes_capacity)
    {
        reader->query_dense_indexes =
            reader->query_dense_indexes == NULL
                ? palloc(query_dense_indexes_size)
                : repalloc(
                    reader->query_dense_indexes,
                    query_dense_indexes_size
                );
        reader->query_dense_indexes_capacity = query_dense_indexes_size;
    }
    if (score_bytes > reader->score_capacity)
    {
        reader->scores = reader->scores == NULL
            ? palloc(score_bytes)
            : repalloc(reader->scores, score_bytes);
        reader->score_capacity = score_bytes;
    }
    MemSet(reader->scores, 0, score_bytes);
    MemSet(scores_out, 0, (Size) header.document_count * sizeof(float));
    if (evoke_test_semantic_accelerator_sparse_block_oracle &&
        allowed_document_bitmap != NULL)
    {
        Size oracle_pages_size = entry->forward_object.page_count;

        if (oracle_pages_size > MaxAllocSize)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator sparse block oracle pages",
                EVOKE_ERR_RANGE
            );
        }
        if (oracle_pages_size > reader->sparse_oracle_pages_capacity)
        {
            reader->sparse_oracle_pages =
                reader->sparse_oracle_pages == NULL
                    ? palloc(oracle_pages_size)
                    : repalloc(
                        reader->sparse_oracle_pages,
                        oracle_pages_size
                    );
            reader->sparse_oracle_pages_capacity = oracle_pages_size;
        }
        MemSet(reader->sparse_oracle_pages, 0, oracle_pages_size);
        active_b64_mask = evoke_segment_pages_forward_active_b64_mask(
            allowed_document_bitmap,
            header.first_document,
            header.document_count
        );
        measure_sparse_block_oracle = true;
    }
    if (evoke_test_semantic_accelerator_dense_subrange_reads &&
        allowed_document_bitmap != NULL)
    {
        document_run_count =
            evoke_segment_pages_forward_build_document_runs(
                reader,
                allowed_document_bitmap,
                header.first_document,
                header.document_count,
                &selected_document_count
            );
        use_document_subranges = document_run_count > 0 &&
            selected_document_count < header.document_count;
    }
    if (use_document_subranges)
    {
        MemSet(reader->scales, 0, scales_size);
        bytes_read = evoke_segment_pages_forward_read_document_runs(
            index_relation,
            context,
            &entry->forward_object,
            header.transpose_scales_offset,
            sizeof(float),
            reader->scales,
            reader->document_runs,
            document_run_count
        );
    }
    else
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            header.transpose_scales_offset,
            scales_size,
            reader->scales
        );
        bytes_read = scales_size;
    }
    for (uint32 document = 0;
         document < header.document_count;
         document++)
    {
        float scale = evoke_segment_pages_forward_read_f32(
            reader->scales + (Size) document * sizeof(float)
        );

        if (use_document_subranges)
        {
            uint32 global_document = header.first_document + document;

            if ((allowed_document_bitmap[global_document >> 3] &
                 (uint8) (UINT8_C(1) <<
                    (global_document & 7U))) == 0)
            {
                continue;
            }
        }
        if (!isfinite(scale) || scale <= 0.0f)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator transpose scales",
                EVOKE_ERR_FORMAT
            );
        }
    }
    if (dense_term_ids_size > reader->dense_term_ids_capacity)
    {
        reader->dense_term_ids = reader->dense_term_ids == NULL
            ? palloc(dense_term_ids_size)
            : repalloc(reader->dense_term_ids, dense_term_ids_size);
        reader->dense_term_ids_capacity = dense_term_ids_size;
    }
    if (dense_term_ids_size > 0)
    {
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            header.transpose_dense_term_ids_offset,
            dense_term_ids_size,
            reader->dense_term_ids
        );
        for (uint32 dense_index = 0;
             dense_index < header.transpose_dense_term_count;
             dense_index++)
        {
            uint32 dense_term = evoke_segment_pages_forward_read_u32(
                reader->dense_term_ids +
                    (Size) dense_index * sizeof(uint32)
            );

            if (dense_term >= header.vocab_size ||
                (dense_index > 0 &&
                 dense_term <= evoke_segment_pages_forward_read_u32(
                    reader->dense_term_ids +
                        (Size) (dense_index - 1U) * sizeof(uint32))))
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator transpose dense terms",
                    EVOKE_ERR_FORMAT
                );
            }
        }
    }
    bytes_read += (uint64) header_size + dense_term_ids_size;
    for (size_t query = 0; query < query_count; query++)
    {
        reader->query_dense_indexes[query] =
            evoke_segment_pages_forward_find_dense_term(
                reader->dense_term_ids,
                header.transpose_dense_term_count,
                query_ids[query]
            );
    }
    for (size_t query = 0; query < query_count;)
    {
        size_t window_first_query = query;
        size_t window_end_query = query + 1U;
        Size window_start =
            (Size) query_ids[query] * sizeof(uint32);
        Size window_end =
            ((Size) query_ids[query] + 2U) * sizeof(uint32);
        Size window_bytes;

        while (window_end_query < query_count)
        {
            Size next_start =
                (Size) query_ids[window_end_query] * sizeof(uint32);
            Size next_end =
                ((Size) query_ids[window_end_query] + 2U) *
                sizeof(uint32);
            Size gap = next_start > window_end
                ? next_start - window_end
                : 0;

            if (gap > 4096U ||
                next_end - window_start > 64U * 1024U)
            {
                break;
            }
            window_end = next_end;
            window_end_query++;
        }
        window_bytes = window_end - window_start;
        if (window_bytes > reader->term_offsets_capacity)
        {
            reader->term_offsets = reader->term_offsets == NULL
                ? palloc(window_bytes)
                : repalloc(reader->term_offsets, window_bytes);
            reader->term_offsets_capacity = window_bytes;
        }
        evoke_segment_pages_read_query_range(
            index_relation,
            context,
            &entry->forward_object,
            header.transpose_term_offsets_offset + window_start,
            window_bytes,
            reader->term_offsets
        );
        bytes_read += window_bytes;
        for (size_t window_query = window_first_query;
             window_query < window_end_query;
             window_query++)
        {
            Size local_offset =
                (Size) query_ids[window_query] * sizeof(uint32) -
                window_start;
            uint32 start = evoke_segment_pages_forward_read_u32(
                reader->term_offsets + local_offset
            );
            uint32 end = evoke_segment_pages_forward_read_u32(
                reader->term_offsets + local_offset + sizeof(uint32)
            );

            if (start > end ||
                end > header.transpose_sparse_posting_count ||
                (reader->query_dense_indexes[window_query] != UINT32_MAX &&
                 start != end) ||
                (window_query > 0 &&
                 start < reader->query_ranges[
                    (window_query - 1U) * 2U + 1U]))
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator transpose query offsets",
                    EVOKE_ERR_FORMAT
                );
            }
            reader->query_ranges[window_query * 2U] = start;
            reader->query_ranges[window_query * 2U + 1U] = end;
        }
        query = window_end_query;
        CHECK_FOR_INTERRUPTS();
    }
    for (size_t query = 0; query < query_count;)
    {
        uint32 dense_index = reader->query_dense_indexes[query];
        size_t range_first_query = query;
        size_t range_end_query = query + 1U;
        uint32 range_start = reader->query_ranges[query * 2U];
        uint32 range_end = reader->query_ranges[query * 2U + 1U];
        Size posting_bytes;

        if (dense_index != UINT32_MAX)
        {
            Size dense_bytes = (Size) header.document_count * sizeof(int8);

            if (dense_bytes > reader->dense_codes_capacity)
            {
                reader->dense_codes = reader->dense_codes == NULL
                    ? palloc(dense_bytes)
                    : repalloc(reader->dense_codes, dense_bytes);
                reader->dense_codes_capacity = dense_bytes;
            }
            if (use_document_subranges)
            {
                MemSet(reader->dense_codes, 0, dense_bytes);
                bytes_read +=
                    evoke_segment_pages_forward_read_document_runs(
                        index_relation,
                        context,
                        &entry->forward_object,
                        header.transpose_dense_codes_offset +
                            (Size) dense_index * dense_bytes,
                        sizeof(int8),
                        reader->dense_codes,
                        reader->document_runs,
                        document_run_count
                    );
            }
            else
            {
                evoke_segment_pages_read_query_range(
                    index_relation,
                    context,
                    &entry->forward_object,
                    header.transpose_dense_codes_offset +
                        (Size) dense_index * dense_bytes,
                    dense_bytes,
                    reader->dense_codes
                );
                bytes_read += dense_bytes;
            }
            if (UINT64_MAX - postings_examined < header.document_count)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator transpose dense telemetry",
                    EVOKE_ERR_RANGE
                );
            }
            postings_examined += evoke_segment_pages_accumulate_dense_codes(
                reader->dense_codes,
                reader->scales,
                header.first_document,
                header.document_count,
                allowed_document_bitmap,
                query_weights[query],
                reader->scores
            );
            query++;
            CHECK_FOR_INTERRUPTS();
            continue;
        }
        while (range_end_query < query_count)
        {
            uint32 next_start =
                reader->query_ranges[range_end_query * 2U];
            uint32 next_end =
                reader->query_ranges[range_end_query * 2U + 1U];
            Size gap_bytes =
                (Size) (next_start - range_end) *
                header.transpose_entry_size;
            Size merged_bytes =
                (Size) (next_end - range_start) *
                header.transpose_entry_size;

            /*
             * Coalesce nearby query slices without turning a selective read
             * into a scan over a large unrelated postings interval.
             */
            if (reader->query_dense_indexes[range_end_query] != UINT32_MAX ||
                gap_bytes > 4096U || merged_bytes > 256U * 1024U)
            {
                break;
            }
            range_end = next_end;
            range_end_query++;
        }
        posting_bytes =
            (Size) (range_end - range_start) *
            header.transpose_entry_size;

        if (posting_bytes > MaxAllocSize)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator transpose postings",
                EVOKE_ERR_RANGE
            );
        }
        if (posting_bytes > reader->postings_capacity)
        {
            reader->postings = reader->postings == NULL
                ? palloc(posting_bytes)
                : repalloc(reader->postings, posting_bytes);
            reader->postings_capacity = posting_bytes;
        }
        if (posting_bytes > 0)
        {
            evoke_segment_pages_read_query_range(
                index_relation,
                context,
                &entry->forward_object,
                header.transpose_postings_offset +
                    (Size) range_start * header.transpose_entry_size,
                posting_bytes,
                reader->postings
            );
        }
        bytes_read += posting_bytes;
        for (size_t range_query = range_first_query;
             range_query < range_end_query;
             range_query++)
        {
            uint32 start = reader->query_ranges[range_query * 2U];
            uint32 end = reader->query_ranges[range_query * 2U + 1U];
            uint16 previous_document = 0;
            uint64 oracle_selected_postings = 0;
            uint64 oracle_ranges = 0;
            bool oracle_range_open = false;

            if (measure_sparse_block_oracle)
            {
                uint64 full_bytes =
                    (uint64) (end - start) *
                    header.transpose_entry_size;

                if (UINT64_MAX - reader->sparse_oracle_full_bytes <
                    full_bytes)
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator sparse block oracle bytes",
                        EVOKE_ERR_RANGE
                    );
                }
                reader->sparse_oracle_full_bytes += full_bytes;
            }

            for (uint32 posting = 0; posting < end - start; posting++)
            {
                Size posting_offset =
                    (Size) (start - range_start + posting) *
                    header.transpose_entry_size;
                uint16 document = evoke_segment_pages_forward_read_u16(
                    reader->postings + posting_offset
                );
                int8 code = (int8) reader->postings[
                    posting_offset + sizeof(uint16)
                ];
                uint32 global_document =
                    header.first_document + (uint32) document;
                float scale;
                bool oracle_selected = false;

                if (document >= header.document_count || code == INT8_MIN ||
                    (posting > 0 && document < previous_document))
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator transpose posting",
                        EVOKE_ERR_FORMAT
                    );
                }
                previous_document = document;
                if (measure_sparse_block_oracle)
                {
                    uint32 block =
                        (uint32) document >>
                        EVOKE_SEMANTIC_FORWARD_SUBRANGE_SHIFT;
                    Size object_offset =
                        header.transpose_postings_offset +
                        ((Size) start + posting) *
                            header.transpose_entry_size;
                    Size payload_capacity =
                        evoke_segment_page_payload_capacity();
                    uint32 page = (uint32) (
                        object_offset / payload_capacity
                    );

                    if (page >= entry->forward_object.page_count)
                    {
                        evoke_segment_pages_report_codec_error(
                            "semantic accelerator sparse block oracle page",
                            EVOKE_ERR_RANGE
                        );
                    }
                    reader->sparse_oracle_pages[page] |= UINT8_C(1);

                    oracle_selected =
                        (active_b64_mask & (UINT64_C(1) << block)) != 0;
                    if (oracle_selected)
                    {
                        reader->sparse_oracle_pages[page] |= UINT8_C(2);
                        oracle_selected_postings++;
                        if (!oracle_range_open)
                        {
                            oracle_ranges++;
                        }
                    }
                    oracle_range_open = oracle_selected;
                }
                if (allowed_document_bitmap != NULL &&
                    (allowed_document_bitmap[global_document >> 3] &
                     (uint8) (UINT8_C(1) <<
                        (global_document & 7U))) == 0)
                {
                    continue;
                }
                scale = evoke_segment_pages_forward_read_f32(
                    reader->scales + (Size) document * sizeof(float)
                );
                reader->scores[document] +=
                    (double) code * scale * query_weights[range_query];
                if (!isfinite(reader->scores[document]) ||
                    reader->scores[document] > FLT_MAX ||
                    reader->scores[document] < -FLT_MAX)
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator transpose score",
                        EVOKE_ERR_RANGE
                    );
                }
            }
            if (measure_sparse_block_oracle)
            {
                uint64 selected_bytes =
                    oracle_selected_postings *
                    header.transpose_entry_size;

                if (UINT64_MAX - reader->sparse_oracle_selected_bytes <
                        selected_bytes ||
                    UINT64_MAX - reader->sparse_oracle_ranges <
                        oracle_ranges)
                {
                    evoke_segment_pages_report_codec_error(
                        "semantic accelerator sparse block oracle totals",
                        EVOKE_ERR_RANGE
                    );
                }
                reader->sparse_oracle_selected_bytes += selected_bytes;
                reader->sparse_oracle_ranges += oracle_ranges;
            }
            postings_examined += (uint64) end - start;
        }
        query = range_end_query;
        CHECK_FOR_INTERRUPTS();
    }
    if (measure_sparse_block_oracle)
    {
        for (uint32 page = 0;
             page < entry->forward_object.page_count;
             page++)
        {
            if ((reader->sparse_oracle_pages[page] & UINT8_C(1)) != 0)
            {
                reader->sparse_oracle_full_pages++;
            }
            if ((reader->sparse_oracle_pages[page] & UINT8_C(2)) != 0)
            {
                reader->sparse_oracle_selected_pages++;
            }
        }
    }
    for (uint32 document = 0;
         document < header.document_count;
         document++)
    {
        scores_out[document] = (float) reader->scores[document];
    }
    *postings_examined_out = postings_examined;
    *bytes_read_out = bytes_read;
    return;
}

static evoke_status
evoke_segment_pages_accumulate_forward_costs(
    const uint8 *bytes,
    size_t size,
    uint64 authority_checksum,
    uint32 expected_vocab_size,
    uint64 *term_work,
    uint64 *term_bytes,
    uint32 *row_offsets_out,
    size_t row_offset_count,
    uint32 *row_data_offset_out,
    uint64 *row_data_bytes_out,
    uint64 *transpose_fixed_bytes_out
)
{
    evoke_semantic_forward_header header;
    evoke_status status;

    if (bytes == NULL || term_work == NULL || term_bytes == NULL ||
        row_offsets_out == NULL || row_data_offset_out == NULL ||
        row_data_bytes_out == NULL || transpose_fixed_bytes_out == NULL ||
        expected_vocab_size == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_forward_header_deserialize(
        bytes,
        size,
        size,
        authority_checksum,
        &header
    );
    if (status != EVOKE_OK || header.vocab_size != expected_vocab_size ||
        row_offset_count != (size_t) header.document_count + 1U ||
        header.row_data_offset > UINT32_MAX)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *row_data_offset_out = (uint32) header.row_data_offset;
    *row_data_bytes_out = header.row_data_end - header.row_data_offset;
    *transpose_fixed_bytes_out = evoke_u64_saturating_add(
        evoke_semantic_forward_header_size(),
        evoke_u64_saturating_add(
            (uint64) header.document_count * sizeof(float),
            (uint64) header.transpose_dense_term_count * sizeof(uint32)
        )
    );
    if (*transpose_fixed_bytes_out == UINT64_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    for (size_t row = 0; row < row_offset_count; row++)
    {
        uint32 offset = evoke_segment_pages_forward_read_u32(
            bytes + header.row_offsets_offset + row * sizeof(uint32)
        );

        if ((row > 0 && offset < row_offsets_out[row - 1U]) ||
            offset > *row_data_bytes_out)
        {
            return EVOKE_ERR_FORMAT;
        }
        row_offsets_out[row] = offset;
    }
    if (row_offsets_out[0] != 0 ||
        row_offsets_out[row_offset_count - 1U] != *row_data_bytes_out)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32 term_id = 0; term_id < header.vocab_size; term_id++)
    {
        uint32 start = evoke_segment_pages_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32)
        );
        uint32 end = evoke_segment_pages_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) (term_id + 1U) * sizeof(uint32)
        );
        uint64 sparse_bytes;
        uint64 sparse_work;

        if (start > end || end > header.transpose_sparse_posting_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        sparse_bytes = evoke_u64_saturating_add(
            ((uint64) end - start) * header.transpose_entry_size,
            2U * sizeof(uint32)
        );
        if (sparse_bytes == UINT64_MAX ||
            UINT64_MAX - term_work[term_id] <
                (uint64) end - start ||
            UINT64_MAX - term_bytes[term_id] < sparse_bytes)
        {
            return EVOKE_ERR_RANGE;
        }
        sparse_work = (uint64) end - start;
        term_work[term_id] += sparse_work;
        term_bytes[term_id] += sparse_bytes;
    }
    for (uint32 dense_index = 0;
         dense_index < header.transpose_dense_term_count;
         dense_index++)
    {
        uint32 term_id = evoke_segment_pages_forward_read_u32(
            bytes + header.transpose_dense_term_ids_offset +
                (size_t) dense_index * sizeof(uint32)
        );

        if (term_id >= header.vocab_size ||
            UINT64_MAX - term_work[term_id] < header.document_count ||
            UINT64_MAX - term_bytes[term_id] < header.document_count)
        {
            return term_id >= header.vocab_size
                ? EVOKE_ERR_FORMAT
                : EVOKE_ERR_RANGE;
        }
        term_work[term_id] += header.document_count;
        term_bytes[term_id] += header.document_count;
    }
    return EVOKE_OK;
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_semantic_accelerator_complete_stream_fork(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    uint32 artifact_count,
    evoke_semantic_accelerator_artifact_producer produce_artifact,
    void *producer_context,
    uint32 forward_chunk_count,
    uint32 forward_document_shift,
    evoke_semantic_forward_artifact_producer produce_forward,
    void *forward_context,
    uint32 forward_bound_shard_count,
    evoke_semantic_forward_bound_artifact_producer produce_forward_bound,
    void *forward_bound_context,
    const uint8 *scope_bytes,
    Size scope_size,
    const uint8 *tid_lookup_bytes,
    Size tid_lookup_size,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_manifest *next_manifest,
    evoke_segment_cow_result *result_out
)
{
    evoke_semantic_accelerator_directory directory;
    evoke_semantic_accelerator_index index;
    evoke_semantic_forward_chunk forward;
    evoke_segment_manifest manifest;
    evoke_segment_cow_result result;
    evoke_segment_page_reuse_arena handoff_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_page_reuse_arena *write_arena;
    uint64 authority_checksum = 0;
    uint64 next_authority_checksum = 0;
    uint8_t *directory_bytes = NULL;
    uint8_t *manifest_bytes = NULL;
    size_t directory_size = 0;
    size_t manifest_size = 0;
    bool reader_fenced = reuse_arena != NULL &&
        reuse_arena->reader_fenced && reuse_arena->allocator.initialized;
    evoke_segment_cow_write_outcome outcome =
        EVOKE_SEGMENT_COW_WRITE_WRITTEN;
    evoke_status status;

    if (index_relation == NULL || build_root == NULL ||
        old_manifest == NULL || produce_artifact == NULL ||
        artifact_count == 0 || next_manifest == NULL ||
        result_out == NULL ||
        (forward_chunk_count > 0 &&
         (forward_document_shift == 0 ||
          forward_document_shift >= 32 || produce_forward == NULL)) ||
        (forward_chunk_count == 0 &&
         (forward_document_shift != 0 || produce_forward != NULL)) ||
        (forward_bound_shard_count > 0 && produce_forward_bound == NULL) ||
        (forward_bound_shard_count == 0 && produce_forward_bound != NULL) ||
        ((scope_bytes == NULL) != (scope_size == 0)) ||
        ((tid_lookup_bytes == NULL) != (tid_lookup_size == 0)) ||
        (fork_number != MAIN_FORKNUM && fork_number != INIT_FORKNUM) ||
        (reuse_arena != NULL && !reader_fenced) ||
        (fork_number != MAIN_FORKNUM && reuse_arena != NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke accelerator publication")));
    }
    evoke_semantic_accelerator_directory_init(&directory);
    evoke_semantic_accelerator_index_init(&index);
    evoke_semantic_forward_chunk_init(&forward);
    evoke_segment_manifest_init(&manifest);
    evoke_segment_cow_result_init(&result);
    evoke_segment_page_reuse_arena_init(&handoff_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    write_arena = reuse_arena != NULL ? reuse_arena : &staging_arena;

    PG_TRY();
    {
        status = evoke_segment_manifest_validate_published(
            old_manifest,
            &build_root->manifest,
            build_root->published_block_high_watermark
        );
        if (status != EVOKE_OK ||
            build_root->root_id != old_manifest->manifest_id ||
            (old_manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_SAE) == 0 ||
            (old_manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
            old_manifest->document_slot_count == 0 ||
            artifact_count > old_manifest->vocab_size)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator source root",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        status = evoke_segment_manifest_authority_checksum(
            old_manifest,
            &authority_checksum
        );
        if (status == EVOKE_OK)
        {
            status = evoke_segment_manifest_build_identity(
                build_root,
                old_manifest,
                &manifest
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator manifest identity",
                status
            );
        }
        manifest.flags &=
            ~EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR;
        memset(
            &manifest.semantic_accelerator_directory,
            0,
            sizeof(manifest.semantic_accelerator_directory)
        );
        manifest.semantic_accelerator_max_sequence = 0;
        if (!reader_fenced &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
                0,
                0,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                reuse_arena,
                &result,
                &manifest))
        {
            outcome =
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED;
        }
        if (reader_fenced)
        {
            free(manifest.retired_ranges);
            manifest.retired_ranges = NULL;
            manifest.retired_range_count = 0;
        }

        directory.source_manifest_id = old_manifest->manifest_id;
        directory.source_authority_checksum = authority_checksum;
        directory.owner_manifest_id = manifest.manifest_id;
        manifest.semantic_accelerator_max_sequence =
            old_manifest->max_sequence;
        directory.document_count =
            (uint32) old_manifest->document_slot_count;
        directory.vocab_size = old_manifest->vocab_size;
        directory.term_count = artifact_count;
        directory.forward_chunk_count = forward_chunk_count;
        directory.forward_bound_shard_count =
            forward_bound_shard_count;
        directory.forward_document_shift = forward_document_shift;
        directory.builder_policy_id =
            EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY;
        directory.retained_document_cap =
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP;
        directory.terms = calloc(
            artifact_count,
            sizeof(*directory.terms)
        );
        if (directory.terms == NULL)
        {
            ereport(ERROR, (errmsg("out of memory")));
        }
        if (forward_chunk_count > 0)
        {
            uint64 forward_row_offset_count =
                (uint64) directory.document_count + forward_chunk_count;

            if (forward_row_offset_count > SIZE_MAX /
                    sizeof(*directory.forward_row_offsets))
            {
                ereport(ERROR, (errmsg("evoke forward row directory too large")));
            }
            directory.forward_chunks = calloc(
                forward_chunk_count,
                sizeof(*directory.forward_chunks)
            );
            if (directory.forward_chunks == NULL)
            {
                ereport(ERROR, (errmsg("out of memory")));
            }
            directory.forward_row_data_bytes = calloc(
                forward_chunk_count,
                sizeof(*directory.forward_row_data_bytes)
            );
            directory.forward_transpose_fixed_bytes = calloc(
                forward_chunk_count,
                sizeof(*directory.forward_transpose_fixed_bytes)
            );
            directory.forward_row_offsets = calloc(
                (size_t) forward_row_offset_count,
                sizeof(*directory.forward_row_offsets)
            );
            directory.forward_term_work = calloc(
                directory.vocab_size,
                sizeof(*directory.forward_term_work)
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
                directory.forward_term_work == NULL ||
                directory.forward_term_bytes == NULL ||
                directory.forward_bound_term_bytes == NULL)
            {
                ereport(ERROR, (errmsg("out of memory")));
            }
        }
        if (forward_bound_shard_count > 0)
        {
            uint32 expected_bound_shards =
                (directory.vocab_size - 1U) /
                    EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U;

            if (forward_bound_shard_count != expected_bound_shards)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke forward bound shard count"))
                );
            }
            directory.forward_bound_shards = calloc(
                forward_bound_shard_count,
                sizeof(*directory.forward_bound_shards)
            );
            if (directory.forward_bound_shards == NULL)
            {
                ereport(ERROR, (errmsg("out of memory")));
            }
        }

        for (uint32 artifact_index = 0;
             artifact_index < artifact_count;
             artifact_index++)
        {
            evoke_semantic_accelerator_term_artifact artifact;
            evoke_semantic_accelerator_directory_entry *entry;

            memset(&artifact, 0, sizeof(artifact));
            status = produce_artifact(
                producer_context,
                artifact_index,
                &artifact
            );

            if (status != EVOKE_OK || artifact.bytes == NULL ||
                artifact.size == 0 ||
                artifact.term_id >= old_manifest->vocab_size ||
                (artifact_index > 0 &&
                 directory.terms[artifact_index - 1].term_id >=
                    artifact.term_id))
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke accelerator term artifact"))
                );
            }
            status = evoke_semantic_accelerator_term_deserialize(
                artifact.bytes,
                artifact.size,
                authority_checksum,
                artifact.term_id,
                &index
            );
            if (status == EVOKE_OK &&
                index.document_count !=
                    old_manifest->document_slot_count)
            {
                status = EVOKE_ERR_FORMAT;
            }
            evoke_semantic_accelerator_index_free(&index);
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator term artifact",
                    status
                );
            }
            entry = &directory.terms[artifact_index];

            entry->term_id = artifact.term_id;
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TERM,
                (uint64) artifact.term_id + 1,
                manifest.manifest_id,
                artifact.bytes,
                artifact.size,
                write_arena,
                &entry->term_object
            );
        }
        for (uint32 forward_index = 0;
             forward_index < forward_chunk_count;
             forward_index++)
        {
            evoke_semantic_forward_artifact artifact;
            evoke_semantic_accelerator_forward_entry *entry;
            uint32 posting_count;
            uint32 row_data_offset = 0;
            uint32 expected_first = forward_index == 0
                ? 0
                : directory.forward_chunks[
                    forward_index - 1U
                ].first_document + directory.forward_chunks[
                    forward_index - 1U
                ].document_count;

            memset(&artifact, 0, sizeof(artifact));
            status = produce_forward(
                forward_context,
                forward_index,
                &artifact
            );
            if (status != EVOKE_OK || artifact.bytes == NULL ||
                artifact.size == 0 || artifact.document_count == 0 ||
                artifact.first_document != expected_first ||
                artifact.document_count >
                    old_manifest->document_slot_count ||
                artifact.first_document >
                    old_manifest->document_slot_count -
                        artifact.document_count)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke accelerator forward artifact"))
                );
            }
            status = evoke_semantic_forward_chunk_deserialize(
                artifact.bytes,
                artifact.size,
                authority_checksum,
                &forward
            );
            if (status == EVOKE_OK &&
                (forward.first_document != artifact.first_document ||
                 forward.document_count != artifact.document_count))
            {
                status = EVOKE_ERR_FORMAT;
            }
            if (status == EVOKE_OK)
            {
                uint64 row_offset_index =
                    (uint64) artifact.first_document + forward_index;
                uint64 row_offset_end = row_offset_index +
                    artifact.document_count + UINT64_C(1);

                if (row_offset_end >
                    (uint64) directory.document_count +
                        directory.forward_chunk_count)
                {
                    status = EVOKE_ERR_FORMAT;
                }

                if (status == EVOKE_OK)
                {
                    status = evoke_segment_pages_accumulate_forward_costs(
                        artifact.bytes,
                        artifact.size,
                        authority_checksum,
                        directory.vocab_size,
                        directory.forward_term_work,
                        directory.forward_term_bytes,
                        directory.forward_row_offsets + row_offset_index,
                        (size_t) artifact.document_count + 1U,
                        &row_data_offset,
                        &directory.forward_row_data_bytes[forward_index],
                        &directory.forward_transpose_fixed_bytes[forward_index]
                    );
                }
            }
            posting_count = forward.posting_count;
            evoke_semantic_forward_chunk_free(&forward);
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator forward artifact",
                    status
                );
            }
            entry = &directory.forward_chunks[forward_index];
            entry->first_document = artifact.first_document;
            entry->document_count = artifact.document_count;
            entry->posting_count = posting_count;
            entry->row_data_offset = row_data_offset;
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD,
                (uint64) forward_index + 1U,
                manifest.manifest_id,
                artifact.bytes,
                artifact.size,
                write_arena,
                &entry->forward_object
            );
        }
        for (uint32 bound_index = 0;
             bound_index < forward_bound_shard_count;
             bound_index++)
        {
            evoke_semantic_forward_bound_artifact artifact;
            evoke_semantic_forward_bound_summary summary;
            uint32 expected_first = bound_index *
                EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD;
            uint32 expected_count = Min(
                EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD,
                directory.vocab_size - expected_first
            );

            memset(&artifact, 0, sizeof(artifact));
            status = produce_forward_bound(
                forward_bound_context,
                bound_index,
                &artifact
            );
            if (status != EVOKE_OK || artifact.bytes == NULL ||
                artifact.size == 0 ||
                artifact.first_term != expected_first ||
                artifact.term_count != expected_count)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke forward bound artifact"))
                );
            }
            status = evoke_semantic_forward_bound_header_deserialize(
                artifact.bytes,
                artifact.size,
                artifact.size,
                authority_checksum,
                &summary
            );
            if (status == EVOKE_OK &&
                (summary.first_term != artifact.first_term ||
                 summary.term_count != artifact.term_count ||
                 summary.document_count != directory.document_count ||
                 summary.vocab_size != directory.vocab_size))
            {
                status = EVOKE_ERR_FORMAT;
            }
            if (status == EVOKE_OK)
            {
                for (uint32 local_term = 0;
                     local_term < summary.term_count;
                     local_term++)
                {
                    size_t payload_offset;
                    size_t payload_size;
                    uint32 term_id = summary.first_term + local_term;

                    status = evoke_semantic_forward_bound_term_slice(
                        &summary,
                        artifact.bytes + summary.offsets_offset +
                            (size_t) local_term * sizeof(uint64),
                        2U * sizeof(uint64),
                        term_id,
                        &payload_offset,
                        &payload_size
                    );
                    if (status != EVOKE_OK)
                    {
                        break;
                    }
                    directory.forward_bound_term_bytes[term_id] =
                        payload_size;
                }
            }
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator forward bound artifact",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND,
                (uint64) bound_index + 1U,
                manifest.manifest_id,
                artifact.bytes,
                artifact.size,
                write_arena,
                &directory.forward_bound_shards[bound_index]
            );
        }
        if (scope_size > 0)
        {
            status = evoke_scope_validate(
                scope_bytes,
                scope_size,
                authority_checksum
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator scope artifact",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_SCOPE,
                1,
                manifest.manifest_id,
                scope_bytes,
                scope_size,
                write_arena,
                &directory.scope_object
            );
        }
        if (tid_lookup_size > 0)
        {
            evoke_document_tid_lookup_view tid_lookup;

            status = evoke_document_tid_lookup_open(
                tid_lookup_bytes,
                tid_lookup_size,
                authority_checksum,
                (uint32) old_manifest->visible_document_count,
                (uint32) old_manifest->document_slot_count,
                &tid_lookup
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "semantic accelerator TID lookup artifact",
                    status
                );
            }
            evoke_segment_pages_write_internal(
                index_relation,
                fork_number,
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TID_LOOKUP,
                1,
                manifest.manifest_id,
                tid_lookup_bytes,
                tid_lookup_size,
                write_arena,
                &directory.tid_lookup_object
            );
        }
        status = evoke_semantic_accelerator_directory_serialize(
            &directory,
            &directory_bytes,
            &directory_size
        );
        if (status != EVOKE_OK || directory_size > MaxAllocSize)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator directory serialization",
                status == EVOKE_OK ? EVOKE_ERR_RANGE : status
            );
        }
        evoke_segment_pages_write_internal(
            index_relation,
            fork_number,
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_DIRECTORY,
            manifest.manifest_id,
            manifest.manifest_id,
            directory_bytes,
            (Size) directory_size,
            write_arena,
            &manifest.semantic_accelerator_directory
        );
        free(directory_bytes);
        directory_bytes = NULL;
        manifest.flags |=
            EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR;

        status = evoke_segment_manifest_authority_checksum(
            &manifest,
            &next_authority_checksum
        );
        if (status != EVOKE_OK ||
            next_authority_checksum != authority_checksum)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator authority closure",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        status = evoke_segment_manifest_serialize(
            &manifest,
            &manifest_bytes,
            &manifest_size
        );
        if (status != EVOKE_OK || manifest_size > MaxAllocSize)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator manifest serialization",
                status == EVOKE_OK ? EVOKE_ERR_RANGE : status
            );
        }
        evoke_segment_pages_write_internal(
            index_relation,
            fork_number,
            EVOKE_SEGMENT_OBJECT_MANIFEST,
            manifest.manifest_id,
            manifest.manifest_id,
            manifest_bytes,
            (Size) manifest_size,
            write_arena,
            &result.manifest
        );
        free(manifest_bytes);
        manifest_bytes = NULL;
        /*
         * The manifest is the final reuse allocation. Snapshot the remaining
         * free ranges only after it is placed, or the handoff can recycle the
         * newly published root page.
         */
        if (outcome ==
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED)
        {
            status = old_manifest->retired_range_count > 0
                ? evoke_segment_page_reuse_arena_build_retired(
                      old_manifest->retired_ranges,
                      old_manifest->retired_range_count,
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  )
                : evoke_segment_page_reuse_arena_build_empty_fenced(
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "prepare accelerator FSM handoff",
                    status
                );
            }
        }
        if ((reader_fenced ||
             outcome ==
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED) &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
                0,
                0,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                reader_fenced ? reuse_arena : &handoff_arena,
                &result,
                &manifest))
        {
            ereport(
                ERROR,
                (errmsg("evoke accelerator retirement handoff overflow"))
            );
        }
        result.published_block_high_watermark =
            RelationGetNumberOfBlocksInFork(index_relation, fork_number);
        evoke_segment_pages_set_staged_writes(
            write_arena,
            result.published_block_high_watermark,
            &result
        );
        result.reused_block_count =
            evoke_segment_page_reuse_arena_reused_blocks(write_arena);
        status = evoke_segment_manifest_validate_published(
            &manifest,
            &result.manifest,
            result.published_block_high_watermark
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "semantic accelerator publication closure",
                status
            );
        }

        evoke_segment_manifest_free(next_manifest);
        *next_manifest = manifest;
        evoke_segment_manifest_init(&manifest);
        evoke_segment_cow_result_free(result_out);
        *result_out = result;
        evoke_segment_cow_result_init(&result);
    }
    PG_FINALLY();
    {
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        free(manifest_bytes);
        free(directory_bytes);
        evoke_segment_cow_result_free(&result);
        evoke_segment_manifest_free(&manifest);
        evoke_semantic_accelerator_index_free(&index);
        evoke_semantic_forward_chunk_free(&forward);
        evoke_semantic_accelerator_directory_free(&directory);
    }
    PG_END_TRY();
    return outcome;
}

static bool
evoke_segment_pages_object_ref_absent(
    const evoke_segment_object_ref *ref
)
{
    static const evoke_segment_object_ref zero_ref = {0};

    return ref != NULL &&
        memcmp(ref, &zero_ref, sizeof(*ref)) == 0;
}

static evoke_status
evoke_segment_pages_inventory_add_object_ref(
    evoke_segment_reachability_context *context,
    const evoke_segment_object_ref *ref
)
{
    evoke_status status;

    if (context == NULL || ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    CHECK_FOR_INTERRUPTS();
    if (evoke_segment_pages_object_ref_absent(ref))
    {
        return EVOKE_OK;
    }
    status = evoke_segment_object_ref_validate(
        ref,
        context->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_block_range_inventory_add(
        context->blocks,
        ref->start_block,
        ref->page_count
    );
}

static evoke_status
evoke_segment_pages_inventory_term_object(
    void *context_pointer,
    const evoke_term_cow_object *object
)
{
    evoke_segment_reachability_context *context = context_pointer;
    evoke_segment_object_ref ref;
    evoke_status status;

    if (context == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_term_cow_ref_as_segment_object_ref(
        &object->ref,
        &ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_pages_inventory_add_object_ref(context, &ref);
    if (status != EVOKE_OK ||
        object->ref.kind != EVOKE_TERM_COW_OBJECT_LEAF)
    {
        return status;
    }

    for (uint32 record_index = 0;
         record_index < object->value.leaf.record_count;
         record_index++)
    {
        const evoke_term_cow_record *record =
            &object->value.leaf.records[record_index];
        const evoke_segment_object_ref *refs[4] = {
            &record->lexical_catalog,
            &record->neutral_fold,
            &record->neutral_minor_fold,
            &record->impact_fold
        };

        if ((record_index & 255) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        for (size_t ref_index = 0; ref_index < 4; ref_index++)
        {
            status = evoke_segment_pages_inventory_add_object_ref(
                context,
                refs[ref_index]
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
evoke_segment_pages_inventory_document_object(
    void *context_pointer,
    const evoke_document_cow_object *object
)
{
    evoke_segment_reachability_context *context = context_pointer;
    evoke_segment_object_ref ref;
    evoke_status status;

    if (context == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_document_cow_ref_as_segment_object_ref(
        &object->ref,
        &ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_segment_pages_inventory_add_object_ref(context, &ref);
}

static evoke_status
evoke_segment_pages_inventory_lexicon_object(
    void *context_pointer,
    const evoke_lexicon_cow_object *object
)
{
    evoke_segment_reachability_context *context = context_pointer;
    evoke_segment_object_ref ref;
    evoke_status status;

    if (context == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_lexicon_cow_ref_as_segment_object_ref(
        &object->ref,
        &ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_segment_pages_inventory_add_object_ref(context, &ref);
}

static evoke_status
evoke_segment_pages_inventory_prefix_object(
    void *context_pointer,
    const evoke_prefix_cow_object *object
)
{
    evoke_segment_reachability_context *context = context_pointer;
    evoke_segment_object_ref ref;
    evoke_status status;

    if (context == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    CHECK_FOR_INTERRUPTS();
    status = evoke_prefix_cow_ref_as_segment_object_ref(
        &object->ref,
        &ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_segment_pages_inventory_add_object_ref(context, &ref);
}

static evoke_status
evoke_segment_pages_inventory_l0_frontier(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_active_l0_frontier *frontier,
    evoke_block_range_inventory *blocks
)
{
    uint32 *visited_blocks = NULL;
    uint32 current_block;
    volatile Buffer buffer = InvalidBuffer;
    evoke_status result = EVOKE_OK;

    if (frontier->record_count == 0)
    {
        return EVOKE_OK;
    }
    visited_blocks = palloc(
        sizeof(*visited_blocks) * (Size) frontier->page_count
    );
    current_block = frontier->head_block;
    PG_TRY();
    {
        for (uint32 page_index = 0;
             page_index < frontier->page_count;
             page_index++)
        {
            Page page;
            const uint8 *content;
            evoke_active_l0_page_header page_header;

            CHECK_FOR_INTERRUPTS();
            if (current_block == 0 ||
                current_block == EVOKE_ACTIVE_L0_NO_NEXT_BLOCK ||
                current_block >= root->published_block_high_watermark)
            {
                result = EVOKE_ERR_FORMAT;
                break;
            }
            for (uint32 visited_index = 0;
                 visited_index < page_index;
                 visited_index++)
            {
                if (visited_blocks[visited_index] == current_block)
                {
                    result = EVOKE_ERR_FORMAT;
                    break;
                }
            }
            if (result != EVOKE_OK)
            {
                break;
            }
            visited_blocks[page_index] = current_block;
            buffer = ReadBufferExtended(
                index_relation,
                MAIN_FORKNUM,
                current_block,
                RBM_NORMAL,
                NULL
            );
            LockBuffer(buffer, BUFFER_LOCK_SHARE);
            page = BufferGetPage(buffer);
            if (PageIsNew(page))
            {
                result = EVOKE_ERR_FORMAT;
            }
            else
            {
                content = (const uint8 *) PageGetContents(page);
                result = evoke_active_l0_page_header_deserialize(
                    content,
                    EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
                    evoke_segment_page_content_bytes(),
                    &page_header
                );
            }
            if (result == EVOKE_OK &&
                (page_header.segment_id != frontier->segment_id ||
                 page_header.ordinal != page_index))
            {
                result = EVOKE_ERR_FORMAT;
            }
            if (result == EVOKE_OK)
            {
                result = evoke_block_range_inventory_add(
                    blocks,
                    current_block,
                    1
                );
            }
            if (result == EVOKE_OK &&
                page_index + 1 == frontier->page_count &&
                current_block != frontier->tail_block)
            {
                result = EVOKE_ERR_FORMAT;
            }
            if (result == EVOKE_OK &&
                page_index + 1 < frontier->page_count &&
                page_header.next_block ==
                    EVOKE_ACTIVE_L0_NO_NEXT_BLOCK)
            {
                result = EVOKE_ERR_FORMAT;
            }
            if (result == EVOKE_OK)
            {
                current_block = page_header.next_block;
            }
            UnlockReleaseBuffer(buffer);
            buffer = InvalidBuffer;
            if (result != EVOKE_OK)
            {
                break;
            }
        }
    }
    PG_FINALLY();
    {
        if (BufferIsValid(buffer))
        {
            UnlockReleaseBuffer(buffer);
        }
        pfree(visited_blocks);
    }
    PG_END_TRY();
    return result;
}

void
evoke_segment_reachability_inventory_init(
    evoke_segment_reachability_inventory *inventory
)
{
    if (inventory != NULL)
    {
        memset(inventory, 0, sizeof(*inventory));
        evoke_block_range_inventory_init(&inventory->blocks);
    }
}

void
evoke_segment_reachability_inventory_free(
    evoke_segment_reachability_inventory *inventory
)
{
    if (inventory == NULL)
    {
        return;
    }
    evoke_block_range_inventory_free(&inventory->blocks);
    memset(inventory, 0, sizeof(*inventory));
}

void
evoke_segment_pages_inventory_reachable(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_reachability_inventory *inventory_out
)
{
    evoke_segment_reachability_inventory inventory;
    evoke_segment_reachability_context context;
    evoke_term_cow_page_loader_context term_loader;
    evoke_document_cow_page_loader_context document_loader;
    evoke_lexicon_cow_page_loader_context lexicon_loader;
    evoke_prefix_cow_page_loader_context prefix_loader;
    evoke_semantic_accelerator_directory accelerator_directory;
    BlockNumber physical_blocks;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        inventory_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke reachability inventory")));
    }
    physical_blocks = RelationGetNumberOfBlocks(index_relation);
    if (physical_blocks < root->published_block_high_watermark)
    {
        ereport(ERROR, (errmsg("evoke segment relation is truncated")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status == EVOKE_OK)
    {
        status = evoke_segment_manifest_validate_published(
            manifest,
            &root->manifest,
            root->published_block_high_watermark
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "reachability root",
            status
        );
    }

    evoke_segment_reachability_inventory_init(&inventory);
    evoke_semantic_accelerator_directory_init(&accelerator_directory);
    context.blocks = &inventory.blocks;
    context.published_block_high_watermark =
        root->published_block_high_watermark;
    PG_TRY();
    {
        status = evoke_block_range_inventory_add(
            &inventory.blocks,
            0,
            1
        );
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_inventory_add_object_ref(
                &context,
                &root->manifest
            );
        }
        if (status == EVOKE_OK)
        {
            const evoke_segment_object_ref *refs[8] = {
                &manifest->query_contract,
                &manifest->term_directory,
                &manifest->neutral_fold,
                &manifest->impact_fold,
                &manifest->document_directory,
                &manifest->lexicon_lookup,
                &manifest->prefix_lookup,
                &manifest->semantic_accelerator_directory
            };

            for (size_t ref_index = 0; ref_index < 8; ref_index++)
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    refs[ref_index]
                );
                if (status != EVOKE_OK)
                {
                    break;
                }
            }
        }
        if (status == EVOKE_OK &&
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0)
        {
            evoke_segment_pages_load_semantic_accelerator_directory(
                index_relation,
                root,
                manifest,
                &accelerator_directory
            );
            for (uint32 term_index = 0;
                 status == EVOKE_OK &&
                 term_index < accelerator_directory.term_count;
                 term_index++)
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &accelerator_directory.terms[term_index].term_object
                );
            }
            for (uint32 forward_index = 0;
                 status == EVOKE_OK &&
                 forward_index <
                    accelerator_directory.forward_chunk_count;
                 forward_index++)
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &accelerator_directory.forward_chunks[
                        forward_index
                    ].forward_object
                );
            }
            for (uint32 bound_index = 0;
                 status == EVOKE_OK &&
                 bound_index <
                    accelerator_directory.forward_bound_shard_count;
                 bound_index++)
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &accelerator_directory.forward_bound_shards[bound_index]
                );
            }
            if (status == EVOKE_OK &&
                evoke_semantic_accelerator_directory_has_scope(
                    &accelerator_directory))
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &accelerator_directory.scope_object
                );
            }
            if (status == EVOKE_OK &&
                evoke_semantic_accelerator_directory_has_tid_lookup(
                    &accelerator_directory))
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &accelerator_directory.tid_lookup_object
                );
            }
        }
        for (uint32 segment_index = 0;
             status == EVOKE_OK &&
             segment_index < manifest->segment_count;
             segment_index++)
        {
            evoke_segment_object_ref payload_ref;

            if ((segment_index & 255) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            status = evoke_segment_descriptor_payload_ref(
                manifest,
                &manifest->segments[segment_index],
                &payload_ref
            );
            if (status == EVOKE_OK)
            {
                status = evoke_segment_pages_inventory_add_object_ref(
                    &context,
                    &payload_ref
                );
            }
        }
        if (status == EVOKE_OK &&
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0)
        {
            status = evoke_segment_pages_open_term_cow(
                index_relation,
                &manifest->term_directory,
                root->published_block_high_watermark,
                manifest->manifest_id,
                &term_loader
            );
            if (status == EVOKE_OK)
            {
                CHECK_FOR_INTERRUPTS();
                status = evoke_term_cow_visit_external(
                    &term_loader.root_object.ref,
                    manifest->vocab_size,
                    evoke_segment_pages_load_term_cow_object,
                    &term_loader,
                    evoke_segment_pages_inventory_term_object,
                    &context
                );
            }
        }
        if (status == EVOKE_OK && manifest->document_slot_count > 0)
        {
            status = evoke_segment_pages_open_document_cow(
                index_relation,
                &manifest->document_directory,
                root->published_block_high_watermark,
                manifest->manifest_id,
                &document_loader
            );
            if (status == EVOKE_OK)
            {
                CHECK_FOR_INTERRUPTS();
                status = evoke_document_cow_visit_external(
                    &document_loader.root_object.ref,
                    manifest->document_slot_count,
                    evoke_segment_pages_load_document_cow_object,
                    &document_loader,
                    evoke_segment_pages_inventory_document_object,
                    &context
                );
            }
        }
        if (status == EVOKE_OK &&
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) != 0)
        {
            status = evoke_segment_pages_open_lexicon_cow(
                index_relation,
                &manifest->lexicon_lookup,
                root->published_block_high_watermark,
                manifest->manifest_id,
                &lexicon_loader
            );
            if (status == EVOKE_OK)
            {
                CHECK_FOR_INTERRUPTS();
                status = evoke_lexicon_cow_visit_external(
                    &lexicon_loader.root_object.ref,
                    manifest->lexicon_hash_seed,
                    manifest->vocab_size,
                    evoke_segment_pages_load_lexicon_cow_object,
                    &lexicon_loader,
                    evoke_segment_pages_inventory_lexicon_object,
                    &context
                );
            }
        }
        if (status == EVOKE_OK &&
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) != 0)
        {
            status = evoke_segment_pages_open_prefix_cow(
                index_relation,
                &manifest->prefix_lookup,
                root->published_block_high_watermark,
                manifest->manifest_id,
                &prefix_loader
            );
            if (status == EVOKE_OK)
            {
                CHECK_FOR_INTERRUPTS();
                status = evoke_prefix_cow_visit_external(
                    &prefix_loader.root_object.ref,
                    manifest->vocab_size,
                    evoke_segment_pages_load_prefix_cow_object,
                    &prefix_loader,
                    evoke_segment_pages_inventory_prefix_object,
                    &context
                );
            }
        }
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_inventory_l0_frontier(
                index_relation,
                root,
                &root->pending_l0,
                &inventory.blocks
            );
        }
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_inventory_l0_frontier(
                index_relation,
                root,
                &root->active_l0,
                &inventory.blocks
            );
        }
        if (status == EVOKE_OK)
        {
            status = evoke_block_range_inventory_finalize(
                &inventory.blocks,
                root->published_block_high_watermark
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "reachability closure",
                status
            );
        }
        inventory.physical_block_count = physical_blocks;
        inventory.trailing_unpublished_block_count =
            physical_blocks - root->published_block_high_watermark;
        *inventory_out = inventory;
        memset(&inventory, 0, sizeof(inventory));
    }
    PG_FINALLY();
    {
        evoke_semantic_accelerator_directory_free(&accelerator_directory);
        evoke_segment_reachability_inventory_free(&inventory);
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_load_sealed_manifest_internal(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_manifest *manifest_out,
    bool validate_term_closure
)
{
    uint8 *bytes;
    Size size;
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke sealed manifest load")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    bytes = evoke_segment_pages_read(
        index_relation,
        &root->manifest,
        &size
    );
    status = evoke_segment_manifest_deserialize(
        bytes,
        size,
        manifest_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("manifest", status);
    }
    status = evoke_segment_manifest_validate_published(
        manifest_out,
        &root->manifest,
        root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "published manifest closure",
            status
        );
    }
    if (validate_term_closure)
    {
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_validate_term_cow_closure(
            index_relation,
            root,
            manifest_out
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_validate_document_cow_closure(
            index_relation,
            root,
            manifest_out
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_validate_lexicon_cow_closure(
            index_relation,
            root,
            manifest_out
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_validate_prefix_cow_closure(
            index_relation,
            root,
            manifest_out
        );
    }
}

void
evoke_segment_pages_load_sealed_manifest(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_manifest *manifest_out
)
{
    evoke_segment_pages_load_sealed_manifest_internal(
        index_relation,
        root,
        manifest_out,
        true
    );
}

void
evoke_segment_pages_load_maintenance_manifest(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_manifest *manifest_out
)
{
    evoke_segment_pages_load_sealed_manifest_internal(
        index_relation,
        root,
        manifest_out,
        false
    );
}

void
evoke_segment_pages_load_query_contract(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract
)
{
    uint8 *bytes;
    Size size;
    evoke_status status;

    bytes = evoke_segment_pages_read(
        index_relation,
        &manifest->query_contract,
        &size
    );
    status = evoke_segment_query_contract_deserialize(
        bytes,
        size,
        manifest,
        contract
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "query contract",
            status
        );
    }
}

void
evoke_segment_pages_load_query_context(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_query_context *context_out
)
{
    evoke_segment_query_context context;
    evoke_status status;

    if (index_relation == NULL || root == NULL || context_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke query context load")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }

    evoke_segment_query_context_init(&context);
    context.root = *root;
    PG_TRY();
    {
        evoke_segment_pages_load_maintenance_manifest(
            index_relation,
            root,
            &context.manifest
        );
        if ((context.manifest.flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
            (context.manifest.flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0 ||
            context.manifest.doc_frequencies != NULL)
        {
            ereport(
                ERROR,
                (errmsg(
                    "evoke foreground query requires COW term storage"
                ))
            );
        }
        if ((context.manifest.document_slot_count > 0) !=
            ((context.manifest.flags &
              EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY) != 0))
        {
            ereport(
                ERROR,
                (errmsg(
                    "evoke foreground query requires COW document storage"
                ))
            );
        }
        evoke_segment_pages_load_query_contract(
            index_relation,
            &context.manifest,
            &context.query_contract
        );
        if (context.query_contract.vocab != NULL)
        {
            ereport(
                ERROR,
                (errmsg(
                    "evoke foreground query materialized a vocabulary"
                ))
            );
        }
        context.semantic_accelerator_compatible =
            evoke_segment_pages_semantic_accelerator_compatible(
                index_relation,
                root,
                &context.manifest,
                NULL
            );
        context.page_validation_cache = palloc0(
            sizeof(*context.page_validation_cache)
        );
        evoke_segment_query_context_free(context_out);
        *context_out = context;
        evoke_segment_query_context_init(&context);
    }
    PG_FINALLY();
    {
        evoke_segment_query_context_free(&context);
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_validate_block_contract(
    const evoke_segment_storage_snapshot *snapshot
)
{
    uint32 block_shift = snapshot->query_contract.block_shift;

    if (block_shift == 0 || block_shift >= 32)
    {
        ereport(ERROR, (errmsg("invalid evoke posting block contract")));
    }
    for (uint32 segment_index = 0;
         segment_index < snapshot->manifest.segment_count;
         segment_index++)
    {
        if (snapshot->payloads[segment_index].block_shift !=
            block_shift)
        {
            ereport(
                ERROR,
                (errmsg("evoke segment posting block contract mismatch"))
            );
        }
    }
    for (uint32 bundle_index = 0;
         bundle_index < snapshot->term_fold_bundle_count;
         bundle_index++)
    {
        if (snapshot->term_fold_bundles[bundle_index].block_shift !=
            block_shift)
        {
            ereport(
                ERROR,
                (errmsg("evoke fold posting block contract mismatch"))
            );
        }
    }
}

static void
evoke_segment_pages_initialize_empty_directory(
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory
)
{
    if (manifest->segment_count != 0 ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0)
    {
        ereport(ERROR, (errmsg("invalid empty evoke term directory")));
    }
    directory->vocab_size = manifest->vocab_size;
    directory->term_offsets = calloc(
        (size_t) manifest->vocab_size + 1,
        sizeof(*directory->term_offsets)
    );
    if (directory->term_offsets == NULL)
    {
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("out of memory")));
    }
}

void
evoke_segment_pages_load_term_directory(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory
)
{
    uint8 *bytes;
    Size size;
    evoke_status status;

    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0)
    {
        ereport(
            ERROR,
            (errmsg(
                "COW evoke term directory requires root-aware loading"
            ))
        );
    }
    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) == 0)
    {
        evoke_segment_pages_initialize_empty_directory(
            manifest,
            directory
        );
        return;
    }
    bytes = evoke_segment_pages_read(
        index_relation,
        &manifest->term_directory,
        &size
    );
    status = evoke_term_directory_deserialize(
        bytes,
        size,
        manifest,
        directory
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "term directory",
            status
        );
    }
}

void
evoke_segment_pages_load_term_accelerator(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory,
    uint32 **doc_frequencies_out,
    evoke_segment_object_ref **lexical_catalog_refs_out,
    evoke_term_cow_fold_state **fold_states_out
)
{
    evoke_term_cow_page_loader_context loader;
    evoke_status status;

    if (doc_frequencies_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke term accelerator load")));
    }
    *doc_frequencies_out = NULL;
    if (lexical_catalog_refs_out != NULL)
    {
        *lexical_catalog_refs_out = NULL;
    }
    if (fold_states_out != NULL)
    {
        *fold_states_out = NULL;
    }
    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0)
    {
        evoke_segment_pages_load_term_directory(
            index_relation,
            manifest,
            directory
        );
        return;
    }

    status = evoke_segment_pages_open_term_cow(
        index_relation,
        &manifest->term_directory,
        root->published_block_high_watermark,
        manifest->manifest_id,
        &loader
    );
    if (status == EVOKE_OK)
    {
        status = evoke_term_cow_materialize_external(
            &loader.root_object.ref,
            manifest,
            evoke_segment_pages_load_term_cow_object,
            &loader,
            directory,
            doc_frequencies_out,
            lexical_catalog_refs_out,
            fold_states_out
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW term accelerator",
            status
        );
    }
}

static void
evoke_segment_pages_attach_query_vocabulary(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *catalog_refs,
    evoke_segment_query_contract *contract
)
{
    bool has_vocabulary;
    uint32 term_id = 0;

    if (index_relation == NULL || manifest == NULL || contract == NULL ||
        contract->vocab_size != manifest->vocab_size)
    {
        ereport(ERROR, (errmsg("invalid evoke lexical catalog attach")));
    }
    has_vocabulary =
        (contract->flags & EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) != 0;
    if (!has_vocabulary)
    {
        if (catalog_refs != NULL)
        {
            for (term_id = 0;
                 term_id < manifest->vocab_size;
                 term_id++)
            {
                if (!evoke_segment_object_ref_is_zero(
                        &catalog_refs[term_id]))
                {
                    ereport(
                        ERROR,
                        (errmsg("unexpected evoke lexical catalog"))
                    );
                }
            }
        }
        return;
    }
    if (catalog_refs == NULL || contract->vocab != NULL)
    {
        ereport(ERROR, (errmsg("missing evoke lexical catalog")));
    }
    contract->vocab = calloc(
        contract->vocab_size,
        sizeof(*contract->vocab)
    );
    if (contract->vocab == NULL)
    {
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("out of memory")));
    }
    while (term_id < contract->vocab_size)
    {
        const evoke_segment_object_ref *ref = &catalog_refs[term_id];
        evoke_lexical_catalog catalog;
        uint8 *bytes;
        Size size;
        evoke_status status;

        if (evoke_segment_object_ref_is_zero(ref) ||
            ref->object_kind != EVOKE_SEGMENT_OBJECT_LEXICAL_CATALOG ||
            ref->owner_manifest_id > manifest->manifest_id)
        {
            ereport(ERROR, (errmsg("invalid evoke lexical catalog ref")));
        }
        bytes = evoke_segment_pages_read(index_relation, ref, &size);
        evoke_lexical_catalog_init(&catalog);
        status = evoke_lexical_catalog_deserialize(
            bytes,
            size,
            ref->owner_manifest_id,
            &catalog
        );
        pfree(bytes);
        if (status == EVOKE_OK &&
            (catalog.first_term_id != term_id ||
             catalog.term_count >
                contract->vocab_size - term_id))
        {
            status = EVOKE_ERR_FORMAT;
        }
        if (status == EVOKE_OK)
        {
            for (uint32 catalog_index = 0;
                 catalog_index < catalog.term_count;
                 catalog_index++)
            {
                uint32 catalog_term_id = term_id + catalog_index;

                if (!evoke_segment_object_refs_equal(
                        ref,
                        &catalog_refs[catalog_term_id]))
                {
                    status = EVOKE_ERR_FORMAT;
                    break;
                }
                contract->vocab[catalog_term_id] =
                    catalog.terms[catalog_index];
                catalog.terms[catalog_index] = NULL;
            }
        }
        if (status != EVOKE_OK)
        {
            evoke_lexical_catalog_free(&catalog);
            evoke_segment_pages_report_codec_error(
                "lexical catalog",
                status
            );
        }
        term_id += catalog.term_count;
        evoke_lexical_catalog_free(&catalog);
    }
}

void
evoke_segment_pages_load_query_vocabulary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract
)
{
    evoke_term_directory directory;
    uint32 *doc_frequencies = NULL;
    evoke_segment_object_ref *catalog_refs = NULL;

    evoke_term_directory_init(&directory);
    PG_TRY();
    {
        evoke_segment_pages_load_term_accelerator(
            index_relation,
            root,
            manifest,
            &directory,
            &doc_frequencies,
            &catalog_refs,
            NULL
        );
        evoke_segment_pages_attach_query_vocabulary(
            index_relation,
            manifest,
            catalog_refs,
            contract
        );
    }
    PG_FINALLY();
    {
        free(catalog_refs);
        free(doc_frequencies);
        evoke_term_directory_free(&directory);
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_validate_optional_object(
    Relation index_relation,
    uint32 manifest_flags,
    uint32 required_flag,
    const evoke_segment_object_ref *ref
)
{
    uint8 *bytes;
    Size size;

    if ((manifest_flags & required_flag) == 0)
    {
        return;
    }
    bytes = evoke_segment_pages_read(index_relation, ref, &size);
    pfree(bytes);
}

void
evoke_segment_pages_free_payloads(
    evoke_segment_payload *payloads,
    uint32 payload_count
)
{
    uint32 payload_index;

    if (payloads == NULL)
    {
        return;
    }
    for (payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        evoke_segment_payload_free(&payloads[payload_index]);
    }
    pfree(payloads);
}

void
evoke_segment_pages_load_payload_range(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    uint32 first_segment_index,
    uint32 segment_count,
    evoke_segment_payload **payloads_out
)
{
    evoke_segment_payload *payloads = NULL;
    evoke_status status;
    uint32 payload_index;

    if (index_relation == NULL || manifest == NULL ||
        payloads_out == NULL ||
        first_segment_index > manifest->segment_count ||
        segment_count >
            manifest->segment_count - first_segment_index)
    {
        ereport(ERROR, (errmsg("invalid evoke payload range load")));
    }
    *payloads_out = NULL;
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "payload range manifest",
            status
        );
    }
    if (segment_count == 0)
    {
        return;
    }
    payloads = palloc0(
        (Size) segment_count * sizeof(*payloads)
    );
    for (payload_index = 0;
         payload_index < segment_count;
         payload_index++)
    {
        evoke_segment_payload_init(&payloads[payload_index]);
    }

    PG_TRY();
    {
        for (payload_index = 0;
             payload_index < segment_count;
             payload_index++)
        {
            uint32 segment_index =
                first_segment_index + payload_index;
            const evoke_segment_descriptor *descriptor =
                &manifest->segments[segment_index];
            evoke_segment_object_ref ref;
            uint8 *bytes;
            Size size;

            status = evoke_segment_descriptor_payload_ref(
                manifest,
                descriptor,
                &ref
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "segment payload reference",
                    status
                );
            }
            bytes = evoke_segment_pages_read(
                index_relation,
                &ref,
                &size
            );
            status = evoke_segment_payload_deserialize(
                bytes,
                size,
                manifest,
                descriptor,
                &payloads[payload_index]
            );
            pfree(bytes);
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "segment payload",
                    status
                );
            }
        }
        *payloads_out = payloads;
    }
    PG_CATCH();
    {
        evoke_segment_pages_free_payloads(
            payloads,
            segment_count
        );
        PG_RE_THROW();
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_load_payloads(
    Relation index_relation,
    evoke_segment_storage_snapshot *snapshot
)
{
    uint32 segment_index;

    if (snapshot->manifest.segment_count == 0)
    {
        return;
    }
    evoke_segment_pages_load_payload_range(
        index_relation,
        &snapshot->manifest,
        0,
        snapshot->manifest.segment_count,
        &snapshot->payloads
    );
    snapshot->payload_views = palloc0(
        (Size) snapshot->manifest.segment_count *
        sizeof(*snapshot->payload_views)
    );

    for (segment_index = 0;
         segment_index < snapshot->manifest.segment_count;
         segment_index++)
    {
        evoke_segment_payload_as_view(
            &snapshot->payloads[segment_index],
            &snapshot->payload_views[segment_index]
        );
    }
}

typedef struct evoke_segment_fold_ref_entry
{
    evoke_segment_object_ref ref;
    uint32 term_id;
    bool minor;
} evoke_segment_fold_ref_entry;

static int
evoke_segment_compare_fold_ref_entries(
    const void *left_pointer,
    const void *right_pointer
)
{
    const evoke_segment_fold_ref_entry *left = left_pointer;
    const evoke_segment_fold_ref_entry *right = right_pointer;

#define EVOKE_COMPARE_FOLD_REF_FIELD(field_name)                         \
    do                                                                 \
    {                                                                  \
        if (left->ref.field_name != right->ref.field_name)             \
        {                                                              \
            return left->ref.field_name < right->ref.field_name        \
                ? -1                                                   \
                : 1;                                                   \
        }                                                              \
    } while (0)

    EVOKE_COMPARE_FOLD_REF_FIELD(object_kind);
    EVOKE_COMPARE_FOLD_REF_FIELD(start_block);
    EVOKE_COMPARE_FOLD_REF_FIELD(page_count);
    EVOKE_COMPARE_FOLD_REF_FIELD(object_id);
    EVOKE_COMPARE_FOLD_REF_FIELD(owner_manifest_id);
    EVOKE_COMPARE_FOLD_REF_FIELD(object_bytes);
    EVOKE_COMPARE_FOLD_REF_FIELD(object_checksum);
#undef EVOKE_COMPARE_FOLD_REF_FIELD

    if (left->term_id != right->term_id)
    {
        return left->term_id < right->term_id ? -1 : 1;
    }
    if (left->minor != right->minor)
    {
        return left->minor ? 1 : -1;
    }
    return 0;
}

static evoke_status
evoke_segment_pages_read_fold_bundle(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_segment_object_kind expected_kind,
    evoke_term_fold_bundle *bundle_out
)
{
    uint8 *bytes;
    Size size = 0;
    evoke_status status;

    if (ref->object_kind != expected_kind ||
        ref->owner_manifest_id > manifest->manifest_id ||
        evoke_segment_object_ref_validate(
            ref,
            root->published_block_high_watermark
        ) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    bytes = evoke_segment_pages_read(index_relation, ref, &size);
    status = evoke_term_fold_bundle_deserialize(
        bytes,
        size,
        bundle_out
    );
    pfree(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (bundle_out->object_kind != expected_kind ||
        bundle_out->owner_manifest_id != ref->owner_manifest_id)
    {
        evoke_term_fold_bundle_free(bundle_out);
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

void
evoke_segment_pages_load_neutral_fold_bundle(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_term_fold_bundle *bundle_out
)
{
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        ref == NULL || bundle_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke neutral fold load")));
    }
    status = evoke_segment_pages_read_fold_bundle(
        index_relation,
        root,
        manifest,
        ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        bundle_out
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "neutral term fold bundle",
            status
        );
    }
}

void
evoke_segment_pages_load_impact_fold_bundle(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_term_fold_bundle *bundle_out
)
{
    evoke_status status;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        ref == NULL || bundle_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke impact fold load")));
    }
    status = evoke_segment_pages_read_fold_bundle(
        index_relation,
        root,
        manifest,
        ref,
        EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
        bundle_out
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "impact term fold bundle",
            status
        );
    }
}

static evoke_status
evoke_segment_collect_fold_entries(
    const evoke_term_cow_fold_state *states,
    uint32 vocab_size,
    bool impact,
    evoke_segment_fold_ref_entry **entries_out,
    uint32 *entry_count_out
)
{
    evoke_segment_fold_ref_entry *entries = NULL;
    uint32 entry_count = 0;
    uint32 entry_capacity;

    if (states == NULL || entries_out == NULL ||
        entry_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *entries_out = NULL;
    *entry_count_out = 0;
    for (uint32 term_id = 0; term_id < vocab_size; term_id++)
    {
        uint64 coverage = impact
            ? states[term_id].impact_coverage
            : states[term_id].neutral_coverage;

        if (coverage != 0)
        {
            entry_count++;
        }
        if (!impact && states[term_id].neutral_minor_coverage != 0)
        {
            entry_count++;
        }
    }
    if (entry_count == 0)
    {
        return EVOKE_OK;
    }
    entry_capacity = entry_count;
    entries = calloc(entry_capacity, sizeof(*entries));
    if (entries == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    entry_count = 0;
    for (uint32 term_id = 0; term_id < vocab_size; term_id++)
    {
        uint64 coverage = impact
            ? states[term_id].impact_coverage
            : states[term_id].neutral_coverage;

        if (coverage != 0)
        {
            entries[entry_count].ref = impact
                ? states[term_id].impact_ref
                : states[term_id].neutral_ref;
            entries[entry_count].term_id = term_id;
            entries[entry_count].minor = false;
            entry_count++;
        }
        if (!impact && states[term_id].neutral_minor_coverage != 0)
        {
            entries[entry_count].ref =
                states[term_id].neutral_minor_ref;
            entries[entry_count].term_id = term_id;
            entries[entry_count].minor = true;
            entry_count++;
        }
    }
    if (entry_count != entry_capacity)
    {
        free(entries);
        return EVOKE_ERR_FORMAT;
    }
    qsort(
        entries,
        entry_count,
        sizeof(*entries),
        evoke_segment_compare_fold_ref_entries
    );
    *entries_out = entries;
    *entry_count_out = entry_count;
    return EVOKE_OK;
}

static uint32
evoke_segment_fold_group_count(
    const evoke_segment_fold_ref_entry *entries,
    uint32 entry_count
)
{
    uint32 group_count = 0;

    for (uint32 entry_index = 0;
         entry_index < entry_count;
         entry_index++)
    {
        if (entry_index == 0 ||
            !evoke_segment_object_refs_equal(
                &entries[entry_index - 1].ref,
                &entries[entry_index].ref
            ))
        {
            group_count++;
        }
    }
    return group_count;
}

static bool
evoke_segment_impact_fold_state_eligible(
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_term_cow_fold_state *state
)
{
    uint64 effective_coverage;

    if (root->active_l0.record_count != 0 ||
        root->pending_l0.record_count != 0)
    {
        return false;
    }
    effective_coverage = state->neutral_minor_coverage != 0
        ? state->neutral_minor_coverage
        : state->neutral_coverage;
    return effective_coverage != 0 &&
        state->impact_coverage == effective_coverage &&
        state->impact_statistics_epoch == manifest->statistics_epoch;
}

static uint32
evoke_segment_eligible_impact_group_count(
    const evoke_segment_fold_ref_entry *entries,
    uint32 entry_count,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_term_cow_fold_state *states
)
{
    uint32 group_count = 0;
    uint32 entry_index = 0;

    while (entry_index < entry_count)
    {
        uint32 group_end = entry_index + 1;
        bool eligible = true;

        while (group_end < entry_count &&
               evoke_segment_object_refs_equal(
                    &entries[entry_index].ref,
                    &entries[group_end].ref
               ))
        {
            group_end++;
        }
        for (uint32 group_index = entry_index;
             group_index < group_end;
             group_index++)
        {
            if (!evoke_segment_impact_fold_state_eligible(
                    root,
                    manifest,
                    &states[entries[group_index].term_id]
                ))
            {
                eligible = false;
                break;
            }
        }
        if (eligible)
        {
            group_count++;
        }
        entry_index = group_end;
    }
    return group_count;
}

static void
evoke_segment_pages_load_term_folds(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_storage_snapshot *snapshot
)
{
    evoke_segment_fold_ref_entry *neutral_entries = NULL;
    evoke_segment_fold_ref_entry *impact_entries = NULL;
    uint32 neutral_entry_count = 0;
    uint32 impact_entry_count = 0;
    uint32 entry_index = 0;
    uint32 neutral_bundle_count;
    uint32 impact_bundle_count;
    uint32 bundle_count = 0;
    uint32 bundle_index = 0;
    uint32 failure_term_id = UINT32_MAX;
    uint32 failure_run_index = UINT32_MAX;
    const char *failure_stage = "none";
    evoke_segment_object_ref failure_ref = {0};
    bool failure_ref_present = false;
    evoke_status status = EVOKE_OK;

    if (snapshot->term_fold_states == NULL)
    {
        return;
    }
    status = evoke_segment_collect_fold_entries(
        snapshot->term_fold_states,
        snapshot->manifest.vocab_size,
        false,
        &neutral_entries,
        &neutral_entry_count
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "neutral fold catalog",
            status
        );
    }
    status = evoke_segment_collect_fold_entries(
        snapshot->term_fold_states,
        snapshot->manifest.vocab_size,
        true,
        &impact_entries,
        &impact_entry_count
    );
    if (status != EVOKE_OK)
    {
        free(neutral_entries);
        evoke_segment_pages_report_codec_error(
            "impact fold catalog",
            status
        );
    }
    neutral_bundle_count = evoke_segment_fold_group_count(
        neutral_entries,
        neutral_entry_count
    );
    impact_bundle_count =
        evoke_segment_eligible_impact_group_count(
            impact_entries,
            impact_entry_count,
            root,
            &snapshot->manifest,
            snapshot->term_fold_states
        );
    bundle_count = neutral_bundle_count + impact_bundle_count;
    if (bundle_count > 0)
    {
        snapshot->term_fold_bundles = calloc(
            bundle_count,
            sizeof(*snapshot->term_fold_bundles)
        );
        snapshot->term_fold_plans = calloc(
            snapshot->manifest.vocab_size,
            sizeof(*snapshot->term_fold_plans)
        );
        if (snapshot->term_fold_bundles == NULL ||
            snapshot->term_fold_plans == NULL)
        {
            free(impact_entries);
            free(neutral_entries);
            evoke_segment_pages_report_codec_error(
                "term fold attachment",
                EVOKE_ERR_NOMEM
            );
        }
        snapshot->term_fold_bundle_count = bundle_count;
    }

    while (entry_index < neutral_entry_count)
    {
        uint32 group_end = entry_index + 1;
        uint32 run_index = 0;
        evoke_term_fold_bundle *bundle =
            &snapshot->term_fold_bundles[bundle_index];

        while (group_end < neutral_entry_count &&
               evoke_segment_object_refs_equal(
                    &neutral_entries[entry_index].ref,
                    &neutral_entries[group_end].ref
               ))
        {
            group_end++;
        }
        status = evoke_segment_pages_read_fold_bundle(
            index_relation,
            root,
            &snapshot->manifest,
            &neutral_entries[entry_index].ref,
            EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
            bundle
        );
        if (status != EVOKE_OK)
        {
            failure_stage = "neutral_bundle_read";
            failure_term_id = neutral_entries[entry_index].term_id;
            failure_ref = neutral_entries[entry_index].ref;
            failure_ref_present = true;
            break;
        }
        for (uint32 group_index = entry_index;
             group_index < group_end && status == EVOKE_OK;
             group_index++)
        {
            uint32 term_id = neutral_entries[group_index].term_id;
            uint32 first_run = run_index;
            bool minor = neutral_entries[group_index].minor;
            uint64 coverage = minor
                ? snapshot->term_fold_states[
                    term_id
                  ].neutral_minor_coverage
                : snapshot->term_fold_states[
                    term_id
                  ].neutral_coverage;
            evoke_term_fold_read_plan *plan =
                &snapshot->term_fold_plans[term_id];

            while (run_index < bundle->run_count &&
                   bundle->runs[run_index].term_id < term_id)
            {
                run_index++;
            }
            first_run = run_index;
            while (run_index < bundle->run_count &&
                   bundle->runs[run_index].term_id == term_id)
            {
                if (bundle->runs[run_index].coverage_sequence !=
                    coverage)
                {
                    status = EVOKE_ERR_FORMAT;
                    break;
                }
                run_index++;
            }
            if (status != EVOKE_OK || run_index == first_run)
            {
                failure_stage = "neutral_run_mapping";
                failure_term_id = term_id;
                failure_run_index = run_index;
                status = EVOKE_ERR_FORMAT;
                break;
            }
            if (minor)
            {
                plan->neutral_minor_coverage = coverage;
                plan->neutral_minor_bundle_index = bundle_index;
                plan->neutral_minor_first_run = first_run;
                plan->neutral_minor_run_count =
                    run_index - first_run;
            }
            else
            {
                plan->neutral_coverage = coverage;
                plan->neutral_bundle_index = bundle_index;
                plan->neutral_first_run = first_run;
                plan->neutral_run_count = run_index - first_run;
            }
        }
        if (status != EVOKE_OK)
        {
            break;
        }
        bundle_index++;
        entry_index = group_end;
    }
    free(neutral_entries);
    entry_index = 0;
    while (entry_index < impact_entry_count && status == EVOKE_OK)
    {
        uint32 group_end = entry_index + 1;
        uint32 run_index = 0;
        bool eligible = true;
        evoke_term_fold_bundle bundle;

        while (group_end < impact_entry_count &&
               evoke_segment_object_refs_equal(
                    &impact_entries[entry_index].ref,
                    &impact_entries[group_end].ref
               ))
        {
            group_end++;
        }
        for (uint32 group_index = entry_index;
             group_index < group_end;
             group_index++)
        {
            if (!evoke_segment_impact_fold_state_eligible(
                    root,
                    &snapshot->manifest,
                    &snapshot->term_fold_states[
                        impact_entries[group_index].term_id
                    ]
                ))
            {
                eligible = false;
                break;
            }
        }
        evoke_term_fold_bundle_init(&bundle);
        status = evoke_segment_pages_read_fold_bundle(
            index_relation,
            root,
            &snapshot->manifest,
            &impact_entries[entry_index].ref,
            EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
            &bundle
        );
        if (status != EVOKE_OK)
        {
            failure_stage = "impact_bundle_read";
            failure_term_id = impact_entries[entry_index].term_id;
            failure_ref = impact_entries[entry_index].ref;
            failure_ref_present = true;
            evoke_term_fold_bundle_free(&bundle);
            break;
        }
        for (uint32 group_index = entry_index;
             group_index < group_end && status == EVOKE_OK;
             group_index++)
        {
            uint32 term_id = impact_entries[group_index].term_id;
            const evoke_term_cow_fold_state *state =
                &snapshot->term_fold_states[term_id];

            while (run_index < bundle.run_count &&
                   bundle.runs[run_index].term_id < term_id)
            {
                run_index++;
            }
            if (run_index >= bundle.run_count ||
                bundle.runs[run_index].term_id != term_id ||
                bundle.runs[run_index].kind !=
                    EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
                bundle.runs[run_index].coverage_sequence !=
                    state->impact_coverage ||
                bundle.statistics_epoch !=
                    state->impact_statistics_epoch)
            {
                failure_stage = "impact_run_mapping";
                failure_term_id = term_id;
                failure_run_index = run_index;
                status = EVOKE_ERR_FORMAT;
                break;
            }
            if (eligible)
            {
                evoke_term_fold_read_plan *plan =
                    &snapshot->term_fold_plans[term_id];

                plan->impact_coverage =
                    state->impact_coverage;
                plan->impact_statistics_epoch =
                    state->impact_statistics_epoch;
                plan->impact_bundle_index = bundle_index;
                plan->impact_first_run = run_index;
                plan->impact_run_count = 1;
            }
            run_index++;
        }
        if (status == EVOKE_OK && eligible)
        {
            snapshot->term_fold_bundles[bundle_index] = bundle;
            evoke_term_fold_bundle_init(&bundle);
            bundle_index++;
        }
        evoke_term_fold_bundle_free(&bundle);
        entry_index = group_end;
    }
    free(impact_entries);
    if (status == EVOKE_OK && bundle_index != bundle_count)
    {
        failure_stage = "bundle_count";
        status = EVOKE_ERR_FORMAT;
    }
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg(
                "invalid evoke term fold closure object: %s",
                evoke_strerror(status)
            ),
             errdetail(
                "stage=%s term_id=%u run_index=%u "
                "bundles=%u/%u neutral_entries=%u impact_entries=%u "
                "ref_kind=%u ref_start=%u ref_pages=%u ref_id=%llu "
                "ref_owner=%llu ref_bytes=%llu ref_checksum=%llu",
                failure_stage,
                failure_term_id,
                failure_run_index,
                bundle_index,
                bundle_count,
                neutral_entry_count,
                impact_entry_count,
                failure_ref_present ? failure_ref.object_kind : 0,
                failure_ref_present ? failure_ref.start_block : 0,
                failure_ref_present ? failure_ref.page_count : 0,
                (unsigned long long) (
                    failure_ref_present ? failure_ref.object_id : 0
                ),
                (unsigned long long) (
                    failure_ref_present
                        ? failure_ref.owner_manifest_id
                        : 0
                ),
                (unsigned long long) (
                    failure_ref_present ? failure_ref.object_bytes : 0
                ),
                (unsigned long long) (
                    failure_ref_present
                        ? failure_ref.object_checksum
                        : 0
                )
            ))
        );
    }
}

void
evoke_segment_pages_load_sealed_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_storage_snapshot *snapshot_out
)
{
    evoke_segment_storage_snapshot *snapshot;
    const uint32 *doc_frequencies;
    evoke_status status;

    if (index_relation == NULL || root == NULL || snapshot_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke sealed snapshot load")));
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }

    snapshot = palloc0(sizeof(*snapshot));
    evoke_segment_storage_snapshot_init(snapshot);
    snapshot->root = *root;

    PG_TRY();
    {
        evoke_segment_pages_load_sealed_manifest_internal(
            index_relation,
            root,
            &snapshot->manifest,
            false
        );
        evoke_segment_pages_load_query_contract(
            index_relation,
            &snapshot->manifest,
            &snapshot->query_contract
        );
        evoke_segment_pages_load_document_block_extrema_all(
            index_relation,
            root,
            &snapshot->manifest,
            &snapshot->query_contract,
            &snapshot->document_block_extrema,
            &snapshot->document_block_count
        );
        evoke_segment_pages_load_term_accelerator(
            index_relation,
            root,
            &snapshot->manifest,
            &snapshot->term_directory,
            &snapshot->materialized_doc_frequencies,
            &snapshot->lexical_catalog_refs,
            &snapshot->term_fold_states
        );
        evoke_segment_pages_attach_query_vocabulary(
            index_relation,
            &snapshot->manifest,
            snapshot->lexical_catalog_refs,
            &snapshot->query_contract
        );
        free(snapshot->lexical_catalog_refs);
        snapshot->lexical_catalog_refs = NULL;
        doc_frequencies =
            snapshot->materialized_doc_frequencies != NULL
                ? snapshot->materialized_doc_frequencies
                : snapshot->manifest.doc_frequencies;
        evoke_segment_pages_validate_optional_object(
            index_relation,
            snapshot->manifest.flags,
            EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD,
            &snapshot->manifest.neutral_fold
        );
        evoke_segment_pages_validate_optional_object(
            index_relation,
            snapshot->manifest.flags,
            EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD,
            &snapshot->manifest.impact_fold
        );
        evoke_segment_pages_load_term_folds(
            index_relation,
            root,
            snapshot
        );
        evoke_segment_pages_load_payloads(index_relation, snapshot);
        evoke_segment_pages_validate_block_contract(snapshot);
        status = evoke_segment_index_metadata_build_base(
            &snapshot->query_contract,
            &snapshot->manifest,
            doc_frequencies,
            &snapshot->index_metadata
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "index metadata",
                status
            );
        }
        evoke_segment_pages_build_document_metadata(
            index_relation,
            root,
            snapshot
        );
        if (snapshot->term_fold_plans == NULL)
        {
            status = evoke_segment_read_view_build(
                &snapshot->index_metadata,
                &snapshot->manifest,
                &snapshot->term_directory,
                snapshot->payload_views,
                snapshot->manifest.segment_count,
                &snapshot->read_view
            );
        }
        else
        {
            status = evoke_segment_read_view_build_folded(
                &snapshot->index_metadata,
                &snapshot->manifest,
                &snapshot->term_directory,
                snapshot->payload_views,
                snapshot->manifest.segment_count,
                snapshot->term_fold_bundles,
                snapshot->term_fold_bundle_count,
                snapshot->term_fold_plans,
                snapshot->manifest.vocab_size,
                &snapshot->read_view
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "sealed read view",
                status
            );
        }
        snapshot->corpus_stats.document_count =
            snapshot->manifest.visible_document_count;
        snapshot->corpus_stats.total_document_length =
            snapshot->manifest.total_document_length;
        snapshot->corpus_stats.doc_frequencies =
            doc_frequencies;
        snapshot->corpus_stats.vocab_size =
            snapshot->manifest.vocab_size;
        evoke_segment_pages_load_l0_snapshot(
            index_relation,
            root,
            &snapshot->l0_snapshot
        );

        evoke_segment_storage_snapshot_free(snapshot_out);
        *snapshot_out = *snapshot;
        pfree(snapshot);
    }
    PG_CATCH();
    {
        evoke_segment_storage_snapshot_free(snapshot);
        pfree(snapshot);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

typedef struct evoke_segment_bundle_write_cleanup
{
    uint8 *serialized_bytes;
    evoke_segment_payload_view *payload_views;
    evoke_segment_object_ref *initial_fold_refs;
    evoke_term_directory term_directory;
    evoke_term_cow_tree term_cow_tree;
    evoke_document_cow_tree document_cow_tree;
    evoke_lexicon_cow_tree lexicon_cow_tree;
    evoke_prefix_cow_tree prefix_cow_tree;
} evoke_segment_bundle_write_cleanup;

static bool
evoke_segment_object_ref_is_zero(
    const evoke_segment_object_ref *ref
)
{
    static const evoke_segment_object_ref zero_ref = {0};

    return memcmp(ref, &zero_ref, sizeof(*ref)) == 0;
}

static bool
evoke_segment_object_refs_equal(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
)
{
    return left != NULL &&
        right != NULL &&
        left->object_kind == right->object_kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->object_checksum == right->object_checksum;
}

static void
evoke_segment_bundle_write_cleanup_free(
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    if (cleanup == NULL)
    {
        return;
    }
    free(cleanup->serialized_bytes);
    if (cleanup->payload_views != NULL)
    {
        pfree(cleanup->payload_views);
    }
    free(cleanup->initial_fold_refs);
    evoke_term_directory_free(&cleanup->term_directory);
    evoke_term_cow_tree_free(&cleanup->term_cow_tree);
    evoke_document_cow_tree_free(&cleanup->document_cow_tree);
    evoke_lexicon_cow_tree_free(&cleanup->lexicon_cow_tree);
    evoke_prefix_cow_tree_free(&cleanup->prefix_cow_tree);
    pfree(cleanup);
}

static void
evoke_segment_pages_require_unpublished_manifest(
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    size_t payload_count
)
{
    uint32 segment_index;

    if (manifest == NULL || manifest->manifest_id == 0 ||
        manifest->statistics_epoch == 0 ||
        manifest->segment_count > EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        (manifest->segment_count > 0 && manifest->segments == NULL) ||
        (manifest->vocab_size > 0 &&
         manifest->doc_frequencies == NULL) ||
        manifest->visible_document_count >
            manifest->document_slot_count ||
        manifest->document_slot_count > UINT32_MAX ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL) ||
        !evoke_segment_object_ref_is_zero(&manifest->query_contract) ||
        !evoke_segment_object_ref_is_zero(&manifest->term_directory) ||
        !evoke_segment_object_ref_is_zero(&manifest->neutral_fold) ||
        !evoke_segment_object_ref_is_zero(&manifest->impact_fold) ||
        !evoke_segment_object_ref_is_zero(
            &manifest->document_directory) ||
        !evoke_segment_object_ref_is_zero(
            &manifest->lexicon_lookup) ||
        !evoke_segment_object_ref_is_zero(
            &manifest->prefix_lookup) ||
        manifest->lexicon_hash_seed != 0 ||
        (manifest->flags &
         (EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY |
          EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD |
          EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD |
          EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY |
          EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP |
          EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP)) != 0)
    {
        ereport(
            ERROR,
            (errmsg("evoke segment manifest is already published"))
        );
    }
    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        const evoke_segment_descriptor *descriptor =
            &manifest->segments[segment_index];

        if (descriptor->start_block != 0 ||
            descriptor->block_count != 0 ||
            descriptor->payload_bytes != 0 ||
            descriptor->payload_checksum != 0 ||
            descriptor->payload_owner_manifest_id != 0)
        {
            ereport(
                ERROR,
                (errmsg(
                    "evoke segment descriptor is already published"
                ))
            );
        }
    }
}

static void
evoke_segment_pages_write_query_contract(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    size_t size = 0;
    evoke_status status;

    status = evoke_segment_query_contract_serialize(
        query_contract,
        manifest,
        &cleanup->serialized_bytes,
        &size
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "query contract",
            status
        );
    }
    evoke_segment_pages_write_internal(
        index_relation,
        fork_number,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        manifest->manifest_id,
        manifest->manifest_id,
        cleanup->serialized_bytes,
        size,
        reuse_arena,
        &manifest->query_contract
    );
    free(cleanup->serialized_bytes);
    cleanup->serialized_bytes = NULL;
}

static void
evoke_segment_pages_write_lexical_catalog(
    Relation index_relation,
    ForkNumber fork_number,
    uint64 owner_manifest_id,
    uint32 first_term_id,
    uint32 term_count,
    const char *const *terms,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *ref_out
)
{
    evoke_lexical_catalog catalog;
    size_t size = 0;
    evoke_status status;

    if (term_count == 0 || terms == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke lexical catalog write")));
    }
    evoke_lexical_catalog_init(&catalog);
    status = evoke_lexical_catalog_build(
        owner_manifest_id,
        first_term_id,
        term_count,
        terms,
        &catalog
    );
    if (status == EVOKE_OK)
    {
        status = evoke_lexical_catalog_serialize(
            &catalog,
            &cleanup->serialized_bytes,
            &size
        );
    }
    evoke_lexical_catalog_free(&catalog);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "lexical catalog",
            status
        );
    }
    evoke_segment_pages_write_internal(
        index_relation,
        fork_number,
        EVOKE_SEGMENT_OBJECT_LEXICAL_CATALOG,
        owner_manifest_id,
        owner_manifest_id,
        cleanup->serialized_bytes,
        size,
        reuse_arena,
        ref_out
    );
    free(cleanup->serialized_bytes);
    cleanup->serialized_bytes = NULL;
}

static void
evoke_segment_pages_write_payload(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    evoke_segment_descriptor *descriptor,
    const evoke_segment_payload *payload,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    evoke_segment_payload encoded_payload;
    evoke_segment_object_ref ref;
    size_t size = 0;
    uint64 checksum = 0;
    evoke_status status;

    encoded_payload = *payload;
    encoded_payload.semantic_impact_precision =
        evoke_am_get_semantic_impact_precision(index_relation);
    status = evoke_segment_payload_serialize(
        &encoded_payload,
        manifest,
        descriptor,
        &cleanup->serialized_bytes,
        &size,
        &checksum
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "segment payload",
            status
        );
    }
    evoke_segment_pages_write_internal(
        index_relation,
        fork_number,
        EVOKE_SEGMENT_OBJECT_PAYLOAD,
        descriptor->segment_id,
        manifest->manifest_id,
        cleanup->serialized_bytes,
        size,
        reuse_arena,
        &ref
    );
    if (ref.object_bytes != size ||
        ref.object_checksum != checksum)
    {
        ereport(
            ERROR,
            (errmsg("evoke segment payload identity mismatch"))
        );
    }
    descriptor->start_block = ref.start_block;
    descriptor->block_count = ref.page_count;
    descriptor->payload_bytes = ref.object_bytes;
    descriptor->payload_checksum = ref.object_checksum;
    descriptor->size_class =
        evoke_segment_size_class(ref.object_bytes);
    descriptor->payload_owner_manifest_id =
        ref.owner_manifest_id;
    free(cleanup->serialized_bytes);
    cleanup->serialized_bytes = NULL;
}

static void
evoke_segment_pages_write_payloads(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    uint32 segment_index;

    if (manifest->segment_count > 0)
    {
        cleanup->payload_views = palloc0(
            (Size) manifest->segment_count *
            sizeof(*cleanup->payload_views)
        );
    }
    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        evoke_segment_descriptor *descriptor =
            &manifest->segments[segment_index];

        evoke_segment_pages_write_payload(
            index_relation,
            fork_number,
            manifest,
            descriptor,
            &payloads[segment_index],
            cleanup,
            reuse_arena
        );
        evoke_segment_payload_as_view(
            &payloads[segment_index],
            &cleanup->payload_views[segment_index]
        );
    }
}

static evoke_status
evoke_segment_pages_initial_fold_group_end(
    const evoke_segment_payload *payload,
    uint32 first_run,
    uint32 *end_run_out
)
{
    Size group_size = EVOKE_TERM_FOLD_HEADER_SIZE;
    uint32 group_end = first_run;

    if (payload == NULL || end_run_out == NULL ||
        first_run >= payload->run_count || payload->indices == NULL ||
        payload->values == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    while (group_end < payload->run_count)
    {
        uint32 term_id = payload->runs[group_end].term_id;
        Size term_size = 0;
        uint32 term_end = group_end;

        while (term_end < payload->run_count &&
               payload->runs[term_end].term_id == term_id)
        {
            const evoke_segment_term_run *run =
                &payload->runs[term_end];
            Size run_size;
            uint64 posting_end;
            uint64 block_count = 0;
            uint32 prior_block_id = 0;

            if (run->posting_count == 0 ||
                run->posting_offset > payload->posting_count ||
                run->posting_count >
                    payload->posting_count - run->posting_offset ||
                (run->kind !=
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 run->kind !=
                     EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT))
            {
                return EVOKE_ERR_FORMAT;
            }
            posting_end = run->posting_offset + run->posting_count;
            for (uint64 posting_index = run->posting_offset;
                 posting_index < posting_end;
                 posting_index++)
            {
                uint32 document_slot =
                    payload->indices[posting_index];
                uint32 block_id = document_slot >>
                    EVOKE_DEFAULT_POSTING_BLOCK_SHIFT;

                if (posting_index == run->posting_offset ||
                    block_id != prior_block_id)
                {
                    block_count++;
                    prior_block_id = block_id;
                }
            }
            if (run->posting_count >
                    (SIZE_MAX - EVOKE_TERM_FOLD_RUN_SIZE) /
                        (sizeof(uint32) * 2) ||
                block_count >
                    (SIZE_MAX - EVOKE_TERM_FOLD_RUN_SIZE -
                     (Size) run->posting_count *
                         (sizeof(uint32) * 2)) /
                        EVOKE_POSTING_BLOCK_RECORD_SIZE)
            {
                return EVOKE_ERR_RANGE;
            }
            run_size = EVOKE_TERM_FOLD_RUN_SIZE +
                (Size) run->posting_count * (sizeof(uint32) * 2) +
                (Size) block_count *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE;
            if (term_size > MaxAllocSize - run_size)
            {
                return EVOKE_ERR_RANGE;
            }
            term_size += run_size;
            term_end++;
        }
        if (term_size > MaxAllocSize - EVOKE_TERM_FOLD_HEADER_SIZE ||
            group_size > MaxAllocSize - term_size)
        {
            return EVOKE_ERR_RANGE;
        }
        if (group_end > first_run &&
            (group_size >= EVOKE_INITIAL_FOLD_TARGET_BYTES ||
             term_size > EVOKE_INITIAL_FOLD_TARGET_BYTES - group_size))
        {
            break;
        }
        group_size += term_size;
        group_end = term_end;
    }
    *end_run_out = group_end;
    return EVOKE_OK;
}

static void
evoke_segment_pages_write_initial_fold_bundle(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_manifest *manifest,
    const evoke_term_fold_bundle *bundle,
    uint64 object_id,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_term_fold_bundle encoded_bundle;
    evoke_segment_object_ref ref;
    uint64 checksum = 0;
    size_t size = 0;
    evoke_status status;

    if (bundle == NULL ||
        bundle->object_kind != EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD ||
        bundle->owner_manifest_id != manifest->manifest_id ||
        bundle->statistics_epoch != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke initial fold bundle")));
    }
    encoded_bundle = *bundle;
    encoded_bundle.semantic_impact_precision =
        evoke_am_get_semantic_impact_precision(index_relation);
    memset(&ref, 0, sizeof(ref));
    status = evoke_term_fold_bundle_serialize(
        &encoded_bundle,
        &cleanup->serialized_bytes,
        &size,
        &checksum
    );
    if (status != EVOKE_OK || size > MaxAllocSize)
    {
        evoke_segment_pages_report_codec_error(
            "initial neutral-fold bundle",
            status == EVOKE_OK ? EVOKE_ERR_RANGE : status
        );
    }
    evoke_segment_pages_write_internal(
        index_relation,
        fork_number,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        object_id,
        manifest->manifest_id,
        cleanup->serialized_bytes,
        size,
        NULL,
        &ref
    );
    if (ref.object_bytes != size ||
        ref.object_checksum != checksum)
    {
        ereport(ERROR, (errmsg("evoke initial fold identity mismatch")));
    }
    for (uint32 run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        uint32 term_id = bundle->runs[run_index].term_id;
        evoke_segment_object_ref *term_ref;

        if (term_id >= manifest->vocab_size)
        {
            ereport(ERROR, (errmsg("invalid evoke initial fold term")));
        }
        term_ref = &cleanup->initial_fold_refs[term_id];
        if (!evoke_segment_object_ref_is_zero(term_ref) &&
            !evoke_segment_object_refs_equal(term_ref, &ref))
        {
            ereport(ERROR, (errmsg("evoke initial term fold split")));
        }
        *term_ref = ref;
    }
    free(cleanup->serialized_bytes);
    cleanup->serialized_bytes = NULL;
}

static void
evoke_segment_pages_write_initial_fold_group(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payload,
    uint32 first_run,
    uint32 end_run,
    uint64 object_id,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_term_fold_bundle bundle;
    uint64 first_posting;
    uint64 posting_end;

    if (first_run >= end_run || end_run > payload->run_count)
    {
        ereport(ERROR, (errmsg("invalid evoke initial fold group")));
    }
    first_posting = payload->runs[first_run].posting_offset;
    posting_end =
        payload->runs[end_run - 1].posting_offset +
        payload->runs[end_run - 1].posting_count;
    if (posting_end <= first_posting ||
        posting_end > payload->posting_count)
    {
        ereport(ERROR, (errmsg("invalid evoke initial fold postings")));
    }

    evoke_term_fold_bundle_init(&bundle);
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    bundle.owner_manifest_id = manifest->manifest_id;
    bundle.run_count = end_run - first_run;
    bundle.posting_count = posting_end - first_posting;
    bundle.runs = calloc(bundle.run_count, sizeof(*bundle.runs));
    bundle.document_slots = &payload->indices[first_posting];
    bundle.values = &payload->values[first_posting];
    if (bundle.runs == NULL)
    {
        ereport(ERROR, (errmsg("out of memory")));
    }
    for (uint32 run_index = first_run;
         run_index < end_run;
         run_index++)
    {
        const evoke_segment_term_run *source =
            &payload->runs[run_index];
        evoke_term_fold_run *target =
            &bundle.runs[run_index - first_run];

        target->term_id = source->term_id;
        target->kind = source->kind;
        target->coverage_sequence = manifest->max_sequence;
        target->posting_offset =
            source->posting_offset - first_posting;
        target->posting_count = source->posting_count;
    }
    evoke_segment_pages_write_initial_fold_bundle(
        index_relation,
        fork_number,
        manifest,
        &bundle,
        object_id,
        cleanup
    );
    free(bundle.runs);
    bundle.runs = NULL;
    bundle.document_slots = NULL;
    bundle.values = NULL;
}

static void
evoke_segment_pages_write_initial_folds(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payload,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    uint32 first_run = 0;
    uint64 object_id = 1;

    if (manifest->vocab_size > 0)
    {
        cleanup->initial_fold_refs = calloc(
            manifest->vocab_size,
            sizeof(*cleanup->initial_fold_refs)
        );
        if (cleanup->initial_fold_refs == NULL)
        {
            ereport(ERROR, (errmsg("out of memory")));
        }
    }
    if (payload->posting_count == 0)
    {
        return;
    }
    if (payload->document_id_base != 0 ||
        payload->local_document_count !=
            manifest->document_slot_count)
    {
        ereport(ERROR, (errmsg("invalid evoke initial fold source")));
    }
    for (uint32 document_slot = 0;
         payload->document_id_map != NULL &&
             document_slot < payload->local_document_count;
         document_slot++)
    {
        if (payload->document_id_map[document_slot] != document_slot)
        {
            ereport(
                ERROR,
                (errmsg("non-identity evoke initial fold document map"))
            );
        }
    }
    while (first_run < payload->run_count)
    {
        uint32 end_run = first_run;
        evoke_status status;

        CHECK_FOR_INTERRUPTS();
        status = evoke_segment_pages_initial_fold_group_end(
            payload,
            first_run,
            &end_run
        );
        if (status != EVOKE_OK || end_run <= first_run)
        {
            evoke_segment_pages_report_codec_error(
                "initial neutral-fold grouping",
                status == EVOKE_OK ? EVOKE_ERR_FORMAT : status
            );
        }
        evoke_segment_pages_write_initial_fold_group(
            index_relation,
            fork_number,
            manifest,
            payload,
            first_run,
            end_run,
            object_id,
            cleanup
        );
        object_id++;
        first_run = end_run;
    }
}

static void
evoke_segment_pages_write_streamed_initial_folds(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_manifest *manifest,
    evoke_initial_fold_bundle_producer producer,
    void *producer_context,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    uint64 object_id = 1;
    bool done = false;

    if (producer == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke initial fold producer")));
    }
    if (manifest->vocab_size > 0)
    {
        cleanup->initial_fold_refs = calloc(
            manifest->vocab_size,
            sizeof(*cleanup->initial_fold_refs)
        );
        if (cleanup->initial_fold_refs == NULL)
        {
            ereport(ERROR, (errmsg("out of memory")));
        }
    }
    while (!done)
    {
        evoke_term_fold_bundle bundle;
        evoke_status status;

        CHECK_FOR_INTERRUPTS();
        evoke_term_fold_bundle_init(&bundle);
        status = producer(producer_context, &bundle, &done);
        if (status != EVOKE_OK)
        {
            evoke_term_fold_bundle_free(&bundle);
            evoke_segment_pages_report_codec_error(
                "initial fold producer",
                status
            );
        }
        if (done)
        {
            evoke_term_fold_bundle_free(&bundle);
            break;
        }
        evoke_segment_pages_write_initial_fold_bundle(
            index_relation,
            fork_number,
            manifest,
            &bundle,
            object_id,
            cleanup
        );
        evoke_term_fold_bundle_free(&bundle);
        if (object_id == UINT64_MAX)
        {
            ereport(ERROR, (errmsg("evoke fold object identity exhausted")));
        }
        object_id++;
    }
}

static evoke_status
evoke_segment_pages_build_document_cow(
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    bool count_event_residency,
    evoke_document_cow_tree *tree
)
{
    evoke_document_cow_record *records = NULL;
    evoke_status status = EVOKE_OK;

    if (manifest == NULL || tree == NULL ||
        (count_event_residency &&
         payload_count != manifest->segment_count) ||
        (payload_count > 0 && payloads == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    if (manifest->document_slot_count == 0)
    {
        return EVOKE_OK;
    }
    if (manifest->document_slot_count >
        SIZE_MAX / sizeof(*records))
    {
        return EVOKE_ERR_RANGE;
    }
    records = calloc(
        (size_t) manifest->document_slot_count,
        sizeof(*records)
    );
    if (records == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (size_t payload_index = 0;
         status == EVOKE_OK && payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_payload *payload =
            &payloads[payload_index];

        for (uint32 version_index = 0;
             version_index < payload->version_count;
             version_index++)
        {
            const evoke_document_version_record *version =
                &payload->versions[version_index];

            if (version->document_slot >=
                    manifest->document_slot_count ||
                records[version->document_slot]
                    .version.born_sequence != 0)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            records[version->document_slot].version = *version;
            if (count_event_residency)
            {
                records[version->document_slot].event_residency++;
            }
        }
        for (uint32 retirement_index = 0;
             status == EVOKE_OK &&
             retirement_index < payload->retirement_count;
             retirement_index++)
        {
            const evoke_document_retirement_record *retirement =
                &payload->retirements[retirement_index];

            if (retirement->document_slot >=
                    manifest->document_slot_count ||
                records[retirement->document_slot]
                    .retirement.retirement_sequence != 0)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            records[retirement->document_slot].retirement =
                *retirement;
            if (count_event_residency)
            {
                records[retirement->document_slot].event_residency++;
            }
        }
        for (uint32 semantic_index = 0;
             status == EVOKE_OK &&
             semantic_index < payload->semantic_state_count;
             semantic_index++)
        {
            const evoke_semantic_state_record *semantic =
                &payload->semantic_states[semantic_index];

            if (semantic->document_slot >=
                    manifest->document_slot_count ||
                records[semantic->document_slot]
                    .semantic_state.transition_sequence != 0)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            records[semantic->document_slot].semantic_state =
                *semantic;
            if (count_event_residency)
            {
                records[semantic->document_slot].event_residency++;
            }
        }
        for (uint32 run_index = 0;
             status == EVOKE_OK && run_index < payload->run_count;
             run_index++)
        {
            const evoke_segment_term_run *run = &payload->runs[run_index];

            if (run->posting_offset > payload->posting_count ||
                run->posting_count >
                    payload->posting_count - run->posting_offset)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            for (uint64 posting_index = run->posting_offset;
                 posting_index <
                    run->posting_offset + run->posting_count;
                 posting_index++)
            {
                uint32 local_document_id =
                    payload->indices[posting_index];
                uint64 document_slot;
                uint64 *residency;

                if (local_document_id >= payload->local_document_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    break;
                }
                document_slot = payload->document_id_map != NULL
                    ? payload->document_id_map[local_document_id]
                    : (uint64) payload->document_id_base +
                        local_document_id;
                if (document_slot >= manifest->document_slot_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    break;
                }
                residency = run->kind ==
                    EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                    ? &records[document_slot].semantic_residency
                    : &records[document_slot].lexical_residency;
                if (*residency == UINT64_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    break;
                }
                (*residency)++;
            }
        }
    }
    for (uint64 document_slot = 0;
         status == EVOKE_OK &&
         document_slot < manifest->document_slot_count;
         document_slot++)
    {
        if (records[document_slot].version.born_sequence == 0)
        {
            status = EVOKE_ERR_FORMAT;
        }
    }
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_tree_build(
            records,
            manifest->document_slot_count,
            manifest->manifest_id,
            tree
        );
    }
    free(records);
    return status;
}

static int
evoke_segment_pages_compare_document_slots(
    const void *left,
    const void *right
)
{
    uint64 left_slot = *(const uint64 *) left;
    uint64 right_slot = *(const uint64 *) right;

    if (left_slot < right_slot)
    {
        return -1;
    }
    if (left_slot > right_slot)
    {
        return 1;
    }
    return 0;
}

static size_t
evoke_segment_pages_find_document_update(
    const evoke_document_cow_record *updates,
    size_t update_count,
    uint64 document_slot
)
{
    size_t low = 0;
    size_t high = update_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2;
        uint64 middle_slot =
            updates[middle].version.document_slot;

        if (middle_slot < document_slot)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < update_count &&
        updates[low].version.document_slot == document_slot
        ? low
        : SIZE_MAX;
}

static evoke_status
evoke_segment_pages_build_document_updates(
    const evoke_segment_payload *payload,
    uint64 old_document_slot_count,
    uint64 next_document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    const evoke_document_cow_ref *old_root,
    evoke_document_cow_record **updates_out,
    size_t *update_count_out
)
{
    evoke_document_cow_record *updates = NULL;
    uint64 *slots = NULL;
    uint8 *version_seen = NULL;
    uint8 *retirement_seen = NULL;
    uint8 *semantic_seen = NULL;
    size_t source_count;
    size_t slot_count = 0;
    size_t update_count = 0;
    evoke_status status = EVOKE_OK;

    if (payload == NULL || updates_out == NULL ||
        update_count_out == NULL ||
        next_document_slot_count < old_document_slot_count ||
        (old_document_slot_count > 0 &&
         (loader == NULL || old_root == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    *updates_out = NULL;
    *update_count_out = 0;
    if ((size_t) payload->version_count >
            SIZE_MAX - payload->retirement_count ||
        (size_t) payload->version_count +
            payload->retirement_count >
            SIZE_MAX - payload->semantic_state_count)
    {
        return EVOKE_ERR_RANGE;
    }
    source_count = (size_t) payload->version_count +
        payload->retirement_count +
        payload->semantic_state_count;
    if (source_count == 0 ||
        source_count > SIZE_MAX / sizeof(*slots))
    {
        return EVOKE_ERR_FORMAT;
    }
    slots = malloc(source_count * sizeof(*slots));
    if (slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32 index = 0; index < payload->version_count; index++)
    {
        slots[slot_count++] =
            payload->versions[index].document_slot;
    }
    for (uint32 index = 0;
         index < payload->retirement_count;
         index++)
    {
        slots[slot_count++] =
            payload->retirements[index].document_slot;
    }
    for (uint32 index = 0;
         index < payload->semantic_state_count;
         index++)
    {
        slots[slot_count++] =
            payload->semantic_states[index].document_slot;
    }
    qsort(
        slots,
        slot_count,
        sizeof(*slots),
        evoke_segment_pages_compare_document_slots
    );
    for (size_t index = 0; index < slot_count; index++)
    {
        if (slots[index] >= next_document_slot_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        if (index == 0 || slots[index - 1] != slots[index])
        {
            slots[update_count++] = slots[index];
        }
    }

    updates = calloc(update_count, sizeof(*updates));
    version_seen = calloc(update_count, sizeof(*version_seen));
    retirement_seen = calloc(
        update_count,
        sizeof(*retirement_seen)
    );
    semantic_seen = calloc(update_count, sizeof(*semantic_seen));
    if (updates == NULL || version_seen == NULL ||
        retirement_seen == NULL || semantic_seen == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto done;
    }
    for (size_t index = 0; index < update_count; index++)
    {
        uint64 document_slot = slots[index];

        if (document_slot < old_document_slot_count)
        {
            status = evoke_document_cow_lookup_external(
                old_root,
                old_document_slot_count,
                document_slot,
                loader,
                loader_context,
                &updates[index]
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
        }
        else
        {
            updates[index].version.document_slot =
                document_slot;
        }
    }

    for (uint32 index = 0; index < payload->version_count; index++)
    {
        const evoke_document_version_record *version =
            &payload->versions[index];
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                version->document_slot
            );

        if (update_index == SIZE_MAX ||
            version_seen[update_index] != 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        if (version->document_slot < old_document_slot_count &&
            evoke_document_cow_record_is_l0_owned(
                &updates[update_index]))
        {
            evoke_document_version_record expected =
                updates[update_index].version;
            bool placeholder =
                (expected.flags &
                 EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0;

            expected.flags &= ~EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
            if (version->document_slot != expected.document_slot ||
                version->born_sequence != expected.born_sequence ||
                (!placeholder &&
                 !evoke_document_cow_versions_equal(
                     version,
                     &expected
                 )))
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            updates[update_index].version = *version;
        }
        else if (version->document_slot < old_document_slot_count)
        {
            memset(
                &updates[update_index],
                0,
                sizeof(updates[update_index])
            );
            updates[update_index].version = *version;
        }
        else
        {
            updates[update_index].version = *version;
        }
        if (updates[update_index].event_residency == UINT64_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        updates[update_index].event_residency++;
        version_seen[update_index] = 1;
    }
    for (uint32 index = 0;
         index < payload->retirement_count;
         index++)
    {
        const evoke_document_retirement_record *retirement =
            &payload->retirements[index];
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                retirement->document_slot
            );

        if (update_index == SIZE_MAX ||
            retirement_seen[update_index] != 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        updates[update_index].retirement = *retirement;
        if (updates[update_index].event_residency == UINT64_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        updates[update_index].event_residency++;
        retirement_seen[update_index] = 1;
    }
    for (uint32 index = 0;
         index < payload->semantic_state_count;
         index++)
    {
        const evoke_semantic_state_record *semantic =
            &payload->semantic_states[index];
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                semantic->document_slot
            );

        if (update_index == SIZE_MAX ||
            semantic_seen[update_index] != 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        updates[update_index].semantic_state = *semantic;
        if (updates[update_index].event_residency == UINT64_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        updates[update_index].event_residency++;
        semantic_seen[update_index] = 1;
    }
    for (uint32 run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];

        if (run->posting_offset > payload->posting_count ||
            run->posting_count >
                payload->posting_count - run->posting_offset)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        for (uint64 posting_index = run->posting_offset;
             posting_index < run->posting_offset + run->posting_count;
             posting_index++)
        {
            uint32 local_document_id = payload->indices[posting_index];
            uint64 document_slot;
            size_t update_index;
            uint64 *residency;

            if (local_document_id >= payload->local_document_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            document_slot = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : (uint64) payload->document_id_base +
                    local_document_id;
            update_index = evoke_segment_pages_find_document_update(
                updates,
                update_count,
                document_slot
            );
            if (update_index == SIZE_MAX)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            residency = run->kind ==
                EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? &updates[update_index].semantic_residency
                : &updates[update_index].lexical_residency;
            if (*residency == UINT64_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            (*residency)++;
        }
    }
    for (size_t index = 0; index < update_count; index++)
    {
        if (slots[index] >= old_document_slot_count &&
            version_seen[index] == 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
    }

done:
    free(semantic_seen);
    free(retirement_seen);
    free(version_seen);
    free(slots);
    if (status != EVOKE_OK)
    {
        free(updates);
        return status;
    }
    *updates_out = updates;
    *update_count_out = update_count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_payload_slot_source_count(
    const evoke_segment_payload *payload,
    size_t *count_out
)
{
    size_t count;

    if (payload == NULL || count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    count = payload->local_document_count;
    if (count > SIZE_MAX - payload->version_count)
    {
        return EVOKE_ERR_RANGE;
    }
    count += payload->version_count;
    if (count > SIZE_MAX - payload->retirement_count)
    {
        return EVOKE_ERR_RANGE;
    }
    count += payload->retirement_count;
    if (count > SIZE_MAX - payload->semantic_state_count)
    {
        return EVOKE_ERR_RANGE;
    }
    *count_out = count + payload->semantic_state_count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_payload_document_slot(
    const evoke_segment_payload *payload,
    uint32 local_document_id,
    uint64 document_slot_count,
    uint64 *document_slot_out
)
{
    uint64 document_slot;

    if (payload == NULL || document_slot_out == NULL ||
        local_document_id >= payload->local_document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    document_slot = payload->document_id_map != NULL
        ? payload->document_id_map[local_document_id]
        : (uint64) payload->document_id_base + local_document_id;
    if (document_slot >= document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *document_slot_out = document_slot;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_collect_payload_document_slots(
    const evoke_segment_payload *payload,
    uint64 document_slot_count,
    uint64 *slots,
    size_t capacity,
    size_t *slot_count
)
{
    size_t required;
    evoke_status status;

    if (payload == NULL || slots == NULL || slot_count == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_pages_payload_slot_source_count(
        payload,
        &required
    );
    if (status != EVOKE_OK || *slot_count > capacity ||
        required > capacity - *slot_count)
    {
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    for (uint32 local_document_id = 0;
         local_document_id < payload->local_document_count;
         local_document_id++)
    {
        status = evoke_segment_pages_payload_document_slot(
            payload,
            local_document_id,
            document_slot_count,
            &slots[(*slot_count)++]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    for (uint32 index = 0; index < payload->version_count; index++)
    {
        if (payload->versions[index].document_slot >= document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        slots[(*slot_count)++] =
            payload->versions[index].document_slot;
    }
    for (uint32 index = 0; index < payload->retirement_count; index++)
    {
        if (payload->retirements[index].document_slot >=
            document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        slots[(*slot_count)++] =
            payload->retirements[index].document_slot;
    }
    for (uint32 index = 0;
         index < payload->semantic_state_count;
         index++)
    {
        if (payload->semantic_states[index].document_slot >=
            document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        slots[(*slot_count)++] =
            payload->semantic_states[index].document_slot;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_adjust_residency(
    uint64 *residency,
    bool add
)
{
    if (residency == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (add)
    {
        if (*residency == UINT64_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        (*residency)++;
    }
    else
    {
        if (*residency == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        (*residency)--;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_apply_payload_residency(
    const evoke_segment_payload *payload,
    uint64 document_slot_count,
    evoke_document_cow_record *updates,
    size_t update_count,
    bool add
)
{
    evoke_status status;

    if (payload == NULL || updates == NULL || update_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32 index = 0; index < payload->version_count; index++)
    {
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                payload->versions[index].document_slot
            );

        if (update_index == SIZE_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_segment_pages_adjust_residency(
            &updates[update_index].event_residency,
            add
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    for (uint32 index = 0; index < payload->retirement_count; index++)
    {
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                payload->retirements[index].document_slot
            );

        if (update_index == SIZE_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_segment_pages_adjust_residency(
            &updates[update_index].event_residency,
            add
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    for (uint32 index = 0;
         index < payload->semantic_state_count;
         index++)
    {
        size_t update_index =
            evoke_segment_pages_find_document_update(
                updates,
                update_count,
                payload->semantic_states[index].document_slot
            );

        if (update_index == SIZE_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_segment_pages_adjust_residency(
            &updates[update_index].event_residency,
            add
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    for (uint32 run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];

        if (run->posting_offset > payload->posting_count ||
            run->posting_count >
                payload->posting_count - run->posting_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint64 posting_index = run->posting_offset;
             posting_index < run->posting_offset + run->posting_count;
             posting_index++)
        {
            uint32 local_document_id = payload->indices[posting_index];
            uint64 document_slot;
            size_t update_index;
            uint64 *residency;

            status = evoke_segment_pages_payload_document_slot(
                payload,
                local_document_id,
                document_slot_count,
                &document_slot
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            update_index = evoke_segment_pages_find_document_update(
                updates,
                update_count,
                document_slot
            );
            if (update_index == SIZE_MAX)
            {
                return EVOKE_ERR_FORMAT;
            }
            residency = run->kind ==
                EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? &updates[update_index].semantic_residency
                : &updates[update_index].lexical_residency;
            status = evoke_segment_pages_adjust_residency(residency, add);
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_build_replacement_document_updates(
    const evoke_segment_payload *replaced_payloads,
    uint32 replaced_payload_count,
    const evoke_segment_payload *replacement_payload,
    uint64 document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    const evoke_document_cow_ref *old_root,
    evoke_document_cow_record **updates_out,
    size_t *update_count_out
)
{
    evoke_document_cow_record *updates = NULL;
    evoke_document_cow_record *originals = NULL;
    uint64 *slots = NULL;
    size_t source_count = 0;
    size_t slot_count = 0;
    size_t update_count = 0;
    size_t changed_count = 0;
    evoke_status status = EVOKE_OK;

    if (replaced_payloads == NULL || replaced_payload_count == 0 ||
        replacement_payload == NULL || updates_out == NULL ||
        update_count_out == NULL ||
        (document_slot_count > 0 &&
         (loader == NULL || old_root == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    *updates_out = NULL;
    *update_count_out = 0;
    for (uint32 index = 0; index < replaced_payload_count; index++)
    {
        size_t payload_count;

        status = evoke_segment_pages_payload_slot_source_count(
            &replaced_payloads[index],
            &payload_count
        );
        if (status != EVOKE_OK || source_count > SIZE_MAX - payload_count)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        source_count += payload_count;
    }
    {
        size_t payload_count;

        status = evoke_segment_pages_payload_slot_source_count(
            replacement_payload,
            &payload_count
        );
        if (status != EVOKE_OK || source_count > SIZE_MAX - payload_count)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        source_count += payload_count;
    }
    if (source_count == 0)
    {
        return EVOKE_OK;
    }
    if (document_slot_count == 0 ||
        source_count > SIZE_MAX / sizeof(*slots))
    {
        return EVOKE_ERR_FORMAT;
    }
    slots = malloc(source_count * sizeof(*slots));
    if (slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32 index = 0; index < replaced_payload_count; index++)
    {
        status = evoke_segment_pages_collect_payload_document_slots(
            &replaced_payloads[index],
            document_slot_count,
            slots,
            source_count,
            &slot_count
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    status = evoke_segment_pages_collect_payload_document_slots(
        replacement_payload,
        document_slot_count,
        slots,
        source_count,
        &slot_count
    );
    if (status != EVOKE_OK || slot_count != source_count)
    {
        status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        goto done;
    }
    qsort(
        slots,
        slot_count,
        sizeof(*slots),
        evoke_segment_pages_compare_document_slots
    );
    for (size_t index = 0; index < slot_count; index++)
    {
        if (index == 0 || slots[index - 1] != slots[index])
        {
            slots[update_count++] = slots[index];
        }
    }
    if (update_count > SIZE_MAX / sizeof(*updates))
    {
        status = EVOKE_ERR_RANGE;
        goto done;
    }
    updates = calloc(update_count, sizeof(*updates));
    originals = calloc(update_count, sizeof(*originals));
    if (updates == NULL || originals == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto done;
    }
    for (size_t index = 0; index < update_count; index++)
    {
        status = evoke_document_cow_lookup_external(
            old_root,
            document_slot_count,
            slots[index],
            loader,
            loader_context,
            &updates[index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        originals[index] = updates[index];
    }
    for (uint32 index = 0; index < replaced_payload_count; index++)
    {
        status = evoke_segment_pages_apply_payload_residency(
            &replaced_payloads[index],
            document_slot_count,
            updates,
            update_count,
            false
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    status = evoke_segment_pages_apply_payload_residency(
        replacement_payload,
        document_slot_count,
        updates,
        update_count,
        true
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    for (size_t index = 0; index < update_count; index++)
    {
        if (updates[index].lexical_residency !=
                originals[index].lexical_residency ||
            updates[index].semantic_residency !=
                originals[index].semantic_residency ||
            updates[index].event_residency !=
                originals[index].event_residency)
        {
            updates[changed_count++] = updates[index];
        }
    }

done:
    free(originals);
    free(slots);
    if (status != EVOKE_OK)
    {
        free(updates);
        return status;
    }
    if (changed_count == 0)
    {
        free(updates);
        return EVOKE_OK;
    }
    *updates_out = updates;
    *update_count_out = changed_count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_fold_term_posting_count(
    const evoke_term_fold_bundle *bundle,
    uint32 term_id,
    size_t *count_out
)
{
    size_t count = 0;
    evoke_status status;

    if (bundle == NULL || count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_fold_bundle_validate(bundle);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (uint32 run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];

        if (run->term_id != term_id)
        {
            continue;
        }
        if (run->posting_count > SIZE_MAX - count)
        {
            return EVOKE_ERR_RANGE;
        }
        count += (size_t) run->posting_count;
    }
    *count_out = count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_collect_fold_document_slots(
    const evoke_term_fold_bundle *bundle,
    uint32 term_id,
    uint64 document_slot_count,
    uint64 *slots,
    size_t capacity,
    size_t *slot_count
)
{
    size_t required;
    evoke_status status;

    if (bundle == NULL || slots == NULL || slot_count == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_pages_fold_term_posting_count(
        bundle,
        term_id,
        &required
    );
    if (status != EVOKE_OK || *slot_count > capacity ||
        required > capacity - *slot_count)
    {
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    for (uint32 run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];

        if (run->term_id != term_id)
        {
            continue;
        }
        for (uint64 posting_index = run->posting_offset;
             posting_index < run->posting_offset + run->posting_count;
             posting_index++)
        {
            uint64 document_slot =
                bundle->document_slots[posting_index];

            if (document_slot >= document_slot_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            slots[(*slot_count)++] = document_slot;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_apply_fold_residency(
    const evoke_term_fold_bundle *bundle,
    uint32 term_id,
    uint64 document_slot_count,
    evoke_document_cow_record *updates,
    size_t update_count,
    bool add
)
{
    if (bundle == NULL || updates == NULL || update_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32 run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];

        if (run->term_id != term_id)
        {
            continue;
        }
        if (run->posting_offset > bundle->posting_count ||
            run->posting_count >
                bundle->posting_count - run->posting_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint64 posting_index = run->posting_offset;
             posting_index < run->posting_offset + run->posting_count;
             posting_index++)
        {
            uint64 document_slot =
                bundle->document_slots[posting_index];
            size_t update_index;
            uint64 *residency;
            evoke_status status;

            if (document_slot >= document_slot_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            update_index = evoke_segment_pages_find_document_update(
                updates,
                update_count,
                document_slot
            );
            if (update_index == SIZE_MAX)
            {
                return EVOKE_ERR_FORMAT;
            }
            residency = run->kind ==
                EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? &updates[update_index].semantic_residency
                : &updates[update_index].lexical_residency;
            status = evoke_segment_pages_adjust_residency(residency, add);
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_pages_build_fold_document_updates(
    const evoke_term_fold_bundle *old_folds,
    size_t old_fold_count,
    const evoke_term_fold_bundle *new_fold,
    uint32 term_id,
    uint64 document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    const evoke_document_cow_ref *old_root,
    evoke_document_cow_record **updates_out,
    size_t *update_count_out
)
{
    evoke_document_cow_record *updates = NULL;
    evoke_document_cow_record *originals = NULL;
    uint64 *slots = NULL;
    size_t source_count = 0;
    size_t slot_count = 0;
    size_t update_count = 0;
    size_t changed_count = 0;
    evoke_status status = EVOKE_OK;

    if ((old_fold_count > 0 && old_folds == NULL) ||
        new_fold == NULL || updates_out == NULL ||
        update_count_out == NULL || document_slot_count == 0 ||
        loader == NULL || old_root == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *updates_out = NULL;
    *update_count_out = 0;
    for (size_t index = 0; index < old_fold_count; index++)
    {
        size_t fold_count;

        status = evoke_segment_pages_fold_term_posting_count(
            &old_folds[index],
            term_id,
            &fold_count
        );
        if (status != EVOKE_OK || source_count > SIZE_MAX - fold_count)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        source_count += fold_count;
    }
    {
        size_t fold_count;

        status = evoke_segment_pages_fold_term_posting_count(
            new_fold,
            term_id,
            &fold_count
        );
        if (status != EVOKE_OK || fold_count == 0 ||
            source_count > SIZE_MAX - fold_count)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        source_count += fold_count;
    }
    if (source_count > SIZE_MAX / sizeof(*slots))
    {
        return EVOKE_ERR_RANGE;
    }
    slots = malloc(source_count * sizeof(*slots));
    if (slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t index = 0; index < old_fold_count; index++)
    {
        status = evoke_segment_pages_collect_fold_document_slots(
            &old_folds[index],
            term_id,
            document_slot_count,
            slots,
            source_count,
            &slot_count
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    status = evoke_segment_pages_collect_fold_document_slots(
        new_fold,
        term_id,
        document_slot_count,
        slots,
        source_count,
        &slot_count
    );
    if (status != EVOKE_OK || slot_count != source_count)
    {
        status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        goto done;
    }
    qsort(
        slots,
        slot_count,
        sizeof(*slots),
        evoke_segment_pages_compare_document_slots
    );
    for (size_t index = 0; index < slot_count; index++)
    {
        if (index == 0 || slots[index - 1] != slots[index])
        {
            slots[update_count++] = slots[index];
        }
    }
    updates = calloc(update_count, sizeof(*updates));
    originals = calloc(update_count, sizeof(*originals));
    if (updates == NULL || originals == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto done;
    }
    for (size_t index = 0; index < update_count; index++)
    {
        status = evoke_document_cow_lookup_external(
            old_root,
            document_slot_count,
            slots[index],
            loader,
            loader_context,
            &updates[index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        originals[index] = updates[index];
    }
    for (size_t index = 0; index < old_fold_count; index++)
    {
        status = evoke_segment_pages_apply_fold_residency(
            &old_folds[index],
            term_id,
            document_slot_count,
            updates,
            update_count,
            false
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    status = evoke_segment_pages_apply_fold_residency(
        new_fold,
        term_id,
        document_slot_count,
        updates,
        update_count,
        true
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    for (size_t index = 0; index < update_count; index++)
    {
        if (updates[index].lexical_residency !=
                originals[index].lexical_residency ||
            updates[index].semantic_residency !=
                originals[index].semantic_residency)
        {
            updates[changed_count++] = updates[index];
        }
    }

done:
    free(originals);
    free(slots);
    if (status != EVOKE_OK)
    {
        free(updates);
        return status;
    }
    if (changed_count == 0)
    {
        free(updates);
        return EVOKE_OK;
    }
    *updates_out = updates;
    *update_count_out = changed_count;
    return EVOKE_OK;
}

static void
evoke_segment_pages_write_document_directory(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    bool count_event_residency,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    evoke_status status;

    if (manifest->document_slot_count == 0)
    {
        return;
    }
    status = evoke_segment_pages_build_document_cow(
        manifest,
        payloads,
        payload_count,
        count_event_residency,
        &cleanup->document_cow_tree
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document-directory build",
            status
        );
    }
    evoke_segment_pages_write_document_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->document_cow_tree,
        manifest->manifest_id,
        reuse_arena,
        &manifest->document_directory
    );
    evoke_document_cow_tree_free(&cleanup->document_cow_tree);
    manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
}

static void
evoke_segment_pages_write_document_directory_records(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_document_cow_record *records,
    size_t record_count,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_status status;

    if (record_count != manifest->document_slot_count ||
        (record_count > 0 && records == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke initial document records")));
    }
    if (record_count == 0)
    {
        return;
    }
    status = evoke_document_cow_tree_build(
        records,
        record_count,
        manifest->manifest_id,
        &cleanup->document_cow_tree
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "initial COW document-directory build",
            status
        );
    }
    evoke_segment_pages_write_document_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->document_cow_tree,
        manifest->manifest_id,
        NULL,
        &manifest->document_directory
    );
    evoke_document_cow_tree_free(&cleanup->document_cow_tree);
    manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
}

static void
evoke_segment_pages_write_initial_folded_term_directory(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *lexical_catalog,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_status status;

    status = evoke_term_cow_tree_build_initial_folds(
        manifest,
        lexical_catalog,
        cleanup->initial_fold_refs,
        &cleanup->term_cow_tree
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "initial folded COW term-directory build",
            status
        );
    }
    evoke_segment_pages_write_term_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->term_cow_tree,
        1,
        manifest->manifest_id,
        NULL,
        &manifest->term_directory
    );
    evoke_term_cow_tree_free(&cleanup->term_cow_tree);
    manifest->flags &=
        ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    free(manifest->doc_frequencies);
    manifest->doc_frequencies = NULL;
}

static void
evoke_segment_pages_write_cow_term_directory(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *lexical_catalog,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    evoke_status status;

    status = evoke_term_directory_build_from_payloads(
        manifest,
        cleanup->payload_views,
        manifest->segment_count,
        &cleanup->term_directory
    );
    if (status == EVOKE_OK)
    {
        status = evoke_term_cow_tree_build(
            &cleanup->term_directory,
            manifest,
            lexical_catalog,
            &cleanup->term_cow_tree
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW term-directory build",
            status
        );
    }
    evoke_segment_pages_write_term_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->term_cow_tree,
        1,
        manifest->manifest_id,
        reuse_arena,
        &manifest->term_directory
    );
    evoke_term_cow_tree_free(&cleanup->term_cow_tree);
    evoke_term_directory_free(&cleanup->term_directory);
    manifest->flags &=
        ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    free(manifest->doc_frequencies);
    manifest->doc_frequencies = NULL;
}

static uint64
evoke_segment_pages_initial_lexicon_hash_seed(
    Relation index_relation,
    const evoke_segment_manifest *manifest
)
{
    uint64 seed;

    seed = ((uint64) RelationGetRelid(index_relation) << 32) ^
        manifest->manifest_id ^ UINT64_C(0x9E3779B97F4A7C15);
    seed = evoke_lexicon_cow_hash(
        manifest->contract_hash,
        sizeof(manifest->contract_hash),
        seed
    );
    return seed == 0 ? UINT64_C(0xA0761D6478BD642F) : seed;
}

static void
evoke_segment_pages_write_lexicon_lookup(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    evoke_lexicon_cow_key *keys;
    evoke_status status;

    if ((query_contract->flags &
         EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) == 0 ||
        manifest->vocab_size == 0)
    {
        return;
    }
    keys = palloc0(sizeof(*keys) * (Size) manifest->vocab_size);
    for (uint32 term_id = 0;
         term_id < manifest->vocab_size;
         term_id++)
    {
        size_t bytes_len = strlen(query_contract->vocab[term_id]);

        if (bytes_len == 0 || bytes_len > UINT32_MAX)
        {
            ereport(ERROR, (errmsg("invalid evoke lexical lookup term")));
        }
        keys[term_id].bytes =
            (const uint8 *) query_contract->vocab[term_id];
        keys[term_id].bytes_len = (uint32) bytes_len;
        keys[term_id].term_id = term_id;
    }
    manifest->lexicon_hash_seed =
        evoke_segment_pages_initial_lexicon_hash_seed(
            index_relation,
            manifest
        );
    status = evoke_lexicon_cow_tree_build(
        keys,
        manifest->vocab_size,
        manifest->lexicon_hash_seed,
        manifest->manifest_id,
        &cleanup->lexicon_cow_tree
    );
    pfree(keys);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW lexical-lookup build",
            status
        );
    }
    evoke_segment_pages_write_lexicon_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->lexicon_cow_tree,
        manifest->manifest_id,
        reuse_arena,
        &manifest->lexicon_lookup
    );
    evoke_lexicon_cow_tree_free(&cleanup->lexicon_cow_tree);
    manifest->flags |= EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP;
}

static void
evoke_segment_pages_write_prefix_lookup(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    evoke_prefix_cow_key *keys;
    evoke_status status;

    if ((query_contract->flags &
         EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) == 0 ||
        manifest->vocab_size == 0)
    {
        return;
    }
    keys = palloc0(sizeof(*keys) * (Size) manifest->vocab_size);
    for (uint32 term_id = 0;
         term_id < manifest->vocab_size;
         term_id++)
    {
        size_t bytes_len = strlen(query_contract->vocab[term_id]);

        if (bytes_len == 0 || bytes_len > UINT32_MAX)
        {
            ereport(ERROR, (errmsg("invalid evoke prefix lookup term")));
        }
        keys[term_id].bytes =
            (const uint8 *) query_contract->vocab[term_id];
        keys[term_id].bytes_len = (uint32) bytes_len;
        keys[term_id].term_id = term_id;
    }
    status = evoke_prefix_cow_tree_build(
        keys,
        manifest->vocab_size,
        manifest->manifest_id,
        &cleanup->prefix_cow_tree
    );
    pfree(keys);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW prefix-lookup build",
            status
        );
    }
    evoke_segment_pages_write_prefix_cow_objects_internal(
        index_relation,
        fork_number,
        &cleanup->prefix_cow_tree,
        manifest->manifest_id,
        reuse_arena,
        &manifest->prefix_lookup
    );
    evoke_prefix_cow_tree_free(&cleanup->prefix_cow_tree);
    manifest->flags |= EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
}

static void
evoke_segment_pages_write_manifest(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_manifest *manifest,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_object_ref *manifest_ref_out
)
{
    size_t size = 0;
    evoke_status status;

    status = evoke_segment_manifest_serialize(
        manifest,
        &cleanup->serialized_bytes,
        &size
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("manifest", status);
    }
    evoke_segment_pages_write_internal(
        index_relation,
        fork_number,
        EVOKE_SEGMENT_OBJECT_MANIFEST,
        manifest->manifest_id,
        manifest->manifest_id,
        cleanup->serialized_bytes,
        size,
        reuse_arena,
        manifest_ref_out
    );
    free(cleanup->serialized_bytes);
    cleanup->serialized_bytes = NULL;
}

static void
evoke_segment_pages_write_sealed_bundle_fork_internal(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    const evoke_segment_payload *initial_fold_payload,
    evoke_initial_fold_bundle_producer initial_fold_producer,
    void *initial_fold_producer_context,
    const evoke_document_cow_record *initial_document_records,
    size_t initial_document_record_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_segment_read_root root;
    evoke_segment_object_ref lexical_catalog = {0};
    uint64 max_segment_id;
    uint32 segment_index;
    evoke_status status;

    if (index_relation == NULL ||
        (fork_number != MAIN_FORKNUM &&
         fork_number != INIT_FORKNUM) ||
        manifest == NULL ||
        query_contract == NULL || root_out == NULL ||
        active_l0_segment_id == 0 ||
        active_l0_segment_id == manifest->manifest_id ||
        next_sequence <= manifest->max_sequence ||
        (initial_fold_payload != NULL &&
         (manifest->segment_count != 0 || payload_count != 0 ||
          initial_fold_producer != NULL)) ||
        (initial_fold_producer != NULL &&
         (manifest->segment_count != 0 || payload_count != 0 ||
          initial_fold_payload != NULL ||
          initial_document_record_count !=
            manifest->document_slot_count ||
          (initial_document_record_count > 0 &&
           initial_document_records == NULL))) ||
        (initial_fold_producer == NULL &&
         (initial_fold_producer_context != NULL ||
          initial_document_records != NULL ||
          initial_document_record_count != 0)))
    {
        ereport(ERROR, (errmsg("invalid evoke sealed bundle write")));
    }
    evoke_segment_pages_require_unpublished_manifest(
        manifest,
        payloads,
        payload_count
    );
    max_segment_id = Max(active_l0_segment_id, manifest->manifest_id);
    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (manifest->segments[segment_index].segment_id ==
            active_l0_segment_id)
        {
            ereport(
                ERROR,
                (errmsg("evoke active L0 identity is not unique"))
            );
        }
        if (manifest->segments[segment_index].segment_id > max_segment_id)
        {
            max_segment_id =
                manifest->segments[segment_index].segment_id;
        }
    }
    if (max_segment_id == UINT64_MAX)
    {
        ereport(ERROR, (errmsg("evoke segment identity is exhausted")));
    }
    status = evoke_segment_query_contract_validate(
        query_contract,
        manifest
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "query contract",
            status
        );
    }

    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    evoke_lexicon_cow_tree_init(&cleanup->lexicon_cow_tree);
    evoke_prefix_cow_tree_init(&cleanup->prefix_cow_tree);
    memset(&root, 0, sizeof(root));

    PG_TRY();
    {
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write query contract"
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_write_query_contract(
            index_relation,
            fork_number,
            manifest,
            query_contract,
            cleanup,
            NULL
        );
        if ((query_contract->flags &
             EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) != 0)
        {
            pgstat_report_activity(
                STATE_RUNNING,
                "evoke build: write lexical catalog"
            );
            CHECK_FOR_INTERRUPTS();
            evoke_segment_pages_write_lexical_catalog(
                index_relation,
                fork_number,
                manifest->manifest_id,
                0,
                manifest->vocab_size,
                (const char *const *) query_contract->vocab,
                cleanup,
                NULL,
                &lexical_catalog
            );
        }
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write lexicon lookup"
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_write_lexicon_lookup(
            index_relation,
            fork_number,
            manifest,
            query_contract,
            cleanup,
            NULL
        );
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write prefix lookup"
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_write_prefix_lookup(
            index_relation,
            fork_number,
            manifest,
            query_contract,
            cleanup,
            NULL
        );
        if (initial_fold_payload == NULL &&
            initial_fold_producer == NULL)
        {
            pgstat_report_activity(
                STATE_RUNNING,
                "evoke build: write payloads"
            );
            CHECK_FOR_INTERRUPTS();
            evoke_segment_pages_write_payloads(
                index_relation,
                fork_number,
                manifest,
                payloads,
                cleanup,
                NULL
            );
        }
        else if (initial_fold_payload != NULL)
        {
            pgstat_report_activity(
                STATE_RUNNING,
                "evoke build: write initial term folds"
            );
            CHECK_FOR_INTERRUPTS();
            evoke_segment_pages_write_initial_folds(
                index_relation,
                fork_number,
                manifest,
                initial_fold_payload,
                cleanup
            );
        }
        else
        {
            pgstat_report_activity(
                STATE_RUNNING,
                "evoke build: stream initial term folds"
            );
            CHECK_FOR_INTERRUPTS();
            evoke_segment_pages_write_streamed_initial_folds(
                index_relation,
                fork_number,
                manifest,
                initial_fold_producer,
                initial_fold_producer_context,
                cleanup
            );
        }
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write document directory"
        );
        CHECK_FOR_INTERRUPTS();
        if (initial_fold_producer != NULL)
        {
            evoke_segment_pages_write_document_directory_records(
                index_relation,
                fork_number,
                manifest,
                initial_document_records,
                initial_document_record_count,
                cleanup
            );
        }
        else
        {
            evoke_segment_pages_write_document_directory(
                index_relation,
                fork_number,
                manifest,
                initial_fold_payload == NULL
                    ? payloads
                    : initial_fold_payload,
                initial_fold_payload == NULL ? payload_count : 1,
                initial_fold_payload == NULL,
                cleanup,
                NULL
            );
        }
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write COW term directory"
        );
        CHECK_FOR_INTERRUPTS();
        if (initial_fold_payload == NULL &&
            initial_fold_producer == NULL)
        {
            evoke_segment_pages_write_cow_term_directory(
                index_relation,
                fork_number,
                manifest,
                evoke_segment_object_ref_is_zero(&lexical_catalog)
                    ? NULL
                    : &lexical_catalog,
                cleanup,
                NULL
            );
        }
        else
        {
            evoke_segment_pages_write_initial_folded_term_directory(
                index_relation,
                fork_number,
                manifest,
                evoke_segment_object_ref_is_zero(&lexical_catalog)
                    ? NULL
                    : &lexical_catalog,
                cleanup
            );
        }
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: write manifest"
        );
        CHECK_FOR_INTERRUPTS();
        evoke_segment_pages_write_manifest(
            index_relation,
            fork_number,
            manifest,
            cleanup,
            NULL,
            &root.manifest
        );
        root.root_id = manifest->manifest_id;
        root.next_sequence = next_sequence;
        root.next_document_slot = manifest->document_slot_count;
        root.next_segment_id = max_segment_id + 1;
        root.active_l0.segment_id = active_l0_segment_id;
        root.published_block_high_watermark =
            RelationGetNumberOfBlocksInFork(
                index_relation,
                fork_number
            );
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke build: validate manifest"
        );
        CHECK_FOR_INTERRUPTS();
        status = evoke_segment_manifest_validate_published(
            manifest,
            &root.manifest,
            root.published_block_high_watermark
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "published manifest closure",
                status
            );
        }
        status = evoke_segment_read_root_validate(&root);
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error("read root", status);
        }
        *root_out = root;
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

void
evoke_segment_pages_write_sealed_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
)
{
    evoke_segment_pages_write_sealed_bundle_fork_internal(
        index_relation,
        fork_number,
        manifest,
        query_contract,
        payloads,
        payload_count,
        NULL,
        NULL,
        NULL,
        NULL,
        0,
        active_l0_segment_id,
        next_sequence,
        root_out
    );
}

void
evoke_segment_pages_write_initial_folded_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payload,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
)
{
    if (payload == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke initial folded bundle")));
    }
    evoke_segment_pages_write_sealed_bundle_fork_internal(
        index_relation,
        fork_number,
        manifest,
        query_contract,
        NULL,
        0,
        payload,
        NULL,
        NULL,
        NULL,
        0,
        active_l0_segment_id,
        next_sequence,
        root_out
    );
}

void
evoke_segment_pages_write_streamed_initial_folded_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_initial_fold_bundle_producer producer,
    void *producer_context,
    const evoke_document_cow_record *document_records,
    size_t document_record_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
)
{
    if (producer == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke streamed initial bundle")));
    }
    evoke_segment_pages_write_sealed_bundle_fork_internal(
        index_relation,
        fork_number,
        manifest,
        query_contract,
        NULL,
        0,
        NULL,
        producer,
        producer_context,
        document_records,
        document_record_count,
        active_l0_segment_id,
        next_sequence,
        root_out
    );
}

void
evoke_segment_pages_write_sealed_bundle(
    Relation index_relation,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
)
{
    evoke_segment_pages_write_sealed_bundle_fork(
        index_relation,
        MAIN_FORKNUM,
        manifest,
        query_contract,
        payloads,
        payload_count,
        active_l0_segment_id,
        next_sequence,
        root_out
    );
}

static void
evoke_segment_pages_require_cow_append(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    const char *const *lexical_terms,
    uint32 lexical_term_count,
    const evoke_segment_payload *new_payload,
    const evoke_l0_storage_snapshot *retired_l0
)
{
    const evoke_segment_descriptor *new_descriptor;
    evoke_status status;
    uint32 segment_index;

    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL || new_payload == NULL ||
        retired_l0 == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW segment append")));
    }
    status = evoke_segment_read_root_validate(build_root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    status = evoke_segment_manifest_validate_published(
        old_manifest,
        &build_root->manifest,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW ancestor manifest",
            status
        );
    }
    if (build_root->root_id != old_manifest->manifest_id ||
        build_root->pending_l0.segment_id == 0 ||
        old_manifest->segment_count >=
            EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        next_manifest->manifest_id != build_root->next_segment_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        next_manifest->segment_count !=
            old_manifest->segment_count + 1 ||
        next_manifest->segments == NULL ||
        next_manifest->max_sequence !=
            build_root->pending_l0.max_sequence ||
        next_manifest->statistics_epoch <=
            old_manifest->statistics_epoch ||
        next_manifest->document_slot_count <
            old_manifest->document_slot_count ||
        next_manifest->document_slot_count >
            build_root->next_document_slot ||
        next_manifest->vocab_size < old_manifest->vocab_size ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0 ||
        next_manifest->flags !=
            (old_manifest->flags &
             (EVOKE_SEGMENT_MANIFEST_FLAG_SAE |
              EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP |
              EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP |
              EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR)) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->term_directory) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->neutral_fold) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->impact_fold) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->document_directory) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->lexicon_lookup) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->prefix_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->semantic_accelerator_directory,
            &old_manifest->semantic_accelerator_directory) ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            next_manifest
        ) !=
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            ) ||
        next_manifest->lexicon_hash_seed !=
            old_manifest->lexicon_hash_seed ||
        next_manifest->neutral_fold_coverage != 0 ||
        next_manifest->impact_fold_coverage != 0 ||
        next_manifest->impact_statistics_epoch != 0 ||
        next_manifest->doc_frequencies != NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW manifest transition")));
    }
    if (retired_l0->record_count !=
            build_root->pending_l0.record_count ||
        retired_l0->page_count != build_root->pending_l0.page_count ||
        retired_l0->payload_bytes !=
            build_root->pending_l0.payload_bytes ||
        retired_l0->page_blocks == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke retired pending L0")));
    }
    for (segment_index = 0;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[segment_index]))
        {
            ereport(
                ERROR,
                (errmsg("evoke COW descriptor prefix changed"))
            );
        }
    }

    new_descriptor =
        &next_manifest->segments[old_manifest->segment_count];
    if (new_descriptor->segment_id !=
            build_root->pending_l0.segment_id ||
        new_descriptor->min_sequence !=
            build_root->pending_l0.min_sequence ||
        new_descriptor->max_sequence !=
            build_root->pending_l0.max_sequence ||
        new_descriptor->start_block != 0 ||
        new_descriptor->block_count != 0 ||
        new_descriptor->payload_bytes != 0 ||
        new_descriptor->payload_checksum != 0 ||
        new_descriptor->payload_owner_manifest_id != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW tail descriptor")));
    }
    if (!evoke_segment_object_refs_equal(
            &next_manifest->query_contract,
            &old_manifest->query_contract))
    {
        ereport(ERROR, (errmsg("invalid evoke reused query contract")));
    }
    if ((lexical_term_count == 0) != (lexical_terms == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke lexical catalog suffix")));
    }
    if ((old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) != 0 &&
        lexical_term_count == 0 &&
        next_manifest->vocab_size != old_manifest->vocab_size)
    {
        ereport(ERROR, (errmsg("missing evoke lexical lookup suffix")));
    }
    if (lexical_term_count > 0)
    {
        if (((old_manifest->flags &
              EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) == 0 &&
             old_manifest->vocab_size != 0) ||
            next_manifest->vocab_size <= old_manifest->vocab_size ||
            lexical_term_count !=
                next_manifest->vocab_size - old_manifest->vocab_size)
        {
            ereport(
                ERROR,
                (errmsg("invalid evoke lexical catalog source"))
            );
        }
        for (uint32 term_index = 0;
             term_index < lexical_term_count;
             term_index++)
        {
            if (lexical_terms[term_index] == NULL)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke lexical catalog term"))
                );
            }
        }
    }
    status = evoke_segment_payload_validate(
        new_payload,
        next_manifest,
        new_descriptor
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW segment payload",
            status
        );
    }
}

static uint64
evoke_segment_pages_build_cow_lexicon_append(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    const char *const *lexical_terms,
    uint32 lexical_term_count,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_lexicon_cow_page_loader_context loader;
    evoke_lexicon_cow_update_stats update_stats;
    evoke_lexicon_cow_key *keys = NULL;
    bool old_has_lookup;
    uint64 hash_seed;
    evoke_status status;

    old_has_lookup =
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) != 0;
    hash_seed = old_has_lookup
        ? old_manifest->lexicon_hash_seed
        : evoke_segment_pages_initial_lexicon_hash_seed(
            index_relation,
            next_manifest
        );
    if (lexical_term_count == 0)
    {
        return hash_seed;
    }

    keys = palloc0(sizeof(*keys) * (Size) lexical_term_count);
    for (uint32 term_index = 0;
         term_index < lexical_term_count;
         term_index++)
    {
        size_t bytes_len = strlen(lexical_terms[term_index]);

        if (bytes_len == 0 || bytes_len > UINT32_MAX)
        {
            ereport(ERROR, (errmsg("invalid evoke lexical lookup term")));
        }
        keys[term_index].bytes =
            (const uint8 *) lexical_terms[term_index];
        keys[term_index].bytes_len = (uint32) bytes_len;
        keys[term_index].term_id =
            old_manifest->vocab_size + term_index;
    }
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    if (old_has_lookup)
    {
        status = evoke_segment_pages_open_lexicon_cow(
            index_relation,
            &old_manifest->lexicon_lookup,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_lexicon_cow_build_external_append_patch(
                &loader.root_object.ref,
                old_manifest->lexicon_hash_seed,
                old_manifest->vocab_size,
                keys,
                lexical_term_count,
                next_manifest->manifest_id,
                evoke_segment_pages_load_lexicon_cow_object,
                &loader,
                &cleanup->lexicon_cow_tree,
                &update_stats
            );
        }
    }
    else
    {
        status = evoke_lexicon_cow_tree_build(
            keys,
            lexical_term_count,
            hash_seed,
            next_manifest->manifest_id,
            &cleanup->lexicon_cow_tree
        );
    }
    pfree(keys);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW lexical-lookup append patch",
            status
        );
    }
    return hash_seed;
}

static void
evoke_segment_pages_write_cow_lexicon_append(
    Relation index_relation,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 lexical_term_count,
    uint64 hash_seed,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    bool old_has_lookup =
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) != 0;

    if (lexical_term_count == 0)
    {
        if (old_has_lookup)
        {
            next_manifest->lexicon_lookup = old_manifest->lexicon_lookup;
            next_manifest->lexicon_hash_seed =
                old_manifest->lexicon_hash_seed;
            next_manifest->flags |=
                EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP;
        }
        return;
    }
    evoke_segment_pages_write_lexicon_cow_objects_internal(
        index_relation,
        MAIN_FORKNUM,
        &cleanup->lexicon_cow_tree,
        next_manifest->manifest_id,
        reuse_arena,
        &next_manifest->lexicon_lookup
    );
    next_manifest->lexicon_hash_seed = hash_seed;
    next_manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP;
}

static void
evoke_segment_pages_build_cow_prefix_append(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    const char *const *lexical_terms,
    uint32 lexical_term_count,
    evoke_segment_bundle_write_cleanup *cleanup
)
{
    evoke_prefix_cow_page_loader_context loader;
    evoke_prefix_cow_update_stats update_stats;
    evoke_prefix_cow_key *keys = NULL;
    bool old_has_lookup;
    evoke_status status;

    old_has_lookup = (old_manifest->flags &
        EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) != 0;
    if (lexical_term_count == 0)
    {
        return;
    }
    keys = palloc0(sizeof(*keys) * (Size) lexical_term_count);
    for (uint32 term_index = 0;
         term_index < lexical_term_count;
         term_index++)
    {
        size_t bytes_len = strlen(lexical_terms[term_index]);

        if (bytes_len == 0 || bytes_len > UINT32_MAX)
        {
            ereport(ERROR, (errmsg("invalid evoke prefix lookup term")));
        }
        keys[term_index].bytes =
            (const uint8 *) lexical_terms[term_index];
        keys[term_index].bytes_len = (uint32) bytes_len;
        keys[term_index].term_id =
            old_manifest->vocab_size + term_index;
    }
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    if (old_has_lookup)
    {
        status = evoke_segment_pages_open_prefix_cow(
            index_relation,
            &old_manifest->prefix_lookup,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_prefix_cow_build_external_append_patch(
                &loader.root_object.ref,
                old_manifest->vocab_size,
                keys,
                lexical_term_count,
                next_manifest->manifest_id,
                evoke_segment_pages_load_prefix_cow_object,
                &loader,
                &cleanup->prefix_cow_tree,
                &update_stats
            );
        }
    }
    else
    {
        status = old_manifest->vocab_size == 0
            ? evoke_prefix_cow_tree_build(
                keys,
                lexical_term_count,
                next_manifest->manifest_id,
                &cleanup->prefix_cow_tree
            )
            : EVOKE_ERR_FORMAT;
    }
    pfree(keys);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW prefix-lookup append patch",
            status
        );
    }
}

static void
evoke_segment_pages_write_cow_prefix_append(
    Relation index_relation,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 lexical_term_count,
    evoke_segment_bundle_write_cleanup *cleanup,
    evoke_segment_page_reuse_arena *reuse_arena
)
{
    bool old_has_lookup = (old_manifest->flags &
        EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) != 0;

    if (lexical_term_count == 0)
    {
        if (old_has_lookup)
        {
            next_manifest->prefix_lookup = old_manifest->prefix_lookup;
            next_manifest->flags |=
                EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
        }
        return;
    }
    evoke_segment_pages_write_prefix_cow_objects_internal(
        index_relation,
        MAIN_FORKNUM,
        &cleanup->prefix_cow_tree,
        next_manifest->manifest_id,
        reuse_arena,
        &next_manifest->prefix_lookup
    );
    next_manifest->flags |= EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
}

static void
evoke_segment_pages_init_preflight_lexical_catalog_ref(
    uint64 owner_manifest_id,
    evoke_segment_object_ref *ref_out
)
{
    evoke_segment_object_ref ref = {0};

    if (owner_manifest_id == 0 || ref_out == NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke lexical-catalog preflight ref"))
        );
    }
    ref.object_kind = EVOKE_SEGMENT_OBJECT_LEXICAL_CATALOG;
    ref.start_block = 1;
    ref.page_count = 1;
    ref.object_id = owner_manifest_id;
    ref.owner_manifest_id = owner_manifest_id;
    ref.object_bytes = 1;
    ref.object_checksum = 1;
    if (evoke_segment_object_ref_validate(&ref, UINT32_MAX) != EVOKE_OK)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke lexical-catalog preflight ref"))
        );
    }
    *ref_out = ref;
}

static bool
evoke_segment_pages_term_retirements_equal(
    const evoke_term_cow_tree *left,
    const evoke_term_cow_tree *right
)
{
    if (left == NULL || right == NULL ||
        left->retired_range_count != right->retired_range_count)
    {
        return false;
    }
    for (size_t index = 0; index < left->retired_range_count; index++)
    {
        if (left->retired_ranges[index].start_block !=
                right->retired_ranges[index].start_block ||
            left->retired_ranges[index].block_count !=
                right->retired_ranges[index].block_count)
        {
            return false;
        }
    }
    return true;
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_append(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    const char *const *lexical_terms,
    uint32 lexical_term_count,
    const evoke_segment_payload *new_payload,
    const evoke_l0_storage_snapshot *retired_l0,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_term_cow_page_loader_context loader;
    evoke_term_cow_update_stats update_stats;
    evoke_document_cow_page_loader_context document_loader;
    evoke_document_cow_update_stats document_update_stats;
    evoke_document_cow_record *document_updates = NULL;
    size_t document_update_count = 0;
    evoke_term_cow_tree preflight_term_cow_tree;
    evoke_segment_page_reuse_arena handoff_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_page_reuse_arena *write_arena;
    evoke_term_cow_tree *term_patch;
    evoke_segment_cow_result result;
    evoke_segment_payload_view payload_view;
    evoke_segment_descriptor *new_descriptor;
    evoke_segment_object_ref lexical_catalog = {0};
    evoke_segment_object_ref preflight_lexical_catalog = {0};
    uint64 lexicon_hash_seed;
    bool has_document_updates;
    bool reader_fenced = reuse_arena != NULL &&
        reuse_arena->reader_fenced;
    bool needs_catalog_rebind = lexical_term_count > 0 && !reader_fenced;
    evoke_segment_cow_write_outcome outcome =
        EVOKE_SEGMENT_COW_WRITE_WRITTEN;
    evoke_status status;

    if (index_relation == NULL || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW segment write")));
    }
    evoke_segment_pages_require_cow_append(
        build_root,
        old_manifest,
        next_manifest,
        lexical_terms,
        lexical_term_count,
        new_payload,
        retired_l0
    );
    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    evoke_lexicon_cow_tree_init(&cleanup->lexicon_cow_tree);
    evoke_prefix_cow_tree_init(&cleanup->prefix_cow_tree);
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    memset(&document_loader, 0, sizeof(document_loader));
    memset(&document_update_stats, 0, sizeof(document_update_stats));
    evoke_term_cow_tree_init(&preflight_term_cow_tree);
    evoke_segment_page_reuse_arena_init(&handoff_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    evoke_segment_cow_result_init(&result);
    write_arena = reuse_arena != NULL ? reuse_arena : &staging_arena;
    term_patch = needs_catalog_rebind
        ? &preflight_term_cow_tree
        : &cleanup->term_cow_tree;
    new_descriptor =
        &next_manifest->segments[old_manifest->segment_count];
    evoke_segment_payload_as_view(new_payload, &payload_view);
    has_document_updates =
        new_payload->version_count > 0 ||
        new_payload->retirement_count > 0 ||
        new_payload->semantic_state_count > 0;

    PG_TRY();
    {
        if (lexical_term_count > 0)
        {
            if (needs_catalog_rebind)
            {
                /*
                 * The catalog ref is fixed-width metadata in each new term
                 * record, so its value cannot change patch topology or the
                 * retired object set. Use a valid unpublished ref to keep the
                 * non-fenced retirement preflight free of page writes.
                 */
                evoke_segment_pages_init_preflight_lexical_catalog_ref(
                    next_manifest->manifest_id,
                    &preflight_lexical_catalog
                );
            }
            else
            {
                evoke_segment_pages_write_lexical_catalog(
                    index_relation,
                    MAIN_FORKNUM,
                    next_manifest->manifest_id,
                    old_manifest->vocab_size,
                    lexical_term_count,
                    lexical_terms,
                    cleanup,
                    write_arena,
                    &lexical_catalog
                );
            }
        }
        status = evoke_segment_pages_open_term_cow(
            index_relation,
            &old_manifest->term_directory,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_term_cow_build_external_append_patch(
                &loader.root_object.ref,
                old_manifest,
                next_manifest,
                &payload_view,
                lexical_term_count == 0
                    ? NULL
                    : (
                        needs_catalog_rebind
                            ? &preflight_lexical_catalog
                            : &lexical_catalog
                    ),
                evoke_segment_pages_load_term_cow_object,
                &loader,
                term_patch,
                &update_stats
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term-directory append patch",
                status
            );
        }
        if (has_document_updates &&
            old_manifest->document_slot_count > 0)
        {
            status = evoke_segment_pages_open_document_cow(
                index_relation,
                &old_manifest->document_directory,
                build_root->published_block_high_watermark,
                old_manifest->manifest_id,
                &document_loader
            );
        }
        if (status == EVOKE_OK && has_document_updates)
        {
            status = evoke_segment_pages_build_document_updates(
                new_payload,
                old_manifest->document_slot_count,
                next_manifest->document_slot_count,
                old_manifest->document_slot_count > 0
                    ? evoke_segment_pages_load_document_cow_object
                    : NULL,
                old_manifest->document_slot_count > 0
                    ? &document_loader
                    : NULL,
                old_manifest->document_slot_count > 0
                    ? &document_loader.root_object.ref
                    : NULL,
                &document_updates,
                &document_update_count
            );
        }
        if (status == EVOKE_OK && has_document_updates &&
            old_manifest->document_slot_count == 0)
        {
            status = evoke_document_cow_tree_build(
                document_updates,
                document_update_count,
                next_manifest->manifest_id,
                &cleanup->document_cow_tree
            );
        }
        else if (status == EVOKE_OK && has_document_updates)
        {
            status = evoke_document_cow_build_external_patch(
                &document_loader.root_object.ref,
                old_manifest->document_slot_count,
                next_manifest->document_slot_count,
                document_updates,
                document_update_count,
                next_manifest->manifest_id,
                evoke_segment_pages_load_document_cow_object,
                &document_loader,
                &cleanup->document_cow_tree,
                &document_update_stats
            );
        }
        free(document_updates);
        document_updates = NULL;
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW document-directory append patch",
                status
            );
        }
        lexicon_hash_seed =
            evoke_segment_pages_build_cow_lexicon_append(
                index_relation,
                build_root,
                old_manifest,
                next_manifest,
                lexical_terms,
                lexical_term_count,
                cleanup
            );
        evoke_segment_pages_build_cow_prefix_append(
            index_relation,
            build_root,
            old_manifest,
            next_manifest,
            lexical_terms,
            lexical_term_count,
            cleanup
        );
        if (!reader_fenced &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
                0,
                0,
                term_patch,
                has_document_updates ? &cleanup->document_cow_tree : NULL,
                lexical_term_count > 0
                    ? &cleanup->lexicon_cow_tree
                    : NULL,
                lexical_term_count > 0
                    ? &cleanup->prefix_cow_tree
                    : NULL,
                retired_l0,
                reuse_arena,
                &result,
                next_manifest
            ))
        {
            status = old_manifest->retired_range_count > 0
                ? evoke_segment_page_reuse_arena_build_retired(
                      old_manifest->retired_ranges,
                      old_manifest->retired_range_count,
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  )
                : evoke_segment_page_reuse_arena_build_empty_fenced(
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "prepare append FSM handoff",
                    status
                );
            }
            if (!evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    term_patch,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    lexical_term_count > 0
                        ? &cleanup->lexicon_cow_tree
                        : NULL,
                    lexical_term_count > 0
                        ? &cleanup->prefix_cow_tree
                        : NULL,
                    retired_l0,
                    &handoff_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke prepared append retirement overflow"))
                );
            }
            outcome =
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED;
        }
        {
            if (needs_catalog_rebind)
            {
                evoke_segment_pages_write_lexical_catalog(
                    index_relation,
                    MAIN_FORKNUM,
                    next_manifest->manifest_id,
                    old_manifest->vocab_size,
                    lexical_term_count,
                    lexical_terms,
                    cleanup,
                    write_arena,
                    &lexical_catalog
                );
                status = evoke_term_cow_build_external_append_patch(
                    &loader.root_object.ref,
                    old_manifest,
                    next_manifest,
                    &payload_view,
                    &lexical_catalog,
                    evoke_segment_pages_load_term_cow_object,
                    &loader,
                    &cleanup->term_cow_tree,
                    &update_stats
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "COW term-directory catalog bind",
                        status
                    );
                }
                if (!evoke_segment_pages_term_retirements_equal(
                        &preflight_term_cow_tree,
                        &cleanup->term_cow_tree))
                {
                    ereport(
                        ERROR,
                        (errmsg(
                            "evoke term retirement changed after catalog bind"
                        ))
                    );
                }
            }
            evoke_segment_pages_write_cow_lexicon_append(
                index_relation,
                old_manifest,
                next_manifest,
                lexical_term_count,
                lexicon_hash_seed,
                cleanup,
                write_arena
            );
            evoke_segment_pages_write_cow_prefix_append(
                index_relation,
                old_manifest,
                next_manifest,
                lexical_term_count,
                cleanup,
                write_arena
            );
            evoke_segment_pages_write_payload(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                new_descriptor,
                new_payload,
                cleanup,
                write_arena
            );
            if (cleanup->term_cow_tree.object_count > 0)
            {
                evoke_segment_pages_write_term_cow_objects_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    &cleanup->term_cow_tree,
                    1,
                    next_manifest->manifest_id,
                    write_arena,
                    &next_manifest->term_directory
                );
            }
            else
            {
                status = evoke_term_cow_ref_as_segment_object_ref(
                    &cleanup->term_cow_tree.root,
                    &next_manifest->term_directory
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "reused COW term root",
                        status
                    );
                }
            }
            if (has_document_updates)
            {
                evoke_segment_pages_write_document_cow_objects_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    &cleanup->document_cow_tree,
                    next_manifest->manifest_id,
                    write_arena,
                    &next_manifest->document_directory
                );
            }
            else
            {
                next_manifest->document_directory =
                    old_manifest->document_directory;
            }
            next_manifest->flags &=
                ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
            next_manifest->flags |=
                EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY |
                EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
            if (reader_fenced &&
                !evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    &cleanup->term_cow_tree,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    lexical_term_count > 0
                        ? &cleanup->lexicon_cow_tree
                        : NULL,
                    lexical_term_count > 0
                        ? &cleanup->prefix_cow_tree
                        : NULL,
                    retired_l0,
                    reuse_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke reader-fenced retirement overflow"))
                );
            }
            evoke_segment_pages_write_manifest(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                cleanup,
                write_arena,
                &result.manifest
            );
            result.published_block_high_watermark =
                RelationGetNumberOfBlocks(index_relation);
            evoke_segment_pages_set_staged_writes(
                write_arena,
                result.published_block_high_watermark,
                &result
            );
            result.reused_block_count =
                evoke_segment_page_reuse_arena_reused_blocks(write_arena);
            status = evoke_segment_manifest_validate_published(
                next_manifest,
                &result.manifest,
                result.published_block_high_watermark
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW published manifest closure",
                    status
                );
            }
            *result_out = result;
            evoke_segment_cow_result_init(&result);
        }
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        free(document_updates);
        evoke_segment_cow_result_free(&result);
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_term_cow_tree_free(&preflight_term_cow_tree);
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
    evoke_segment_cow_result_free(&result);
    evoke_segment_page_reuse_arena_free(&handoff_arena);
    evoke_segment_page_reuse_arena_free(&staging_arena);
    evoke_term_cow_tree_free(&preflight_term_cow_tree);
    return outcome;
}

static void
evoke_segment_pages_require_cow_replace(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    uint32 first_segment_index,
    uint32 replaced_segment_count,
    const evoke_segment_payload *replaced_payloads,
    uint32 replaced_payload_count,
    const evoke_segment_payload *replacement_payload
)
{
    const evoke_segment_descriptor *replacement_descriptor;
    uint32 replaced_end;
    uint32 segment_index;
    evoke_status status;

    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL || replaced_payloads == NULL ||
        replacement_payload == NULL || replaced_segment_count < 2 ||
        replaced_payload_count != replaced_segment_count ||
        first_segment_index >= old_manifest->segment_count ||
        replaced_segment_count >
            old_manifest->segment_count - first_segment_index)
    {
        ereport(ERROR, (errmsg("invalid evoke COW segment replacement")));
    }
    status = evoke_segment_read_root_validate(build_root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    status = evoke_segment_manifest_validate_published(
        old_manifest,
        &build_root->manifest,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW replacement ancestor manifest",
            status
        );
    }
    replaced_end = first_segment_index + replaced_segment_count;
    if (build_root->root_id != old_manifest->manifest_id ||
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0 ||
        next_manifest->manifest_id != build_root->next_segment_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        next_manifest->segment_count !=
            old_manifest->segment_count - replaced_segment_count + 1 ||
        next_manifest->segments == NULL ||
        next_manifest->flags !=
            (old_manifest->flags &
             ~(EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY |
               EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY)) ||
        next_manifest->max_sequence != old_manifest->max_sequence ||
        next_manifest->statistics_epoch !=
            old_manifest->statistics_epoch ||
        next_manifest->visible_document_count !=
            old_manifest->visible_document_count ||
        next_manifest->document_slot_count !=
            old_manifest->document_slot_count ||
        next_manifest->total_document_length !=
            old_manifest->total_document_length ||
        next_manifest->reclaim_before_sequence !=
            old_manifest->reclaim_before_sequence ||
        next_manifest->vocab_size != old_manifest->vocab_size ||
        !evoke_segment_object_refs_equal(
            &next_manifest->query_contract,
            &old_manifest->query_contract) ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0 ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->term_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->neutral_fold,
            &old_manifest->neutral_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->impact_fold,
            &old_manifest->impact_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->document_directory,
            &old_manifest->document_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->lexicon_lookup,
            &old_manifest->lexicon_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->prefix_lookup,
            &old_manifest->prefix_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->semantic_accelerator_directory,
            &old_manifest->semantic_accelerator_directory) ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            next_manifest
        ) !=
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            ) ||
        next_manifest->lexicon_hash_seed !=
            old_manifest->lexicon_hash_seed ||
        next_manifest->neutral_fold_coverage !=
            old_manifest->neutral_fold_coverage ||
        next_manifest->impact_fold_coverage !=
            old_manifest->impact_fold_coverage ||
        next_manifest->impact_statistics_epoch !=
            old_manifest->impact_statistics_epoch ||
        next_manifest->doc_frequencies != NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW replacement transition"))
        );
    }
    for (segment_index = 0;
         segment_index < first_segment_index;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[segment_index]))
        {
            ereport(
                ERROR,
                (errmsg("evoke COW replacement prefix changed"))
            );
        }
    }
    for (segment_index = replaced_end;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        uint32 next_segment_index =
            segment_index - replaced_segment_count + 1;

        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[next_segment_index]))
        {
            ereport(
                ERROR,
                (errmsg("evoke COW replacement suffix changed"))
            );
        }
    }

    replacement_descriptor =
        &next_manifest->segments[first_segment_index];
    if (replacement_descriptor->segment_id !=
            next_manifest->manifest_id ||
        replacement_descriptor->start_block != 0 ||
        replacement_descriptor->block_count != 0 ||
        replacement_descriptor->payload_bytes != 0 ||
        replacement_descriptor->payload_checksum != 0 ||
        replacement_descriptor->payload_owner_manifest_id != 0)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW replacement descriptor"))
        );
    }
    status = evoke_segment_payload_validate(
        replacement_payload,
        next_manifest,
        replacement_descriptor
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW replacement payload",
            status
        );
    }
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_replace(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 first_segment_index,
    uint32 replaced_segment_count,
    const evoke_segment_payload *replaced_payloads,
    uint32 replaced_payload_count,
    const evoke_segment_payload *replacement_payload,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_term_cow_page_loader_context loader;
    evoke_term_cow_update_stats update_stats;
    evoke_document_cow_page_loader_context document_loader;
    evoke_document_cow_update_stats document_update_stats;
    evoke_document_cow_record *document_updates = NULL;
    size_t document_update_count = 0;
    evoke_segment_page_reuse_arena handoff_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_page_reuse_arena *write_arena;
    evoke_segment_cow_result result;
    evoke_segment_payload_view payload_view;
    evoke_segment_descriptor *replacement_descriptor;
    bool has_document_updates;
    bool reader_fenced = reuse_arena != NULL &&
        reuse_arena->reader_fenced;
    evoke_segment_cow_write_outcome outcome =
        EVOKE_SEGMENT_COW_WRITE_WRITTEN;
    evoke_status status;

    if (index_relation == NULL || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW replacement write")));
    }
    evoke_segment_pages_require_cow_replace(
        build_root,
        old_manifest,
        next_manifest,
        first_segment_index,
        replaced_segment_count,
        replaced_payloads,
        replaced_payload_count,
        replacement_payload
    );
    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    memset(&document_loader, 0, sizeof(document_loader));
    memset(&document_update_stats, 0, sizeof(document_update_stats));
    evoke_segment_page_reuse_arena_init(&handoff_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    evoke_segment_cow_result_init(&result);
    write_arena = reuse_arena != NULL ? reuse_arena : &staging_arena;
    has_document_updates = false;
    replacement_descriptor =
        &next_manifest->segments[first_segment_index];
    evoke_segment_payload_as_view(
        replacement_payload,
        &payload_view
    );

    PG_TRY();
    {
        cleanup->payload_views = palloc0(
            (Size) replaced_payload_count *
            sizeof(*cleanup->payload_views)
        );
        for (uint32 payload_index = 0;
             payload_index < replaced_payload_count;
             payload_index++)
        {
            evoke_segment_payload_as_view(
                &replaced_payloads[payload_index],
                &cleanup->payload_views[payload_index]
            );
        }
        status = evoke_segment_pages_open_term_cow(
            index_relation,
            &old_manifest->term_directory,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_term_cow_build_external_replace_patch(
                &loader.root_object.ref,
                old_manifest,
                next_manifest,
                first_segment_index,
                cleanup->payload_views,
                replaced_payload_count,
                &payload_view,
                evoke_segment_pages_load_term_cow_object,
                &loader,
                &cleanup->term_cow_tree,
                &update_stats
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term-directory replacement patch",
                status
            );
        }
        if (old_manifest->document_slot_count > 0)
        {
            status = evoke_segment_pages_open_document_cow(
                index_relation,
                &old_manifest->document_directory,
                build_root->published_block_high_watermark,
                old_manifest->manifest_id,
                &document_loader
            );
        }
        if (status == EVOKE_OK)
        {
            status =
                evoke_segment_pages_build_replacement_document_updates(
                    replaced_payloads,
                    replaced_payload_count,
                    replacement_payload,
                    old_manifest->document_slot_count,
                    old_manifest->document_slot_count > 0
                        ? evoke_segment_pages_load_document_cow_object
                        : NULL,
                    old_manifest->document_slot_count > 0
                        ? &document_loader
                        : NULL,
                    old_manifest->document_slot_count > 0
                        ? &document_loader.root_object.ref
                        : NULL,
                    &document_updates,
                    &document_update_count
                );
        }
        has_document_updates = document_update_count > 0;
        if (status == EVOKE_OK && has_document_updates)
        {
            status = evoke_document_cow_build_external_patch(
                &document_loader.root_object.ref,
                old_manifest->document_slot_count,
                next_manifest->document_slot_count,
                document_updates,
                document_update_count,
                next_manifest->manifest_id,
                evoke_segment_pages_load_document_cow_object,
                &document_loader,
                &cleanup->document_cow_tree,
                &document_update_stats
            );
        }
        free(document_updates);
        document_updates = NULL;
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW document-directory replacement patch",
                status
            );
        }
        if (!reader_fenced &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
                first_segment_index,
                replaced_segment_count,
                &cleanup->term_cow_tree,
                has_document_updates ? &cleanup->document_cow_tree : NULL,
                NULL,
                NULL,
                NULL,
                reuse_arena,
                &result,
                next_manifest
            ))
        {
            status = old_manifest->retired_range_count > 0
                ? evoke_segment_page_reuse_arena_build_retired(
                      old_manifest->retired_ranges,
                      old_manifest->retired_range_count,
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  )
                : evoke_segment_page_reuse_arena_build_empty_fenced(
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "prepare replacement FSM handoff",
                    status
                );
            }
            if (!evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    first_segment_index,
                    replaced_segment_count,
                    &cleanup->term_cow_tree,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    NULL,
                    NULL,
                    NULL,
                    &handoff_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg(
                        "evoke prepared replacement retirement overflow"
                    ))
                );
            }
            outcome =
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED;
        }
        {
            evoke_segment_pages_write_payload(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                replacement_descriptor,
                replacement_payload,
                cleanup,
                write_arena
            );
            if (cleanup->term_cow_tree.object_count > 0)
            {
                evoke_segment_pages_write_term_cow_objects_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    &cleanup->term_cow_tree,
                    1,
                    next_manifest->manifest_id,
                    write_arena,
                    &next_manifest->term_directory
                );
            }
            else
            {
                status = evoke_term_cow_ref_as_segment_object_ref(
                    &cleanup->term_cow_tree.root,
                    &next_manifest->term_directory
                );
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "reused COW replacement root",
                        status
                    );
                }
            }
            if (has_document_updates)
            {
                evoke_segment_pages_write_document_cow_objects_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    &cleanup->document_cow_tree,
                    next_manifest->manifest_id,
                    write_arena,
                    &next_manifest->document_directory
                );
            }
            else
            {
                next_manifest->document_directory =
                    old_manifest->document_directory;
            }
            next_manifest->flags &=
                ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
            next_manifest->flags |=
                EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY |
                EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
            free(next_manifest->doc_frequencies);
            next_manifest->doc_frequencies = NULL;
            if (reader_fenced &&
                !evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    first_segment_index,
                    replaced_segment_count,
                    &cleanup->term_cow_tree,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    NULL,
                    NULL,
                    NULL,
                    reuse_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke reader-fenced retirement overflow"))
                );
            }
            evoke_segment_pages_write_manifest(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                cleanup,
                write_arena,
                &result.manifest
            );
            result.published_block_high_watermark =
                RelationGetNumberOfBlocks(index_relation);
            evoke_segment_pages_set_staged_writes(
                write_arena,
                result.published_block_high_watermark,
                &result
            );
            result.reused_block_count =
                evoke_segment_page_reuse_arena_reused_blocks(write_arena);
            status = evoke_segment_manifest_validate_published(
                next_manifest,
                &result.manifest,
                result.published_block_high_watermark
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW replacement manifest closure",
                    status
                );
            }
            *result_out = result;
            evoke_segment_cow_result_init(&result);
        }
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        free(document_updates);
        evoke_segment_cow_result_free(&result);
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
    evoke_segment_page_reuse_arena_free(&handoff_arena);
    evoke_segment_page_reuse_arena_free(&staging_arena);
    return outcome;
}

static void
evoke_segment_pages_require_cow_document_patch(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    const evoke_document_cow_record *updates,
    size_t update_count
)
{
    evoke_status status;

    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL || updates == NULL || update_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document patch")));
    }
    status = evoke_segment_read_root_validate(build_root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    status = evoke_segment_manifest_validate_published(
        old_manifest,
        &build_root->manifest,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW document-patch ancestor manifest",
            status
        );
    }
    if (build_root->root_id != old_manifest->manifest_id ||
        next_manifest->manifest_id != build_root->next_segment_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        next_manifest->flags != old_manifest->flags ||
        next_manifest->max_sequence != old_manifest->max_sequence ||
        old_manifest->statistics_epoch == UINT64_MAX ||
        next_manifest->statistics_epoch !=
            old_manifest->statistics_epoch + 1 ||
        next_manifest->visible_document_count >
            old_manifest->visible_document_count ||
        next_manifest->document_slot_count <
            old_manifest->document_slot_count ||
        next_manifest->document_slot_count >
            build_root->next_document_slot ||
        next_manifest->total_document_length >
            old_manifest->total_document_length ||
        next_manifest->reclaim_before_sequence !=
            old_manifest->reclaim_before_sequence ||
        next_manifest->neutral_fold_coverage !=
            old_manifest->neutral_fold_coverage ||
        next_manifest->impact_fold_coverage !=
            old_manifest->impact_fold_coverage ||
        next_manifest->impact_statistics_epoch !=
            old_manifest->impact_statistics_epoch ||
        next_manifest->vocab_size != old_manifest->vocab_size ||
        !evoke_segment_object_refs_equal(
            &next_manifest->query_contract,
            &old_manifest->query_contract) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->term_directory,
            &old_manifest->term_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->neutral_fold,
            &old_manifest->neutral_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->impact_fold,
            &old_manifest->impact_fold) ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->document_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->lexicon_lookup,
            &old_manifest->lexicon_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->prefix_lookup,
            &old_manifest->prefix_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->semantic_accelerator_directory,
            &old_manifest->semantic_accelerator_directory) ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            next_manifest
        ) !=
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            ) ||
        next_manifest->lexicon_hash_seed !=
            old_manifest->lexicon_hash_seed ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0 ||
        next_manifest->retired_ranges != NULL ||
        next_manifest->retired_range_count != 0 ||
        next_manifest->segment_count != old_manifest->segment_count ||
        (next_manifest->segment_count > 0 &&
         next_manifest->segments == NULL) ||
        next_manifest->doc_frequencies != NULL ||
        old_manifest->doc_frequencies != NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW document-patch transition"))
        );
    }
    for (uint32 segment_index = 0;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &next_manifest->segments[segment_index],
                &old_manifest->segments[segment_index]))
        {
            ereport(
                ERROR,
                (errmsg("evoke COW document patch changed a segment"))
            );
        }
    }
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_document_patch(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    const evoke_document_cow_record *updates,
    size_t update_count,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_document_cow_page_loader_context loader;
    evoke_document_cow_update_stats update_stats;
    evoke_segment_page_reuse_arena handoff_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_page_reuse_arena *write_arena;
    evoke_segment_cow_result result;
    evoke_segment_cow_write_outcome outcome =
        EVOKE_SEGMENT_COW_WRITE_WRITTEN;
    bool reader_fenced = reuse_arena != NULL &&
        reuse_arena->reader_fenced;
    evoke_status status;

    if (index_relation == NULL || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW document-patch write")));
    }
    evoke_segment_pages_require_cow_document_patch(
        build_root,
        old_manifest,
        next_manifest,
        updates,
        update_count
    );
    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    evoke_lexicon_cow_tree_init(&cleanup->lexicon_cow_tree);
    evoke_prefix_cow_tree_init(&cleanup->prefix_cow_tree);
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    evoke_segment_page_reuse_arena_init(&handoff_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    evoke_segment_cow_result_init(&result);
    write_arena = reuse_arena != NULL ? reuse_arena : &staging_arena;

    PG_TRY();
    {
        if (old_manifest->document_slot_count == 0)
        {
            status = evoke_document_cow_tree_build(
                updates,
                update_count,
                next_manifest->manifest_id,
                &cleanup->document_cow_tree
            );
        }
        else
        {
            status = evoke_segment_pages_open_document_cow(
                index_relation,
                &old_manifest->document_directory,
                build_root->published_block_high_watermark,
                old_manifest->manifest_id,
                &loader
            );
            if (status == EVOKE_OK)
            {
                status = evoke_document_cow_build_external_patch(
                    &loader.root_object.ref,
                    old_manifest->document_slot_count,
                    next_manifest->document_slot_count,
                    updates,
                    update_count,
                    next_manifest->manifest_id,
                    evoke_segment_pages_load_document_cow_object,
                    &loader,
                    &cleanup->document_cow_tree,
                    &update_stats
                );
            }
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW document patch",
                status
            );
        }
        if (!reader_fenced &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
            0,
            0,
            NULL,
            &cleanup->document_cow_tree,
            NULL,
            NULL,
            NULL,
            reuse_arena,
            &result,
            next_manifest
        ))
        {
            status = old_manifest->retired_range_count > 0
                ? evoke_segment_page_reuse_arena_build_retired(
                      old_manifest->retired_ranges,
                      old_manifest->retired_range_count,
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  )
                : evoke_segment_page_reuse_arena_build_empty_fenced(
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "prepare document-patch FSM handoff",
                    status
                );
            }
            if (!evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    NULL,
                    &cleanup->document_cow_tree,
                    NULL,
                    NULL,
                    NULL,
                    &handoff_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg(
                        "evoke prepared document-patch retirement overflow"
                    ))
                );
            }
            outcome =
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED;
        }
        {
            evoke_segment_pages_write_document_cow_objects_internal(
                index_relation,
                MAIN_FORKNUM,
                &cleanup->document_cow_tree,
                next_manifest->manifest_id,
                write_arena,
                &next_manifest->document_directory
            );
            if (reader_fenced &&
                !evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    NULL,
                    &cleanup->document_cow_tree,
                    NULL,
                    NULL,
                    NULL,
                    reuse_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke reader-fenced retirement overflow"))
                );
            }
            evoke_segment_pages_write_manifest(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                cleanup,
                write_arena,
                &result.manifest
            );
            result.published_block_high_watermark =
                RelationGetNumberOfBlocks(index_relation);
            evoke_segment_pages_set_staged_writes(
                write_arena,
                result.published_block_high_watermark,
                &result
            );
            result.reused_block_count =
                evoke_segment_page_reuse_arena_reused_blocks(write_arena);
            status = evoke_segment_manifest_validate_published(
                next_manifest,
                &result.manifest,
                result.published_block_high_watermark
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW document-patch manifest closure",
                    status
                );
            }
            *result_out = result;
            evoke_segment_cow_result_init(&result);
        }
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        evoke_segment_cow_result_free(&result);
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
    evoke_segment_page_reuse_arena_free(&handoff_arena);
    evoke_segment_page_reuse_arena_free(&staging_arena);
    return outcome;
}

static void
evoke_segment_pages_require_cow_reclaim(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    uint32 reclaim_block_limit
)
{
    evoke_status status;

    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL || reclaim_block_limit == 0 ||
        old_manifest->retired_range_count == 0 ||
        build_root->published_block_high_watermark == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke COW reclamation request")));
    }
    status = evoke_segment_read_root_validate(build_root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    status = evoke_segment_manifest_validate_published(
        old_manifest,
        &build_root->manifest,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW reclamation ancestor manifest",
            status
        );
    }
    if (build_root->root_id != old_manifest->manifest_id ||
        next_manifest->manifest_id != build_root->next_segment_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        next_manifest->flags != old_manifest->flags ||
        next_manifest->max_sequence != old_manifest->max_sequence ||
        next_manifest->statistics_epoch !=
            old_manifest->statistics_epoch ||
        next_manifest->visible_document_count !=
            old_manifest->visible_document_count ||
        next_manifest->document_slot_count !=
            old_manifest->document_slot_count ||
        next_manifest->total_document_length !=
            old_manifest->total_document_length ||
        next_manifest->reclaim_before_sequence !=
            old_manifest->reclaim_before_sequence ||
        next_manifest->neutral_fold_coverage !=
            old_manifest->neutral_fold_coverage ||
        next_manifest->impact_fold_coverage !=
            old_manifest->impact_fold_coverage ||
        next_manifest->impact_statistics_epoch !=
            old_manifest->impact_statistics_epoch ||
        next_manifest->vocab_size != old_manifest->vocab_size ||
        !evoke_segment_object_refs_equal(
            &next_manifest->query_contract,
            &old_manifest->query_contract) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->term_directory,
            &old_manifest->term_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->neutral_fold,
            &old_manifest->neutral_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->impact_fold,
            &old_manifest->impact_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->document_directory,
            &old_manifest->document_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->lexicon_lookup,
            &old_manifest->lexicon_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->prefix_lookup,
            &old_manifest->prefix_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->semantic_accelerator_directory,
            &old_manifest->semantic_accelerator_directory) ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            next_manifest
        ) !=
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            ) ||
        next_manifest->lexicon_hash_seed !=
            old_manifest->lexicon_hash_seed ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0 ||
        next_manifest->retired_ranges != NULL ||
        next_manifest->retired_range_count != 0 ||
        next_manifest->segment_count != old_manifest->segment_count ||
        (next_manifest->segment_count > 0 &&
         next_manifest->segments == NULL) ||
        next_manifest->doc_frequencies != NULL ||
        old_manifest->doc_frequencies != NULL)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke COW reclamation transition"))
        );
    }
    for (uint32 segment_index = 0;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &next_manifest->segments[segment_index],
                &old_manifest->segments[segment_index]))
        {
            ereport(
                ERROR,
                (errmsg("evoke COW reclamation changed a segment"))
            );
        }
    }
}

static void
evoke_segment_pages_prepare_cow_reclaim_ranges(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    const evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result
)
{
    evoke_block_range_inventory remaining;
    evoke_block_range_inventory handoff;
    size_t reclaim_index = 0;
    evoke_status status = EVOKE_OK;

    evoke_block_range_inventory_init(&remaining);
    evoke_block_range_inventory_init(&handoff);
    for (uint32 old_index = 0;
         status == EVOKE_OK &&
         old_index < old_manifest->retired_range_count;
         old_index++)
    {
        const evoke_block_range *old_range =
            &old_manifest->retired_ranges[old_index];
        uint64 cursor = old_range->start_block;
        uint64 old_end = cursor + old_range->block_count;

        while (reclaim_index < reuse_arena->allocator.range_count)
        {
            const evoke_block_range *reclaimed =
                &reuse_arena->allocator.ranges[reclaim_index];
            uint64 reclaimed_end =
                (uint64) reclaimed->start_block + reclaimed->block_count;

            if (reclaimed->start_block >= old_end)
            {
                break;
            }
            if (reclaimed->start_block < cursor ||
                reclaimed_end > old_end)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            if (reclaimed->start_block > cursor)
            {
                status = evoke_block_range_inventory_add(
                    &remaining,
                    (uint32) cursor,
                    reclaimed->start_block - (uint32) cursor
                );
                if (status != EVOKE_OK)
                {
                    break;
                }
            }
            cursor = reclaimed_end;
            reclaim_index++;
        }
        if (status == EVOKE_OK && cursor < old_end)
        {
            status = evoke_block_range_inventory_add(
                &remaining,
                (uint32) cursor,
                (uint32) (old_end - cursor)
            );
        }
    }
    if (status == EVOKE_OK &&
        reclaim_index != reuse_arena->allocator.range_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        status = evoke_block_range_inventory_add(
            &remaining,
            build_root->manifest.start_block,
            build_root->manifest.page_count
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_block_range_inventory_finalize(
            &remaining,
            reuse_arena->source_high_watermark
        );
    }
    if (status == EVOKE_OK &&
        remaining.range_count > EVOKE_SEGMENT_MANIFEST_MAX_RETIRED_RANGES)
    {
        status = EVOKE_ERR_RANGE;
    }
    if (status == EVOKE_OK && remaining.range_count > 0)
    {
        next_manifest->retired_ranges = calloc(
            remaining.range_count,
            sizeof(*next_manifest->retired_ranges)
        );
        if (next_manifest->retired_ranges == NULL)
        {
            status = EVOKE_ERR_NOMEM;
        }
        else
        {
            memcpy(
                next_manifest->retired_ranges,
                remaining.ranges,
                remaining.range_count *
                    sizeof(*next_manifest->retired_ranges)
            );
            next_manifest->retired_range_count =
                (uint32) remaining.range_count;
        }
    }
    if (status == EVOKE_OK)
    {
        evoke_segment_pages_add_retired_ranges(
            &handoff,
            reuse_arena->allocator.ranges,
            reuse_arena->allocator.range_count,
            &status
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_block_range_inventory_finalize(
            &handoff,
            reuse_arena->source_high_watermark
        );
    }
    if (status == EVOKE_OK)
    {
        evoke_segment_pages_set_fsm_handoff(&handoff, result);
    }
    evoke_block_range_inventory_free(&handoff);
    evoke_block_range_inventory_free(&remaining);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW reclamation range partition",
            status
        );
    }
}

void
evoke_segment_pages_write_cow_reclaim(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 reclaim_block_limit,
    evoke_segment_cow_result *result_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_segment_page_reuse_arena reclaim_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_cow_result result;
    evoke_block_range *bounded_ranges = NULL;
    size_t bounded_range_count = 0;
    uint32 remaining = reclaim_block_limit;
    evoke_status status;

    if (index_relation == NULL || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW reclamation write")));
    }
    evoke_segment_pages_require_cow_reclaim(
        build_root,
        old_manifest,
        next_manifest,
        reclaim_block_limit
    );
    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    evoke_lexicon_cow_tree_init(&cleanup->lexicon_cow_tree);
    evoke_prefix_cow_tree_init(&cleanup->prefix_cow_tree);
    evoke_segment_page_reuse_arena_init(&reclaim_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    evoke_segment_cow_result_init(&result);

    PG_TRY();
    {
        bounded_ranges = calloc(
            old_manifest->retired_range_count,
            sizeof(*bounded_ranges)
        );
        if (bounded_ranges == NULL)
        {
            ereport(ERROR, (errmsg("out of memory")));
        }
        for (uint32 range_index = 0;
             range_index < old_manifest->retired_range_count &&
             remaining > 0;
             range_index++)
        {
            const evoke_block_range *source =
                &old_manifest->retired_ranges[range_index];
            evoke_block_range *bounded =
                &bounded_ranges[bounded_range_count++];

            bounded->start_block = source->start_block;
            bounded->block_count = Min(source->block_count, remaining);
            remaining -= bounded->block_count;
        }
        status = evoke_segment_page_reuse_arena_build_retired(
            bounded_ranges,
            bounded_range_count,
            build_root->published_block_high_watermark,
            &reclaim_arena
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "prepare COW reclamation handoff",
                status
            );
        }
        evoke_segment_pages_prepare_cow_reclaim_ranges(
            build_root,
            old_manifest,
            next_manifest,
            &reclaim_arena,
            &result
        );
        evoke_segment_pages_write_manifest(
            index_relation,
            MAIN_FORKNUM,
            next_manifest,
            cleanup,
            &staging_arena,
            &result.manifest
        );
        result.published_block_high_watermark =
            RelationGetNumberOfBlocks(index_relation);
        evoke_segment_pages_set_staged_writes(
            &staging_arena,
            result.published_block_high_watermark,
            &result
        );
        result.reused_block_count =
            evoke_segment_page_reuse_arena_reused_blocks(&staging_arena);
        status = evoke_segment_manifest_validate_published(
            next_manifest,
            &result.manifest,
            result.published_block_high_watermark
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW reclamation manifest closure",
                status
            );
        }
        *result_out = result;
        evoke_segment_cow_result_init(&result);
        free(bounded_ranges);
        bounded_ranges = NULL;
        evoke_segment_page_reuse_arena_free(&reclaim_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        free(bounded_ranges);
        evoke_segment_cow_result_free(&result);
        evoke_segment_page_reuse_arena_free(&reclaim_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

static void
evoke_segment_pages_require_cow_term_fold(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    uint32 term_id,
    bool impact,
    evoke_term_cow_neutral_fold_level fold_level,
    const evoke_term_fold_bundle *fold_bundle
)
{
    uint64 coverage_sequence;
    evoke_status status;

    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL || fold_bundle == NULL ||
        fold_bundle->run_count == 0 ||
        term_id >= old_manifest->vocab_size ||
        (!impact &&
         fold_level != EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR &&
         fold_level != EVOKE_TERM_COW_NEUTRAL_FOLD_MINOR))
    {
        ereport(ERROR, (errmsg("invalid evoke COW term fold")));
    }
    status = evoke_segment_read_root_validate(build_root);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error("read root", status);
    }
    status = evoke_segment_manifest_validate_published(
        old_manifest,
        &build_root->manifest,
        build_root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW fold ancestor manifest",
            status
        );
    }
    status = evoke_term_fold_bundle_validate(fold_bundle);
    if (status != EVOKE_OK)
    {
        evoke_segment_pages_report_codec_error(
            "COW term fold bundle",
            status
        );
    }
    coverage_sequence = fold_bundle->runs[0].coverage_sequence;
    if (build_root->root_id != old_manifest->manifest_id ||
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0 ||
        next_manifest->manifest_id != build_root->next_segment_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        next_manifest->segment_count != old_manifest->segment_count ||
        (next_manifest->segment_count > 0 &&
         next_manifest->segments == NULL) ||
        next_manifest->flags !=
            (old_manifest->flags &
             ~(EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY |
               EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY)) ||
        next_manifest->max_sequence != old_manifest->max_sequence ||
        next_manifest->statistics_epoch !=
            old_manifest->statistics_epoch ||
        next_manifest->visible_document_count !=
            old_manifest->visible_document_count ||
        next_manifest->document_slot_count !=
            old_manifest->document_slot_count ||
        next_manifest->total_document_length !=
            old_manifest->total_document_length ||
        next_manifest->reclaim_before_sequence !=
            old_manifest->reclaim_before_sequence ||
        next_manifest->vocab_size != old_manifest->vocab_size ||
        !evoke_segment_object_refs_equal(
            &next_manifest->query_contract,
            &old_manifest->query_contract) ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0 ||
        !evoke_segment_object_ref_is_zero(
            &next_manifest->term_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->neutral_fold,
            &old_manifest->neutral_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->impact_fold,
            &old_manifest->impact_fold) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->document_directory,
            &old_manifest->document_directory) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->lexicon_lookup,
            &old_manifest->lexicon_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->prefix_lookup,
            &old_manifest->prefix_lookup) ||
        !evoke_segment_object_refs_equal(
            &next_manifest->semantic_accelerator_directory,
            &old_manifest->semantic_accelerator_directory) ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            next_manifest
        ) !=
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            ) ||
        next_manifest->lexicon_hash_seed !=
            old_manifest->lexicon_hash_seed ||
        next_manifest->neutral_fold_coverage !=
            old_manifest->neutral_fold_coverage ||
        next_manifest->impact_fold_coverage !=
            old_manifest->impact_fold_coverage ||
        next_manifest->impact_statistics_epoch !=
            old_manifest->impact_statistics_epoch ||
        next_manifest->doc_frequencies != NULL ||
        fold_bundle->object_kind != (impact
            ? EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
            : EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD) ||
        fold_bundle->owner_manifest_id != next_manifest->manifest_id ||
        (impact &&
         (build_root->active_l0.record_count != 0 ||
          build_root->pending_l0.record_count != 0 ||
          fold_bundle->statistics_epoch !=
            old_manifest->statistics_epoch)) ||
        coverage_sequence == 0 ||
        coverage_sequence > old_manifest->max_sequence)
    {
        ereport(ERROR, (errmsg("invalid evoke COW fold transition")));
    }
    for (uint32 segment_index = 0;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[segment_index]))
        {
            ereport(ERROR, (errmsg("evoke COW fold segment set changed")));
        }
    }
    for (uint32 run_index = 0;
         run_index < fold_bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run =
            &fold_bundle->runs[run_index];

        if (run->term_id != term_id ||
            run->coverage_sequence != coverage_sequence)
        {
            ereport(
                ERROR,
                (errmsg("evoke COW fold bundle spans multiple terms"))
            );
        }
    }
}

static evoke_status
evoke_segment_pages_load_replaced_term_folds(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_term_cow_record *record,
    uint32 term_id,
    bool impact,
    evoke_term_cow_neutral_fold_level fold_level,
    evoke_term_fold_bundle *folds,
    size_t *fold_count_out
)
{
    const evoke_segment_object_ref *refs[2] = {NULL, NULL};
    evoke_segment_object_kind expected_kind;
    size_t ref_count;
    size_t fold_count = 0;

    if (index_relation == NULL || root == NULL || manifest == NULL ||
        record == NULL || folds == NULL || fold_count_out == NULL ||
        record->term_id != term_id)
    {
        return EVOKE_ERR_INVALID;
    }
    if (impact)
    {
        refs[0] = &record->impact_fold;
        ref_count = 1;
        expected_kind = EVOKE_SEGMENT_OBJECT_IMPACT_FOLD;
    }
    else if (fold_level == EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR)
    {
        refs[0] = &record->neutral_fold;
        refs[1] = &record->neutral_minor_fold;
        ref_count = 2;
        expected_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    }
    else if (fold_level == EVOKE_TERM_COW_NEUTRAL_FOLD_MINOR)
    {
        refs[0] = &record->neutral_minor_fold;
        ref_count = 1;
        expected_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    }
    else
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t ref_index = 0; ref_index < ref_count; ref_index++)
    {
        size_t term_posting_count;
        evoke_status status;

        if (evoke_segment_object_ref_is_zero(refs[ref_index]))
        {
            continue;
        }
        if (ref_index > 0 &&
            evoke_segment_object_refs_equal(refs[0], refs[ref_index]))
        {
            continue;
        }
        status = evoke_segment_pages_read_fold_bundle(
            index_relation,
            root,
            manifest,
            refs[ref_index],
            expected_kind,
            &folds[fold_count]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_segment_pages_fold_term_posting_count(
            &folds[fold_count],
            term_id,
            &term_posting_count
        );
        if (status != EVOKE_OK || term_posting_count == 0)
        {
            evoke_term_fold_bundle_free(&folds[fold_count]);
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        fold_count++;
    }
    *fold_count_out = fold_count;
    return EVOKE_OK;
}

static void
evoke_segment_pages_init_preflight_fold_ref(
    uint64 owner_manifest_id,
    uint32 term_id,
    bool impact,
    evoke_segment_object_ref *ref_out
)
{
    evoke_segment_object_ref ref = {0};

    if (owner_manifest_id == 0 || ref_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke fold preflight ref")));
    }
    ref.object_kind = impact
        ? EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
        : EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    ref.start_block = 1;
    ref.page_count = 1;
    ref.object_id = (uint64) term_id + 1;
    ref.owner_manifest_id = owner_manifest_id;
    ref.object_bytes = 1;
    ref.object_checksum = 1;
    if (evoke_segment_object_ref_validate(&ref, UINT32_MAX) != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("invalid evoke fold preflight ref")));
    }
    *ref_out = ref;
}

static evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_term_fold(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 term_id,
    bool impact,
    evoke_term_cow_neutral_fold_level fold_level,
    const evoke_term_fold_bundle *fold_bundle,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    evoke_segment_bundle_write_cleanup *cleanup;
    evoke_term_cow_page_loader_context loader;
    evoke_term_cow_update_stats update_stats;
    evoke_document_cow_page_loader_context document_loader;
    evoke_document_cow_update_stats document_update_stats;
    evoke_document_cow_record *document_updates = NULL;
    size_t document_update_count = 0;
    evoke_term_cow_tree preflight_term_cow_tree;
    evoke_segment_page_reuse_arena handoff_arena;
    evoke_segment_page_reuse_arena staging_arena;
    evoke_segment_page_reuse_arena *write_arena;
    evoke_term_cow_tree *term_patch;
    evoke_segment_cow_result result;
    evoke_segment_object_ref fold_ref;
    evoke_segment_object_ref preflight_fold_ref;
    evoke_segment_read_root next_root;
    evoke_segment_manifest checked_manifest;
    evoke_term_cow_record old_term_record;
    evoke_term_fold_bundle old_folds[2];
    size_t old_fold_count = 0;
    evoke_term_fold_bundle checked_fold;
    uint64 fold_checksum = 0;
    size_t fold_size = 0;
    bool has_document_updates = false;
    bool reader_fenced = reuse_arena != NULL &&
        reuse_arena->reader_fenced && reuse_arena->allocator.initialized;
    evoke_segment_cow_write_outcome outcome =
        EVOKE_SEGMENT_COW_WRITE_WRITTEN;
    evoke_status status;

    if (index_relation == NULL || result_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW term fold write")));
    }
    if (reuse_arena != NULL && !reader_fenced)
    {
        ereport(ERROR, (errmsg("invalid evoke fold reuse arena")));
    }
    evoke_segment_pages_require_cow_term_fold(
        build_root,
        old_manifest,
        next_manifest,
        term_id,
        impact,
        fold_level,
        fold_bundle
    );
    cleanup = palloc0(sizeof(*cleanup));
    evoke_term_directory_init(&cleanup->term_directory);
    evoke_term_cow_tree_init(&cleanup->term_cow_tree);
    evoke_document_cow_tree_init(&cleanup->document_cow_tree);
    memset(&loader, 0, sizeof(loader));
    memset(&update_stats, 0, sizeof(update_stats));
    memset(&document_loader, 0, sizeof(document_loader));
    memset(&document_update_stats, 0, sizeof(document_update_stats));
    evoke_term_cow_tree_init(&preflight_term_cow_tree);
    evoke_segment_page_reuse_arena_init(&handoff_arena);
    evoke_segment_page_reuse_arena_init(&staging_arena);
    evoke_segment_cow_result_init(&result);
    write_arena = reuse_arena != NULL ? reuse_arena : &staging_arena;
    memset(&fold_ref, 0, sizeof(fold_ref));
    memset(&preflight_fold_ref, 0, sizeof(preflight_fold_ref));
    memset(&old_term_record, 0, sizeof(old_term_record));
    evoke_segment_manifest_init(&checked_manifest);
    for (size_t index = 0; index < 2; index++)
    {
        evoke_term_fold_bundle_init(&old_folds[index]);
    }
    evoke_term_fold_bundle_init(&checked_fold);
    term_patch = reader_fenced
        ? &cleanup->term_cow_tree
        : &preflight_term_cow_tree;

    PG_TRY();
    {
        evoke_term_fold_bundle encoded_fold = *fold_bundle;

        encoded_fold.semantic_impact_precision =
            evoke_am_get_semantic_impact_precision(index_relation);
        status = evoke_term_fold_bundle_serialize(
            &encoded_fold,
            &cleanup->serialized_bytes,
            &fold_size,
            &fold_checksum
        );
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term fold bundle",
                status
            );
        }
        if (reader_fenced)
        {
            evoke_segment_pages_write_internal(
                index_relation,
                MAIN_FORKNUM,
                impact
                    ? EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
                    : EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                (uint64) term_id + 1,
                next_manifest->manifest_id,
                cleanup->serialized_bytes,
                fold_size,
                write_arena,
                &fold_ref
            );
        }
        else
        {
            evoke_segment_pages_init_preflight_fold_ref(
                next_manifest->manifest_id,
                term_id,
                impact,
                &preflight_fold_ref
            );
        }

        status = evoke_segment_pages_open_term_cow(
            index_relation,
            &old_manifest->term_directory,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_term_cow_lookup_external(
                &loader.root_object.ref,
                old_manifest->vocab_size,
                term_id,
                evoke_segment_pages_load_term_cow_object,
                &loader,
                &old_term_record
            );
        }
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_load_replaced_term_folds(
                index_relation,
                build_root,
                old_manifest,
                &old_term_record,
                term_id,
                impact,
                fold_level,
                old_folds,
                &old_fold_count
            );
        }
        if (status == EVOKE_OK && impact)
        {
            status = evoke_term_cow_build_external_impact_fold_patch(
                &loader.root_object.ref,
                old_manifest,
                next_manifest->manifest_id,
                term_id,
                reader_fenced ? &fold_ref : &preflight_fold_ref,
                fold_bundle->runs[0].coverage_sequence,
                fold_bundle->statistics_epoch,
                evoke_segment_pages_load_term_cow_object,
                &loader,
                term_patch,
                &update_stats
            );
        }
        else if (status == EVOKE_OK)
        {
            status = evoke_term_cow_build_external_neutral_fold_patch(
                &loader.root_object.ref,
                old_manifest,
                next_manifest->manifest_id,
                term_id,
                reader_fenced ? &fold_ref : &preflight_fold_ref,
                fold_bundle->runs[0].coverage_sequence,
                fold_level,
                evoke_segment_pages_load_term_cow_object,
                &loader,
                term_patch,
                &update_stats
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term fold term patch",
                status
            );
        }
        status = evoke_segment_pages_open_document_cow(
            index_relation,
            &old_manifest->document_directory,
            build_root->published_block_high_watermark,
            old_manifest->manifest_id,
            &document_loader
        );
        if (status == EVOKE_OK)
        {
            status = evoke_segment_pages_build_fold_document_updates(
                old_folds,
                old_fold_count,
                fold_bundle,
                term_id,
                old_manifest->document_slot_count,
                evoke_segment_pages_load_document_cow_object,
                &document_loader,
                &document_loader.root_object.ref,
                &document_updates,
                &document_update_count
            );
        }
        has_document_updates = document_update_count > 0;
        if (status == EVOKE_OK && has_document_updates)
        {
            status = evoke_document_cow_build_external_patch(
                &document_loader.root_object.ref,
                old_manifest->document_slot_count,
                next_manifest->document_slot_count,
                document_updates,
                document_update_count,
                next_manifest->manifest_id,
                evoke_segment_pages_load_document_cow_object,
                &document_loader,
                &cleanup->document_cow_tree,
                &document_update_stats
            );
        }
        free(document_updates);
        document_updates = NULL;
        if (status != EVOKE_OK)
        {
            evoke_segment_pages_report_codec_error(
                "COW term fold document patch",
                status
            );
        }
        if (!reader_fenced &&
            !evoke_segment_pages_prepare_retired_ranges(
                index_relation,
                build_root,
                old_manifest,
                0,
                0,
                term_patch,
                has_document_updates
                    ? &cleanup->document_cow_tree
                    : NULL,
                NULL,
                NULL,
                NULL,
                reuse_arena,
                &result,
                next_manifest
            ))
        {
            status = old_manifest->retired_range_count > 0
                ? evoke_segment_page_reuse_arena_build_retired(
                      old_manifest->retired_ranges,
                      old_manifest->retired_range_count,
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  )
                : evoke_segment_page_reuse_arena_build_empty_fenced(
                      build_root->published_block_high_watermark,
                      &handoff_arena
                  );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "prepare term fold FSM handoff",
                    status
                );
            }
            if (!evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    term_patch,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    NULL,
                    NULL,
                    NULL,
                    &handoff_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke prepared term fold retirement overflow"))
                );
            }
            outcome =
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED;
        }
        /*
         * The new fold closure remains unreachable until the root swap. The
         * prepared path only needs a reader fence around that publication.
         */
        {
            if (!reader_fenced)
            {
                evoke_segment_pages_write_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    impact
                        ? EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
                        : EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                    (uint64) term_id + 1,
                    next_manifest->manifest_id,
                    cleanup->serialized_bytes,
                    fold_size,
                    write_arena,
                    &fold_ref
                );
                if (impact)
                {
                    status =
                        evoke_term_cow_build_external_impact_fold_patch(
                            &loader.root_object.ref,
                            old_manifest,
                            next_manifest->manifest_id,
                            term_id,
                            &fold_ref,
                            fold_bundle->runs[0].coverage_sequence,
                            fold_bundle->statistics_epoch,
                            evoke_segment_pages_load_term_cow_object,
                            &loader,
                            &cleanup->term_cow_tree,
                            &update_stats
                        );
                }
                else
                {
                    status =
                        evoke_term_cow_build_external_neutral_fold_patch(
                            &loader.root_object.ref,
                            old_manifest,
                            next_manifest->manifest_id,
                            term_id,
                            &fold_ref,
                            fold_bundle->runs[0].coverage_sequence,
                            fold_level,
                            evoke_segment_pages_load_term_cow_object,
                            &loader,
                            &cleanup->term_cow_tree,
                            &update_stats
                        );
                }
                if (status != EVOKE_OK)
                {
                    evoke_segment_pages_report_codec_error(
                        "COW term fold ref bind",
                        status
                    );
                }
                if (!evoke_segment_pages_term_retirements_equal(
                        &preflight_term_cow_tree,
                        &cleanup->term_cow_tree))
                {
                    ereport(
                        ERROR,
                        (errmsg(
                            "evoke term fold retirement changed after bind"
                        ))
                    );
                }
            }
            free(cleanup->serialized_bytes);
            cleanup->serialized_bytes = NULL;
            if (fold_ref.object_bytes != fold_size ||
                fold_ref.object_checksum != fold_checksum)
            {
                ereport(
                    ERROR,
                    (errmsg("evoke COW term fold identity mismatch"))
                );
            }
            evoke_segment_pages_write_term_cow_objects_internal(
                index_relation,
                MAIN_FORKNUM,
                &cleanup->term_cow_tree,
                1,
                next_manifest->manifest_id,
                write_arena,
                &next_manifest->term_directory
            );
            if (has_document_updates)
            {
                evoke_segment_pages_write_document_cow_objects_internal(
                    index_relation,
                    MAIN_FORKNUM,
                    &cleanup->document_cow_tree,
                    next_manifest->manifest_id,
                    write_arena,
                    &next_manifest->document_directory
                );
            }
            else
            {
                next_manifest->document_directory =
                    old_manifest->document_directory;
            }
            next_manifest->flags &=
                ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
            next_manifest->flags |=
                EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY |
                EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
            if (reader_fenced &&
                !evoke_segment_pages_prepare_retired_ranges(
                    index_relation,
                    build_root,
                    old_manifest,
                    0,
                    0,
                    &cleanup->term_cow_tree,
                    has_document_updates
                        ? &cleanup->document_cow_tree
                        : NULL,
                    NULL,
                    NULL,
                    NULL,
                    reuse_arena,
                    &result,
                    next_manifest
                ))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke reader-fenced retirement overflow"))
                );
            }
            evoke_segment_pages_write_manifest(
                index_relation,
                MAIN_FORKNUM,
                next_manifest,
                cleanup,
                write_arena,
                &result.manifest
            );
            result.published_block_high_watermark =
                RelationGetNumberOfBlocks(index_relation);
            evoke_segment_pages_set_staged_writes(
                write_arena,
                result.published_block_high_watermark,
                &result
            );
            result.reused_block_count =
                evoke_segment_page_reuse_arena_reused_blocks(write_arena);
            status = evoke_segment_manifest_validate_published(
                next_manifest,
                &result.manifest,
                result.published_block_high_watermark
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW term fold manifest closure",
                    status
                );
            }

            next_root = *build_root;
            status = evoke_segment_read_root_replace_manifest(
                &next_root,
                &result.manifest,
                result.published_block_high_watermark
            );
            if (status != EVOKE_OK)
            {
                evoke_segment_pages_report_codec_error(
                    "COW term fold descendant root",
                    status
                );
            }
            evoke_segment_pages_load_maintenance_manifest(
                index_relation,
                &next_root,
                &checked_manifest
            );
            if (checked_manifest.manifest_id !=
                    next_manifest->manifest_id ||
                !evoke_segment_object_refs_equal(
                    &checked_manifest.term_directory,
                    &next_manifest->term_directory))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke COW term fold cold closure differs"))
                );
            }
            status = evoke_segment_pages_read_fold_bundle(
                index_relation,
                &next_root,
                &checked_manifest,
                &fold_ref,
                impact
                    ? EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
                    : EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
                &checked_fold
            );
            if (status != EVOKE_OK ||
                checked_fold.owner_manifest_id !=
                    fold_bundle->owner_manifest_id ||
                checked_fold.statistics_epoch !=
                    fold_bundle->statistics_epoch ||
                checked_fold.run_count != fold_bundle->run_count ||
                checked_fold.posting_count != fold_bundle->posting_count)
            {
                ereport(
                    ERROR,
                    (errmsg("evoke COW term fold cold payload differs"))
                );
            }
            *result_out = result;
            evoke_segment_cow_result_init(&result);
        }
        for (size_t index = 0; index < old_fold_count; index++)
        {
            evoke_term_fold_bundle_free(&old_folds[index]);
        }
        evoke_term_fold_bundle_free(&checked_fold);
        evoke_segment_manifest_free(&checked_manifest);
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_term_cow_tree_free(&preflight_term_cow_tree);
        evoke_segment_bundle_write_cleanup_free(cleanup);
    }
    PG_CATCH();
    {
        free(document_updates);
        evoke_segment_cow_result_free(&result);
        for (size_t index = 0; index < old_fold_count; index++)
        {
            evoke_term_fold_bundle_free(&old_folds[index]);
        }
        evoke_term_fold_bundle_free(&checked_fold);
        evoke_segment_manifest_free(&checked_manifest);
        evoke_segment_page_reuse_arena_free(&handoff_arena);
        evoke_segment_page_reuse_arena_free(&staging_arena);
        evoke_term_cow_tree_free(&preflight_term_cow_tree);
        evoke_segment_bundle_write_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return outcome;
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_neutral_fold(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 term_id,
    evoke_term_cow_neutral_fold_level fold_level,
    const evoke_term_fold_bundle *fold_bundle,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    return evoke_segment_pages_write_cow_term_fold(
        index_relation,
        build_root,
        old_manifest,
        next_manifest,
        term_id,
        false,
        fold_level,
        fold_bundle,
        reuse_arena,
        result_out
    );
}

evoke_segment_cow_write_outcome
evoke_segment_pages_write_cow_impact_fold(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 term_id,
    const evoke_term_fold_bundle *fold_bundle,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
)
{
    return evoke_segment_pages_write_cow_term_fold(
        index_relation,
        build_root,
        old_manifest,
        next_manifest,
        term_id,
        true,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
        fold_bundle,
        reuse_arena,
        result_out
    );
}
