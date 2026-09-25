#include "postgres.h"

#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "evoke_am_maintenance.h"
#include "evoke_am_meta.h"
#include "evoke_am_reclamation.h"
#include "evoke_am_test_support.h"

/*
 * Reuse is opportunistic and never part of publication correctness. Routine
 * maintenance consumes only bounded retirement evidence authenticated by the
 * current manifest. Full-root orphan discovery belongs to explicit scrub.
 */
static bool
evoke_am_acquire_convergent_reader_fence_internal(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    bool require_fence,
    uint32 reusable_block_limit,
    evoke_segment_page_reuse_arena *arena
)
{
    evoke_am_meta_page latest_meta;
    evoke_segment_read_root latest_root;
    BlockNumber physical_blocks;
    volatile bool relation_locked = false;
    evoke_block_range *bounded_ranges = NULL;
    size_t bounded_range_count = 0;
    bool prepared = false;
    evoke_status status;

    if (index_relation == NULL || build_root == NULL ||
        old_manifest == NULL || arena == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke page-reuse preparation")));
    }

    if (old_manifest->retired_range_count == 0 && !require_fence)
    {
        return false;
    }
    PG_TRY();
    {
        if (ConditionalLockRelation(
                index_relation,
                AccessExclusiveLock))
        {
            relation_locked = true;
            evoke_am_read_meta(index_relation, &latest_meta);
            status = evoke_am_segment_read_root_from_meta(
                &latest_meta,
                &latest_root
            );
            physical_blocks =
                RelationGetNumberOfBlocks(index_relation);
            if (evoke_am_meta_uses_convergent_segment_storage(
                    &latest_meta) &&
                status == EVOKE_OK &&
                latest_root.root_id == build_root->root_id &&
                latest_root.next_segment_id ==
                    build_root->next_segment_id &&
                evoke_segment_object_ref_equal(
                    &latest_root.manifest,
                    &build_root->manifest) &&
                latest_root.published_block_high_watermark >=
                    build_root->published_block_high_watermark &&
                physical_blocks >=
                    latest_root.published_block_high_watermark)
            {
                if (old_manifest->retired_range_count > 0 &&
                    reusable_block_limit > 0)
                {
                    uint32 remaining = reusable_block_limit;

                    bounded_ranges = palloc0(
                        old_manifest->retired_range_count *
                            sizeof(*bounded_ranges)
                    );
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
                        bounded->block_count = Min(
                            source->block_count,
                            remaining
                        );
                        remaining -= bounded->block_count;
                    }
                }
                status = old_manifest->retired_range_count > 0
                    ? evoke_segment_page_reuse_arena_build_retired(
                          bounded_ranges != NULL
                              ? bounded_ranges
                              : old_manifest->retired_ranges,
                          bounded_ranges != NULL
                              ? bounded_range_count
                              : old_manifest->retired_range_count,
                          latest_root.published_block_high_watermark,
                          arena
                      )
                    : evoke_segment_page_reuse_arena_build_empty_fenced(
                          latest_root.published_block_high_watermark,
                          arena
                      );
                if (status != EVOKE_OK)
                {
                    evoke_am_maintenance_codec_error(
                        "build the page-reuse arena",
                        status
                    );
                }
                prepared = require_fence ||
                    arena->allocator.available_block_count > 0;
                if (prepared)
                {
                    evoke_am_test_pause_ms(
                        "evoke.test_reuse_reader_fence_pause_ms",
                        "evoke retired-page reader fence"
                    );
                    /*
                     * A true return transfers the relation fence to the
                     * caller. It must cover every reused-page write and the
                     * replacement-root publication before being released.
                     */
                    relation_locked = false;
                }
            }
        }
    }
    PG_FINALLY();
    {
        if (bounded_ranges != NULL)
        {
            pfree(bounded_ranges);
        }
        if (relation_locked)
        {
            UnlockRelation(index_relation, AccessExclusiveLock);
        }
    }
    PG_END_TRY();
    return prepared;
}

bool
evoke_am_acquire_convergent_reader_fence(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_page_reuse_arena *arena
)
{
    return evoke_am_acquire_convergent_reader_fence_internal(
        index_relation,
        build_root,
        old_manifest,
        true,
        0,
        arena
    );
}
