#include "postgres.h"

#include "access/tableam.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"

#include "evoke_am_scan.h"

bool
evoke_am_tid_visible_as(
    evoke_am_visibility_ctx *visibility,
    const ItemPointerData *tid,
    ItemPointerData *visible_tid_out
)
{
    ItemPointerData heap_tid;
    bool call_again = false;
    bool all_dead = false;

    if (tid == NULL || !ItemPointerIsValid(tid))
    {
        return false;
    }
    heap_tid = *tid;
    table_index_fetch_reset(visibility->fetch);
    do
    {
        if (table_index_fetch_tuple(
                visibility->fetch,
                &heap_tid,
                visibility->snapshot,
                visibility->slot,
                &call_again,
                &all_dead))
        {
            if (visible_tid_out != NULL)
            {
                if (ItemPointerIsValid(&visibility->slot->tts_tid))
                {
                    *visible_tid_out = visibility->slot->tts_tid;
                }
                else
                {
                    *visible_tid_out = heap_tid;
                }
            }
            ExecClearTuple(visibility->slot);
            return true;
        }
        ExecClearTuple(visibility->slot);
    } while (call_again);

    return false;
}

bool
evoke_am_tid_visible(
    evoke_am_visibility_ctx *visibility,
    const ItemPointerData *tid
)
{
    return evoke_am_tid_visible_as(visibility, tid, NULL);
}

void
evoke_am_visibility_begin(
    Relation heap_relation,
    evoke_am_visibility_ctx *visibility_out
)
{
    evoke_am_visibility_begin_with_snapshot(
        heap_relation,
        GetActiveSnapshot(),
        visibility_out
    );
}

void
evoke_am_visibility_begin_with_snapshot(
    Relation heap_relation,
    Snapshot snapshot,
    evoke_am_visibility_ctx *visibility_out
)
{
    memset(visibility_out, 0, sizeof(*visibility_out));
    visibility_out->fetch = table_index_fetch_begin(heap_relation);
    visibility_out->snapshot = snapshot;
    visibility_out->slot = MakeSingleTupleTableSlot(
        RelationGetDescr(heap_relation),
        &TTSOpsBufferHeapTuple
    );
}

void
evoke_am_visibility_end(evoke_am_visibility_ctx *visibility)
{
    if (visibility == NULL)
    {
        return;
    }

    if (visibility->slot != NULL)
    {
        ExecDropSingleTupleTableSlot(visibility->slot);
        visibility->slot = NULL;
    }
    if (visibility->fetch != NULL)
    {
        table_index_fetch_end(visibility->fetch);
        visibility->fetch = NULL;
    }
    visibility->snapshot = NULL;
}
