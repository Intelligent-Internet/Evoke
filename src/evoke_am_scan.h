#ifndef EVOKE_AM_SCAN_H
#define EVOKE_AM_SCAN_H

#include "postgres.h"

#include "access/tableam.h"
#include "executor/tuptable.h"
#include "storage/itemptr.h"
#include "utils/rel.h"
#include "utils/snapshot.h"

typedef struct evoke_am_visibility_ctx
{
    IndexFetchTableData *fetch;
    Snapshot snapshot;
    TupleTableSlot *slot;
} evoke_am_visibility_ctx;

void evoke_am_visibility_begin(
    Relation heap_relation,
    evoke_am_visibility_ctx *visibility_out
);
void evoke_am_visibility_begin_with_snapshot(
    Relation heap_relation,
    Snapshot snapshot,
    evoke_am_visibility_ctx *visibility_out
);
bool evoke_am_tid_visible(
    evoke_am_visibility_ctx *visibility,
    const ItemPointerData *tid
);
bool evoke_am_tid_visible_as(
    evoke_am_visibility_ctx *visibility,
    const ItemPointerData *tid,
    ItemPointerData *visible_tid_out
);
void evoke_am_visibility_end(evoke_am_visibility_ctx *visibility);

#endif
