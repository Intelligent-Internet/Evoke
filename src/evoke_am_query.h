#ifndef EVOKE_AM_QUERY_H
#define EVOKE_AM_QUERY_H

#include "postgres.h"

#include "storage/itemptr.h"
#include "utils/jsonb.h"

typedef struct evoke_am_query_hits
{
    ItemPointerData *tids;
    float *scores;
    size_t len;
} evoke_am_query_hits;

bool evoke_am_query_text(
    Oid index_oid,
    text *query_text,
    ArrayType *field_names,
    ArrayType *field_weights,
    Jsonb *planner_scope_filters,
    const uint64 *allowed_tid_keys,
    size_t allowed_tid_count,
    int32 top_k,
    MemoryContext result_context,
    evoke_am_query_hits *hits_out
);

#endif
