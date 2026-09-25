#ifndef EVOKE_AM_RESIDENT_FOLD_H
#define EVOKE_AM_RESIDENT_FOLD_H

#include "postgres.h"

#include "utils/rel.h"

#include "evoke_am_meta.h"
#include "evoke_am_preload.h"
#include "evoke_segment_pages.h"

typedef enum evoke_am_resident_fold_publish_result
{
    EVOKE_AM_RESIDENT_FOLD_PUBLISH_FAILED = 0,
    EVOKE_AM_RESIDENT_FOLD_PUBLISH_READY,
    EVOKE_AM_RESIDENT_FOLD_PUBLISH_LOADING,
    EVOKE_AM_RESIDENT_FOLD_PUBLISH_OVERSIZED,
    EVOKE_AM_RESIDENT_FOLD_PUBLISH_DONE
} evoke_am_resident_fold_publish_result;

typedef struct evoke_am_resident_fold_view
{
    evoke_am_preload_lease private_lease;
    const void *private_header;
    const void *private_terms;
    const void *private_extents;
    const void *private_vocab_offsets;
    const uint32 *private_sorted_vocab_ids;
} evoke_am_resident_fold_view;

void evoke_am_resident_fold_view_init(
    evoke_am_resident_fold_view *view
);

void evoke_am_resident_fold_view_release(
    evoke_am_resident_fold_view *view
);

bool evoke_am_resident_fold_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_resident_fold_view *view
);

bool evoke_am_resident_fold_lookup_token(
    const evoke_am_resident_fold_view *view,
    const char *token,
    uint32 *term_id_out
);

evoke_status evoke_am_resident_fold_topk(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *stats_out
);

evoke_status evoke_am_resident_fold_topk_filtered(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    const uint8 *allowed_document_bitmap,
    size_t allowed_document_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *stats_out
);

const ItemPointerData *evoke_am_resident_fold_document_tids(
    const evoke_am_resident_fold_view *view
);

uint32 evoke_am_resident_fold_document_count(
    const evoke_am_resident_fold_view *view
);

uint64 evoke_am_resident_fold_root_id(
    const evoke_am_resident_fold_view *view
);

Size evoke_am_resident_fold_required_size(
    const evoke_segment_storage_snapshot *snapshot
);

bool evoke_am_resident_fold_materialization_fits(
    uint64 relation_bytes,
    uint64 arena_bytes,
    uint64 physical_bytes,
    uint64 shared_buffer_bytes
);

evoke_am_resident_fold_publish_result evoke_am_resident_fold_publish(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    const evoke_segment_storage_snapshot *snapshot,
    Size *published_size_out
);

#endif
