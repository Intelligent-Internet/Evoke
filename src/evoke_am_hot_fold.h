#ifndef EVOKE_AM_HOT_FOLD_H
#define EVOKE_AM_HOT_FOLD_H

#include "postgres.h"

#include "utils/rel.h"

#include "evoke_am_preload.h"

#define EVOKE_AM_HOT_FOLD_MAGIC UINT32_C(0x46323449)
#define EVOKE_AM_HOT_FOLD_VERSION UINT32_C(1)

typedef struct evoke_segment_storage_snapshot
    evoke_segment_storage_snapshot;
typedef struct evoke_term_fold_bundle evoke_term_fold_bundle;

/* Worker-published, exact-root read image for converged hot impact folds. */
typedef struct evoke_am_hot_fold_header
{
    uint32 magic;
    uint32 version;
    Size total_size;
    uint64 checksum;
    uint32 term_count;
    uint32 reserved;
    uint64 entry_count;
    Size terms_offset;
    Size entries_offset;
} evoke_am_hot_fold_header;

typedef struct evoke_am_hot_fold_term
{
    uint32 term_id;
    uint32 reserved;
    uint64 first_entry;
    uint64 entry_count;
} evoke_am_hot_fold_term;

typedef struct evoke_am_hot_fold_entry
{
    uint64 tie_break_key;
    uint32 document_slot;
    float score;
    ItemPointerData tid;
    uint16 reserved;
} evoke_am_hot_fold_entry;

/* Value handle only; the HOT_FOLD module owns payload interpretation. */
typedef struct evoke_am_hot_fold_view
{
    const void *private_header;
    const void *private_terms;
    const void *private_entries;
    evoke_am_preload_lease private_lease;
} evoke_am_hot_fold_view;

void evoke_am_hot_fold_view_init(evoke_am_hot_fold_view *view);
void evoke_am_hot_fold_view_release(evoke_am_hot_fold_view *view);

bool evoke_am_hot_fold_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_hot_fold_view *view
);

const evoke_am_hot_fold_term *evoke_am_hot_fold_find_term(
    const evoke_am_hot_fold_view *view,
    uint32 term_id
);

const evoke_am_hot_fold_entry *evoke_am_hot_fold_term_entries(
    const evoke_am_hot_fold_view *view,
    const evoke_am_hot_fold_term *term
);

bool evoke_am_hot_fold_publish(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_hot_fold_view *prior_view,
    uint32 term_id,
    const evoke_term_fold_bundle *impact_fold,
    const evoke_segment_storage_snapshot *snapshot
);

#endif
