#ifndef EVOKE_AM_RECLAMATION_H
#define EVOKE_AM_RECLAMATION_H

#include "postgres.h"

#include "utils/rel.h"

#include "evoke_segment_pages.h"
#include "evoke_segments.h"

bool evoke_am_acquire_convergent_reader_fence(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_page_reuse_arena *arena
);

#endif
