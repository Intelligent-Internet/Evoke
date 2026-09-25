#ifndef EVOKE_AM_ACCELERATOR_H
#define EVOKE_AM_ACCELERATOR_H

#include "postgres.h"

#include "common/relpath.h"
#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_segment_pages.h"

#define EVOKE_AM_ACCELERATOR_SORT_MEMORY_MAX_KB \
    (UINT64_C(1024) * 1024)
#define EVOKE_AM_ACCELERATOR_MAX_DOCUMENT_SHIFT 12U

extern double evoke_test_semantic_accelerator_posting_mass;
extern int evoke_test_semantic_accelerator_forward_document_shift;

/*
 * Derive an accelerator from one immutable query-authority snapshot. This
 * never runs the encoder. The caller publishes the baseline with the normal
 * root compare-and-swap path while preserving any newer active-L0 frontier.
 */
bool evoke_am_prepare_accelerator_baseline(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_query_context *context,
    evoke_segment_page_reuse_arena *reuse_arena,
    uint64 scope_output_budget,
    volatile bool *reader_fence_locked_out,
    bool *memory_blocked_out,
    uint64 *scope_required_bytes_out,
    evoke_segment_manifest *next_manifest,
    evoke_segment_cow_result *result_out
);

#endif
