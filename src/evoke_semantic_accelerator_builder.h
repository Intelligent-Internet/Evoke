#ifndef EVOKE_SEMANTIC_ACCELERATOR_BUILDER_H
#define EVOKE_SEMANTIC_ACCELERATOR_BUILDER_H

#include <stddef.h>
#include <stdint.h>

#include "evoke_semantic_accelerator.h"

typedef struct evoke_semantic_accelerator_document_view
{
    const uint32_t *term_ids;
    const float *impacts;
    uint32_t term_count;
} evoke_semantic_accelerator_document_view;

typedef evoke_status (*evoke_semantic_accelerator_document_cb)(
    void *context,
    uint32_t document_id,
    evoke_semantic_accelerator_document_view *view_out
);

typedef struct evoke_semantic_accelerator_builder_options
{
    float centroid_fraction;
    uint32_t minimum_cluster_size;
    uint32_t document_cut;
    float summary_energy;
    uint32_t random_seed;
} evoke_semantic_accelerator_builder_options;

typedef struct evoke_semantic_accelerator_owned_term
{
    evoke_semantic_accelerator_term_input input;
    evoke_semantic_accelerator_cluster_input *clusters;
    uint32_t *document_ids;
    evoke_semantic_accelerator_summary_input *summaries;
} evoke_semantic_accelerator_owned_term;

/*
 * Conservatively estimate the resident workspace required to derive the
 * disposable accelerator from one immutable query-authority snapshot. The
 * builder streams one term at a time and spills the corpus transpose, so
 * corpus-wide posting and source-object bytes are not resident. The estimate
 * covers document metadata, vocabulary selection state, the largest decoded
 * term, bounded tuplesort memory, and fixed publication scratch.
 */
uint64_t evoke_semantic_accelerator_workspace_estimate(
    uint64_t document_count,
    uint64_t vocab_size,
    uint64_t largest_term_posting_count,
    uint64_t sort_memory_bytes
);

/* Initialize every owned term before passing it to build. */

void evoke_semantic_accelerator_owned_term_init(
    evoke_semantic_accelerator_owned_term *term
);

void evoke_semantic_accelerator_owned_term_free(
    evoke_semantic_accelerator_owned_term *term
);

/* Select the smallest high-DF term set covering the requested posting mass. */
evoke_status evoke_semantic_accelerator_select_terms(
    const uint32_t *document_frequencies,
    uint32_t term_count,
    float target_posting_mass,
    uint32_t **term_ids_out,
    uint32_t *selected_count_out,
    float *actual_posting_mass_out
);

/*
 * Build one retained posting list at a time. The document callback may be
 * backed by bounded external scratch; this function never owns corpus-wide
 * document geometry.
 */
evoke_status evoke_semantic_accelerator_build_term(
    uint32_t term_id,
    const uint32_t *document_ids,
    uint32_t document_count,
    const evoke_semantic_accelerator_builder_options *options,
    evoke_semantic_accelerator_document_cb read_document,
    void *document_context,
    evoke_semantic_accelerator_owned_term *term_out
);

#endif
