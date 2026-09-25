#include "evoke_semantic_accelerator_builder.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_ACCELERATOR_WORKSPACE_FIXED_BYTES \
    (UINT64_C(64) * 1024 * 1024)
#define EVOKE_ACCELERATOR_DOCUMENT_BYTES UINT64_C(72)
#define EVOKE_ACCELERATOR_VOCABULARY_BYTES UINT64_C(640)
#define EVOKE_ACCELERATOR_DECODED_POSTING_BYTES UINT64_C(32)

static uint64_t
evoke_accelerator_workspace_add(uint64_t left, uint64_t right)
{
    return left > UINT64_MAX - right ? UINT64_MAX : left + right;
}

static uint64_t
evoke_accelerator_workspace_mul(uint64_t left, uint64_t right)
{
    return left != 0 && right > UINT64_MAX / left
        ? UINT64_MAX
        : left * right;
}

uint64_t
evoke_semantic_accelerator_workspace_estimate(
    uint64_t document_count,
    uint64_t vocab_size,
    uint64_t largest_term_posting_count,
    uint64_t sort_memory_bytes
)
{
    uint64_t bytes = EVOKE_ACCELERATOR_WORKSPACE_FIXED_BYTES;

    /*
     * The page-native publisher retains document offsets, spill positions,
     * TIDs, lengths, and liveness for every slot. The TID reverse-directory
     * phase additionally owns sortable pairs, key/slot arrays, and its
     * serialized output while those source arrays remain live. Seventy-two
     * bytes per document conservatively covers that measured ownership.
     * Selected-term heaps and directory state are bounded by vocabulary size.
     * Only one term is decoded at once; the corpus transpose is owned by
     * BufFile and tuplesort is explicitly bounded by sort_memory_bytes.
     */
    bytes = evoke_accelerator_workspace_add(
        bytes,
        evoke_accelerator_workspace_mul(
            document_count,
            EVOKE_ACCELERATOR_DOCUMENT_BYTES
        )
    );
    bytes = evoke_accelerator_workspace_add(
        bytes,
        evoke_accelerator_workspace_mul(
            vocab_size,
            EVOKE_ACCELERATOR_VOCABULARY_BYTES
        )
    );
    bytes = evoke_accelerator_workspace_add(
        bytes,
        evoke_accelerator_workspace_mul(
            largest_term_posting_count,
            EVOKE_ACCELERATOR_DECODED_POSTING_BYTES
        )
    );
    return evoke_accelerator_workspace_add(bytes, sort_memory_bytes);
}

typedef struct evoke_accelerator_builder_pair
{
    uint32_t term_id;
    float impact;
} evoke_accelerator_builder_pair;

typedef struct evoke_accelerator_builder_seed
{
    uint64_t key;
    uint32_t document_index;
    uint32_t document_id;
} evoke_accelerator_builder_seed;

typedef struct evoke_accelerator_builder_frequency
{
    uint32_t term_id;
    uint32_t document_frequency;
} evoke_accelerator_builder_frequency;

typedef struct evoke_accelerator_builder_centroid_posting
{
    uint32_t term_id;
    uint32_t cluster;
    float impact;
} evoke_accelerator_builder_centroid_posting;

static int
evoke_accelerator_builder_compare_u32(const void *left, const void *right)
{
    const uint32_t a = *(const uint32_t *) left;
    const uint32_t b = *(const uint32_t *) right;

    return a < b ? -1 : a > b ? 1 : 0;
}

static int
evoke_accelerator_builder_compare_term(const void *left, const void *right)
{
    const evoke_accelerator_builder_pair *a = left;
    const evoke_accelerator_builder_pair *b = right;

    return a->term_id < b->term_id
        ? -1
        : a->term_id > b->term_id ? 1 : 0;
}

static int
evoke_accelerator_builder_compare_impact(const void *left, const void *right)
{
    const evoke_accelerator_builder_pair *a = left;
    const evoke_accelerator_builder_pair *b = right;

    if (a->impact > b->impact)
    {
        return -1;
    }
    if (a->impact < b->impact)
    {
        return 1;
    }
    return a->term_id < b->term_id
        ? -1
        : a->term_id > b->term_id ? 1 : 0;
}

static int
evoke_accelerator_builder_compare_seed(const void *left, const void *right)
{
    const evoke_accelerator_builder_seed *a = left;
    const evoke_accelerator_builder_seed *b = right;

    if (a->key < b->key)
    {
        return -1;
    }
    if (a->key > b->key)
    {
        return 1;
    }
    return a->document_id < b->document_id
        ? -1
        : a->document_id > b->document_id ? 1 : 0;
}

static int
evoke_accelerator_builder_compare_frequency(
    const void *left,
    const void *right
)
{
    const evoke_accelerator_builder_frequency *a = left;
    const evoke_accelerator_builder_frequency *b = right;

    if (a->document_frequency > b->document_frequency)
    {
        return -1;
    }
    if (a->document_frequency < b->document_frequency)
    {
        return 1;
    }
    return a->term_id < b->term_id
        ? -1
        : a->term_id > b->term_id ? 1 : 0;
}

static int
evoke_accelerator_builder_compare_centroid_posting(
    const void *left,
    const void *right
)
{
    const evoke_accelerator_builder_centroid_posting *a = left;
    const evoke_accelerator_builder_centroid_posting *b = right;

    if (a->term_id != b->term_id)
    {
        return a->term_id < b->term_id ? -1 : 1;
    }
    return a->cluster < b->cluster
        ? -1
        : a->cluster > b->cluster ? 1 : 0;
}

static uint64_t
evoke_accelerator_builder_mix64(uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

static evoke_status
evoke_accelerator_builder_validate_view(
    const evoke_semantic_accelerator_document_view *view
)
{
    if (view == NULL || view->term_count == 0 ||
        view->term_ids == NULL || view->impacts == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t index = 0; index < view->term_count; index++)
    {
        if (!isfinite(view->impacts[index]) || view->impacts[index] <= 0.0f ||
            (index > 0 && view->term_ids[index - 1U] >= view->term_ids[index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

static size_t
evoke_accelerator_builder_centroid_lower_bound(
    const evoke_accelerator_builder_centroid_posting *postings,
    size_t posting_count,
    uint32_t term_id
)
{
    size_t left = 0;
    size_t right = posting_count;

    while (left < right)
    {
        size_t middle = left + (right - left) / 2U;

        if (postings[middle].term_id < term_id)
        {
            left = middle + 1U;
        }
        else
        {
            right = middle;
        }
    }
    return left;
}

static evoke_status
evoke_accelerator_builder_assign_documents(
    const uint32_t *document_indexes,
    uint32_t document_count,
    const uint32_t *centroid_documents,
    uint32_t centroid_count,
    const uint32_t *top_ids,
    const float *top_impacts,
    const uint32_t *top_counts,
    uint32_t document_cut,
    uint32_t *assignments
)
{
    evoke_accelerator_builder_centroid_posting *postings = NULL;
    double *scores = NULL;
    uint32_t *epochs = NULL;
    uint32_t *touched = NULL;
    size_t posting_count = 0;
    size_t posting_capacity;
    uint32_t epoch = 1;

    if (document_indexes == NULL || document_count == 0 ||
        centroid_documents == NULL || centroid_count == 0 ||
        top_ids == NULL || top_impacts == NULL || top_counts == NULL ||
        document_cut == 0 || assignments == NULL ||
        centroid_count > SIZE_MAX / document_cut)
    {
        return EVOKE_ERR_INVALID;
    }
    posting_capacity = (size_t) centroid_count * document_cut;
    postings = malloc(posting_capacity * sizeof(*postings));
    scores = calloc(centroid_count, sizeof(*scores));
    epochs = calloc(centroid_count, sizeof(*epochs));
    touched = malloc((size_t) centroid_count * sizeof(*touched));
    if (postings == NULL || scores == NULL || epochs == NULL ||
        touched == NULL)
    {
        free(postings);
        free(scores);
        free(epochs);
        free(touched);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t cluster = 0; cluster < centroid_count; cluster++)
    {
        uint32_t centroid = centroid_documents[cluster];
        size_t offset = (size_t) centroid * document_cut;

        for (uint32_t item = 0; item < top_counts[centroid]; item++)
        {
            postings[posting_count].term_id = top_ids[offset + item];
            postings[posting_count].cluster = cluster;
            postings[posting_count].impact = top_impacts[offset + item];
            posting_count++;
        }
    }
    qsort(
        postings,
        posting_count,
        sizeof(*postings),
        evoke_accelerator_builder_compare_centroid_posting
    );
    for (uint32_t input = 0; input < document_count; input++)
    {
        uint32_t document = document_indexes[input];
        size_t document_offset = (size_t) document * document_cut;
        uint32_t touched_count = 0;
        uint32_t best_cluster = 0;
        double best_score = -1.0;

        if (epoch == 0)
        {
            memset(epochs, 0, (size_t) centroid_count * sizeof(*epochs));
            epoch = 1;
        }
        for (uint32_t item = 0; item < top_counts[document]; item++)
        {
            uint32_t term_id = top_ids[document_offset + item];
            float document_impact = top_impacts[document_offset + item];
            size_t position = evoke_accelerator_builder_centroid_lower_bound(
                postings,
                posting_count,
                term_id
            );

            while (position < posting_count &&
                postings[position].term_id == term_id)
            {
                uint32_t cluster = postings[position].cluster;

                if (epochs[cluster] != epoch)
                {
                    epochs[cluster] = epoch;
                    scores[cluster] = 0.0;
                    touched[touched_count++] = cluster;
                }
                scores[cluster] +=
                    (double) document_impact * postings[position].impact;
                position++;
            }
        }
        for (uint32_t item = 0; item < touched_count; item++)
        {
            uint32_t cluster = touched[item];

            if (scores[cluster] > best_score ||
                (scores[cluster] == best_score && cluster < best_cluster))
            {
                best_score = scores[cluster];
                best_cluster = cluster;
            }
        }
        assignments[input] = best_cluster;
        epoch++;
    }
    free(postings);
    free(scores);
    free(epochs);
    free(touched);
    return EVOKE_OK;
}

static evoke_status
evoke_accelerator_builder_reserve_pairs(
    evoke_accelerator_builder_pair **pairs,
    size_t *capacity,
    size_t required
)
{
    size_t next_capacity;
    evoke_accelerator_builder_pair *next;

    if (required <= *capacity)
    {
        return EVOKE_OK;
    }
    next_capacity = *capacity == 0 ? 64U : *capacity;
    while (next_capacity < required)
    {
        if (next_capacity > SIZE_MAX / 2U)
        {
            return EVOKE_ERR_RANGE;
        }
        next_capacity *= 2U;
    }
    if (next_capacity > SIZE_MAX / sizeof(*next))
    {
        return EVOKE_ERR_RANGE;
    }
    next = realloc(*pairs, next_capacity * sizeof(*next));
    if (next == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    *pairs = next;
    *capacity = next_capacity;
    return EVOKE_OK;
}

static bool
evoke_accelerator_builder_multiply_size(
    size_t left,
    size_t right,
    size_t *result_out
)
{
    if (result_out == NULL || (left > 0 && right > SIZE_MAX / left))
    {
        return false;
    }
    *result_out = left * right;
    return true;
}

static evoke_status
evoke_accelerator_builder_append_summaries(
    evoke_semantic_accelerator_summary_input **summaries,
    size_t *count,
    size_t *capacity,
    const evoke_accelerator_builder_pair *pairs,
    size_t pair_count
)
{
    size_t required;
    size_t next_capacity;
    evoke_semantic_accelerator_summary_input *next;

    if (pair_count > SIZE_MAX - *count)
    {
        return EVOKE_ERR_RANGE;
    }
    required = *count + pair_count;
    if (required > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (required > *capacity)
    {
        next_capacity = *capacity == 0 ? 64U : *capacity;
        while (next_capacity < required)
        {
            if (next_capacity > SIZE_MAX / 2U)
            {
                return EVOKE_ERR_RANGE;
            }
            next_capacity *= 2U;
        }
        if (next_capacity > SIZE_MAX / sizeof(*next))
        {
            return EVOKE_ERR_RANGE;
        }
        next = realloc(*summaries, next_capacity * sizeof(*next));
        if (next == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        *summaries = next;
        *capacity = next_capacity;
    }
    for (size_t index = 0; index < pair_count; index++)
    {
        (*summaries)[*count + index].term_id = pairs[index].term_id;
        (*summaries)[*count + index].max_impact = pairs[index].impact;
    }
    *count = required;
    return EVOKE_OK;
}

void
evoke_semantic_accelerator_owned_term_init(
    evoke_semantic_accelerator_owned_term *term
)
{
    if (term != NULL)
    {
        memset(term, 0, sizeof(*term));
    }
}

void
evoke_semantic_accelerator_owned_term_free(
    evoke_semantic_accelerator_owned_term *term
)
{
    if (term == NULL)
    {
        return;
    }
    free(term->clusters);
    free(term->document_ids);
    free(term->summaries);
    evoke_semantic_accelerator_owned_term_init(term);
}

evoke_status
evoke_semantic_accelerator_select_terms(
    const uint32_t *document_frequencies,
    uint32_t term_count,
    float target_posting_mass,
    uint32_t **term_ids_out,
    uint32_t *selected_count_out,
    float *actual_posting_mass_out
)
{
    evoke_accelerator_builder_frequency *frequencies = NULL;
    uint32_t *selected = NULL;
    uint64_t total = 0;
    uint64_t retained = 0;
    uint32_t nonempty_count = 0;
    uint32_t selected_count = 0;

    if ((term_count > 0 && document_frequencies == NULL) ||
        term_ids_out == NULL || selected_count_out == NULL ||
        actual_posting_mass_out == NULL ||
        !isfinite(target_posting_mass) || target_posting_mass <= 0.0f ||
        target_posting_mass > 1.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    *term_ids_out = NULL;
    *selected_count_out = 0;
    *actual_posting_mass_out = 1.0f;
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        if (UINT64_MAX - total < document_frequencies[term_id])
        {
            return EVOKE_ERR_RANGE;
        }
        total += document_frequencies[term_id];
        nonempty_count += document_frequencies[term_id] > 0;
    }
    if (total == 0)
    {
        return EVOKE_OK;
    }
    frequencies = malloc(
        (size_t) nonempty_count * sizeof(*frequencies)
    );
    selected = malloc((size_t) nonempty_count * sizeof(*selected));
    if (frequencies == NULL || selected == NULL)
    {
        free(frequencies);
        free(selected);
        return EVOKE_ERR_NOMEM;
    }
    nonempty_count = 0;
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        if (document_frequencies[term_id] == 0)
        {
            continue;
        }
        frequencies[nonempty_count].term_id = term_id;
        frequencies[nonempty_count].document_frequency =
            document_frequencies[term_id];
        nonempty_count++;
    }
    qsort(
        frequencies,
        nonempty_count,
        sizeof(*frequencies),
        evoke_accelerator_builder_compare_frequency
    );
    while (selected_count < nonempty_count &&
        (selected_count == 0 ||
         (double) retained / total < target_posting_mass))
    {
        retained += frequencies[selected_count].document_frequency;
        selected[selected_count] = frequencies[selected_count].term_id;
        selected_count++;
    }
    qsort(
        selected,
        selected_count,
        sizeof(*selected),
        evoke_accelerator_builder_compare_u32
    );
    free(frequencies);
    *term_ids_out = selected;
    *selected_count_out = selected_count;
    *actual_posting_mass_out = (float) ((double) retained / total);
    return EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_build_term(
    uint32_t term_id,
    const uint32_t *document_ids,
    uint32_t document_count,
    const evoke_semantic_accelerator_builder_options *options,
    evoke_semantic_accelerator_document_cb read_document,
    void *document_context,
    evoke_semantic_accelerator_owned_term *term_out
)
{
    uint32_t *sorted_documents = NULL;
    uint32_t *top_ids = NULL;
    float *top_impacts = NULL;
    uint32_t *top_counts = NULL;
    uint32_t *document_indexes = NULL;
    evoke_accelerator_builder_seed *seeds = NULL;
    uint32_t *centroid_documents = NULL;
    uint32_t *assignments = NULL;
    uint32_t *cluster_sizes = NULL;
    uint32_t *survivors = NULL;
    uint32_t *survivor_map = NULL;
    uint32_t *surviving_centroids = NULL;
    uint32_t *reassign_documents = NULL;
    uint32_t *reassign_assignments = NULL;
    uint32_t *cluster_offsets = NULL;
    uint32_t *cluster_cursors = NULL;
    uint32_t *summary_offsets = NULL;
    uint32_t *summary_counts = NULL;
    evoke_accelerator_builder_pair *pair_scratch = NULL;
    size_t pair_capacity = 0;
    evoke_semantic_accelerator_summary_input *summaries = NULL;
    size_t summary_count = 0;
    size_t summary_capacity = 0;
    evoke_semantic_accelerator_owned_term result;
    uint32_t initial_cluster_count;
    uint32_t survivor_count = 0;
    uint32_t reassign_count = 0;
    size_t document_bytes;
    size_t top_slots;
    size_t top_id_bytes;
    size_t top_impact_bytes;
    evoke_status status = EVOKE_OK;

    evoke_semantic_accelerator_owned_term_init(&result);
    if (document_ids == NULL || document_count == 0 || options == NULL ||
        read_document == NULL || term_out == NULL ||
        !isfinite(options->centroid_fraction) ||
        options->centroid_fraction <= 0.0f ||
        options->centroid_fraction > 1.0f || options->document_cut == 0 ||
        !isfinite(options->summary_energy) ||
        options->summary_energy <= 0.0f || options->summary_energy > 1.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    if (!evoke_accelerator_builder_multiply_size(
            document_count,
            sizeof(*sorted_documents),
            &document_bytes
        ) ||
        !evoke_accelerator_builder_multiply_size(
            document_count,
            options->document_cut,
            &top_slots
        ) ||
        !evoke_accelerator_builder_multiply_size(
            top_slots,
            sizeof(*top_ids),
            &top_id_bytes
        ) ||
        !evoke_accelerator_builder_multiply_size(
            top_slots,
            sizeof(*top_impacts),
            &top_impact_bytes
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    sorted_documents = malloc(document_bytes);
    top_ids = malloc(top_id_bytes);
    top_impacts = malloc(top_impact_bytes);
    top_counts = calloc(document_count, sizeof(*top_counts));
    document_indexes = malloc(
        (size_t) document_count * sizeof(*document_indexes)
    );
    seeds = malloc((size_t) document_count * sizeof(*seeds));
    assignments = malloc((size_t) document_count * sizeof(*assignments));
    if (sorted_documents == NULL || top_ids == NULL || top_impacts == NULL ||
        top_counts == NULL || document_indexes == NULL || seeds == NULL ||
        assignments == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    memcpy(
        sorted_documents,
        document_ids,
        (size_t) document_count * sizeof(*sorted_documents)
    );
    qsort(
        sorted_documents,
        document_count,
        sizeof(*sorted_documents),
        evoke_accelerator_builder_compare_u32
    );
    for (uint32_t document_index = 0;
         document_index < document_count;
         document_index++)
    {
        evoke_semantic_accelerator_document_view view;
        uint32_t retained_count;

        if (document_index > 0 &&
            sorted_documents[document_index - 1U] ==
                sorted_documents[document_index])
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        memset(&view, 0, sizeof(view));
        status = read_document(
            document_context,
            sorted_documents[document_index],
            &view
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        status = evoke_accelerator_builder_validate_view(&view);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        status = evoke_accelerator_builder_reserve_pairs(
            &pair_scratch,
            &pair_capacity,
            view.term_count
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        for (uint32_t item = 0; item < view.term_count; item++)
        {
            pair_scratch[item].term_id = view.term_ids[item];
            pair_scratch[item].impact = view.impacts[item];
        }
        qsort(
            pair_scratch,
            view.term_count,
            sizeof(*pair_scratch),
            evoke_accelerator_builder_compare_impact
        );
        retained_count = view.term_count < options->document_cut
            ? view.term_count
            : options->document_cut;
        qsort(
            pair_scratch,
            retained_count,
            sizeof(*pair_scratch),
            evoke_accelerator_builder_compare_term
        );
        top_counts[document_index] = retained_count;
        for (uint32_t item = 0; item < retained_count; item++)
        {
            size_t offset =
                (size_t) document_index * options->document_cut + item;

            top_ids[offset] = pair_scratch[item].term_id;
            top_impacts[offset] = pair_scratch[item].impact;
        }
        seeds[document_index].document_index = document_index;
        seeds[document_index].document_id = sorted_documents[document_index];
        seeds[document_index].key = evoke_accelerator_builder_mix64(
            ((uint64_t) options->random_seed << 32U) ^
            ((uint64_t) term_id << 1U) ^ sorted_documents[document_index]
        );
        document_indexes[document_index] = document_index;
    }
    initial_cluster_count = (uint32_t) floorf(
        options->centroid_fraction * document_count
    );
    if (initial_cluster_count == 0)
    {
        initial_cluster_count = 1;
    }
    if (initial_cluster_count > document_count)
    {
        initial_cluster_count = document_count;
    }
    qsort(
        seeds,
        document_count,
        sizeof(*seeds),
        evoke_accelerator_builder_compare_seed
    );
    centroid_documents = malloc(
        (size_t) initial_cluster_count * sizeof(*centroid_documents)
    );
    cluster_sizes = calloc(initial_cluster_count, sizeof(*cluster_sizes));
    survivors = malloc(
        (size_t) initial_cluster_count * sizeof(*survivors)
    );
    survivor_map = malloc(
        (size_t) initial_cluster_count * sizeof(*survivor_map)
    );
    if (centroid_documents == NULL || cluster_sizes == NULL ||
        survivors == NULL || survivor_map == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t cluster = 0; cluster < initial_cluster_count; cluster++)
    {
        centroid_documents[cluster] = seeds[cluster].document_index;
        survivor_map[cluster] = UINT32_MAX;
    }
    status = evoke_accelerator_builder_assign_documents(
        document_indexes,
        document_count,
        centroid_documents,
        initial_cluster_count,
        top_ids,
        top_impacts,
        top_counts,
        options->document_cut,
        assignments
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (uint32_t document_index = 0;
         document_index < document_count;
         document_index++)
    {
        cluster_sizes[assignments[document_index]]++;
    }
    for (uint32_t cluster = 0; cluster < initial_cluster_count; cluster++)
    {
        if (cluster_sizes[cluster] > options->minimum_cluster_size)
        {
            survivor_map[cluster] = survivor_count;
            survivors[survivor_count++] = cluster;
        }
    }
    if (survivor_count == 0)
    {
        uint32_t largest = 0;

        for (uint32_t cluster = 1;
             cluster < initial_cluster_count;
             cluster++)
        {
            if (cluster_sizes[cluster] > cluster_sizes[largest])
            {
                largest = cluster;
            }
        }
        survivor_map[largest] = 0;
        survivors[0] = largest;
        survivor_count = 1;
    }
    surviving_centroids = malloc(
        (size_t) survivor_count * sizeof(*surviving_centroids)
    );
    reassign_documents = malloc(
        (size_t) document_count * sizeof(*reassign_documents)
    );
    reassign_assignments = malloc(
        (size_t) document_count * sizeof(*reassign_assignments)
    );
    if (surviving_centroids == NULL || reassign_documents == NULL ||
        reassign_assignments == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t cluster = 0; cluster < survivor_count; cluster++)
    {
        surviving_centroids[cluster] =
            centroid_documents[survivors[cluster]];
    }
    for (uint32_t document_index = 0;
         document_index < document_count;
         document_index++)
    {
        if (survivor_map[assignments[document_index]] == UINT32_MAX)
        {
            reassign_documents[reassign_count++] = document_index;
        }
    }
    if (reassign_count > 0)
    {
        status = evoke_accelerator_builder_assign_documents(
            reassign_documents,
            reassign_count,
            surviving_centroids,
            survivor_count,
            top_ids,
            top_impacts,
            top_counts,
            options->document_cut,
            reassign_assignments
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    memset(cluster_sizes, 0, (size_t) survivor_count * sizeof(*cluster_sizes));
    reassign_count = 0;
    for (uint32_t document_index = 0;
         document_index < document_count;
         document_index++)
    {
        uint32_t original_cluster = assignments[document_index];
        uint32_t output_cluster = survivor_map[original_cluster];

        if (output_cluster == UINT32_MAX)
        {
            output_cluster = reassign_assignments[reassign_count++];
        }
        assignments[document_index] = output_cluster;
        cluster_sizes[output_cluster]++;
    }
    result.clusters = calloc(survivor_count, sizeof(*result.clusters));
    result.document_ids = malloc(
        (size_t) document_count * sizeof(*result.document_ids)
    );
    cluster_offsets = calloc(survivor_count + 1U, sizeof(*cluster_offsets));
    cluster_cursors = calloc(survivor_count, sizeof(*cluster_cursors));
    summary_offsets = calloc(survivor_count, sizeof(*summary_offsets));
    summary_counts = calloc(survivor_count, sizeof(*summary_counts));
    if (result.clusters == NULL || result.document_ids == NULL ||
        cluster_offsets == NULL || cluster_cursors == NULL ||
        summary_offsets == NULL || summary_counts == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t cluster = 0; cluster < survivor_count; cluster++)
    {
        cluster_offsets[cluster + 1U] =
            cluster_offsets[cluster] + cluster_sizes[cluster];
        cluster_cursors[cluster] = cluster_offsets[cluster];
    }
    for (uint32_t document_index = 0;
         document_index < document_count;
         document_index++)
    {
        uint32_t cluster = assignments[document_index];

        result.document_ids[cluster_cursors[cluster]++] =
            sorted_documents[document_index];
    }
    for (uint32_t cluster = 0; cluster < survivor_count; cluster++)
    {
        size_t pair_count = 0;
        size_t distinct_count = 0;
        double total = 0.0;
        double retained = 0.0;
        size_t retained_count = 0;

        for (uint32_t position = cluster_offsets[cluster];
             position < cluster_offsets[cluster + 1U];
             position++)
        {
            evoke_semantic_accelerator_document_view view;

            memset(&view, 0, sizeof(view));
            status = read_document(
                document_context,
                result.document_ids[position],
                &view
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            status = evoke_accelerator_builder_validate_view(&view);
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (view.term_count > SIZE_MAX - pair_count)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            status = evoke_accelerator_builder_reserve_pairs(
                &pair_scratch,
                &pair_capacity,
                pair_count + view.term_count
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            for (uint32_t item = 0; item < view.term_count; item++)
            {
                pair_scratch[pair_count + item].term_id =
                    view.term_ids[item];
                pair_scratch[pair_count + item].impact = view.impacts[item];
            }
            pair_count += view.term_count;
        }
        qsort(
            pair_scratch,
            pair_count,
            sizeof(*pair_scratch),
            evoke_accelerator_builder_compare_term
        );
        for (size_t item = 0; item < pair_count; item++)
        {
            if (distinct_count == 0 ||
                pair_scratch[distinct_count - 1U].term_id !=
                    pair_scratch[item].term_id)
            {
                pair_scratch[distinct_count++] = pair_scratch[item];
            }
            else if (pair_scratch[item].impact >
                pair_scratch[distinct_count - 1U].impact)
            {
                pair_scratch[distinct_count - 1U].impact =
                    pair_scratch[item].impact;
            }
        }
        qsort(
            pair_scratch,
            distinct_count,
            sizeof(*pair_scratch),
            evoke_accelerator_builder_compare_impact
        );
        for (size_t item = 0; item < distinct_count; item++)
        {
            total += pair_scratch[item].impact;
        }
        while (retained_count < distinct_count &&
            (retained_count == 0 ||
             retained < options->summary_energy * total))
        {
            retained += pair_scratch[retained_count].impact;
            retained_count++;
        }
        if (retained_count > UINT16_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        qsort(
            pair_scratch,
            retained_count,
            sizeof(*pair_scratch),
            evoke_accelerator_builder_compare_term
        );
        summary_offsets[cluster] = (uint32_t) summary_count;
        summary_counts[cluster] = (uint32_t) retained_count;
        status = evoke_accelerator_builder_append_summaries(
            &summaries,
            &summary_count,
            &summary_capacity,
            pair_scratch,
            retained_count
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    result.summaries = summaries;
    summaries = NULL;
    for (uint32_t cluster = 0; cluster < survivor_count; cluster++)
    {
        result.clusters[cluster].document_ids =
            &result.document_ids[cluster_offsets[cluster]];
        result.clusters[cluster].document_count = cluster_sizes[cluster];
        result.clusters[cluster].summary =
            &result.summaries[summary_offsets[cluster]];
        result.clusters[cluster].summary_count = summary_counts[cluster];
    }
    result.input.term_id = term_id;
    result.input.clusters = result.clusters;
    result.input.cluster_count = survivor_count;
    evoke_semantic_accelerator_owned_term_free(term_out);
    *term_out = result;
    evoke_semantic_accelerator_owned_term_init(&result);

cleanup:
    free(sorted_documents);
    free(top_ids);
    free(top_impacts);
    free(top_counts);
    free(document_indexes);
    free(seeds);
    free(centroid_documents);
    free(assignments);
    free(cluster_sizes);
    free(survivors);
    free(survivor_map);
    free(surviving_centroids);
    free(reassign_documents);
    free(reassign_assignments);
    free(cluster_offsets);
    free(cluster_cursors);
    free(summary_offsets);
    free(summary_counts);
    free(pair_scratch);
    free(summaries);
    evoke_semantic_accelerator_owned_term_free(&result);
    return status;
}
