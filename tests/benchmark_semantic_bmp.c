#define _POSIX_C_SOURCE 200809L

#include "evoke_semantic_bmp.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_DOCUMENT_COUNT UINT32_C(1000000)
#define DEFAULT_TERM_COUNT UINT32_C(64)
#define DEFAULT_POSTING_STRIDE UINT32_C(20)
#define DEFAULT_TOP_K 100U
#define DEFAULT_PREFIX_COUNT 32U
#define BENCHMARK_REPETITIONS 5U

typedef struct semantic_prefix_entry
{
    uint32_t document_id;
    float impact;
} semantic_prefix_entry;

typedef struct semantic_term_prefix
{
    semantic_prefix_entry *lowest;
    semantic_prefix_entry *highest;
    uint32_t count;
} semantic_term_prefix;

static double
elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    return (double) (end->tv_sec - start->tv_sec) * 1000.0 +
        (double) (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static uint32_t
parse_u32(const char *value, const char *name)
{
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);

    if (value[0] == '\0' || end == NULL || *end != '\0' ||
        parsed == 0 || parsed > UINT32_MAX)
    {
        fprintf(stderr, "invalid %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t) parsed;
}

static float
posting_impact(uint32_t term_id, uint32_t posting_index)
{
    uint32_t value = posting_index ^
        (term_id + UINT32_C(1)) * UINT32_C(0x9e3779b9);

    value ^= value >> 16;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15;
    return 0.01f + (float) (value % UINT32_C(9901)) / 10000.0f;
}

static int
compare_prefix_entry(const void *left, const void *right)
{
    const semantic_prefix_entry *a = left;
    const semantic_prefix_entry *b = right;

    if (a->impact < b->impact)
    {
        return -1;
    }
    if (a->impact > b->impact)
    {
        return 1;
    }
    return (a->document_id > b->document_id) -
        (a->document_id < b->document_id);
}

static int
compare_u32(const void *left, const void *right)
{
    uint32_t a = *(const uint32_t *) left;
    uint32_t b = *(const uint32_t *) right;

    return (a > b) - (a < b);
}

static int
compare_seed_score_desc(const void *left, const void *right)
{
    const evoke_semantic_bmp_seed *a = left;
    const evoke_semantic_bmp_seed *b = right;

    if (a->score > b->score)
    {
        return -1;
    }
    if (a->score < b->score)
    {
        return 1;
    }
    return (a->document_id > b->document_id) -
        (a->document_id < b->document_id);
}

static double
topk_overlap(
    const evoke_topk_result *expected,
    const evoke_topk_result *actual
)
{
    size_t matches = 0;

    if (expected->len == 0)
    {
        return actual->len == 0 ? 1.0 : 0.0;
    }
    for (size_t expected_index = 0;
         expected_index < expected->len;
         expected_index++)
    {
        for (size_t actual_index = 0;
             actual_index < actual->len;
             actual_index++)
        {
            if (expected->doc_ids[expected_index] ==
                actual->doc_ids[actual_index])
            {
                matches++;
                break;
            }
        }
    }
    return (double) matches / (double) expected->len;
}

static evoke_status
build_term_prefixes(
    const evoke_semantic_bmp_run *runs,
    uint32_t term_count,
    uint32_t prefix_count,
    semantic_term_prefix **prefixes_out,
    uint64_t *bytes_out
)
{
    semantic_term_prefix *prefixes = NULL;
    semantic_prefix_entry *storage = NULL;
    semantic_prefix_entry *scratch = NULL;
    uint64_t storage_count;
    uint64_t max_postings = 0;

    if (prefixes_out == NULL || bytes_out == NULL || prefix_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    *prefixes_out = NULL;
    *bytes_out = 0;
    storage_count = (uint64_t) term_count * prefix_count * UINT64_C(2);
    if (storage_count > SIZE_MAX / sizeof(*storage))
    {
        return EVOKE_ERR_RANGE;
    }
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        if (runs[term_id].posting_count > max_postings)
        {
            max_postings = runs[term_id].posting_count;
        }
    }
    if (max_postings > SIZE_MAX / sizeof(*scratch))
    {
        return EVOKE_ERR_RANGE;
    }
    prefixes = calloc(term_count, sizeof(*prefixes));
    storage = malloc((size_t) storage_count * sizeof(*storage));
    scratch = malloc((size_t) max_postings * sizeof(*scratch));
    if (prefixes == NULL || storage == NULL || scratch == NULL)
    {
        free(scratch);
        free(storage);
        free(prefixes);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        const evoke_semantic_bmp_run *run = &runs[term_id];
        uint32_t retained = run->posting_count < prefix_count
            ? (uint32_t) run->posting_count
            : prefix_count;
        semantic_term_prefix *prefix = &prefixes[term_id];

        prefix->lowest = storage +
            (size_t) term_id * prefix_count * UINT32_C(2);
        prefix->highest = prefix->lowest + prefix_count;
        prefix->count = retained;
        for (uint64_t posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            scratch[posting_index].document_id =
                run->local_document_ids[posting_index];
            scratch[posting_index].impact =
                run->values[posting_index].impact;
        }
        qsort(
            scratch,
            (size_t) run->posting_count,
            sizeof(*scratch),
            compare_prefix_entry
        );
        memcpy(
            prefix->lowest,
            scratch,
            retained * sizeof(*scratch)
        );
        memcpy(
            prefix->highest,
            scratch + run->posting_count - retained,
            retained * sizeof(*scratch)
        );
    }
    free(scratch);
    *prefixes_out = prefixes;
    *bytes_out = storage_count * sizeof(*storage) +
        (uint64_t) term_count * sizeof(*prefixes);
    return EVOKE_OK;
}

static bool
find_run_impact(
    const evoke_semantic_bmp_run *run,
    uint32_t document_id,
    float *impact_out
)
{
    uint64_t low = 0;
    uint64_t high = run->posting_count;

    while (low < high)
    {
        uint64_t middle = low + (high - low) / UINT64_C(2);
        uint32_t candidate = run->local_document_ids[middle];

        if (candidate < document_id)
        {
            low = middle + UINT64_C(1);
        }
        else
        {
            high = middle;
        }
    }
    if (low >= run->posting_count ||
        run->local_document_ids[low] != document_id)
    {
        return false;
    }
    *impact_out = run->values[low].impact;
    return true;
}

static evoke_status
prepare_exact_seeds(
    const semantic_term_prefix *prefixes,
    const evoke_semantic_bmp_run *runs,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    uint32_t *candidate_ids,
    size_t candidate_capacity,
    evoke_semantic_bmp_seed *seeds,
    size_t *seed_count_out,
    uint64_t *lookup_count_out
)
{
    size_t candidate_count = 0;
    size_t unique_count = 0;
    size_t seed_count = 0;
    uint64_t lookup_count = 0;

    for (size_t query_index = 0; query_index < query_count; query_index++)
    {
        const semantic_term_prefix *prefix =
            &prefixes[query_ids[query_index]];
        const semantic_prefix_entry *entries = query_weights[query_index] < 0
            ? prefix->lowest
            : prefix->highest;

        if (prefix->count > candidate_capacity - candidate_count)
        {
            return EVOKE_ERR_RANGE;
        }
        for (uint32_t offset = 0; offset < prefix->count; offset++)
        {
            candidate_ids[candidate_count++] = entries[offset].document_id;
        }
    }
    qsort(candidate_ids, candidate_count, sizeof(*candidate_ids), compare_u32);
    for (size_t index = 0; index < candidate_count; index++)
    {
        if (unique_count == 0 ||
            candidate_ids[unique_count - 1U] != candidate_ids[index])
        {
            candidate_ids[unique_count++] = candidate_ids[index];
        }
    }
    for (size_t candidate_index = 0;
         candidate_index < unique_count;
         candidate_index++)
    {
        uint32_t document_id = candidate_ids[candidate_index];
        float score = 0.0f;

        for (size_t query_index = 0;
             query_index < query_count;
             query_index++)
        {
            float impact;

            lookup_count++;
            if (find_run_impact(
                    &runs[query_ids[query_index]],
                    document_id,
                    &impact
                ))
            {
                score += query_weights[query_index] * impact;
            }
        }
        if (score > 0.0f)
        {
            seeds[seed_count].document_id = document_id;
            seeds[seed_count].score = score;
            seed_count++;
        }
    }
    *seed_count_out = seed_count;
    *lookup_count_out = lookup_count;
    return EVOKE_OK;
}

static evoke_status
candidate_topk_from_seeds(
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    size_t top_k,
    evoke_topk_result *result_out
)
{
    size_t result_count = seed_count < top_k ? seed_count : top_k;

    evoke_topk_result_free(result_out);
    if (result_count == 0)
    {
        return EVOKE_OK;
    }
    result_out->doc_ids = malloc(
        result_count * sizeof(*result_out->doc_ids)
    );
    result_out->scores = malloc(
        result_count * sizeof(*result_out->scores)
    );
    if (result_out->doc_ids == NULL || result_out->scores == NULL)
    {
        evoke_topk_result_free(result_out);
        return EVOKE_ERR_NOMEM;
    }
    for (size_t rank = 0; rank < result_count; rank++)
    {
        result_out->doc_ids[rank] = seeds[rank].document_id;
        result_out->scores[rank] = seeds[rank].score;
    }
    result_out->len = result_count;
    return EVOKE_OK;
}

static void
free_norm_bounds(evoke_semantic_bmp_norm_bounds *bounds)
{
    free((void *) bounds->block_max_l1);
    free((void *) bounds->block_max_l2);
    free((void *) bounds->superblock_max_l1);
    free((void *) bounds->superblock_max_l2);
    memset(bounds, 0, sizeof(*bounds));
}

static float
outward_float(double value)
{
    if (value > FLT_MAX)
    {
        return INFINITY;
    }
    return nextafterf((float) nextafter(value, INFINITY), INFINITY);
}

static evoke_status
build_norm_bounds(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    uint32_t term_count,
    evoke_semantic_bmp_norm_bounds *bounds_out,
    uint64_t *bytes_out
)
{
    double *document_l1 = NULL;
    double *document_l2_squared = NULL;
    float *block_l1 = NULL;
    float *block_l2 = NULL;
    float *super_l1 = NULL;
    float *super_l2 = NULL;
    uint32_t block_count = (document_count + UINT32_C(15)) >> 4U;
    uint32_t superblock_count = (document_count + UINT32_C(255)) >> 8U;

    memset(bounds_out, 0, sizeof(*bounds_out));
    document_l1 = calloc(document_count, sizeof(*document_l1));
    document_l2_squared = calloc(
        document_count,
        sizeof(*document_l2_squared)
    );
    block_l1 = calloc(block_count, sizeof(*block_l1));
    block_l2 = calloc(block_count, sizeof(*block_l2));
    super_l1 = calloc(superblock_count, sizeof(*super_l1));
    super_l2 = calloc(superblock_count, sizeof(*super_l2));
    if (document_l1 == NULL || document_l2_squared == NULL ||
        block_l1 == NULL || block_l2 == NULL || super_l1 == NULL ||
        super_l2 == NULL)
    {
        free(document_l1);
        free(document_l2_squared);
        free(block_l1);
        free(block_l2);
        free(super_l1);
        free(super_l2);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        const evoke_semantic_bmp_run *run = &runs[term_id];

        for (uint64_t posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            uint32_t document_id = run->local_document_ids[posting_index];
            double impact = run->values[posting_index].impact;

            document_l1[document_id] += fabs(impact);
            document_l2_squared[document_id] += impact * impact;
        }
    }
    for (uint32_t document_id = 0;
         document_id < document_count;
         document_id++)
    {
        uint32_t block_id = document_id >> 4U;
        float l1 = outward_float(document_l1[document_id]);
        float l2 = outward_float(sqrt(document_l2_squared[document_id]));

        block_l1[block_id] = fmaxf(block_l1[block_id], l1);
        block_l2[block_id] = fmaxf(block_l2[block_id], l2);
    }
    for (uint32_t block_id = 0; block_id < block_count; block_id++)
    {
        uint32_t superblock_id = block_id >> 4U;

        super_l1[superblock_id] = fmaxf(
            super_l1[superblock_id],
            block_l1[block_id]
        );
        super_l2[superblock_id] = fmaxf(
            super_l2[superblock_id],
            block_l2[block_id]
        );
    }
    free(document_l1);
    free(document_l2_squared);
    bounds_out->block_max_l1 = block_l1;
    bounds_out->block_max_l2 = block_l2;
    bounds_out->superblock_max_l1 = super_l1;
    bounds_out->superblock_max_l2 = super_l2;
    bounds_out->block_count = block_count;
    bounds_out->superblock_count = superblock_count;
    *bytes_out = (
        (uint64_t) block_count + superblock_count
    ) * sizeof(float) * UINT64_C(2);
    return EVOKE_OK;
}

static int
compare_double(const void *left, const void *right)
{
    double lhs = *(const double *) left;
    double rhs = *(const double *) right;

    return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static uint64_t
packed_block_hybrid_size(const evoke_semantic_bmp_packed_index *index)
{
    uint64_t bytes;

    bytes = EVOKE_SEMANTIC_BMP_HEADER_SIZE;
    bytes += (uint64_t) index->term_count *
        EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE;
    bytes += (uint64_t) index->super_ref_count *
        EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
    bytes += (uint64_t) index->ref_count *
        EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
    bytes += index->doc_delta_bytes;
    bytes += index->posting_count * sizeof(uint32_t);
    return bytes;
}

static bool
same_topk(const evoke_topk_result *left, const evoke_topk_result *right)
{
    if (left->len != right->len)
    {
        return false;
    }
    for (size_t index = 0; index < left->len; index++)
    {
        if (left->doc_ids[index] != right->doc_ids[index] ||
            memcmp(
                &left->scores[index],
                &right->scores[index],
                sizeof(left->scores[index])
            ) != 0)
        {
            return false;
        }
    }
    return true;
}

int
main(int argc, char **argv)
{
    uint32_t document_count = DEFAULT_DOCUMENT_COUNT;
    uint32_t term_count = DEFAULT_TERM_COUNT;
    uint32_t query_term_count = 0;
    uint32_t posting_stride = DEFAULT_POSTING_STRIDE;
    size_t top_k = DEFAULT_TOP_K;
    uint32_t prefix_count = DEFAULT_PREFIX_COUNT;
    double boundary_error_ratio = 0.0;
    const char *distribution = "uniform";
    uint64_t postings_per_term;
    uint64_t posting_count;
    uint64_t query_posting_count = 0;
    uint32_t *document_ids = NULL;
    evoke_posting_value *values = NULL;
    evoke_semantic_bmp_run *runs = NULL;
    uint32_t *query_ids = NULL;
    float *query_weights = NULL;
    float *scores = NULL;
    uint64_t *tie_breaks = NULL;
    evoke_semantic_bmp_index index;
    evoke_semantic_bmp_packed_index packed_index;
    evoke_semantic_bmp_norm_bounds norm_bounds = {0};
    semantic_term_prefix *prefixes = NULL;
    uint32_t *seed_candidate_ids = NULL;
    evoke_semantic_bmp_seed *seeds = NULL;
    uint64_t prefix_bytes = 0;
    uint64_t norm_bound_bytes = 0;
    uint64_t seed_lookup_count = 0;
    size_t seed_count = 0;
    evoke_topk_result expected;
    evoke_topk_result actual;
    evoke_topk_result packed_actual;
    evoke_topk_result packed_taat_actual;
    evoke_topk_result candidate_actual;
    evoke_topk_result seeded_actual;
    evoke_topk_result oracle_seeded_actual;
    evoke_topk_result bounded_actual;
    evoke_topk_result oracle_bounded_actual;
    evoke_topk_result norm_actual;
    evoke_semantic_bmp_stats stats;
    evoke_semantic_bmp_stats packed_stats;
    evoke_semantic_bmp_stats packed_taat_stats;
    evoke_semantic_bmp_stats seeded_stats;
    evoke_semantic_bmp_stats oracle_seeded_stats;
    evoke_semantic_bmp_stats bounded_stats;
    evoke_semantic_bmp_stats oracle_bounded_stats;
    evoke_semantic_bmp_stats norm_stats;
    struct timespec started;
    struct timespec finished;
    double build_ms;
    double prefix_build_ms;
    double norm_build_ms;
    double taat_ms[BENCHMARK_REPETITIONS];
    double bmp_ms[BENCHMARK_REPETITIONS];
    double packed_ms[BENCHMARK_REPETITIONS];
    double packed_taat_ms[BENCHMARK_REPETITIONS];
    double candidate_ms[BENCHMARK_REPETITIONS];
    double seeded_ms[BENCHMARK_REPETITIONS];
    double oracle_seeded_ms[BENCHMARK_REPETITIONS];
    double bounded_ms[BENCHMARK_REPETITIONS];
    double bounded_overlap = 0.0;
    double candidate_overlap = 0.0;
    double oracle_bounded_overlap = 0.0;
    double oracle_bounded_ms[BENCHMARK_REPETITIONS];
    double norm_ms[BENCHMARK_REPETITIONS];
    float boundary_error = 0.0f;
    float oracle_boundary_error = 0.0f;
    double packed_build_ms;
    uint64_t bound_directory_bytes;
    uint64_t term_major_exact_bytes;
    uint64_t hybrid_authoritative_bytes;
    uint64_t packed_hybrid_bytes;
    uint8_t *packed_serialized = NULL;
    size_t serialized_size = 0;
    size_t packed_serialized_size = 0;
    evoke_status status = EVOKE_OK;
    bool exact = false;

    if (argc > 1)
    {
        document_count = parse_u32(argv[1], "document count");
    }
    if (argc > 2)
    {
        term_count = parse_u32(argv[2], "term count");
    }
    if (argc > 3)
    {
        posting_stride = parse_u32(argv[3], "posting stride");
    }
    if (argc > 4)
    {
        top_k = parse_u32(argv[4], "top k");
    }
    if (argc > 5)
    {
        distribution = argv[5];
        if (strcmp(distribution, "uniform") != 0 &&
            strcmp(distribution, "clustered") != 0)
        {
            fprintf(stderr, "distribution must be uniform or clustered\n");
            return EXIT_FAILURE;
        }
    }
    if (argc > 6)
    {
        query_term_count = parse_u32(argv[6], "query term count");
    }
    else
    {
        query_term_count = term_count;
    }
    if (argc > 7)
    {
        prefix_count = parse_u32(argv[7], "prefix count");
    }
    if (argc > 8)
    {
        char *end = NULL;

        boundary_error_ratio = strtod(argv[8], &end);
        if (argv[8][0] == '\0' || end == NULL || *end != '\0' ||
            !isfinite(boundary_error_ratio) || boundary_error_ratio < 0.0)
        {
            fprintf(stderr, "invalid boundary error ratio: %s\n", argv[8]);
            return EXIT_FAILURE;
        }
    }
    if (query_term_count > term_count)
    {
        fprintf(stderr, "query term count exceeds indexed term count\n");
        return EXIT_FAILURE;
    }
    if (top_k > document_count)
    {
        fprintf(stderr, "top k exceeds document count\n");
        return EXIT_FAILURE;
    }
    postings_per_term =
        (document_count + posting_stride - UINT32_C(1)) /
        posting_stride;
    if (postings_per_term > SIZE_MAX / term_count)
    {
        fprintf(stderr, "posting count exceeds address space\n");
        return EXIT_FAILURE;
    }
    posting_count = postings_per_term * term_count;

    document_ids = malloc((size_t) posting_count * sizeof(*document_ids));
    values = malloc((size_t) posting_count * sizeof(*values));
    runs = calloc(term_count, sizeof(*runs));
    query_ids = malloc(query_term_count * sizeof(*query_ids));
    query_weights = malloc(query_term_count * sizeof(*query_weights));
    scores = malloc(document_count * sizeof(*scores));
    tie_breaks = malloc(document_count * sizeof(*tie_breaks));
    if (document_ids == NULL || values == NULL || runs == NULL ||
        query_ids == NULL || query_weights == NULL || scores == NULL ||
        tie_breaks == NULL)
    {
        fprintf(stderr, "benchmark allocation failed\n");
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t document_id = 0;
         document_id < document_count;
         document_id++)
    {
        tie_breaks[document_id] = document_id;
    }
    for (uint32_t term_id = 0; term_id < term_count; term_id++)
    {
        uint64_t first_posting = postings_per_term * term_id;
        uint32_t residue = term_id % posting_stride;
        uint64_t local_count = 0;

        for (uint32_t document_id = residue;
             document_id < document_count;
             document_id += posting_stride)
        {
            uint64_t posting_index = first_posting + local_count;

            document_ids[posting_index] = document_id;
            values[posting_index].impact = posting_impact(
                term_id,
                (uint32_t) local_count
            );
            if (strcmp(distribution, "clustered") == 0)
            {
                values[posting_index].impact = document_id <
                    (document_count / UINT32_C(100) > 0
                        ? document_count / UINT32_C(100)
                        : UINT32_C(1))
                    ? values[posting_index].impact + 1.0f
                    : values[posting_index].impact * 0.001f;
            }
            local_count++;
        }
        runs[term_id].term_id = term_id;
        runs[term_id].posting_count = local_count;
        runs[term_id].local_document_ids = &document_ids[first_posting];
        runs[term_id].values = &values[first_posting];
        runs[term_id].local_document_count = document_count;
        if (term_id < query_term_count)
        {
            query_ids[term_id] = term_id;
            query_weights[term_id] =
                0.25f +
                (float) ((term_id * UINT32_C(37)) % 101) / 100.0f;
            query_posting_count += local_count;
        }
        if (local_count < postings_per_term)
        {
            posting_count -= postings_per_term - local_count;
        }
    }

    if ((uint64_t) query_term_count * prefix_count >
        SIZE_MAX / sizeof(*seed_candidate_ids))
    {
        fprintf(stderr, "seed candidate count exceeds address space\n");
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    seed_candidate_ids = malloc(
        (size_t) query_term_count * prefix_count *
        sizeof(*seed_candidate_ids)
    );
    seeds = malloc(
        (size_t) query_term_count * prefix_count * sizeof(*seeds)
    );
    if (seed_candidate_ids == NULL || seeds == NULL)
    {
        fprintf(stderr, "seed oracle allocation failed\n");
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &started);
    status = build_term_prefixes(
        runs,
        term_count,
        prefix_count,
        &prefixes,
        &prefix_bytes
    );
    clock_gettime(CLOCK_MONOTONIC, &finished);
    prefix_build_ms = elapsed_ms(&started, &finished);
    if (status != EVOKE_OK)
    {
        fprintf(stderr, "semantic prefix build failed: %s\n",
                evoke_strerror(status));
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &started);
    status = build_norm_bounds(
        document_count,
        runs,
        term_count,
        &norm_bounds,
        &norm_bound_bytes
    );
    clock_gettime(CLOCK_MONOTONIC, &finished);
    norm_build_ms = elapsed_ms(&started, &finished);
    if (status != EVOKE_OK)
    {
        fprintf(stderr, "semantic norm bounds build failed: %s\n",
                evoke_strerror(status));
        goto cleanup;
    }

    evoke_semantic_bmp_index_init(&index);
    evoke_semantic_bmp_packed_index_init(&packed_index);
    memset(&expected, 0, sizeof(expected));
    memset(&actual, 0, sizeof(actual));
    memset(&packed_actual, 0, sizeof(packed_actual));
    memset(&packed_taat_actual, 0, sizeof(packed_taat_actual));
    memset(&candidate_actual, 0, sizeof(candidate_actual));
    memset(&seeded_actual, 0, sizeof(seeded_actual));
    memset(&oracle_seeded_actual, 0, sizeof(oracle_seeded_actual));
    memset(&bounded_actual, 0, sizeof(bounded_actual));
    memset(&oracle_bounded_actual, 0, sizeof(oracle_bounded_actual));
    memset(&norm_actual, 0, sizeof(norm_actual));
    memset(&stats, 0, sizeof(stats));
    memset(&packed_stats, 0, sizeof(packed_stats));
    memset(&packed_taat_stats, 0, sizeof(packed_taat_stats));
    memset(&seeded_stats, 0, sizeof(seeded_stats));
    memset(&oracle_seeded_stats, 0, sizeof(oracle_seeded_stats));
    memset(&bounded_stats, 0, sizeof(bounded_stats));
    memset(&oracle_bounded_stats, 0, sizeof(oracle_bounded_stats));
    memset(&norm_stats, 0, sizeof(norm_stats));
    clock_gettime(CLOCK_MONOTONIC, &started);
    status = evoke_semantic_bmp_index_build_runs(
        document_count,
        runs,
        term_count,
        &index
    );
    clock_gettime(CLOCK_MONOTONIC, &finished);
    build_ms = elapsed_ms(&started, &finished);
    if (status != EVOKE_OK)
    {
        fprintf(stderr, "BMP build failed: %s\n", evoke_strerror(status));
        goto cleanup_index;
    }
    status = evoke_semantic_bmp_serialized_size(&index, &serialized_size);
    if (status != EVOKE_OK)
    {
        goto cleanup_index;
    }
    clock_gettime(CLOCK_MONOTONIC, &started);
    status = evoke_semantic_bmp_packed_index_build_runs(
        document_count,
        runs,
        term_count,
        &packed_index
    );
    clock_gettime(CLOCK_MONOTONIC, &finished);
    packed_build_ms = elapsed_ms(&started, &finished);
    if (status != EVOKE_OK)
    {
        fprintf(
            stderr,
            "packed BMP build failed: %s\n",
            evoke_strerror(status)
        );
        goto cleanup_index;
    }
    status = evoke_semantic_bmp_packed_serialized_size(
        &packed_index,
        &packed_serialized_size
    );
    if (status != EVOKE_OK)
    {
        goto cleanup_index;
    }
    {
        size_t estimated_size = packed_serialized_size;

        status = evoke_semantic_bmp_packed_serialize(
            &packed_index,
            &packed_serialized,
            &packed_serialized_size
        );
        if (status != EVOKE_OK || packed_serialized_size != estimated_size)
        {
            fprintf(
                stderr,
                "packed serialization failed: status=%s "
                "estimated=%zu actual=%zu\n",
                evoke_strerror(status),
                estimated_size,
                packed_serialized_size
            );
            if (status == EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
            }
            goto cleanup_index;
        }
    }

    for (size_t repetition = 0;
         repetition < BENCHMARK_REPETITIONS;
         repetition++)
    {
        memset(scores, 0, document_count * sizeof(*scores));
        clock_gettime(CLOCK_MONOTONIC, &started);
        for (uint32_t term_id = 0;
             term_id < query_term_count;
             term_id++)
        {
            const evoke_semantic_bmp_run *run = &runs[term_id];

            for (uint64_t posting_index = 0;
                 posting_index < run->posting_count;
                 posting_index++)
            {
                scores[run->local_document_ids[posting_index]] +=
                    query_weights[term_id] *
                    run->values[posting_index].impact;
            }
        }
        evoke_topk_result_free(&expected);
        status = evoke_topk_with_tie_breaks(
            scores,
            document_count,
            top_k,
            true,
            tie_breaks,
            &expected
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        taat_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            goto cleanup_index;
        }

        evoke_topk_result_free(&actual);
        memset(&stats, 0, sizeof(stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_topk(
            &index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            &actual,
            &stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        bmp_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            goto cleanup_index;
        }
        exact = same_topk(&expected, &actual);
        if (!exact)
        {
            fprintf(stderr, "BMP result differs from exact TAAT\n");
            status = EVOKE_ERR_FORMAT;
            goto cleanup_index;
        }

        evoke_topk_result_free(&packed_actual);
        memset(&packed_stats, 0, sizeof(packed_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_topk(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            &packed_actual,
            &packed_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        packed_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            goto cleanup_index;
        }
        if (!same_topk(&expected, &packed_actual))
        {
            fprintf(stderr, "packed BMP result differs from exact TAAT\n");
            status = EVOKE_ERR_FORMAT;
            goto cleanup_index;
        }

        evoke_topk_result_free(&norm_actual);
        memset(&norm_stats, 0, sizeof(norm_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_topk_norm(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            &norm_bounds,
            &norm_actual,
            &norm_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        norm_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK || !same_topk(&expected, &norm_actual))
        {
            fprintf(stderr, "norm-bound packed BMP differs from exact TAAT\n");
            if (status == EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
            }
            goto cleanup_index;
        }

        evoke_topk_result_free(&seeded_actual);
        memset(&seeded_stats, 0, sizeof(seeded_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = prepare_exact_seeds(
            prefixes,
            runs,
            query_ids,
            query_weights,
            query_term_count,
            seed_candidate_ids,
            (size_t) query_term_count * prefix_count,
            seeds,
            &seed_count,
            &seed_lookup_count
        );
        if (status == EVOKE_OK)
        {
            qsort(
                seeds,
                seed_count,
                sizeof(*seeds),
                compare_seed_score_desc
            );
            status = candidate_topk_from_seeds(
                seeds,
                seed_count,
                top_k,
                &candidate_actual
            );
        }
        clock_gettime(CLOCK_MONOTONIC, &finished);
        candidate_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            fprintf(stderr, "candidate-only rerank failed\n");
            goto cleanup_index;
        }
        candidate_overlap = topk_overlap(&expected, &candidate_actual);
        if (status == EVOKE_OK)
        {
            status = evoke_semantic_bmp_packed_topk_seeded(
                &packed_index,
                query_ids,
                query_weights,
                query_term_count,
                top_k,
                seeds,
                seed_count,
                &seeded_actual,
                &seeded_stats
            );
        }
        clock_gettime(CLOCK_MONOTONIC, &finished);
        seeded_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK || !same_topk(&expected, &seeded_actual))
        {
            fprintf(stderr, "seeded packed BMP differs from exact TAAT\n");
            if (status == EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
            }
            goto cleanup_index;
        }

        boundary_error = seed_count >= top_k
            ? (float) (boundary_error_ratio * seeds[top_k - 1U].score)
            : 0.0f;
        evoke_topk_result_free(&bounded_actual);
        memset(&bounded_stats, 0, sizeof(bounded_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_topk_seeded_bounded(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            seeds,
            seed_count,
            boundary_error,
            &bounded_actual,
            &bounded_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        bounded_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            fprintf(stderr, "bounded seeded packed BMP failed\n");
            goto cleanup_index;
        }
        bounded_overlap = topk_overlap(&expected, &bounded_actual);

        evoke_topk_result_free(&oracle_seeded_actual);
        memset(&oracle_seeded_stats, 0, sizeof(oracle_seeded_stats));
        for (size_t rank = 0; rank < expected.len; rank++)
        {
            seeds[rank].document_id = expected.doc_ids[rank];
            seeds[rank].score = expected.scores[rank];
        }
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_topk_seeded(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            seeds,
            expected.len,
            &oracle_seeded_actual,
            &oracle_seeded_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        oracle_seeded_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK ||
            !same_topk(&expected, &oracle_seeded_actual))
        {
            fprintf(stderr, "oracle-seeded BMP differs from exact TAAT\n");
            if (status == EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
            }
            goto cleanup_index;
        }

        oracle_boundary_error = expected.len >= top_k
            ? (float) (
                boundary_error_ratio * expected.scores[top_k - 1U]
            )
            : 0.0f;
        evoke_topk_result_free(&oracle_bounded_actual);
        memset(&oracle_bounded_stats, 0, sizeof(oracle_bounded_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_topk_seeded_bounded(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            seeds,
            expected.len,
            oracle_boundary_error,
            &oracle_bounded_actual,
            &oracle_bounded_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        oracle_bounded_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK)
        {
            fprintf(stderr, "oracle-bounded packed BMP failed\n");
            goto cleanup_index;
        }
        oracle_bounded_overlap = topk_overlap(
            &expected,
            &oracle_bounded_actual
        );

        evoke_topk_result_free(&packed_taat_actual);
        memset(&packed_taat_stats, 0, sizeof(packed_taat_stats));
        clock_gettime(CLOCK_MONOTONIC, &started);
        status = evoke_semantic_bmp_packed_taat_topk(
            &packed_index,
            query_ids,
            query_weights,
            query_term_count,
            top_k,
            &packed_taat_actual,
            &packed_taat_stats
        );
        clock_gettime(CLOCK_MONOTONIC, &finished);
        packed_taat_ms[repetition] = elapsed_ms(&started, &finished);
        if (status != EVOKE_OK ||
            !same_topk(&expected, &packed_taat_actual))
        {
            fprintf(stderr, "packed TAAT result differs from exact TAAT\n");
            if (status == EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
            }
            goto cleanup_index;
        }
    }

    qsort(
        taat_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*taat_ms),
        compare_double
    );
    qsort(
        bmp_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*bmp_ms),
        compare_double
    );
    qsort(
        packed_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*packed_ms),
        compare_double
    );
    qsort(
        packed_taat_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*packed_taat_ms),
        compare_double
    );
    qsort(
        candidate_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*candidate_ms),
        compare_double
    );
    qsort(
        seeded_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*seeded_ms),
        compare_double
    );
    qsort(
        oracle_seeded_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*oracle_seeded_ms),
        compare_double
    );
    qsort(
        bounded_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*bounded_ms),
        compare_double
    );
    qsort(
        oracle_bounded_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*oracle_bounded_ms),
        compare_double
    );
    qsort(
        norm_ms,
        BENCHMARK_REPETITIONS,
        sizeof(*norm_ms),
        compare_double
    );
    bound_directory_bytes =
        (uint64_t) index.term_count * EVOKE_SEMANTIC_BMP_TERM_SIZE +
        (uint64_t) index.super_ref_count *
            EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE +
        (uint64_t) index.ref_count * EVOKE_SEMANTIC_BMP_REF_SIZE;
    term_major_exact_bytes = index.posting_count *
        (sizeof(uint32_t) + sizeof(uint32_t));
    hybrid_authoritative_bytes = bound_directory_bytes +
        term_major_exact_bytes;
    packed_hybrid_bytes = packed_block_hybrid_size(&packed_index);
    if (packed_hybrid_bytes != packed_serialized_size)
    {
        fprintf(
            stderr,
            "packed size mismatch: estimate=%" PRIu64 " actual=%zu\n",
            packed_hybrid_bytes,
            packed_serialized_size
        );
        status = EVOKE_ERR_FORMAT;
        goto cleanup_index;
    }
    printf(
        "{\n"
        "  \"documents\": %u,\n"
        "  \"indexed_terms\": %u,\n"
        "  \"query_terms\": %u,\n"
        "  \"distribution\": \"%s\",\n"
        "  \"postings\": %" PRIu64 ",\n"
        "  \"query_postings\": %" PRIu64 ",\n"
        "  \"term_entries\": %u,\n"
        "  \"super_bound_entries\": %u,\n"
        "  \"fine_bound_entries\": %u,\n"
        "  \"active_blocks\": %u,\n"
        "  \"forward_records\": %u,\n"
        "  \"term_bytes\": %" PRIu64 ",\n"
        "  \"super_bound_bytes\": %" PRIu64 ",\n"
        "  \"fine_bound_bytes\": %" PRIu64 ",\n"
        "  \"block_directory_bytes\": %" PRIu64 ",\n"
        "  \"forward_record_bytes\": %" PRIu64 ",\n"
        "  \"impact_bytes\": %" PRIu64 ",\n"
        "  \"bound_directory_bytes\": %" PRIu64 ",\n"
        "  \"term_major_exact_bytes\": %" PRIu64 ",\n"
        "  \"hybrid_authoritative_bytes\": %" PRIu64 ",\n"
        "  \"hybrid_authoritative_bytes_per_posting\": %.3f,\n"
        "  \"packed_block_hybrid_bytes\": %" PRIu64 ",\n"
        "  \"packed_block_hybrid_bytes_per_posting\": %.3f,\n"
        "  \"packed_serialized_bytes\": %zu,\n"
        "  \"packed_super_ref_bytes\": %u,\n"
        "  \"packed_ref_bytes\": %u,\n"
        "  \"serialized_bytes\": %zu,\n"
        "  \"serialized_bytes_per_posting\": %.3f,\n"
        "  \"build_ms\": %.3f,\n"
        "  \"packed_build_ms\": %.3f,\n"
        "  \"prefix_count\": %u,\n"
        "  \"prefix_bytes\": %" PRIu64 ",\n"
        "  \"prefix_build_ms\": %.3f,\n"
        "  \"norm_bound_bytes\": %" PRIu64 ",\n"
        "  \"norm_build_ms\": %.3f,\n"
        "  \"norm_median_ms\": %.3f,\n"
        "  \"norm_postings_examined\": %" PRIu64 ",\n"
        "  \"norm_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"norm_bound_reductions\": %" PRIu64 ",\n"
        "  \"seed_candidates\": %zu,\n"
        "  \"seed_exact_lookups\": %" PRIu64 ",\n"
        "  \"candidate_only_median_ms\": %.3f,\n"
        "  \"candidate_only_overlap_at_k\": %.6f,\n"
        "  \"seeded_median_ms\": %.3f,\n"
        "  \"seeded_postings_examined\": %" PRIu64 ",\n"
        "  \"seeded_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"oracle_seeded_median_ms\": %.3f,\n"
        "  \"oracle_seeded_postings_examined\": %" PRIu64 ",\n"
        "  \"oracle_seeded_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"boundary_error_ratio\": %.6f,\n"
        "  \"boundary_error\": %.9g,\n"
        "  \"bounded_median_ms\": %.3f,\n"
        "  \"bounded_overlap_at_k\": %.6f,\n"
        "  \"bounded_postings_examined\": %" PRIu64 ",\n"
        "  \"bounded_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"bounded_approximate_prune_events\": %" PRIu64 ",\n"
        "  \"bounded_max_skipped_bound\": %.9g,\n"
        "  \"bounded_final_kth_score\": %.9g,\n"
        "  \"exact_kth_score\": %.9g,\n"
        "  \"oracle_boundary_error\": %.9g,\n"
        "  \"oracle_bounded_median_ms\": %.3f,\n"
        "  \"oracle_bounded_overlap_at_k\": %.6f,\n"
        "  \"oracle_bounded_postings_examined\": %" PRIu64 ",\n"
        "  \"oracle_bounded_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"oracle_bounded_approximate_prune_events\": %" PRIu64 ",\n"
        "  \"oracle_bounded_max_skipped_bound\": %.9g,\n"
        "  \"taat_median_ms\": %.3f,\n"
        "  \"bmp_median_ms\": %.3f,\n"
        "  \"speedup\": %.3f,\n"
        "  \"packed_median_ms\": %.3f,\n"
        "  \"packed_speedup\": %.3f,\n"
        "  \"packed_taat_median_ms\": %.3f,\n"
        "  \"packed_taat_postings_examined\": %" PRIu64 ",\n"
        "  \"super_bound_entries_visited\": %" PRIu64 ",\n"
        "  \"superblocks_scored\": %" PRIu64 ",\n"
        "  \"superblocks_skipped\": %" PRIu64 ",\n"
        "  \"bound_entries_visited\": %" PRIu64 ",\n"
        "  \"blocks_scored\": %" PRIu64 ",\n"
        "  \"blocks_skipped\": %" PRIu64 ",\n"
        "  \"forward_records_examined\": %" PRIu64 ",\n"
        "  \"forward_records_per_query_posting\": %.6f,\n"
        "  \"postings_examined\": %" PRIu64 ",\n"
        "  \"query_posting_fraction\": %.9f,\n"
        "  \"packed_bound_entries_visited\": %" PRIu64 ",\n"
        "  \"packed_blocks_scored\": %" PRIu64 ",\n"
        "  \"packed_blocks_skipped\": %" PRIu64 ",\n"
        "  \"packed_forward_records_examined\": %" PRIu64 ",\n"
        "  \"packed_postings_examined\": %" PRIu64 ",\n"
        "  \"packed_adaptive_fallbacks\": %" PRIu64 ",\n"
        "  \"adaptive_record_guard_would_fallback\": %s,\n"
        "  \"exact\": %s\n"
        "}\n",
        document_count,
        term_count,
        query_term_count,
        distribution,
        index.posting_count,
        query_posting_count,
        index.term_count,
        index.super_ref_count,
        index.ref_count,
        index.active_block_count,
        index.record_count,
        (uint64_t) index.term_count * EVOKE_SEMANTIC_BMP_TERM_SIZE,
        (uint64_t) index.super_ref_count *
            EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE,
        (uint64_t) index.ref_count * EVOKE_SEMANTIC_BMP_REF_SIZE,
        (uint64_t) index.active_block_count *
            EVOKE_SEMANTIC_BMP_BLOCK_SIZE,
        (uint64_t) index.record_count * EVOKE_SEMANTIC_BMP_RECORD_SIZE,
        index.posting_count * sizeof(uint32_t),
        bound_directory_bytes,
        term_major_exact_bytes,
        hybrid_authoritative_bytes,
        index.posting_count == 0
            ? 0.0
            : (double) hybrid_authoritative_bytes /
                (double) index.posting_count,
        packed_hybrid_bytes,
        index.posting_count == 0
            ? 0.0
            : (double) packed_hybrid_bytes /
                (double) index.posting_count,
        packed_serialized_size,
        packed_index.super_ref_bytes,
        packed_index.ref_bytes,
        serialized_size,
        index.posting_count == 0
            ? 0.0
            : (double) serialized_size / (double) index.posting_count,
        build_ms,
        packed_build_ms,
        prefix_count,
        prefix_bytes,
        prefix_build_ms,
        norm_bound_bytes,
        norm_build_ms,
        norm_ms[BENCHMARK_REPETITIONS / 2],
        norm_stats.postings_examined,
        norm_stats.adaptive_fallbacks,
        norm_stats.norm_bound_reductions,
        seed_count,
        seed_lookup_count,
        candidate_ms[BENCHMARK_REPETITIONS / 2],
        candidate_overlap,
        seeded_ms[BENCHMARK_REPETITIONS / 2],
        seeded_stats.postings_examined,
        seeded_stats.adaptive_fallbacks,
        oracle_seeded_ms[BENCHMARK_REPETITIONS / 2],
        oracle_seeded_stats.postings_examined,
        oracle_seeded_stats.adaptive_fallbacks,
        boundary_error_ratio,
        boundary_error,
        bounded_ms[BENCHMARK_REPETITIONS / 2],
        bounded_overlap,
        bounded_stats.postings_examined,
        bounded_stats.adaptive_fallbacks,
        bounded_stats.approximate_prune_events,
        bounded_stats.max_approximate_skipped_bound,
        bounded_stats.final_kth_score,
        expected.len >= top_k ? expected.scores[top_k - 1U] : 0.0f,
        oracle_boundary_error,
        oracle_bounded_ms[BENCHMARK_REPETITIONS / 2],
        oracle_bounded_overlap,
        oracle_bounded_stats.postings_examined,
        oracle_bounded_stats.adaptive_fallbacks,
        oracle_bounded_stats.approximate_prune_events,
        oracle_bounded_stats.max_approximate_skipped_bound,
        taat_ms[BENCHMARK_REPETITIONS / 2],
        bmp_ms[BENCHMARK_REPETITIONS / 2],
        taat_ms[BENCHMARK_REPETITIONS / 2] /
            bmp_ms[BENCHMARK_REPETITIONS / 2],
        packed_ms[BENCHMARK_REPETITIONS / 2],
        taat_ms[BENCHMARK_REPETITIONS / 2] /
            packed_ms[BENCHMARK_REPETITIONS / 2],
        packed_taat_ms[BENCHMARK_REPETITIONS / 2],
        packed_taat_stats.postings_examined,
        stats.super_bound_entries_visited,
        stats.superblocks_scored,
        stats.superblocks_skipped,
        stats.bound_entries_visited,
        stats.blocks_scored,
        stats.blocks_skipped,
        stats.forward_records_examined,
        query_posting_count == 0
            ? 0.0
            : (double) stats.forward_records_examined /
                (double) query_posting_count,
        stats.postings_examined,
        query_posting_count == 0
            ? 0.0
            : (double) stats.postings_examined /
                (double) query_posting_count,
        packed_stats.bound_entries_visited,
        packed_stats.blocks_scored,
        packed_stats.blocks_skipped,
        packed_stats.forward_records_examined,
        packed_stats.postings_examined,
        packed_stats.adaptive_fallbacks,
        stats.forward_records_examined >
            query_posting_count + UINT32_C(4096)
            ? "true"
            : "false",
        exact ? "true" : "false"
    );

cleanup_index:
    free(packed_serialized);
    evoke_topk_result_free(&candidate_actual);
    evoke_topk_result_free(&norm_actual);
    evoke_topk_result_free(&oracle_bounded_actual);
    evoke_topk_result_free(&bounded_actual);
    evoke_topk_result_free(&oracle_seeded_actual);
    evoke_topk_result_free(&seeded_actual);
    evoke_topk_result_free(&packed_taat_actual);
    evoke_topk_result_free(&packed_actual);
    evoke_topk_result_free(&actual);
    evoke_topk_result_free(&expected);
    evoke_semantic_bmp_index_free(&index);
    evoke_semantic_bmp_packed_index_free(&packed_index);
cleanup:
    free_norm_bounds(&norm_bounds);
    if (prefixes != NULL)
    {
        free(prefixes[0].lowest);
    }
    free(prefixes);
    free(seeds);
    free(seed_candidate_ids);
    free(tie_breaks);
    free(scores);
    free(query_weights);
    free(query_ids);
    free(runs);
    free(values);
    free(document_ids);
    return status == EVOKE_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
