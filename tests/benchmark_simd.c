#define _POSIX_C_SOURCE 200809L

#include "evoke_core.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool
checked_mul_size(size_t lhs, size_t rhs, size_t *result_out)
{
    if (result_out == NULL)
    {
        return false;
    }
    if (lhs == 0 || rhs == 0)
    {
        *result_out = 0;
        return true;
    }
    if (lhs > SIZE_MAX / rhs)
    {
        return false;
    }
    *result_out = lhs * rhs;
    return true;
}

static bool
parse_size_arg(const char *text, size_t *value_out)
{
    char *endptr = NULL;
    unsigned long long parsed = 0;

    if (text == NULL || text[0] == '\0' || value_out == NULL)
    {
        return false;
    }

    parsed = strtoull(text, &endptr, 10);
    if (endptr == NULL || *endptr != '\0')
    {
        return false;
    }
    if ((unsigned long long) ((size_t) parsed) != parsed)
    {
        return false;
    }

    *value_out = (size_t) parsed;
    return true;
}

static bool
mono_now_ns(uint64_t *ns_out)
{
    struct timespec ts;

    if (ns_out == NULL)
    {
        return false;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        return false;
    }
    *ns_out = ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
    return true;
}

static uint32_t
lcg_next_u32(uint64_t *state)
{
    *state = (*state * 6364136223846793005ULL) + 1ULL;
    return (uint32_t) (*state >> 32);
}

static evoke_status
run_benchmark(
    const evoke_index *index,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    size_t warmup_iters,
    size_t timed_iters,
    double *avg_ms_out,
    double *qps_out,
    double *checksum_out
)
{
    size_t i;
    uint64_t total_ns = 0;
    double checksum = 0.0;

    for (i = 0; i < warmup_iters + timed_iters; i++)
    {
        uint64_t started_ns;
        uint64_t finished_ns;
        uint64_t elapsed_ns;
        float *scores = NULL;
        evoke_status status;

        if (!mono_now_ns(&started_ns))
        {
            return EVOKE_ERR_INVALID;
        }
        status = evoke_scores_from_ids(
            index,
            query_ids,
            query_len,
            weight_mask,
            &scores
        );
        if (status != EVOKE_OK)
        {
            free(scores);
            return status;
        }
        if (!mono_now_ns(&finished_ns) || finished_ns < started_ns)
        {
            free(scores);
            return EVOKE_ERR_INVALID;
        }
        elapsed_ns = finished_ns - started_ns;

        checksum += (double) scores[(i * 2654435761U) % index->num_docs];
        free(scores);

        if (i >= warmup_iters)
        {
            total_ns += elapsed_ns;
        }
    }

    *avg_ms_out = ((double) total_ns / (double) timed_iters) / 1000000.0;
    if (*avg_ms_out > 0.0)
    {
        *qps_out = 1000.0 / *avg_ms_out;
    }
    else
    {
        *qps_out = 0.0;
    }
    *checksum_out = checksum;

    return EVOKE_OK;
}

int
main(int argc, char **argv)
{
    const char *simd_env = NULL;
    evoke_doc_ids *docs = NULL;
    uint32_t *all_tokens = NULL;
    uint32_t *query_ids = NULL;
    float *weight_mask = NULL;
    evoke_index index;
    evoke_params params;
    size_t num_docs = 250000;
    size_t terms_per_doc = 24;
    size_t vocab_size = 4096;
    size_t query_len = 12;
    size_t warmup_iters = 25;
    size_t timed_iters = 200;
    size_t total_tokens = 0;
    size_t i;
    uint64_t seed = 0xC0FFEEULL;
    double avg_ms = 0.0;
    double qps = 0.0;
    double checksum = 0.0;
    evoke_status status;

    if (argc > 1 && !parse_size_arg(argv[1], &num_docs))
    {
        fprintf(stderr, "invalid num_docs argument: %s\n", argv[1]);
        return 2;
    }
    if (argc > 2 && !parse_size_arg(argv[2], &timed_iters))
    {
        fprintf(stderr, "invalid timed_iters argument: %s\n", argv[2]);
        return 2;
    }
    if (argc > 3 && !parse_size_arg(argv[3], &warmup_iters))
    {
        fprintf(stderr, "invalid warmup_iters argument: %s\n", argv[3]);
        return 2;
    }
    if (num_docs == 0 || timed_iters == 0 || warmup_iters == 0)
    {
        fprintf(stderr, "num_docs/timed_iters/warmup_iters must be > 0\n");
        return 2;
    }
    if (query_len > vocab_size)
    {
        fprintf(stderr, "query_len must be <= vocab_size\n");
        return 2;
    }
    if (!checked_mul_size(num_docs, terms_per_doc, &total_tokens))
    {
        fprintf(stderr, "benchmark shape is too large\n");
        return 2;
    }

    docs = calloc(num_docs, sizeof(*docs));
    all_tokens = malloc(total_tokens * sizeof(*all_tokens));
    query_ids = malloc(query_len * sizeof(*query_ids));
    weight_mask = malloc(num_docs * sizeof(*weight_mask));
    if (docs == NULL || all_tokens == NULL || query_ids == NULL ||
        weight_mask == NULL)
    {
        fprintf(stderr, "allocation failed\n");
        free(docs);
        free(all_tokens);
        free(query_ids);
        free(weight_mask);
        return 2;
    }

    for (i = 0; i < num_docs; i++)
    {
        size_t base = i * terms_per_doc;
        size_t j;

        docs[i].token_ids = &all_tokens[base];
        docs[i].len = terms_per_doc;
        for (j = 0; j < terms_per_doc; j++)
        {
            all_tokens[base + j] = lcg_next_u32(&seed) % (uint32_t) vocab_size;
        }
        weight_mask[i] = 0.95f + ((float) (i % 101U) * 0.001f);
    }
    for (i = 0; i < query_len; i++)
    {
        query_ids[i] = (uint32_t) i;
    }

    memset(&params, 0, sizeof(params));
    params.k1 = 1.5f;
    params.b = 0.75f;
    params.delta = 0.5f;
    params.method = EVOKE_METHOD_BM25PLUS;
    params.idf_method = EVOKE_METHOD_LUCENE;

    evoke_index_init(&index);
    status = evoke_build_index_from_ids(
        docs,
        num_docs,
        &params,
        false,
        &index
    );
    if (status != EVOKE_OK)
    {
        fprintf(
            stderr,
            "failed to build benchmark index: %s\n",
            evoke_strerror(status)
        );
        free(docs);
        free(all_tokens);
        free(query_ids);
        free(weight_mask);
        return 1;
    }

    status = run_benchmark(
        &index,
        query_ids,
        query_len,
        weight_mask,
        warmup_iters,
        timed_iters,
        &avg_ms,
        &qps,
        &checksum
    );
    if (status != EVOKE_OK)
    {
        fprintf(stderr, "benchmark failed: %s\n", evoke_strerror(status));
        evoke_index_free(&index);
        free(docs);
        free(all_tokens);
        free(query_ids);
        free(weight_mask);
        return 1;
    }

    simd_env = getenv("EVOKE_SIMD_MODE");
    if (simd_env == NULL || simd_env[0] == '\0')
    {
        simd_env = "auto";
    }

    printf("simd_env=%s\n", simd_env);
    printf("simd_active=%s\n", evoke_active_simd_path());
    printf(
        "docs=%zu terms_per_doc=%zu vocab_size=%zu query_len=%zu "
        "timed_iters=%zu warmup_iters=%zu\n",
        num_docs,
        terms_per_doc,
        vocab_size,
        query_len,
        timed_iters,
        warmup_iters
    );
    printf(
        "avg_ms=%.4f qps=%.2f checksum=%.6f\n",
        avg_ms,
        qps,
        checksum
    );

    evoke_index_free(&index);
    free(docs);
    free(all_tokens);
    free(query_ids);
    free(weight_mask);
    return 0;
}
