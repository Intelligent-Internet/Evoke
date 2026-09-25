#define _POSIX_C_SOURCE 200809L

#include "evoke_core.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct extent_view
{
    evoke_term_extent_list *terms;
    evoke_posting_extent *extents;
    evoke_posting_value *values;
} extent_view;

typedef enum benchmark_score_mode
{
    BENCHMARK_SCORE_IMPACT = 0,
    BENCHMARK_SCORE_NEUTRAL = 1,
    BENCHMARK_SCORE_MIXED = 2
} benchmark_score_mode;

static bool
parse_size_arg(const char *text, size_t *value_out)
{
    char *endptr = NULL;
    unsigned long long parsed;

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

    if (ns_out == NULL || clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
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

static void
extent_view_free(extent_view *view)
{
    if (view == NULL)
    {
        return;
    }
    free(view->terms);
    free(view->extents);
    free(view->values);
    memset(view, 0, sizeof(*view));
}

static evoke_status
extent_view_use_canonical_values(
    const evoke_index *index,
    extent_view *view
)
{
    uint32_t term_id;
    uint64_t posting_index;

    if (index == NULL || view == NULL ||
        index->term_frequencies == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    view->values = calloc(index->data_len, sizeof(*view->values));
    if (view->values == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (posting_index = 0;
         posting_index < index->data_len;
         posting_index++)
    {
        view->values[posting_index].term_frequency =
            index->term_frequencies[posting_index];
    }
    for (term_id = 0; term_id < index->vocab_size; term_id++)
    {
        evoke_term_extent_list *list = &view->terms[term_id];
        size_t extent_index;

        for (extent_index = 0; extent_index < list->len; extent_index++)
        {
            evoke_posting_extent *extent =
                (evoke_posting_extent *) &list->extents[extent_index];
            ptrdiff_t offset =
                extent->term_frequencies - index->term_frequencies;

            if (offset < 0 || (uint64_t) offset >= index->data_len)
            {
                return EVOKE_ERR_FORMAT;
            }
            extent->values = &view->values[offset];
            extent->term_frequencies = NULL;
        }
    }
    return EVOKE_OK;
}

static evoke_status
extent_view_build(
    const evoke_index *index,
    size_t target_extents_per_term,
    extent_view *view_out
)
{
    size_t max_extents;
    size_t extent_cursor = 0;
    uint32_t term_id;

    if (index == NULL || target_extents_per_term == 0 || view_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if ((size_t) index->vocab_size > SIZE_MAX / target_extents_per_term)
    {
        return EVOKE_ERR_RANGE;
    }
    max_extents = (size_t) index->vocab_size * target_extents_per_term;

    memset(view_out, 0, sizeof(*view_out));
    view_out->terms = calloc(index->vocab_size, sizeof(*view_out->terms));
    view_out->extents = calloc(max_extents, sizeof(*view_out->extents));
    if (view_out->terms == NULL || view_out->extents == NULL)
    {
        extent_view_free(view_out);
        return EVOKE_ERR_NOMEM;
    }

    for (term_id = 0; term_id < index->vocab_size; term_id++)
    {
        uint64_t start = index->indptr[term_id];
        uint64_t end = index->indptr[term_id + 1];
        uint64_t posting_count = end - start;
        size_t extent_count = target_extents_per_term;
        uint64_t cursor = start;
        size_t i;

        if (posting_count == 0)
        {
            continue;
        }
        if ((uint64_t) extent_count > posting_count)
        {
            extent_count = (size_t) posting_count;
        }

        view_out->terms[term_id].extents =
            &view_out->extents[extent_cursor];
        view_out->terms[term_id].len = extent_count;

        for (i = 0; i < extent_count; i++)
        {
            uint64_t remaining = end - cursor;
            uint64_t remaining_extents = (uint64_t) (extent_count - i);
            uint64_t extent_len =
                (remaining + remaining_extents - 1) / remaining_extents;
            evoke_posting_extent *extent =
                &view_out->extents[extent_cursor++];

            extent->data = &index->data[cursor];
            extent->indices = &index->indices[cursor];
            extent->term_frequencies = &index->term_frequencies[cursor];
            extent->len = extent_len;
            extent->local_document_count = index->num_docs;
            extent->kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
            if (evoke_posting_extent_validate_layout(index, extent) !=
                EVOKE_OK)
            {
                extent_view_free(view_out);
                return EVOKE_ERR_FORMAT;
            }
            cursor += extent_len;
        }
        if (cursor != end)
        {
            extent_view_free(view_out);
            return EVOKE_ERR_FORMAT;
        }
    }

    return EVOKE_OK;
}

static evoke_status
run_benchmark(
    const evoke_index *index,
    const extent_view *view,
    const evoke_corpus_stats *stats,
    benchmark_score_mode mode,
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
    uint64_t total_ns = 0;
    double checksum = 0.0;
    size_t i;

    for (i = 0; i < warmup_iters + timed_iters; i++)
    {
        uint64_t started_ns;
        uint64_t finished_ns;
        float *scores = NULL;
        evoke_topk_result topk;
        evoke_status status;

        memset(&topk, 0, sizeof(topk));
        if (!mono_now_ns(&started_ns))
        {
            return EVOKE_ERR_INVALID;
        }
        if (mode == BENCHMARK_SCORE_MIXED)
        {
            status = evoke_scores_from_ids_mixed(
                index,
                stats,
                view == NULL ? NULL : view->terms,
                view == NULL ? 0 : index->vocab_size,
                query_ids,
                query_len,
                weight_mask,
                &scores
            );
        }
        else if (mode == BENCHMARK_SCORE_NEUTRAL)
        {
            status = evoke_scores_from_ids_neutral(
                index,
                stats,
                view == NULL ? NULL : view->terms,
                view == NULL ? 0 : index->vocab_size,
                query_ids,
                query_len,
                weight_mask,
                &scores
            );
        }
        else if (view == NULL)
        {
            status = evoke_scores_from_ids(
                index,
                query_ids,
                query_len,
                weight_mask,
                &scores
            );
        }
        else
        {
            status = evoke_scores_from_ids_extents(
                index,
                view->terms,
                index->vocab_size,
                query_ids,
                query_len,
                weight_mask,
                &scores
            );
        }
        if (status != EVOKE_OK)
        {
            free(scores);
            return status;
        }
        status = evoke_topk(
            scores,
            index->num_docs,
            index->num_docs < 100 ? index->num_docs : 100,
            true,
            &topk
        );
        if (status != EVOKE_OK)
        {
            free(scores);
            evoke_topk_result_free(&topk);
            return status;
        }
        if (!mono_now_ns(&finished_ns) || finished_ns < started_ns)
        {
            free(scores);
            evoke_topk_result_free(&topk);
            return EVOKE_ERR_INVALID;
        }

        if (topk.len > 0)
        {
            checksum += (double) topk.scores[i % topk.len];
            checksum += (double) topk.doc_ids[i % topk.len] * 0.000001;
        }
        free(scores);
        evoke_topk_result_free(&topk);

        if (i >= warmup_iters)
        {
            total_ns += finished_ns - started_ns;
        }
    }

    *avg_ms_out = ((double) total_ns / (double) timed_iters) / 1000000.0;
    *qps_out = *avg_ms_out > 0.0 ? 1000.0 / *avg_ms_out : 0.0;
    *checksum_out = checksum;
    return EVOKE_OK;
}

static evoke_status
verify_impact_exact(
    const evoke_index *index,
    const extent_view *view,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask
)
{
    float *contiguous = NULL;
    float *fragmented = NULL;
    evoke_status status;

    status = evoke_scores_from_ids(
        index,
        query_ids,
        query_len,
        weight_mask,
        &contiguous
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_scores_from_ids_extents(
        index,
        view->terms,
        index->vocab_size,
        query_ids,
        query_len,
        weight_mask,
        &fragmented
    );
    if (status == EVOKE_OK &&
        memcmp(
            contiguous,
            fragmented,
            index->num_docs * sizeof(*contiguous)
        ) != 0)
    {
        status = EVOKE_ERR_FORMAT;
    }

    free(contiguous);
    free(fragmented);
    return status;
}

static evoke_status
verify_neutral_exact(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const extent_view *view,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask
)
{
    float *contiguous = NULL;
    float *fragmented = NULL;
    evoke_topk_result impact_topk;
    evoke_topk_result neutral_topk;
    evoke_status status;

    memset(&impact_topk, 0, sizeof(impact_topk));
    memset(&neutral_topk, 0, sizeof(neutral_topk));

    status = evoke_scores_from_ids_neutral(
        index,
        stats,
        NULL,
        0,
        query_ids,
        query_len,
        weight_mask,
        &contiguous
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_scores_from_ids_neutral(
        index,
        stats,
        view->terms,
        index->vocab_size,
        query_ids,
        query_len,
        weight_mask,
        &fragmented
    );
    if (status == EVOKE_OK &&
        memcmp(
            contiguous,
            fragmented,
            index->num_docs * sizeof(*contiguous)
        ) != 0)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        float *impact = NULL;

        status = evoke_scores_from_ids(
            index,
            query_ids,
            query_len,
            weight_mask,
            &impact
        );
        if (status == EVOKE_OK)
        {
            size_t topk_len = index->num_docs < 100 ?
                index->num_docs :
                100;

            status = evoke_topk(
                impact,
                index->num_docs,
                topk_len,
                true,
                &impact_topk
            );
            if (status == EVOKE_OK)
            {
                status = evoke_topk(
                    contiguous,
                    index->num_docs,
                    topk_len,
                    true,
                    &neutral_topk
                );
            }
            if (status == EVOKE_OK &&
                (impact_topk.len != neutral_topk.len ||
                 memcmp(
                     impact_topk.doc_ids,
                     neutral_topk.doc_ids,
                     impact_topk.len * sizeof(*impact_topk.doc_ids)
                 ) != 0))
            {
                status = EVOKE_ERR_FORMAT;
            }
        }
        free(impact);
    }

    free(contiguous);
    free(fragmented);
    evoke_topk_result_free(&impact_topk);
    evoke_topk_result_free(&neutral_topk);
    return status;
}

static evoke_status
verify_mixed_exact(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const extent_view *view,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask
)
{
    float *neutral = NULL;
    float *mixed = NULL;
    evoke_status status;

    status = evoke_scores_from_ids_neutral(
        index,
        stats,
        view->terms,
        index->vocab_size,
        query_ids,
        query_len,
        weight_mask,
        &neutral
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_scores_from_ids_mixed(
        index,
        stats,
        view->terms,
        index->vocab_size,
        query_ids,
        query_len,
        weight_mask,
        &mixed
    );
    if (status == EVOKE_OK &&
        memcmp(
            neutral,
            mixed,
            index->num_docs * sizeof(*neutral)
        ) != 0)
    {
        status = EVOKE_ERR_FORMAT;
    }

    free(neutral);
    free(mixed);
    return status;
}

int
main(int argc, char **argv)
{
    const size_t extent_counts[] = {1, 2, 4, 8};
    evoke_doc_ids *docs = NULL;
    uint32_t *all_tokens = NULL;
    uint32_t *query_ids = NULL;
    float *weight_mask = NULL;
    evoke_index index;
    evoke_params params;
    evoke_corpus_stats stats;
    size_t num_docs = 250000;
    size_t timed_iters = 200;
    size_t warmup_iters = 25;
    size_t terms_per_doc = 24;
    size_t vocab_size = 4096;
    size_t query_len = 12;
    size_t total_tokens;
    uint64_t seed = 0xC0FFEEULL;
    double baseline_ms = 0.0;
    double neutral_baseline_ms = 0.0;
    size_t i;
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
    if (num_docs == 0 || timed_iters == 0 || warmup_iters == 0 ||
        num_docs > SIZE_MAX / terms_per_doc)
    {
        fprintf(stderr, "invalid benchmark shape\n");
        return 2;
    }
    total_tokens = num_docs * terms_per_doc;

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
            all_tokens[base + j] =
                lcg_next_u32(&seed) % (uint32_t) vocab_size;
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
        fprintf(stderr, "index build failed: %s\n", evoke_strerror(status));
        goto fail;
    }

    memset(&stats, 0, sizeof(stats));
    stats.document_count = index.num_docs;
    stats.doc_frequencies = index.doc_frequencies;
    stats.vocab_size = index.vocab_size;
    for (i = 0; i < index.num_docs; i++)
    {
        stats.total_document_length += index.doc_lengths[i];
    }

    for (i = 0; i <= sizeof(extent_counts) / sizeof(extent_counts[0]); i++)
    {
        extent_view view;
        const extent_view *view_ptr = NULL;
        size_t extent_count = 0;
        double avg_ms;
        double qps;
        double checksum;

        memset(&view, 0, sizeof(view));
        if (i > 0)
        {
            extent_count = extent_counts[i - 1];
            status = extent_view_build(&index, extent_count, &view);
            if (status != EVOKE_OK)
            {
                fprintf(
                    stderr,
                    "extent view build failed: %s\n",
                    evoke_strerror(status)
                );
                goto fail;
            }
            status = verify_impact_exact(
                &index,
                &view,
                query_ids,
                query_len,
                weight_mask
            );
            if (status != EVOKE_OK)
            {
                fprintf(
                    stderr,
                    "extent parity failed for %zu extents: %s\n",
                    extent_count,
                    evoke_strerror(status)
                );
                extent_view_free(&view);
                goto fail;
            }
            view_ptr = &view;
        }

        status = run_benchmark(
            &index,
            view_ptr,
            &stats,
            BENCHMARK_SCORE_IMPACT,
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
            extent_view_free(&view);
            goto fail;
        }

        if (i == 0)
        {
            baseline_ms = avg_ms;
            printf(
                "mode=contiguous extents=1 avg_ms=%.6f qps=%.2f "
                "ratio=1.000000 checksum=%.6f exact=true\n",
                avg_ms,
                qps,
                checksum
            );
        }
        else
        {
            printf(
                "mode=fragmented extents=%zu avg_ms=%.6f qps=%.2f "
                "ratio=%.6f checksum=%.6f exact=true\n",
                extent_count,
                avg_ms,
                qps,
                baseline_ms > 0.0 ? avg_ms / baseline_ms : 0.0,
                checksum
            );
        }
        extent_view_free(&view);
    }

    for (i = 0; i <= sizeof(extent_counts) / sizeof(extent_counts[0]); i++)
    {
        extent_view view;
        const extent_view *view_ptr = NULL;
        size_t extent_count = 0;
        double avg_ms;
        double qps;
        double checksum;

        memset(&view, 0, sizeof(view));
        if (i > 0)
        {
            extent_count = extent_counts[i - 1];
            status = extent_view_build(&index, extent_count, &view);
            if (status != EVOKE_OK)
            {
                fprintf(
                    stderr,
                    "neutral extent view build failed: %s\n",
                    evoke_strerror(status)
                );
                goto fail;
            }
            status = verify_neutral_exact(
                &index,
                &stats,
                &view,
                query_ids,
                query_len,
                weight_mask
            );
            if (status != EVOKE_OK)
            {
                fprintf(
                    stderr,
                    "neutral parity failed for %zu extents: %s\n",
                    extent_count,
                    evoke_strerror(status)
                );
                extent_view_free(&view);
                goto fail;
            }
            view_ptr = &view;
        }

        status = run_benchmark(
            &index,
            view_ptr,
            &stats,
            BENCHMARK_SCORE_NEUTRAL,
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
            fprintf(
                stderr,
                "neutral benchmark failed: %s\n",
                evoke_strerror(status)
            );
            extent_view_free(&view);
            goto fail;
        }

        if (i == 0)
        {
            neutral_baseline_ms = avg_ms;
            printf(
                "mode=neutral_contiguous extents=1 avg_ms=%.6f qps=%.2f "
                "ratio_to_impact=%.6f checksum=%.6f exact=true\n",
                avg_ms,
                qps,
                baseline_ms > 0.0 ? avg_ms / baseline_ms : 0.0,
                checksum
            );
        }
        else
        {
            printf(
                "mode=neutral_fragmented extents=%zu avg_ms=%.6f qps=%.2f "
                "ratio_to_neutral=%.6f ratio_to_impact=%.6f "
                "checksum=%.6f exact=true\n",
                extent_count,
                avg_ms,
                qps,
                neutral_baseline_ms > 0.0 ?
                    avg_ms / neutral_baseline_ms :
                    0.0,
                baseline_ms > 0.0 ? avg_ms / baseline_ms : 0.0,
                checksum
            );
        }
        extent_view_free(&view);
    }

    for (i = 0; i < sizeof(extent_counts) / sizeof(extent_counts[0]); i++)
    {
        extent_view view;
        size_t extent_count = extent_counts[i];
        double avg_ms;
        double qps;
        double checksum;

        memset(&view, 0, sizeof(view));
        status = extent_view_build(&index, extent_count, &view);
        if (status != EVOKE_OK)
        {
            fprintf(
                stderr,
                "mixed extent view build failed: %s\n",
                evoke_strerror(status)
            );
            goto fail;
        }
        status = extent_view_use_canonical_values(&index, &view);
        if (status != EVOKE_OK)
        {
            fprintf(
                stderr,
                "canonical value view failed: %s\n",
                evoke_strerror(status)
            );
            extent_view_free(&view);
            goto fail;
        }
        status = verify_mixed_exact(
            &index,
            &stats,
            &view,
            query_ids,
            query_len,
            weight_mask
        );
        if (status != EVOKE_OK)
        {
            fprintf(
                stderr,
                "mixed parity failed for %zu extents: %s\n",
                extent_count,
                evoke_strerror(status)
            );
            extent_view_free(&view);
            goto fail;
        }
        status = run_benchmark(
            &index,
            &view,
            &stats,
            BENCHMARK_SCORE_MIXED,
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
            fprintf(
                stderr,
                "mixed benchmark failed: %s\n",
                evoke_strerror(status)
            );
            extent_view_free(&view);
            goto fail;
        }
        printf(
            "mode=canonical_mixed extents=%zu avg_ms=%.6f qps=%.2f "
            "ratio_to_neutral=%.6f ratio_to_impact=%.6f "
            "checksum=%.6f exact=true\n",
            extent_count,
            avg_ms,
            qps,
            neutral_baseline_ms > 0.0 ?
                avg_ms / neutral_baseline_ms :
                0.0,
            baseline_ms > 0.0 ? avg_ms / baseline_ms : 0.0,
            checksum
        );
        extent_view_free(&view);
    }

    evoke_index_free(&index);
    free(docs);
    free(all_tokens);
    free(query_ids);
    free(weight_mask);
    return 0;

fail:
    evoke_index_free(&index);
    free(docs);
    free(all_tokens);
    free(query_ids);
    free(weight_mask);
    return 1;
}
