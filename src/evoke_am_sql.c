#include "postgres.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "access/htup_details.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "funcapi.h"
#include "storage/itemptr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/typcache.h"

#include "evoke_pg_common.h"

typedef enum evoke_hybrid_fusion_method
{
    EVOKE_HYBRID_FUSION_RRF = 0,
    EVOKE_HYBRID_FUSION_SCORE
} evoke_hybrid_fusion_method;

typedef enum evoke_hybrid_normalizer
{
    EVOKE_HYBRID_NORMALIZER_IDENTITY = 0,
    EVOKE_HYBRID_NORMALIZER_NEGATIVE_DISTANCE,
    EVOKE_HYBRID_NORMALIZER_INVERSE_DISTANCE,
    EVOKE_HYBRID_NORMALIZER_MINMAX,
    EVOKE_HYBRID_NORMALIZER_ZSCORE,
    EVOKE_HYBRID_NORMALIZER_RANK
} evoke_hybrid_normalizer;

typedef enum evoke_hybrid_direction
{
    EVOKE_HYBRID_DIRECTION_HIGHER = 0,
    EVOKE_HYBRID_DIRECTION_LOWER
} evoke_hybrid_direction;

typedef struct evoke_hybrid_candidate_state
{
    char *source_name;
    char *tid_text;
    ItemPointerData tid;
    double raw_value;
    int32 source_rank_input;
    bool source_rank_is_null;
    int32 source_rank;
    double weight;
    evoke_hybrid_normalizer normalizer;
    evoke_hybrid_direction direction;
    size_t source_id;
    double normalized_score;
    double weighted_score;
    bool scored;
} evoke_hybrid_candidate_state;

typedef struct evoke_hybrid_source_state
{
    char *source_name;
    double min_value;
    double max_value;
    double sum_value;
    double sumsq_value;
    size_t count;
} evoke_hybrid_source_state;

typedef struct evoke_hybrid_hit_state
{
    ItemPointerData tid;
    char *tid_text;
    float4 score;
    int32 source_count;
    char **source_names;
    float4 *raw_values;
    float4 *normalized_scores;
    float4 *weighted_scores;
    int32 *ranks;
} evoke_hybrid_hit_state;

typedef struct evoke_hybrid_search_state
{
    evoke_hybrid_hit_state *hits;
    size_t len;
    size_t pos;
} evoke_hybrid_search_state;

static Oid
evoke_am_text_array_element_type(
    ArrayType *array,
    const char *context
)
{
    Oid element_type = ARR_ELEMTYPE(array);

    if (element_type != TEXTOID && element_type != VARCHAROID)
    {
        ereport(
            ERROR,
            (
                errmsg(
                    "%s must use text[] or varchar[] values",
                    context
                )
            )
        );
    }

    return element_type;
}

static Datum
evoke_am_score_overlap_int4(ArrayType *doc_array, ArrayType *query_array)
{
    Datum *doc_datums = NULL;
    Datum *query_datums = NULL;
    bool *doc_nulls = NULL;
    bool *query_nulls = NULL;
    int doc_nelems = 0;
    int query_nelems = 0;
    int i;
    int overlap = 0;

    if (ARR_NDIM(doc_array) != 1 || ARR_NDIM(query_array) != 1)
    {
        ereport(
            ERROR,
            (
                errmsg("bm25 distance operator requires one-dimensional int4[] values")
            )
        );
    }

    deconstruct_array(
        doc_array,
        INT4OID,
        4,
        true,
        TYPALIGN_INT,
        &doc_datums,
        &doc_nulls,
        &doc_nelems
    );
    deconstruct_array(
        query_array,
        INT4OID,
        4,
        true,
        TYPALIGN_INT,
        &query_datums,
        &query_nulls,
        &query_nelems
    );

    for (i = 0; i < query_nelems; i++)
    {
        int j;

        if (query_nulls[i])
        {
            continue;
        }

        for (j = 0; j < doc_nelems; j++)
        {
            if (doc_nulls[j])
            {
                continue;
            }
            if (DatumGetInt32(doc_datums[j]) == DatumGetInt32(query_datums[i]))
            {
                overlap++;
                break;
            }
        }
    }

    if (doc_datums != NULL)
    {
        pfree(doc_datums);
    }
    if (doc_nulls != NULL)
    {
        pfree(doc_nulls);
    }
    if (query_datums != NULL)
    {
        pfree(query_datums);
    }
    if (query_nulls != NULL)
    {
        pfree(query_nulls);
    }

    return Float8GetDatum(-(double) overlap);
}

static Datum
evoke_am_score_overlap_text(ArrayType *doc_array, ArrayType *query_array)
{
    Datum *doc_datums = NULL;
    Datum *query_datums = NULL;
    bool *doc_nulls = NULL;
    bool *query_nulls = NULL;
    int doc_nelems = 0;
    int query_nelems = 0;
    int i;
    int overlap = 0;

    if (ARR_NDIM(doc_array) != 1 || ARR_NDIM(query_array) != 1)
    {
        ereport(
            ERROR,
            (
                errmsg(
                    "bm25 distance operator requires one-dimensional text[] or varchar[] values"
                )
            )
        );
    }

    (void) evoke_am_text_array_element_type(
        doc_array,
        "bm25 distance operator"
    );
    (void) evoke_am_text_array_element_type(
        query_array,
        "bm25 distance operator"
    );
    deconstruct_array(
        doc_array,
        ARR_ELEMTYPE(doc_array),
        -1,
        false,
        TYPALIGN_INT,
        &doc_datums,
        &doc_nulls,
        &doc_nelems
    );
    deconstruct_array(
        query_array,
        ARR_ELEMTYPE(query_array),
        -1,
        false,
        TYPALIGN_INT,
        &query_datums,
        &query_nulls,
        &query_nelems
    );

    for (i = 0; i < query_nelems; i++)
    {
        int j;
        text *query_text;

        if (query_nulls[i])
        {
            continue;
        }

        query_text = DatumGetTextPP(query_datums[i]);
        for (j = 0; j < doc_nelems; j++)
        {
            text *doc_text;

            if (doc_nulls[j])
            {
                continue;
            }

            doc_text = DatumGetTextPP(doc_datums[j]);
            if (VARSIZE_ANY_EXHDR(doc_text) == VARSIZE_ANY_EXHDR(query_text) &&
                memcmp(
                    VARDATA_ANY(doc_text),
                    VARDATA_ANY(query_text),
                    VARSIZE_ANY_EXHDR(doc_text)
                ) == 0)
            {
                overlap++;
                break;
            }
        }
    }

    if (doc_datums != NULL)
    {
        pfree(doc_datums);
    }
    if (doc_nulls != NULL)
    {
        pfree(doc_nulls);
    }
    if (query_datums != NULL)
    {
        pfree(query_datums);
    }
    if (query_nulls != NULL)
    {
        pfree(query_nulls);
    }

    return Float8GetDatum(-(double) overlap);
}

PG_FUNCTION_INFO_V1(evoke_score_ids_op);
Datum
evoke_score_ids_op(PG_FUNCTION_ARGS)
{
    ArrayType *doc_array = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *query_array = PG_GETARG_ARRAYTYPE_P(1);

    PG_RETURN_DATUM(evoke_am_score_overlap_int4(doc_array, query_array));
}

PG_FUNCTION_INFO_V1(evoke_score_tokens_op);
Datum
evoke_score_tokens_op(PG_FUNCTION_ARGS)
{
    ArrayType *doc_array = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *query_array = PG_GETARG_ARRAYTYPE_P(1);

    PG_RETURN_DATUM(evoke_am_score_overlap_text(doc_array, query_array));
}

static char *
evoke_hybrid_tid_text(const ItemPointerData *tid)
{
    char buffer[64];

    snprintf(
        buffer,
        sizeof(buffer),
        "(%u,%u)",
        BlockIdGetBlockNumber(&tid->ip_blkid),
        ItemPointerGetOffsetNumber(tid)
    );
    return pstrdup(buffer);
}

static int
evoke_hybrid_strcmp(const char *left, const char *right)
{
    if (left == NULL && right == NULL)
    {
        return 0;
    }
    if (left == NULL)
    {
        return -1;
    }
    if (right == NULL)
    {
        return 1;
    }
    return strcmp(left, right);
}

static int
evoke_hybrid_cmp_double_asc(double left, double right)
{
    if (left < right)
    {
        return -1;
    }
    if (left > right)
    {
        return 1;
    }
    return 0;
}

static int
evoke_hybrid_cmp_double_desc(double left, double right)
{
    return evoke_hybrid_cmp_double_asc(right, left);
}

static int
evoke_hybrid_parse_fusion(const char *value)
{
    const char *effective = value == NULL ? "rrf" : value;

    if (pg_strcasecmp(effective, "rrf") == 0)
    {
        return EVOKE_HYBRID_FUSION_RRF;
    }
    if (pg_strcasecmp(effective, "score") == 0)
    {
        return EVOKE_HYBRID_FUSION_SCORE;
    }
    ereport(
        ERROR,
        (
            errmsg("unsupported evoke hybrid fusion method: %s",
                   effective),
            errhint("Use rrf or score.")
        )
    );
    return EVOKE_HYBRID_FUSION_RRF;
}

static int
evoke_hybrid_parse_normalizer(const char *value)
{
    const char *effective = value == NULL ? "identity" : value;

    if (pg_strcasecmp(effective, "identity") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_IDENTITY;
    }
    if (pg_strcasecmp(effective, "negative_distance") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_NEGATIVE_DISTANCE;
    }
    if (pg_strcasecmp(effective, "inverse_distance") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_INVERSE_DISTANCE;
    }
    if (pg_strcasecmp(effective, "minmax") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_MINMAX;
    }
    if (pg_strcasecmp(effective, "zscore") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_ZSCORE;
    }
    if (pg_strcasecmp(effective, "rank") == 0)
    {
        return EVOKE_HYBRID_NORMALIZER_RANK;
    }
    ereport(
        ERROR,
        (
            errmsg("unsupported evoke hybrid normalizer: %s",
                   effective),
            errhint("Use identity, negative_distance, inverse_distance, "
                    "minmax, zscore, or rank.")
        )
    );
    return EVOKE_HYBRID_NORMALIZER_IDENTITY;
}

static int
evoke_hybrid_parse_direction(const char *value)
{
    const char *effective = value == NULL ? "higher_is_better" : value;

    if (pg_strcasecmp(effective, "higher_is_better") == 0)
    {
        return EVOKE_HYBRID_DIRECTION_HIGHER;
    }
    if (pg_strcasecmp(effective, "lower_is_better") == 0)
    {
        return EVOKE_HYBRID_DIRECTION_LOWER;
    }
    ereport(
        ERROR,
        (
            errmsg("unsupported evoke hybrid direction: %s",
                   effective),
            errhint("Use higher_is_better or lower_is_better.")
        )
    );
    return EVOKE_HYBRID_DIRECTION_HIGHER;
}

static int
evoke_hybrid_candidate_rank_cmp(const void *left, const void *right)
{
    const evoke_hybrid_candidate_state *a = left;
    const evoke_hybrid_candidate_state *b = right;
    int cmp;

    cmp = evoke_hybrid_strcmp(a->source_name, b->source_name);
    if (cmp != 0)
    {
        return cmp;
    }
    if (a->direction != b->direction)
    {
        return a->direction == EVOKE_HYBRID_DIRECTION_LOWER ? -1 : 1;
    }
    if (a->direction == EVOKE_HYBRID_DIRECTION_LOWER)
    {
        cmp = evoke_hybrid_cmp_double_asc(a->raw_value, b->raw_value);
    }
    else
    {
        cmp = evoke_hybrid_cmp_double_desc(a->raw_value, b->raw_value);
    }
    if (cmp != 0)
    {
        return cmp;
    }
    return evoke_hybrid_strcmp(a->tid_text, b->tid_text);
}

static int
evoke_hybrid_candidate_dedup_cmp(const void *left, const void *right)
{
    const evoke_hybrid_candidate_state *a = left;
    const evoke_hybrid_candidate_state *b = right;
    int cmp;

    cmp = evoke_hybrid_strcmp(a->source_name, b->source_name);
    if (cmp != 0)
    {
        return cmp;
    }
    cmp = evoke_hybrid_strcmp(a->tid_text, b->tid_text);
    if (cmp != 0)
    {
        return cmp;
    }
    cmp = evoke_hybrid_cmp_double_desc(
        a->weighted_score,
        b->weighted_score
    );
    if (cmp != 0)
    {
        return cmp;
    }
    if (a->source_rank < b->source_rank)
    {
        return -1;
    }
    if (a->source_rank > b->source_rank)
    {
        return 1;
    }
    if (a->direction != b->direction)
    {
        return a->direction == EVOKE_HYBRID_DIRECTION_LOWER ? -1 : 1;
    }
    if (a->direction == EVOKE_HYBRID_DIRECTION_LOWER)
    {
        return evoke_hybrid_cmp_double_asc(
            a->raw_value,
            b->raw_value
        );
    }
    return evoke_hybrid_cmp_double_desc(a->raw_value, b->raw_value);
}

static int
evoke_hybrid_candidate_tid_source_cmp(
    const void *left,
    const void *right
)
{
    const evoke_hybrid_candidate_state *a = left;
    const evoke_hybrid_candidate_state *b = right;
    int cmp;

    cmp = evoke_hybrid_strcmp(a->tid_text, b->tid_text);
    if (cmp != 0)
    {
        return cmp;
    }
    return evoke_hybrid_strcmp(a->source_name, b->source_name);
}

static int
evoke_hybrid_hit_cmp(const void *left, const void *right)
{
    const evoke_hybrid_hit_state *a = left;
    const evoke_hybrid_hit_state *b = right;
    int cmp;

    cmp = evoke_hybrid_cmp_double_desc(a->score, b->score);
    if (cmp != 0)
    {
        return cmp;
    }
    return evoke_hybrid_strcmp(a->tid_text, b->tid_text);
}

static size_t
evoke_hybrid_find_or_add_source(
    evoke_hybrid_source_state *sources,
    size_t *source_len,
    const char *source_name
)
{
    size_t i;

    for (i = 0; i < *source_len; i++)
    {
        if (strcmp(sources[i].source_name, source_name) == 0)
        {
            return i;
        }
    }

    sources[*source_len].source_name = pstrdup(source_name);
    sources[*source_len].min_value = DBL_MAX;
    sources[*source_len].max_value = -DBL_MAX;
    sources[*source_len].sum_value = 0.0;
    sources[*source_len].sumsq_value = 0.0;
    sources[*source_len].count = 0;
    (*source_len)++;
    return *source_len - 1;
}

static void
evoke_hybrid_deform_candidate(
    Datum candidate_datum,
    evoke_hybrid_candidate_state *candidate_out,
    evoke_hybrid_source_state *sources,
    size_t *source_len
)
{
    HeapTupleHeader header = DatumGetHeapTupleHeader(candidate_datum);
    Oid tuple_type = HeapTupleHeaderGetTypeId(header);
    int32 tuple_typmod = HeapTupleHeaderGetTypMod(header);
    TupleDesc tupdesc = lookup_rowtype_tupdesc(tuple_type, tuple_typmod);
    HeapTupleData tuple;
    Datum values[7];
    bool nulls[7];
    char *source_name = NULL;
    char *normalizer = NULL;
    char *direction = NULL;

    memset(candidate_out, 0, sizeof(*candidate_out));
    if (tupdesc->natts != 7)
    {
        ReleaseTupleDesc(tupdesc);
        ereport(ERROR, (errmsg("invalid evoke hybrid candidate type")));
    }

    tuple.t_len = HeapTupleHeaderGetDatumLength(header);
    ItemPointerSetInvalid(&tuple.t_self);
    tuple.t_tableOid = InvalidOid;
    tuple.t_data = header;
    heap_deform_tuple(&tuple, tupdesc, values, nulls);

    if (nulls[1] || nulls[2])
    {
        candidate_out->scored = false;
        ReleaseTupleDesc(tupdesc);
        return;
    }

    if (!nulls[0])
    {
        source_name = text_to_cstring(DatumGetTextPP(values[0]));
    }
    if (source_name == NULL || source_name[0] == '\0')
    {
        source_name = pstrdup("source");
    }

    candidate_out->source_name = source_name;
    candidate_out->tid = *DatumGetItemPointer(values[1]);
    candidate_out->tid_text = evoke_hybrid_tid_text(&candidate_out->tid);
    candidate_out->raw_value = (double) DatumGetFloat4(values[2]);
    if (!isfinite(candidate_out->raw_value))
    {
        candidate_out->scored = false;
        ReleaseTupleDesc(tupdesc);
        return;
    }

    candidate_out->source_rank_is_null = nulls[3];
    candidate_out->source_rank_input = nulls[3] ? 0 : DatumGetInt32(values[3]);
    candidate_out->weight = nulls[4] ? 1.0 : (double) DatumGetFloat4(values[4]);

    if (!nulls[6])
    {
        direction = text_to_cstring(DatumGetTextPP(values[6]));
    }
    candidate_out->direction = evoke_hybrid_parse_direction(direction);

    if (!nulls[5])
    {
        normalizer = text_to_cstring(DatumGetTextPP(values[5]));
    }
    if (normalizer == NULL)
    {
        normalizer = candidate_out->direction ==
            EVOKE_HYBRID_DIRECTION_LOWER
            ? "negative_distance"
            : "identity";
    }
    candidate_out->normalizer =
        evoke_hybrid_parse_normalizer(normalizer);
    candidate_out->source_id = evoke_hybrid_find_or_add_source(
        sources,
        source_len,
        candidate_out->source_name
    );

    candidate_out->scored = true;
    ReleaseTupleDesc(tupdesc);
}

static void
evoke_hybrid_prepare_candidates(
    evoke_hybrid_candidate_state *candidates,
    size_t candidate_len,
    evoke_hybrid_source_state *sources,
    evoke_hybrid_fusion_method fusion_method,
    double rrf_k,
    double epsilon
)
{
    size_t i;
    size_t current_source = (size_t) -1;
    int32 rank_position = 0;

    if (candidate_len == 0)
    {
        return;
    }

    qsort(
        candidates,
        candidate_len,
        sizeof(*candidates),
        evoke_hybrid_candidate_rank_cmp
    );

    for (i = 0; i < candidate_len; i++)
    {
        evoke_hybrid_source_state *source;

        if (i == 0 || candidates[i].source_id != current_source)
        {
            current_source = candidates[i].source_id;
            rank_position = 1;
        }
        else
        {
            rank_position++;
        }

        if (candidates[i].source_rank_is_null ||
            candidates[i].source_rank_input == 0)
        {
            candidates[i].source_rank = rank_position;
        }
        else
        {
            candidates[i].source_rank = candidates[i].source_rank_input;
        }

        source = &sources[candidates[i].source_id];
        source->min_value = Min(source->min_value, candidates[i].raw_value);
        source->max_value = Max(source->max_value, candidates[i].raw_value);
        source->sum_value += candidates[i].raw_value;
        source->sumsq_value +=
            candidates[i].raw_value * candidates[i].raw_value;
        source->count++;
    }

    for (i = 0; i < candidate_len; i++)
    {
        evoke_hybrid_source_state *source;
        double normalized_score = 0.0;
        double avg_value;
        double variance;
        double stddev_value;

        if (candidates[i].source_rank <= 0)
        {
            candidates[i].scored = false;
            continue;
        }

        source = &sources[candidates[i].source_id];
        avg_value = source->sum_value / (double) source->count;
        variance = source->sumsq_value / (double) source->count
            - avg_value * avg_value;
        if (variance < 0.0 && variance > -1e-12)
        {
            variance = 0.0;
        }
        stddev_value = variance <= 0.0 ? 0.0 : sqrt(variance);

        if (fusion_method == EVOKE_HYBRID_FUSION_RRF)
        {
            normalized_score = 1.0 /
                (rrf_k + (double) candidates[i].source_rank);
        }
        else if (candidates[i].normalizer ==
                 EVOKE_HYBRID_NORMALIZER_IDENTITY)
        {
            normalized_score = candidates[i].direction ==
                EVOKE_HYBRID_DIRECTION_LOWER
                ? -candidates[i].raw_value
                : candidates[i].raw_value;
        }
        else if (candidates[i].normalizer ==
                 EVOKE_HYBRID_NORMALIZER_NEGATIVE_DISTANCE)
        {
            normalized_score = -candidates[i].raw_value;
        }
        else if (candidates[i].normalizer ==
                 EVOKE_HYBRID_NORMALIZER_INVERSE_DISTANCE)
        {
            normalized_score = 1.0 /
                Max(epsilon, candidates[i].raw_value + epsilon);
        }
        else if (candidates[i].normalizer ==
                 EVOKE_HYBRID_NORMALIZER_MINMAX)
        {
            if (source->max_value == source->min_value)
            {
                normalized_score = 1.0;
            }
            else if (candidates[i].direction ==
                     EVOKE_HYBRID_DIRECTION_LOWER)
            {
                normalized_score = (source->max_value - candidates[i].raw_value)
                    / (source->max_value - source->min_value);
            }
            else
            {
                normalized_score = (candidates[i].raw_value - source->min_value)
                    / (source->max_value - source->min_value);
            }
        }
        else if (candidates[i].normalizer ==
                 EVOKE_HYBRID_NORMALIZER_ZSCORE)
        {
            if (stddev_value == 0.0)
            {
                normalized_score = 0.0;
            }
            else if (candidates[i].direction ==
                     EVOKE_HYBRID_DIRECTION_LOWER)
            {
                normalized_score = (avg_value - candidates[i].raw_value)
                    / stddev_value;
            }
            else
            {
                normalized_score = (candidates[i].raw_value - avg_value)
                    / stddev_value;
            }
        }
        else
        {
            normalized_score = 1.0 /
                Max((double) candidates[i].source_rank, 1.0);
        }

        candidates[i].normalized_score = normalized_score;
        candidates[i].weighted_score = normalized_score * candidates[i].weight;
        candidates[i].scored = true;
    }
}

static evoke_hybrid_candidate_state *
evoke_hybrid_deduplicate_candidates(
    evoke_hybrid_candidate_state *candidates,
    size_t candidate_len,
    size_t *deduped_len_out
)
{
    evoke_hybrid_candidate_state *scored;
    evoke_hybrid_candidate_state *deduped;
    size_t scored_len = 0;
    size_t i;

    *deduped_len_out = 0;
    scored = palloc(sizeof(*scored) * Max(candidate_len, (size_t) 1));
    deduped = palloc(sizeof(*deduped) * Max(candidate_len, (size_t) 1));

    for (i = 0; i < candidate_len; i++)
    {
        if (candidates[i].scored)
        {
            scored[scored_len++] = candidates[i];
        }
    }
    if (scored_len == 0)
    {
        *deduped_len_out = 0;
        return deduped;
    }

    qsort(
        scored,
        scored_len,
        sizeof(*scored),
        evoke_hybrid_candidate_dedup_cmp
    );

    for (i = 0; i < scored_len; i++)
    {
        if (i > 0 &&
            strcmp(scored[i].source_name, scored[i - 1].source_name) == 0 &&
            strcmp(scored[i].tid_text, scored[i - 1].tid_text) == 0)
        {
            continue;
        }
        deduped[*deduped_len_out] = scored[i];
        (*deduped_len_out)++;
    }

    return deduped;
}

static evoke_hybrid_hit_state *
evoke_hybrid_build_hits(
    evoke_hybrid_candidate_state *deduped,
    size_t deduped_len,
    size_t *hit_len_out
)
{
    evoke_hybrid_hit_state *hits;
    size_t i;

    *hit_len_out = 0;
    hits = palloc0(sizeof(*hits) * Max(deduped_len, (size_t) 1));
    if (deduped_len == 0)
    {
        return hits;
    }

    qsort(
        deduped,
        deduped_len,
        sizeof(*deduped),
        evoke_hybrid_candidate_tid_source_cmp
    );

    i = 0;
    while (i < deduped_len)
    {
        size_t start = i;
        size_t end;
        size_t count;
        size_t j;
        evoke_hybrid_hit_state *hit;
        double score = 0.0;

        while (i < deduped_len &&
               strcmp(deduped[i].tid_text, deduped[start].tid_text) == 0)
        {
            i++;
        }
        end = i;
        count = end - start;
        hit = &hits[*hit_len_out];
        hit->tid = deduped[start].tid;
        hit->tid_text = deduped[start].tid_text;
        hit->source_count = (int32) count;
        hit->source_names = palloc(sizeof(*hit->source_names) * count);
        hit->raw_values = palloc(sizeof(*hit->raw_values) * count);
        hit->normalized_scores =
            palloc(sizeof(*hit->normalized_scores) * count);
        hit->weighted_scores = palloc(sizeof(*hit->weighted_scores) * count);
        hit->ranks = palloc(sizeof(*hit->ranks) * count);

        for (j = 0; j < count; j++)
        {
            const evoke_hybrid_candidate_state *candidate =
                &deduped[start + j];

            score += candidate->weighted_score;
            hit->source_names[j] = candidate->source_name;
            hit->raw_values[j] = (float4) candidate->raw_value;
            hit->normalized_scores[j] =
                (float4) candidate->normalized_score;
            hit->weighted_scores[j] = (float4) candidate->weighted_score;
            hit->ranks[j] = candidate->source_rank;
        }
        hit->score = (float4) score;
        (*hit_len_out)++;
    }

    qsort(
        hits,
        *hit_len_out,
        sizeof(*hits),
        evoke_hybrid_hit_cmp
    );
    return hits;
}

static void
evoke_hybrid_prepare_search_state(
    ArrayType *candidate_array,
    int32 requested_k,
    evoke_hybrid_fusion_method fusion_method,
    double rrf_k,
    double epsilon,
    evoke_hybrid_search_state *state_out
)
{
    Datum *items = NULL;
    bool *nulls = NULL;
    int item_count = 0;
    Oid element_type;
    int16 element_len;
    bool element_byval;
    char element_align;
    evoke_hybrid_candidate_state *candidates;
    evoke_hybrid_source_state *sources;
    evoke_hybrid_candidate_state *deduped;
    size_t candidate_len = 0;
    size_t source_len = 0;
    size_t deduped_len = 0;
    size_t hit_len = 0;
    int i;

    memset(state_out, 0, sizeof(*state_out));
    if (candidate_array == NULL || ArrayGetNItems(
            ARR_NDIM(candidate_array),
            ARR_DIMS(candidate_array)
        ) == 0)
    {
        return;
    }

    element_type = ARR_ELEMTYPE(candidate_array);
    get_typlenbyvalalign(
        element_type,
        &element_len,
        &element_byval,
        &element_align
    );
    deconstruct_array(
        candidate_array,
        element_type,
        element_len,
        element_byval,
        element_align,
        &items,
        &nulls,
        &item_count
    );

    candidates = palloc0(sizeof(*candidates) * Max(item_count, 1));
    sources = palloc0(sizeof(*sources) * Max(item_count, 1));
    for (i = 0; i < item_count; i++)
    {
        evoke_hybrid_candidate_state candidate;

        if (nulls[i])
        {
            continue;
        }
        evoke_hybrid_deform_candidate(
            items[i],
            &candidate,
            sources,
            &source_len
        );
        if (!candidate.scored)
        {
            continue;
        }
        candidates[candidate_len++] = candidate;
    }

    evoke_hybrid_prepare_candidates(
        candidates,
        candidate_len,
        sources,
        fusion_method,
        rrf_k,
        epsilon
    );
    deduped = evoke_hybrid_deduplicate_candidates(
        candidates,
        candidate_len,
        &deduped_len
    );
    state_out->hits = evoke_hybrid_build_hits(
        deduped,
        deduped_len,
        &hit_len
    );
    state_out->len = Min((size_t) Max(requested_k, 0), hit_len);
}

PG_FUNCTION_INFO_V1(evoke_hybrid_fuse_candidates);
Datum
evoke_hybrid_fuse_candidates(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    evoke_hybrid_search_state *state;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        ArrayType *candidate_array = PG_ARGISNULL(0)
            ? NULL
            : PG_GETARG_ARRAYTYPE_P(0);
        int32 k = PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1);
        char *fusion_text = PG_ARGISNULL(2)
            ? NULL
            : text_to_cstring(PG_GETARG_TEXT_PP(2));
        float4 rrf_arg = PG_ARGISNULL(3) ? 60.0f : PG_GETARG_FLOAT4(3);
        float4 epsilon_arg = PG_ARGISNULL(4)
            ? 0.000001f
            : PG_GETARG_FLOAT4(4);
        evoke_hybrid_fusion_method fusion_method;
        double rrf_k;
        double epsilon;

        k = Max(k, 0);
        fusion_method = evoke_hybrid_parse_fusion(fusion_text);
        rrf_k = Max((double) rrf_arg, 0.000001);
        epsilon = Max((double) epsilon_arg, 0.000000000001);

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        state = palloc0(sizeof(*state));
        evoke_hybrid_prepare_search_state(
            candidate_array,
            k,
            fusion_method,
            rrf_k,
            epsilon,
            state
        );

        funcctx->user_fctx = state;
        if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc) !=
            TYPEFUNC_COMPOSITE)
        {
            ereport(ERROR, (errmsg("could not resolve result record type")));
        }
        BlessTupleDesc(funcctx->tuple_desc);
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    state = funcctx->user_fctx;

    if (state->pos < state->len)
    {
        Datum values[8];
        bool nulls[8] = {
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false
        };
        HeapTuple tuple;
        evoke_hybrid_hit_state *hit = &state->hits[state->pos++];

        values[0] = ItemPointerGetDatum(&hit->tid);
        values[1] = Float4GetDatum(hit->score);
        values[2] = Int32GetDatum(hit->source_count);
        values[3] = PointerGetDatum(
            evoke_array_from_cstrings(
                hit->source_names,
                hit->source_count
            )
        );
        values[4] = PointerGetDatum(
            evoke_array_from_float4_values(
                hit->raw_values,
                hit->source_count
            )
        );
        values[5] = PointerGetDatum(
            evoke_array_from_float4_values(
                hit->normalized_scores,
                hit->source_count
            )
        );
        values[6] = PointerGetDatum(
            evoke_array_from_float4_values(
                hit->weighted_scores,
                hit->source_count
            )
        );
        values[7] = PointerGetDatum(
            evoke_array_from_int4_values(
                hit->ranks,
                hit->source_count
            )
        );

        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }

    pfree(state);
    funcctx->user_fctx = NULL;
    SRF_RETURN_DONE(funcctx);
}
