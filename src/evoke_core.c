#include "evoke_core.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if !defined(EVOKE_FORCE_SCALAR) && defined(__GNUC__) && \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define EVOKE_HAVE_X86_AVX2_DISPATCH 1
#endif

typedef struct evoke_term_freq
{
    uint32_t token_id;
    uint32_t tf;
} evoke_term_freq;

typedef struct evoke_term_freqs
{
    evoke_term_freq *terms;
    size_t len;
} evoke_term_freqs;

typedef struct evoke_strmap_slot
{
    char *key;
    uint32_t value;
    bool used;
} evoke_strmap_slot;

typedef struct evoke_strmap
{
    evoke_strmap_slot *slots;
    size_t size;
    size_t capacity;
} evoke_strmap;

typedef evoke_topk_item evoke_heap_item;

typedef struct evoke_compact_topk_item
{
    float score;
    uint32_t doc_id;
} evoke_compact_topk_item;

typedef void (*evoke_apply_weight_mask_fn)(
    float *scores,
    const float *weight_mask,
    size_t len
);

typedef void (*evoke_add_constant_fn)(
    float *scores,
    float value,
    size_t len
);

typedef enum evoke_simd_override
{
    EVOKE_SIMD_OVERRIDE_AUTO = 0,
    EVOKE_SIMD_OVERRIDE_SCALAR = 1,
    EVOKE_SIMD_OVERRIDE_AVX2 = 2
} evoke_simd_override;

typedef enum evoke_simd_path
{
    EVOKE_SIMD_PATH_SCALAR = 0,
    EVOKE_SIMD_PATH_AVX2 = 1
} evoke_simd_path;

#ifdef EVOKE_HAVE_X86_AVX2_DISPATCH
static evoke_simd_override
evoke_simd_override_from_env(void)
{
    const char *mode = getenv("EVOKE_SIMD_MODE");

    if (mode == NULL || mode[0] == '\0' || strcmp(mode, "auto") == 0)
    {
        return EVOKE_SIMD_OVERRIDE_AUTO;
    }
    if (strcmp(mode, "scalar") == 0 || strcmp(mode, "fallback") == 0)
    {
        return EVOKE_SIMD_OVERRIDE_SCALAR;
    }
    if (strcmp(mode, "avx2") == 0)
    {
        return EVOKE_SIMD_OVERRIDE_AVX2;
    }

    return EVOKE_SIMD_OVERRIDE_AUTO;
}
#endif

uint32_t
evoke_u32_saturating_add(uint32_t left, uint32_t right)
{
    if (UINT32_MAX - left < right)
    {
        return UINT32_MAX;
    }
    return left + right;
}

uint64_t
evoke_u64_saturating_add(uint64_t left, uint64_t right)
{
    if (UINT64_MAX - left < right)
    {
        return UINT64_MAX;
    }
    return left + right;
}

uint64_t
evoke_u64_saturating_mul(uint64_t left, uint64_t right)
{
    if (left != 0 && right > UINT64_MAX / left)
    {
        return UINT64_MAX;
    }
    return left * right;
}

static void
evoke_apply_weight_mask_scalar(
    float *scores,
    const float *weight_mask,
    size_t len
)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        scores[i] *= weight_mask[i];
    }
}

static void
evoke_add_constant_scalar(float *scores, float value, size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        scores[i] += value;
    }
}

#ifdef EVOKE_HAVE_X86_AVX2_DISPATCH
__attribute__((target("avx2")))
static void
evoke_apply_weight_mask_avx2(
    float *scores,
    const float *weight_mask,
    size_t len
)
{
    size_t i = 0;
    size_t vec_len = len & ~((size_t) 7);

    for (; i < vec_len; i += 8)
    {
        __m256 scores_vec = _mm256_loadu_ps(scores + i);
        __m256 mask_vec = _mm256_loadu_ps(weight_mask + i);
        scores_vec = _mm256_mul_ps(scores_vec, mask_vec);
        _mm256_storeu_ps(scores + i, scores_vec);
    }

    for (; i < len; i++)
    {
        scores[i] *= weight_mask[i];
    }
}

__attribute__((target("avx2")))
static void
evoke_add_constant_avx2(float *scores, float value, size_t len)
{
    size_t i = 0;
    size_t vec_len = len & ~((size_t) 7);
    __m256 value_vec = _mm256_set1_ps(value);

    for (; i < vec_len; i += 8)
    {
        __m256 scores_vec = _mm256_loadu_ps(scores + i);
        scores_vec = _mm256_add_ps(scores_vec, value_vec);
        _mm256_storeu_ps(scores + i, scores_vec);
    }

    for (; i < len; i++)
    {
        scores[i] += value;
    }
}

static bool
evoke_runtime_has_avx2(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#endif

static evoke_apply_weight_mask_fn evoke_apply_weight_mask_impl =
    evoke_apply_weight_mask_scalar;
static evoke_add_constant_fn evoke_add_constant_impl =
    evoke_add_constant_scalar;
static evoke_simd_path evoke_simd_active_path =
    EVOKE_SIMD_PATH_SCALAR;
static bool evoke_simd_dispatch_initialized = false;

static void
evoke_simd_dispatch_init(void)
{
#ifdef EVOKE_HAVE_X86_AVX2_DISPATCH
    evoke_simd_override override_mode = EVOKE_SIMD_OVERRIDE_AUTO;
#endif

    if (evoke_simd_dispatch_initialized)
    {
        return;
    }

#ifdef EVOKE_HAVE_X86_AVX2_DISPATCH
    override_mode = evoke_simd_override_from_env();

    if (override_mode != EVOKE_SIMD_OVERRIDE_SCALAR &&
        evoke_runtime_has_avx2())
    {
        evoke_apply_weight_mask_impl = evoke_apply_weight_mask_avx2;
        evoke_add_constant_impl = evoke_add_constant_avx2;
        evoke_simd_active_path = EVOKE_SIMD_PATH_AVX2;
    }
#endif

    evoke_simd_dispatch_initialized = true;
}

static inline void
evoke_apply_weight_mask_inplace(
    float *scores,
    const float *weight_mask,
    size_t len
)
{
    if (weight_mask == NULL || len == 0)
    {
        return;
    }

    evoke_simd_dispatch_init();
    evoke_apply_weight_mask_impl(scores, weight_mask, len);
}

static inline void
evoke_add_constant_inplace(float *scores, float value, size_t len)
{
    if (value == 0.0f || len == 0)
    {
        return;
    }

    evoke_simd_dispatch_init();
    evoke_add_constant_impl(scores, value, len);
}

bool
evoke_method_requires_nonoccurrence(evoke_method method)
{
    return method == EVOKE_METHOD_BM25L ||
           method == EVOKE_METHOD_BM25PLUS;
}

const char *
evoke_strerror(evoke_status status)
{
    switch (status)
    {
        case EVOKE_OK:
            return "ok";
        case EVOKE_ERR_NOMEM:
            return "out of memory";
        case EVOKE_ERR_INVALID:
            return "invalid input";
        case EVOKE_ERR_RANGE:
            return "value out of range";
        case EVOKE_ERR_FORMAT:
            return "invalid serialized format";
    }

    return "unknown error";
}

bool
evoke_parse_method(const char *name, evoke_method *method_out)
{
    if (name == NULL || method_out == NULL)
    {
        return false;
    }

    if (strcmp(name, "robertson") == 0)
    {
        *method_out = EVOKE_METHOD_ROBERTSON;
        return true;
    }
    if (strcmp(name, "lucene") == 0)
    {
        *method_out = EVOKE_METHOD_LUCENE;
        return true;
    }
    if (strcmp(name, "atire") == 0)
    {
        *method_out = EVOKE_METHOD_ATIRE;
        return true;
    }
    if (strcmp(name, "bm25l") == 0)
    {
        *method_out = EVOKE_METHOD_BM25L;
        return true;
    }
    if (strcmp(name, "bm25+") == 0)
    {
        *method_out = EVOKE_METHOD_BM25PLUS;
        return true;
    }

    return false;
}

const char *
evoke_method_name(evoke_method method)
{
    switch (method)
    {
        case EVOKE_METHOD_ROBERTSON:
            return "robertson";
        case EVOKE_METHOD_LUCENE:
            return "lucene";
        case EVOKE_METHOD_ATIRE:
            return "atire";
        case EVOKE_METHOD_BM25L:
            return "bm25l";
        case EVOKE_METHOD_BM25PLUS:
            return "bm25+";
    }

    return "unknown";
}

const char *
evoke_active_simd_path(void)
{
    evoke_simd_dispatch_init();
    if (evoke_simd_active_path == EVOKE_SIMD_PATH_AVX2)
    {
        return "avx2";
    }
    return "scalar";
}

void
evoke_index_init(evoke_index *index)
{
    if (index == NULL)
    {
        return;
    }

    memset(index, 0, sizeof(*index));
}

void
evoke_index_free(evoke_index *index)
{
    uint32_t i;

    if (index == NULL)
    {
        return;
    }

    free(index->data);
    free(index->indices);
    free(index->indptr);
    free(index->term_frequencies);
    free(index->doc_lengths);
    free(index->doc_frequencies);
    free(index->nonoccurrence);

    if (index->vocab != NULL)
    {
        for (i = 0; i < index->vocab_size; i++)
        {
            free(index->vocab[i]);
        }
    }
    free(index->vocab);

    memset(index, 0, sizeof(*index));
}

void
evoke_topk_result_free(evoke_topk_result *result)
{
    if (result == NULL)
    {
        return;
    }

    free(result->doc_ids);
    free(result->scores);
    memset(result, 0, sizeof(*result));
}

static bool
evoke_checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (a == 0 || b == 0)
    {
        *out = 0;
        return true;
    }

    if (a > ((size_t) -1) / b)
    {
        return false;
    }

    *out = a * b;
    return true;
}

static bool
evoke_method_is_valid(evoke_method method)
{
    switch (method)
    {
        case EVOKE_METHOD_ROBERTSON:
        case EVOKE_METHOD_LUCENE:
        case EVOKE_METHOD_ATIRE:
        case EVOKE_METHOD_BM25L:
        case EVOKE_METHOD_BM25PLUS:
            return true;
    }

    return false;
}

bool
evoke_params_are_valid(const evoke_params *params)
{
    return params != NULL &&
           evoke_method_is_valid(params->method) &&
           evoke_method_is_valid(params->idf_method) &&
           isfinite(params->k1) &&
           isfinite(params->b) &&
           isfinite(params->delta) &&
           params->k1 >= 0.0f &&
           params->b >= 0.0f &&
           params->b <= 1.0f &&
           params->delta >= 0.0f;
}

static uint64_t
evoke_hash_bytes(const char *s)
{
    uint64_t hash = 1469598103934665603ULL;
    const unsigned char *ptr = (const unsigned char *) s;

    while (*ptr != '\0')
    {
        hash ^= (uint64_t) *ptr;
        hash *= 1099511628211ULL;
        ptr++;
    }

    return hash;
}

static void
evoke_strmap_destroy(evoke_strmap *map)
{
    size_t i;

    if (map == NULL)
    {
        return;
    }

    if (map->slots != NULL)
    {
        for (i = 0; i < map->capacity; i++)
        {
            free(map->slots[i].key);
        }
    }

    free(map->slots);
    memset(map, 0, sizeof(*map));
}

static evoke_status
evoke_strmap_init(evoke_strmap *map, size_t min_capacity)
{
    size_t capacity = 8;

    if (map == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    while (capacity < min_capacity)
    {
        if (capacity > ((size_t) -1) / 2)
        {
            return EVOKE_ERR_RANGE;
        }
        capacity *= 2;
    }

    map->slots = calloc(capacity, sizeof(*map->slots));
    if (map->slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    map->capacity = capacity;
    map->size = 0;
    return EVOKE_OK;
}

static evoke_status
evoke_strmap_rehash(evoke_strmap *map, size_t new_capacity)
{
    evoke_strmap_slot *old_slots;
    size_t old_capacity;
    size_t i;

    old_slots = map->slots;
    old_capacity = map->capacity;
    map->slots = calloc(new_capacity, sizeof(*map->slots));
    if (map->slots == NULL)
    {
        map->slots = old_slots;
        return EVOKE_ERR_NOMEM;
    }

    map->capacity = new_capacity;
    map->size = 0;

    for (i = 0; i < old_capacity; i++)
    {
        evoke_strmap_slot slot = old_slots[i];

        if (!slot.used)
        {
            continue;
        }

        {
            uint64_t hash = evoke_hash_bytes(slot.key);
            size_t mask = map->capacity - 1;
            size_t idx = (size_t) hash & mask;

            while (map->slots[idx].used)
            {
                idx = (idx + 1) & mask;
            }

            map->slots[idx] = slot;
            map->slots[idx].used = true;
            map->size++;
        }
    }

    free(old_slots);
    return EVOKE_OK;
}

static evoke_status
evoke_strmap_ensure_capacity(evoke_strmap *map)
{
    if ((map->size + 1) * 10 < map->capacity * 7)
    {
        return EVOKE_OK;
    }

    if (map->capacity > ((size_t) -1) / 2)
    {
        return EVOKE_ERR_RANGE;
    }

    return evoke_strmap_rehash(map, map->capacity * 2);
}

static char *
evoke_strdup(const char *s)
{
    size_t len = strlen(s);
    char *copy = malloc(len + 1);

    if (copy == NULL)
    {
        return NULL;
    }

    memcpy(copy, s, len + 1);
    return copy;
}

static evoke_status
evoke_strmap_get(
    const evoke_strmap *map,
    const char *key,
    uint32_t *value_out,
    bool *found_out
)
{
    uint64_t hash;
    size_t mask;
    size_t idx;

    if (map == NULL || key == NULL || found_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    if (map->capacity == 0)
    {
        *found_out = false;
        return EVOKE_OK;
    }

    hash = evoke_hash_bytes(key);
    mask = map->capacity - 1;
    idx = (size_t) hash & mask;

    while (map->slots[idx].used)
    {
        if (strcmp(map->slots[idx].key, key) == 0)
        {
            *found_out = true;
            if (value_out != NULL)
            {
                *value_out = map->slots[idx].value;
            }
            return EVOKE_OK;
        }

        idx = (idx + 1) & mask;
    }

    *found_out = false;
    return EVOKE_OK;
}

static evoke_status
evoke_strmap_put(
    evoke_strmap *map,
    const char *key,
    uint32_t value
)
{
    uint64_t hash;
    size_t mask;
    size_t idx;
    evoke_status status;

    status = evoke_strmap_ensure_capacity(map);
    if (status != EVOKE_OK)
    {
        return status;
    }

    hash = evoke_hash_bytes(key);
    mask = map->capacity - 1;
    idx = (size_t) hash & mask;

    while (map->slots[idx].used)
    {
        if (strcmp(map->slots[idx].key, key) == 0)
        {
            map->slots[idx].value = value;
            return EVOKE_OK;
        }

        idx = (idx + 1) & mask;
    }

    map->slots[idx].key = evoke_strdup(key);
    if (map->slots[idx].key == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    map->slots[idx].value = value;
    map->slots[idx].used = true;
    map->size++;
    return EVOKE_OK;
}

static int
evoke_cmp_uint32(const void *lhs, const void *rhs)
{
    uint32_t a = *(const uint32_t *) lhs;
    uint32_t b = *(const uint32_t *) rhs;

    if (a < b)
    {
        return -1;
    }
    if (a > b)
    {
        return 1;
    }
    return 0;
}

static void
evoke_term_freqs_free(evoke_term_freqs *term_freqs)
{
    if (term_freqs == NULL)
    {
        return;
    }

    free(term_freqs->terms);
    term_freqs->terms = NULL;
    term_freqs->len = 0;
}

static evoke_status
evoke_count_doc_terms(
    const uint32_t *token_ids,
    size_t len,
    evoke_term_freqs *term_freqs_out
)
{
    uint32_t *sorted_ids = NULL;
    evoke_term_freq *terms = NULL;
    size_t i;
    size_t unique_count = 0;
    size_t bytes;

    if (term_freqs_out == NULL || (len > 0 && token_ids == NULL))
    {
        return EVOKE_ERR_INVALID;
    }

    term_freqs_out->terms = NULL;
    term_freqs_out->len = 0;

    if (len == 0)
    {
        return EVOKE_OK;
    }

    if (!evoke_checked_mul_size(len, sizeof(*sorted_ids), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    sorted_ids = malloc(bytes);
    if (sorted_ids == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    memcpy(sorted_ids, token_ids, bytes);
    qsort(sorted_ids, len, sizeof(*sorted_ids), evoke_cmp_uint32);

    if (!evoke_checked_mul_size(len, sizeof(*terms), &bytes))
    {
        free(sorted_ids);
        return EVOKE_ERR_RANGE;
    }
    terms = malloc(bytes);
    if (terms == NULL)
    {
        free(sorted_ids);
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < len; i++)
    {
        if (unique_count == 0 ||
            terms[unique_count - 1].token_id != sorted_ids[i])
        {
            terms[unique_count].token_id = sorted_ids[i];
            terms[unique_count].tf = 1;
            unique_count++;
        }
        else
        {
            terms[unique_count - 1].tf++;
        }
    }

    free(sorted_ids);
    term_freqs_out->terms = terms;
    term_freqs_out->len = unique_count;
    return EVOKE_OK;
}

double
evoke_score_tfc(
    evoke_method method,
    double tf,
    double l_d,
    double l_avg,
    double k1,
    double b,
    double delta
)
{
    switch (method)
    {
        case EVOKE_METHOD_ROBERTSON:
        case EVOKE_METHOD_LUCENE:
            return tf / (k1 * ((1.0 - b) + (b * l_d / l_avg)) + tf);
        case EVOKE_METHOD_ATIRE:
            return (tf * (k1 + 1.0)) /
                   (tf + k1 * (1.0 - b + (b * l_d / l_avg)));
        case EVOKE_METHOD_BM25L:
        {
            double c = tf / (1.0 - b + (b * l_d / l_avg));
            return ((k1 + 1.0) * (c + delta)) / (k1 + c + delta);
        }
        case EVOKE_METHOD_BM25PLUS:
        {
            double num = (k1 + 1.0) * tf;
            double den = k1 * (1.0 - b + (b * l_d / l_avg)) + tf;
            return (num / den) + delta;
        }
    }

    return 0.0;
}

double
evoke_score_idf(
    evoke_method method,
    double df,
    double n_docs
)
{
    switch (method)
    {
        case EVOKE_METHOD_ROBERTSON:
        {
            double inner = (n_docs - df + 0.5) / (df + 0.5);
            if (inner < 1.0)
            {
                inner = 1.0;
            }
            return log(inner);
        }
        case EVOKE_METHOD_LUCENE:
            return log(1.0 + (n_docs - df + 0.5) / (df + 0.5));
        case EVOKE_METHOD_ATIRE:
            return log(n_docs / df);
        case EVOKE_METHOD_BM25L:
            return log((n_docs + 1.0) / (df + 0.5));
        case EVOKE_METHOD_BM25PLUS:
            return log((n_docs + 1.0) / df);
    }

    return 0.0;
}

static evoke_status
evoke_build_csc(
    const float *scores,
    const uint32_t *rows,
    const uint32_t *cols,
    const uint32_t *term_freqs,
    uint64_t data_len,
    uint32_t n_cols,
    float **data_out,
    uint32_t **indices_out,
    uint64_t **indptr_out,
    uint32_t **term_freqs_out
)
{
    uint64_t *col_counts = NULL;
    uint64_t *heads = NULL;
    uint64_t *indptr = NULL;
    float *sorted_data = NULL;
    uint32_t *sorted_indices = NULL;
    uint32_t *sorted_term_freqs = NULL;
    size_t bytes;
    uint64_t i;
    uint64_t acc = 0;

    *data_out = NULL;
    *indices_out = NULL;
    *indptr_out = NULL;
    if (term_freqs_out != NULL)
    {
        *term_freqs_out = NULL;
    }

    if (!evoke_checked_mul_size(
            n_cols,
            sizeof(*col_counts),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    col_counts = calloc(n_cols, sizeof(*col_counts));
    heads = calloc(n_cols, sizeof(*heads));
    if (col_counts == NULL || heads == NULL)
    {
        free(col_counts);
        free(heads);
        return EVOKE_ERR_NOMEM;
    }

    if (!evoke_checked_mul_size(
            n_cols + 1,
            sizeof(*indptr),
            &bytes))
    {
        free(col_counts);
        free(heads);
        return EVOKE_ERR_RANGE;
    }
    indptr = malloc(bytes);
    if (indptr == NULL)
    {
        free(col_counts);
        free(heads);
        return EVOKE_ERR_NOMEM;
    }

    if (data_len > 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) data_len,
                sizeof(*sorted_data),
                &bytes))
        {
            free(col_counts);
            free(heads);
            free(indptr);
            return EVOKE_ERR_RANGE;
        }
        sorted_data = malloc(bytes);
        if (sorted_data == NULL)
        {
            free(col_counts);
            free(heads);
            free(indptr);
            return EVOKE_ERR_NOMEM;
        }

        if (!evoke_checked_mul_size(
                (size_t) data_len,
                sizeof(*sorted_indices),
                &bytes))
        {
            free(col_counts);
            free(heads);
            free(indptr);
            free(sorted_data);
            return EVOKE_ERR_RANGE;
        }
        sorted_indices = malloc(bytes);
        if (sorted_indices == NULL)
        {
            free(col_counts);
            free(heads);
            free(indptr);
            free(sorted_data);
            return EVOKE_ERR_NOMEM;
        }

        if (term_freqs_out != NULL)
        {
            if (!evoke_checked_mul_size(
                    (size_t) data_len,
                    sizeof(*sorted_term_freqs),
                    &bytes))
            {
                free(col_counts);
                free(heads);
                free(indptr);
                free(sorted_data);
                free(sorted_indices);
                return EVOKE_ERR_RANGE;
            }
            sorted_term_freqs = malloc(bytes);
            if (sorted_term_freqs == NULL)
            {
                free(col_counts);
                free(heads);
                free(indptr);
                free(sorted_data);
                free(sorted_indices);
                return EVOKE_ERR_NOMEM;
            }
        }
    }

    for (i = 0; i < data_len; i++)
    {
        col_counts[cols[i]]++;
    }

    indptr[0] = 0;
    for (i = 0; i < n_cols; i++)
    {
        heads[i] = acc;
        acc += col_counts[i];
        indptr[i + 1] = acc;
    }

    for (i = 0; i < data_len; i++)
    {
        uint32_t col = cols[i];
        uint64_t pos = heads[col];
        sorted_data[pos] = scores[i];
        sorted_indices[pos] = rows[i];
        if (sorted_term_freqs != NULL && term_freqs != NULL)
        {
            sorted_term_freqs[pos] = term_freqs[i];
        }
        heads[col]++;
    }

    free(col_counts);
    free(heads);

    *data_out = sorted_data;
    *indices_out = sorted_indices;
    *indptr_out = indptr;
    if (term_freqs_out != NULL)
    {
        *term_freqs_out = sorted_term_freqs;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_build_empty_index(
    const evoke_params *params,
    uint32_t vocab_size,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    char **vocab,
    evoke_index *index_out
)
{
    uint64_t *indptr;

    if (!evoke_params_are_valid(params) ||
        index_out == NULL ||
        vocab_size == UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    if (create_empty_token && has_empty_token &&
        empty_token_id >= vocab_size)
    {
        return EVOKE_ERR_RANGE;
    }

    indptr = calloc((size_t) vocab_size + 1, sizeof(*indptr));
    if (indptr == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    evoke_index_init(index_out);
    index_out->params = *params;
    index_out->vocab_size = vocab_size;
    index_out->has_empty_token = create_empty_token && has_empty_token;
    index_out->empty_token_id = empty_token_id;
    index_out->indptr = indptr;
    index_out->vocab = vocab;
    return EVOKE_OK;
}

static evoke_status
evoke_build_index_internal(
    const evoke_doc_ids *docs,
    size_t num_docs,
    uint32_t vocab_size,
    const evoke_params *params,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    char **vocab,
    evoke_index *index_out
)
{
    evoke_term_freqs *doc_terms = NULL;
    uint32_t *doc_frequencies = NULL;
    float *idf_array = NULL;
    float *nonoccurrence = NULL;
    float *scores = NULL;
    uint32_t *doc_indices = NULL;
    uint32_t *voc_indices = NULL;
    uint32_t *term_frequencies = NULL;
    uint32_t *doc_lengths = NULL;
    size_t i;
    size_t bytes;
    uint64_t array_size = 0;
    double total_doc_len = 0.0;
    double avg_doc_len;
    uint64_t cursor = 0;
    evoke_status status = EVOKE_OK;

    if (!evoke_params_are_valid(params) ||
        index_out == NULL ||
        (num_docs > 0 && docs == NULL))
    {
        return EVOKE_ERR_INVALID;
    }

    if (num_docs == 0)
    {
        return evoke_build_empty_index(
            params,
            vocab_size,
            create_empty_token,
            has_empty_token,
            empty_token_id,
            vocab,
            index_out
        );
    }

    doc_terms = calloc(num_docs, sizeof(*doc_terms));
    if (doc_terms == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    if (!evoke_checked_mul_size(num_docs, sizeof(*doc_lengths), &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    doc_lengths = calloc(num_docs, sizeof(*doc_lengths));
    if (doc_lengths == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    if (!evoke_checked_mul_size(vocab_size, sizeof(*doc_frequencies), &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    doc_frequencies = calloc(vocab_size, sizeof(*doc_frequencies));
    if (doc_frequencies == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (i = 0; i < num_docs; i++)
    {
        size_t j;

        total_doc_len += (double) docs[i].len;
        doc_lengths[i] = (uint32_t) docs[i].len;
        status = evoke_count_doc_terms(
            docs[i].token_ids,
            docs[i].len,
            &doc_terms[i]
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }

        array_size += (uint64_t) doc_terms[i].len;
        for (j = 0; j < doc_terms[i].len; j++)
        {
            uint32_t token_id = doc_terms[i].terms[j].token_id;
            if (token_id >= vocab_size)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            doc_frequencies[token_id]++;
        }
    }

    avg_doc_len = total_doc_len / (double) num_docs;

    if (!evoke_checked_mul_size(vocab_size, sizeof(*idf_array), &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    idf_array = calloc(vocab_size, sizeof(*idf_array));
    if (idf_array == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (i = 0; i < vocab_size; i++)
    {
        if (doc_frequencies[i] != 0)
        {
            idf_array[i] = (float) evoke_score_idf(
                params->idf_method,
                (double) doc_frequencies[i],
                (double) num_docs
            );
        }
    }

    if (evoke_method_requires_nonoccurrence(params->method))
    {
        nonoccurrence = calloc(vocab_size, sizeof(*nonoccurrence));
        if (nonoccurrence == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }

        for (i = 0; i < vocab_size; i++)
        {
            if (doc_frequencies[i] != 0)
            {
                double idf = evoke_score_idf(
                    params->idf_method,
                    (double) doc_frequencies[i],
                    (double) num_docs
                );
                double tfc = evoke_score_tfc(
                    params->method,
                    0.0,
                    avg_doc_len,
                    avg_doc_len,
                    params->k1,
                    params->b,
                    params->delta
                );
                nonoccurrence[i] = (float) (idf * tfc);
            }
        }
    }

    if (array_size > 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) array_size,
                sizeof(*scores),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        scores = malloc(bytes);
        if (scores == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }

        if (!evoke_checked_mul_size(
                (size_t) array_size,
                sizeof(*doc_indices),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        doc_indices = malloc(bytes);
        if (doc_indices == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }

        if (!evoke_checked_mul_size(
                (size_t) array_size,
                sizeof(*voc_indices),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        voc_indices = malloc(bytes);
        if (voc_indices == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }

        if (!evoke_checked_mul_size(
                (size_t) array_size,
                sizeof(*term_frequencies),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        term_frequencies = malloc(bytes);
        if (term_frequencies == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (i = 0; i < num_docs; i++)
    {
        size_t j;
        double doc_len = (double) docs[i].len;

        for (j = 0; j < doc_terms[i].len; j++)
        {
            uint32_t token_id = doc_terms[i].terms[j].token_id;
            double tf = (double) doc_terms[i].terms[j].tf;
            double tfc = evoke_score_tfc(
                params->method,
                tf,
                doc_len,
                avg_doc_len,
                params->k1,
                params->b,
                params->delta
            );
            double score = (double) idf_array[token_id] * tfc;

            if (evoke_method_requires_nonoccurrence(params->method))
            {
                score -= (double) nonoccurrence[token_id];
            }

            scores[cursor] = (float) score;
            doc_indices[cursor] = (uint32_t) i;
            voc_indices[cursor] = token_id;
            term_frequencies[cursor] = doc_terms[i].terms[j].tf;
            cursor++;
        }
    }

    evoke_index_init(index_out);
    index_out->params = *params;
    index_out->num_docs = (uint32_t) num_docs;
    index_out->vocab_size = vocab_size;
    index_out->data_len = array_size;
    index_out->has_empty_token = create_empty_token && has_empty_token;
    index_out->empty_token_id = empty_token_id;
    index_out->doc_lengths = doc_lengths;
    index_out->doc_frequencies = doc_frequencies;
    index_out->nonoccurrence = nonoccurrence;
    doc_lengths = NULL;
    doc_frequencies = NULL;
    nonoccurrence = NULL;

    status = evoke_build_csc(
        scores,
        doc_indices,
        voc_indices,
        term_frequencies,
        array_size,
        vocab_size,
        &index_out->data,
        &index_out->indices,
        &index_out->indptr,
        &index_out->term_frequencies
    );
    if (status != EVOKE_OK)
    {
        evoke_index_free(index_out);
        goto cleanup;
    }
    /* The caller owns vocab until every other index allocation succeeds. */
    index_out->vocab = vocab;

cleanup:
    if (doc_terms != NULL)
    {
        for (i = 0; i < num_docs; i++)
        {
            evoke_term_freqs_free(&doc_terms[i]);
        }
    }
    free(doc_terms);
    free(doc_frequencies);
    free(idf_array);
    free(nonoccurrence);
    free(scores);
    free(doc_indices);
    free(voc_indices);
    free(term_frequencies);
    free(doc_lengths);
    return status;
}

typedef struct evoke_term_entry_builder
{
    evoke_term_entry *entries;
    uint64_t len;
    uint64_t capacity;
} evoke_term_entry_builder;

static void
evoke_term_entry_builder_free(evoke_term_entry_builder *builder)
{
    if (builder == NULL)
    {
        return;
    }

    free(builder->entries);
    memset(builder, 0, sizeof(*builder));
}

static evoke_status
evoke_term_entry_builder_append(
    evoke_term_entry_builder *builder,
    uint32_t token_id,
    uint32_t doc_id,
    uint32_t tf
)
{
    size_t bytes;

    if (builder->len == builder->capacity)
    {
        uint64_t new_capacity = builder->capacity == 0
            ? 1024
            : builder->capacity * 2;
        evoke_term_entry *new_entries;

        if (new_capacity < builder->capacity ||
            new_capacity > (uint64_t) SIZE_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        if (!evoke_checked_mul_size(
                (size_t) new_capacity,
                sizeof(*builder->entries),
                &bytes))
        {
            return EVOKE_ERR_RANGE;
        }
        new_entries = realloc(builder->entries, bytes);
        if (new_entries == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        builder->entries = new_entries;
        builder->capacity = new_capacity;
    }

    builder->entries[builder->len].token_id = token_id;
    builder->entries[builder->len].doc_id = doc_id;
    builder->entries[builder->len].tf = tf;
    builder->len++;
    return EVOKE_OK;
}

static evoke_status
evoke_copy_vocab(
    const char *const *vocab,
    uint32_t vocab_size,
    char ***vocab_out
)
{
    char **copy = NULL;
    uint32_t i;
    size_t bytes;

    *vocab_out = NULL;
    if (vocab == NULL || vocab_size == 0)
    {
        return EVOKE_OK;
    }
    if (!evoke_checked_mul_size(vocab_size, sizeof(*copy), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    copy = calloc(vocab_size, sizeof(*copy));
    if (copy == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (i = 0; i < vocab_size; i++)
    {
        if (vocab[i] == NULL)
        {
            uint32_t j;

            for (j = 0; j < i; j++)
            {
                free(copy[j]);
            }
            free(copy);
            return EVOKE_ERR_INVALID;
        }
        copy[i] = evoke_strdup(vocab[i]);
        if (copy[i] == NULL)
        {
            uint32_t j;

            for (j = 0; j < i; j++)
            {
                free(copy[j]);
            }
            free(copy);
            return EVOKE_ERR_NOMEM;
        }
    }

    *vocab_out = copy;
    return EVOKE_OK;
}

evoke_status
evoke_build_index_from_term_entry_reader(
    uint64_t num_entries,
    evoke_term_entry_reader_cb read_cb,
    evoke_term_entry_rewind_cb rewind_cb,
    void *reader_ctx,
    const uint32_t *doc_lengths_in,
    uint32_t num_docs,
    uint32_t vocab_size,
    const evoke_params *params,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    const char *const *vocab,
    evoke_index *index_out
)
{
    uint32_t *doc_lengths = NULL;
    uint32_t *doc_frequencies = NULL;
    float *idf_array = NULL;
    float *nonoccurrence = NULL;
    float *data = NULL;
    uint32_t *indices = NULL;
    uint64_t *indptr = NULL;
    uint64_t *heads = NULL;
    uint32_t *term_frequencies = NULL;
    char **vocab_copy = NULL;
    uint32_t i;
    uint64_t j;
    size_t bytes;
    double total_doc_len = 0.0;
    double avg_doc_len = 0.0;
    evoke_status status = EVOKE_OK;

    if (!evoke_params_are_valid(params) || index_out == NULL ||
        vocab_size == UINT32_MAX ||
        (num_docs > 0 && doc_lengths_in == NULL) ||
        (num_entries > 0 && (read_cb == NULL || rewind_cb == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_entries > (uint64_t) SIZE_MAX ||
        (num_docs == 0 && num_entries > 0))
    {
        return EVOKE_ERR_INVALID;
    }
    if (vocab_size == 0 && num_entries > 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (has_empty_token && empty_token_id >= vocab_size)
    {
        return EVOKE_ERR_RANGE;
    }
    if (num_docs == 0)
    {
        status = evoke_copy_vocab(vocab, vocab_size, &vocab_copy);
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_build_empty_index(
            params,
            vocab_size,
            create_empty_token,
            has_empty_token,
            empty_token_id,
            vocab_copy,
            index_out
        );
        if (status != EVOKE_OK && vocab_copy != NULL)
        {
            for (i = 0; i < vocab_size; i++)
            {
                free(vocab_copy[i]);
            }
            free(vocab_copy);
        }
        return status;
    }

    if (!evoke_checked_mul_size(num_docs, sizeof(*doc_lengths), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    doc_lengths = malloc(bytes);
    if (doc_lengths == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (i = 0; i < num_docs; i++)
    {
        doc_lengths[i] = doc_lengths_in[i];
        total_doc_len += (double) doc_lengths[i];
    }
    if (num_entries > 0 && total_doc_len == 0.0)
    {
        status = EVOKE_ERR_INVALID;
        goto cleanup;
    }
    avg_doc_len = total_doc_len / (double) num_docs;

    if (vocab_size > 0)
    {
        if (!evoke_checked_mul_size(
                vocab_size,
                sizeof(*doc_frequencies),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        doc_frequencies = calloc(vocab_size, sizeof(*doc_frequencies));
        idf_array = calloc(vocab_size, sizeof(*idf_array));
        if (doc_frequencies == NULL || idf_array == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (j = 0; j < num_entries; j++)
    {
        evoke_term_entry entry;

        status = read_cb(reader_ctx, &entry);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (entry.doc_id >= num_docs || entry.token_id >= vocab_size)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        if (entry.tf == 0)
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        if (entry.tf > doc_lengths[entry.doc_id] ||
            doc_frequencies[entry.token_id] == UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        doc_frequencies[entry.token_id]++;
    }

    status = rewind_cb(reader_ctx);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }

    for (i = 0; i < vocab_size; i++)
    {
        if (doc_frequencies[i] != 0)
        {
            idf_array[i] = (float) evoke_score_idf(
                params->idf_method,
                (double) doc_frequencies[i],
                (double) num_docs
            );
        }
    }

    if (evoke_method_requires_nonoccurrence(params->method) &&
        vocab_size > 0)
    {
        nonoccurrence = calloc(vocab_size, sizeof(*nonoccurrence));
        if (nonoccurrence == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        for (i = 0; i < vocab_size; i++)
        {
            if (doc_frequencies[i] != 0)
            {
                double idf = evoke_score_idf(
                    params->idf_method,
                    (double) doc_frequencies[i],
                    (double) num_docs
                );
                double tfc = evoke_score_tfc(
                    params->method,
                    0.0,
                    avg_doc_len,
                    avg_doc_len,
                    params->k1,
                    params->b,
                    params->delta
                );

                nonoccurrence[i] = (float) (idf * tfc);
            }
        }
    }

    if (!evoke_checked_mul_size(
            (size_t) vocab_size + 1,
            sizeof(*indptr),
            &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    indptr = calloc((size_t) vocab_size + 1, sizeof(*indptr));
    heads = calloc((size_t) vocab_size + 1, sizeof(*heads));
    if (indptr == NULL || heads == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (j = 0; j < num_entries; j++)
    {
        evoke_term_entry entry;

        status = read_cb(reader_ctx, &entry);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (entry.doc_id >= num_docs ||
            entry.token_id >= vocab_size ||
            entry.tf == 0 ||
            entry.tf > doc_lengths[entry.doc_id])
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        indptr[(size_t) entry.token_id + 1]++;
    }
    status = rewind_cb(reader_ctx);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (i = 0; i < vocab_size; i++)
    {
        indptr[(size_t) i + 1] += indptr[i];
        if (indptr[(size_t) i + 1] - indptr[i] !=
            doc_frequencies[i])
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        heads[i] = indptr[i];
    }

    if (num_entries > 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) num_entries,
                sizeof(*data),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        data = malloc(bytes);
        if (data == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        if (!evoke_checked_mul_size(
                (size_t) num_entries,
                sizeof(*indices),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        indices = malloc(bytes);
        if (indices == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        if (!evoke_checked_mul_size(
                (size_t) num_entries,
                sizeof(*term_frequencies),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        term_frequencies = malloc(bytes);
        if (term_frequencies == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (j = 0; j < num_entries; j++)
    {
        evoke_term_entry entry;
        uint32_t token_id;
        uint32_t doc_id;
        uint64_t pos;
        double doc_len;
        double tf;
        double tfc;
        double score;

        status = read_cb(reader_ctx, &entry);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (entry.doc_id >= num_docs ||
            entry.token_id >= vocab_size ||
            entry.tf == 0 ||
            entry.tf > doc_lengths[entry.doc_id])
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        token_id = entry.token_id;
        doc_id = entry.doc_id;
        pos = heads[token_id]++;
        if (pos >= indptr[(size_t) token_id + 1])
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        doc_len = (double) doc_lengths[doc_id];
        tf = (double) entry.tf;
        tfc = evoke_score_tfc(
            params->method,
            tf,
            doc_len,
            avg_doc_len,
            params->k1,
            params->b,
            params->delta
        );
        score = (double) idf_array[token_id] * tfc;

        if (evoke_method_requires_nonoccurrence(params->method))
        {
            score -= (double) nonoccurrence[token_id];
        }

        data[pos] = (float) score;
        indices[pos] = doc_id;
        term_frequencies[pos] = entry.tf;
    }
    for (i = 0; i < vocab_size; i++)
    {
        if (heads[i] != indptr[(size_t) i + 1])
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
    }

    status = evoke_copy_vocab(vocab, vocab_size, &vocab_copy);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }

    evoke_index_init(index_out);
    index_out->params = *params;
    index_out->num_docs = num_docs;
    index_out->vocab_size = vocab_size;
    index_out->data_len = num_entries;
    index_out->has_empty_token = create_empty_token && has_empty_token;
    index_out->empty_token_id = empty_token_id;
    index_out->data = data;
    index_out->indices = indices;
    index_out->indptr = indptr;
    index_out->term_frequencies = term_frequencies;
    index_out->doc_lengths = doc_lengths;
    index_out->doc_frequencies = doc_frequencies;
    index_out->nonoccurrence = nonoccurrence;
    index_out->vocab = vocab_copy;

    data = NULL;
    indices = NULL;
    indptr = NULL;
    term_frequencies = NULL;
    doc_lengths = NULL;
    doc_frequencies = NULL;
    nonoccurrence = NULL;
    vocab_copy = NULL;

cleanup:
    free(heads);
    free(data);
    free(indices);
    free(indptr);
    free(term_frequencies);
    free(doc_lengths);
    free(doc_frequencies);
    free(idf_array);
    free(nonoccurrence);
    if (vocab_copy != NULL)
    {
        for (i = 0; i < vocab_size; i++)
        {
            free(vocab_copy[i]);
        }
        free(vocab_copy);
    }
    return status;
}

typedef struct evoke_memory_term_entry_reader
{
    const evoke_term_entry *entries;
    uint64_t len;
    uint64_t pos;
} evoke_memory_term_entry_reader;

static evoke_status
evoke_memory_term_entry_read(
    void *ctx,
    evoke_term_entry *entry_out
)
{
    evoke_memory_term_entry_reader *reader = ctx;

    if (reader == NULL || entry_out == NULL || reader->entries == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (reader->pos >= reader->len)
    {
        return EVOKE_ERR_RANGE;
    }

    *entry_out = reader->entries[reader->pos++];
    return EVOKE_OK;
}

static evoke_status
evoke_memory_term_entry_rewind(void *ctx)
{
    evoke_memory_term_entry_reader *reader = ctx;

    if (reader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    reader->pos = 0;
    return EVOKE_OK;
}

evoke_status
evoke_build_index_from_term_entries(
    const evoke_term_entry *entries,
    uint64_t num_entries,
    const uint32_t *doc_lengths,
    uint32_t num_docs,
    uint32_t vocab_size,
    const evoke_params *params,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    const char *const *vocab,
    evoke_index *index_out
)
{
    evoke_memory_term_entry_reader reader;

    if (num_entries > 0 && entries == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    reader.entries = entries;
    reader.len = num_entries;
    reader.pos = 0;
    return evoke_build_index_from_term_entry_reader(
        num_entries,
        evoke_memory_term_entry_read,
        evoke_memory_term_entry_rewind,
        &reader,
        doc_lengths,
        num_docs,
        vocab_size,
        params,
        create_empty_token,
        has_empty_token,
        empty_token_id,
        vocab,
        index_out
    );
}

evoke_status
evoke_build_index_from_ids(
    const evoke_doc_ids *docs,
    size_t num_docs,
    const evoke_params *params,
    bool create_empty_token,
    evoke_index *index_out
)
{
    uint32_t vocab_size = 0;
    uint32_t max_token_id = 0;
    bool has_terms = false;
    bool zero_present = false;
    uint32_t empty_token_id = 0;
    bool has_empty_token = false;
    size_t i;
    size_t j;

    if ((num_docs > 0 && docs == NULL) ||
        params == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_docs > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    for (i = 0; i < num_docs; i++)
    {
        if (docs[i].len > UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        if (docs[i].len > 0 && docs[i].token_ids == NULL)
        {
            return EVOKE_ERR_INVALID;
        }
        for (j = 0; j < docs[i].len; j++)
        {
            uint32_t token_id = docs[i].token_ids[j];
            if (!has_terms || token_id > max_token_id)
            {
                max_token_id = token_id;
            }
            if (token_id == 0)
            {
                zero_present = true;
            }
            has_terms = true;
        }
    }

    if (has_terms)
    {
        if (max_token_id == UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        vocab_size = max_token_id + 1;
    }

    if (create_empty_token)
    {
        has_empty_token = true;
        if (!zero_present)
        {
            empty_token_id = 0;
        }
        else
        {
            if (!has_terms)
            {
                empty_token_id = 0;
            }
            else if (max_token_id == UINT32_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            else
            {
                empty_token_id = max_token_id + 1;
            }
        }

        if (empty_token_id + 1 > vocab_size)
        {
            vocab_size = empty_token_id + 1;
        }
    }

    if (num_docs > 0 && !has_terms && !create_empty_token)
    {
        return EVOKE_ERR_INVALID;
    }

    return evoke_build_index_internal(
        docs,
        num_docs,
        vocab_size,
        params,
        create_empty_token,
        has_empty_token,
        empty_token_id,
        NULL,
        index_out
    );
}

evoke_status
evoke_build_index_from_ids_compact(
    const evoke_doc_ids *docs,
    size_t num_docs,
    const evoke_params *params,
    bool create_empty_token,
    evoke_index *index_out
)
{
    uint32_t *doc_lengths = NULL;
    uint32_t vocab_size = 0;
    uint32_t max_token_id = 0;
    bool has_terms = false;
    bool zero_present = false;
    uint32_t empty_token_id = 0;
    bool has_empty_token = false;
    evoke_term_entry_builder entries = {0};
    size_t bytes;
    size_t i;
    size_t j;
    evoke_status status = EVOKE_OK;

    if ((num_docs > 0 && docs == NULL) ||
        params == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_docs == 0)
    {
        return evoke_build_empty_index(
            params,
            create_empty_token ? 1 : 0,
            create_empty_token,
            create_empty_token,
            0,
            NULL,
            index_out
        );
    }
    if (num_docs > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (!evoke_checked_mul_size(num_docs, sizeof(*doc_lengths), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    doc_lengths = malloc(bytes);
    if (doc_lengths == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_docs; i++)
    {
        if (docs[i].len > 0 && docs[i].token_ids == NULL)
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        if (docs[i].len > UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        doc_lengths[i] = (uint32_t) docs[i].len;
        for (j = 0; j < docs[i].len; j++)
        {
            uint32_t token_id = docs[i].token_ids[j];

            if (!has_terms || token_id > max_token_id)
            {
                max_token_id = token_id;
            }
            if (token_id == 0)
            {
                zero_present = true;
            }
            has_terms = true;
        }
    }

    if (has_terms)
    {
        if (max_token_id == UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        vocab_size = max_token_id + 1;
    }
    if (create_empty_token)
    {
        has_empty_token = true;
        if (!zero_present)
        {
            empty_token_id = 0;
        }
        else if (!has_terms)
        {
            empty_token_id = 0;
        }
        else if (max_token_id == UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        else
        {
            empty_token_id = max_token_id + 1;
        }
        if (empty_token_id + 1 > vocab_size)
        {
            vocab_size = empty_token_id + 1;
        }
    }
    if (!has_terms && !create_empty_token)
    {
        status = EVOKE_ERR_INVALID;
        goto cleanup;
    }

    for (i = 0; i < num_docs; i++)
    {
        evoke_term_freqs terms = {0};

        status = evoke_count_doc_terms(
            docs[i].token_ids,
            docs[i].len,
            &terms
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        for (j = 0; j < terms.len; j++)
        {
            status = evoke_term_entry_builder_append(
                &entries,
                terms.terms[j].token_id,
                (uint32_t) i,
                terms.terms[j].tf
            );
            if (status != EVOKE_OK)
            {
                evoke_term_freqs_free(&terms);
                goto cleanup;
            }
        }
        evoke_term_freqs_free(&terms);
    }

    status = evoke_build_index_from_term_entries(
        entries.entries,
        entries.len,
        doc_lengths,
        (uint32_t) num_docs,
        vocab_size,
        params,
        create_empty_token,
        has_empty_token,
        empty_token_id,
        NULL,
        index_out
    );

cleanup:
    free(doc_lengths);
    evoke_term_entry_builder_free(&entries);
    return status;
}

evoke_status
evoke_build_index_from_tokens(
    const evoke_doc_tokens *docs,
    size_t num_docs,
    const evoke_params *params,
    evoke_index *index_out
)
{
    evoke_doc_ids *docs_as_ids = NULL;
    evoke_strmap vocab_map;
    char **vocab = NULL;
    size_t i;
    evoke_status status;

    if ((num_docs > 0 && docs == NULL) ||
        params == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_docs > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (num_docs == 0)
    {
        return evoke_build_empty_index(
            params,
            0,
            false,
            false,
            0,
            NULL,
            index_out
        );
    }
    memset(&vocab_map, 0, sizeof(vocab_map));
    status = evoke_strmap_init(&vocab_map, 8);
    if (status != EVOKE_OK)
    {
        return status;
    }

    docs_as_ids = calloc(num_docs, sizeof(*docs_as_ids));
    if (docs_as_ids == NULL)
    {
        evoke_strmap_destroy(&vocab_map);
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_docs; i++)
    {
        size_t j;
        size_t bytes;

        if (docs[i].len > UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        if (docs[i].len > 0 && docs[i].tokens == NULL)
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        docs_as_ids[i].len = docs[i].len;
        if (!evoke_checked_mul_size(
                docs[i].len,
                sizeof(*docs_as_ids[i].token_ids),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        docs_as_ids[i].token_ids = NULL;
        if (docs[i].len > 0)
        {
            docs_as_ids[i].token_ids = malloc(bytes);
            if (docs_as_ids[i].token_ids == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto cleanup;
            }
        }

        for (j = 0; j < docs[i].len; j++)
        {
            uint32_t token_id = 0;
            bool found = false;

            if (docs[i].tokens[j] == NULL)
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }

            status = evoke_strmap_get(
                &vocab_map,
                docs[i].tokens[j],
                &token_id,
                &found
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }

            if (!found)
            {
                if (vocab_map.size >= UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                token_id = (uint32_t) vocab_map.size;
                status = evoke_strmap_put(
                    &vocab_map,
                    docs[i].tokens[j],
                    token_id
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
            }

            docs_as_ids[i].token_ids[j] = token_id;
        }
    }

    if (vocab_map.size == 0)
    {
        status = EVOKE_ERR_INVALID;
        goto cleanup;
    }

    vocab = calloc(vocab_map.size, sizeof(*vocab));
    if (vocab == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (i = 0; i < vocab_map.capacity; i++)
    {
        if (vocab_map.slots[i].used)
        {
            vocab[vocab_map.slots[i].value] = vocab_map.slots[i].key;
            vocab_map.slots[i].key = NULL;
        }
    }

    status = evoke_build_index_internal(
        docs_as_ids,
        num_docs,
        (uint32_t) vocab_map.size,
        params,
        false,
        false,
        0,
        vocab,
        index_out
    );
    if (status == EVOKE_OK)
    {
        vocab = NULL;
    }

cleanup:
    if (docs_as_ids != NULL)
    {
        for (i = 0; i < num_docs; i++)
        {
            free(docs_as_ids[i].token_ids);
        }
    }
    free(docs_as_ids);
    if (vocab != NULL)
    {
        for (i = 0; i < vocab_map.size; i++)
        {
            free(vocab[i]);
        }
    }
    free(vocab);
    evoke_strmap_destroy(&vocab_map);
    return status;
}

evoke_status
evoke_build_index_from_tokens_compact(
    const evoke_doc_tokens *docs,
    size_t num_docs,
    const evoke_params *params,
    evoke_index *index_out
)
{
    evoke_doc_ids *docs_as_ids = NULL;
    evoke_strmap vocab_map;
    char **vocab = NULL;
    size_t i;
    evoke_status status;

    if ((num_docs > 0 && docs == NULL) ||
        params == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_docs > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (num_docs == 0)
    {
        return evoke_build_empty_index(
            params,
            0,
            false,
            false,
            0,
            NULL,
            index_out
        );
    }

    memset(&vocab_map, 0, sizeof(vocab_map));
    status = evoke_strmap_init(&vocab_map, 8);
    if (status != EVOKE_OK)
    {
        return status;
    }

    docs_as_ids = calloc(num_docs, sizeof(*docs_as_ids));
    if (docs_as_ids == NULL)
    {
        evoke_strmap_destroy(&vocab_map);
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_docs; i++)
    {
        size_t j;
        size_t bytes;

        if (docs[i].len > UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        if (docs[i].len > 0 && docs[i].tokens == NULL)
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        docs_as_ids[i].len = docs[i].len;
        if (!evoke_checked_mul_size(
                docs[i].len,
                sizeof(*docs_as_ids[i].token_ids),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        if (docs[i].len > 0)
        {
            docs_as_ids[i].token_ids = malloc(bytes);
            if (docs_as_ids[i].token_ids == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto cleanup;
            }
        }

        for (j = 0; j < docs[i].len; j++)
        {
            uint32_t token_id = 0;
            bool found = false;

            if (docs[i].tokens[j] == NULL)
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }

            status = evoke_strmap_get(
                &vocab_map,
                docs[i].tokens[j],
                &token_id,
                &found
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }

            if (!found)
            {
                if (vocab_map.size >= UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                token_id = (uint32_t) vocab_map.size;
                status = evoke_strmap_put(
                    &vocab_map,
                    docs[i].tokens[j],
                    token_id
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
            }

            docs_as_ids[i].token_ids[j] = token_id;
        }
    }

    if (vocab_map.size == 0)
    {
        status = EVOKE_ERR_INVALID;
        goto cleanup;
    }

    vocab = calloc(vocab_map.size, sizeof(*vocab));
    if (vocab == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (i = 0; i < vocab_map.capacity; i++)
    {
        if (vocab_map.slots[i].used)
        {
            vocab[vocab_map.slots[i].value] = vocab_map.slots[i].key;
            vocab_map.slots[i].key = NULL;
        }
    }

    status = evoke_build_index_from_ids_compact(
        docs_as_ids,
        num_docs,
        params,
        false,
        index_out
    );
    if (status == EVOKE_OK)
    {
        index_out->vocab = vocab;
        vocab = NULL;
    }

cleanup:
    if (docs_as_ids != NULL)
    {
        for (i = 0; i < num_docs; i++)
        {
            free(docs_as_ids[i].token_ids);
        }
    }
    free(docs_as_ids);
    if (vocab != NULL)
    {
        for (i = 0; i < vocab_map.size; i++)
        {
            free(vocab[i]);
        }
    }
    free(vocab);
    evoke_strmap_destroy(&vocab_map);
    return status;
}

evoke_status
evoke_query_token_ids(
    const evoke_index *index,
    const char **tokens,
    size_t num_tokens,
    uint32_t **query_ids_out,
    size_t *query_len_out
)
{
    evoke_strmap vocab_map;
    uint32_t *query_ids = NULL;
    size_t bytes;
    size_t i;
    evoke_status status;

    if (index == NULL || query_ids_out == NULL || query_len_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    *query_ids_out = NULL;
    *query_len_out = 0;

    if (index->vocab == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    memset(&vocab_map, 0, sizeof(vocab_map));
    status = evoke_strmap_init(
        &vocab_map,
        index->vocab_size > 8 ? index->vocab_size * 2 : 8
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    for (i = 0; i < index->vocab_size; i++)
    {
        status = evoke_strmap_put(&vocab_map, index->vocab[i], i);
        if (status != EVOKE_OK)
        {
            evoke_strmap_destroy(&vocab_map);
            return status;
        }
    }

    if (!evoke_checked_mul_size(num_tokens, sizeof(*query_ids), &bytes))
    {
        evoke_strmap_destroy(&vocab_map);
        return EVOKE_ERR_RANGE;
    }
    if (num_tokens > 0)
    {
        query_ids = malloc(bytes);
        if (query_ids == NULL)
        {
            evoke_strmap_destroy(&vocab_map);
            return EVOKE_ERR_NOMEM;
        }
    }

    *query_len_out = 0;
    for (i = 0; i < num_tokens; i++)
    {
        uint32_t token_id = 0;
        bool found = false;

        if (tokens[i] == NULL)
        {
            free(query_ids);
            evoke_strmap_destroy(&vocab_map);
            return EVOKE_ERR_INVALID;
        }

        status = evoke_strmap_get(
            &vocab_map,
            tokens[i],
            &token_id,
            &found
        );
        if (status != EVOKE_OK)
        {
            free(query_ids);
            evoke_strmap_destroy(&vocab_map);
            return status;
        }

        if (found)
        {
            query_ids[*query_len_out] = token_id;
            (*query_len_out)++;
        }
    }

    evoke_strmap_destroy(&vocab_map);
    *query_ids_out = query_ids;
    return EVOKE_OK;
}

static bool
evoke_posting_extent_layout_is_valid(
    const evoke_index *index,
    const evoke_posting_extent *extent
)
{
    if (index == NULL || extent == NULL ||
        (extent->len > 0 && extent->indices == NULL) ||
        extent->local_document_count == 0)
    {
        return false;
    }
    if (extent->document_id_map == NULL &&
        ((uint64_t) extent->document_id_base +
         extent->local_document_count >
         index->num_docs))
    {
        return false;
    }
    return true;
}

evoke_status
evoke_posting_extent_validate_layout(
    const evoke_index *index,
    const evoke_posting_extent *extent
)
{
    bool has_blocks;
    uint32_t i;
    uint64_t posting_index;

    if (!evoke_posting_extent_layout_is_valid(index, extent))
    {
        return EVOKE_ERR_INVALID;
    }
    has_blocks = extent->blocks != NULL ||
        extent->block_count != 0 || extent->block_shift != 0;
    if (has_blocks &&
        (extent->blocks == NULL ||
         extent->block_count == 0 ||
         extent->block_shift == 0 ||
         extent->block_shift >= 32))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (posting_index = 0; posting_index < extent->len; posting_index++)
    {
        if (extent->indices[posting_index] >=
            extent->local_document_count)
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if (extent->document_id_map != NULL)
    {
        for (i = 0; i < extent->local_document_count; i++)
        {
            if (extent->document_id_map[i] >= index->num_docs)
            {
                return EVOKE_ERR_RANGE;
            }
        }
    }
    if (has_blocks &&
        evoke_posting_extent_validate_block_records(
            extent,
            extent->block_shift,
            extent->blocks,
            extent->block_count) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static uint32_t
evoke_posting_extent_document_id(
    const evoke_posting_extent *extent,
    uint64_t posting_index
)
{
    uint32_t local_id = extent->indices[posting_index];

    return extent->document_id_map == NULL
        ? extent->document_id_base + local_id
        : extent->document_id_map[local_id];
}

static bool
evoke_posting_block_record_equal(
    const evoke_posting_block_record *left,
    const evoke_posting_block_record *right
)
{
    return left->posting_offset == right->posting_offset &&
        left->posting_count == right->posting_count &&
        left->block_id == right->block_id &&
        left->first_document_id == right->first_document_id &&
        left->last_document_id == right->last_document_id &&
        left->min_term_frequency == right->min_term_frequency &&
        left->max_term_frequency == right->max_term_frequency &&
        left->min_impact == right->min_impact &&
        left->max_impact == right->max_impact &&
        left->kind == right->kind;
}

evoke_status
evoke_posting_extent_build_block_records(
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    evoke_posting_block_record **records_out,
    size_t *record_count_out
)
{
    evoke_posting_block_record *records = NULL;
    size_t record_count = 0;
    size_t bytes;
    uint32_t prior_document_id = 0;

    if (extent == NULL || records_out == NULL ||
        record_count_out == NULL || block_shift >= 32 ||
        extent->local_document_count == 0 ||
        (extent->len > 0 && extent->indices == NULL) ||
        (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->term_frequencies == NULL && extent->values == NULL) ||
        ((extent->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
          extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT) &&
         extent->data == NULL && extent->values == NULL) ||
        (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
         extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT))
    {
        return EVOKE_ERR_INVALID;
    }
    *records_out = NULL;
    *record_count_out = 0;
    if (extent->len == 0)
    {
        return EVOKE_OK;
    }
    if (extent->len > SIZE_MAX ||
        !evoke_checked_mul_size(
            (size_t) extent->len,
            sizeof(*records),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    records = calloc(1, bytes);
    if (records == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (uint64_t posting_index = 0;
         posting_index < extent->len;
         posting_index++)
    {
        uint32_t local_document_id = extent->indices[posting_index];
        uint64_t global_document_id;
        uint32_t document_id;
        uint32_t block_id;
        evoke_posting_block_record *record;

        if (local_document_id >= extent->local_document_count)
        {
            free(records);
            return EVOKE_ERR_FORMAT;
        }
        global_document_id = extent->document_id_map == NULL
            ? (uint64_t) extent->document_id_base + local_document_id
            : extent->document_id_map[local_document_id];
        if (global_document_id > UINT32_MAX)
        {
            free(records);
            return EVOKE_ERR_RANGE;
        }
        document_id = (uint32_t) global_document_id;
        if (posting_index > 0 && document_id <= prior_document_id)
        {
            free(records);
            return EVOKE_ERR_FORMAT;
        }
        block_id = document_id >> block_shift;
        if (record_count == 0 ||
            records[record_count - 1].block_id != block_id)
        {
            record = &records[record_count++];
            record->posting_offset = posting_index;
            record->block_id = block_id;
            record->first_document_id = document_id;
            record->last_document_id = document_id;
            record->kind = extent->kind;
        }
        record = &records[record_count - 1];
        if (record->posting_count == UINT32_MAX)
        {
            free(records);
            return EVOKE_ERR_RANGE;
        }
        record->posting_count++;
        record->last_document_id = document_id;
        if (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            uint32_t term_frequency =
                extent->term_frequencies == NULL
                ? extent->values[posting_index].term_frequency
                : extent->term_frequencies[posting_index];

            if (term_frequency == 0)
            {
                free(records);
                return EVOKE_ERR_FORMAT;
            }
            if (record->posting_count == 1)
            {
                record->min_term_frequency = term_frequency;
                record->max_term_frequency = term_frequency;
            }
            else
            {
                if (term_frequency < record->min_term_frequency)
                {
                    record->min_term_frequency = term_frequency;
                }
                if (term_frequency > record->max_term_frequency)
                {
                    record->max_term_frequency = term_frequency;
                }
            }
        }
        else
        {
            float impact = extent->data == NULL
                ? extent->values[posting_index].impact
                : extent->data[posting_index];

            if (!isfinite(impact))
            {
                free(records);
                return EVOKE_ERR_FORMAT;
            }
            if (record->posting_count == 1)
            {
                record->min_impact = impact;
                record->max_impact = impact;
            }
            else
            {
                if (impact < record->min_impact)
                {
                    record->min_impact = impact;
                }
                if (impact > record->max_impact)
                {
                    record->max_impact = impact;
                }
            }
        }
        prior_document_id = document_id;
    }

    *records_out = records;
    *record_count_out = record_count;
    return EVOKE_OK;
}

void
evoke_posting_block_records_free(evoke_posting_block_record *records)
{
    free(records);
}

evoke_status
evoke_posting_extent_validate_block_records(
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    const evoke_posting_block_record *records,
    size_t record_count
)
{
    evoke_posting_block_record *expected = NULL;
    size_t expected_count = 0;
    evoke_status status;

    if ((record_count > 0 && records == NULL) ||
        record_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_posting_extent_build_block_records(
        extent,
        block_shift,
        &expected,
        &expected_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (record_count != expected_count)
    {
        free(expected);
        return EVOKE_ERR_FORMAT;
    }
    for (size_t record_index = 0;
         record_index < record_count;
         record_index++)
    {
        if (!evoke_posting_block_record_equal(
                &records[record_index],
                &expected[record_index]))
        {
            free(expected);
            return EVOKE_ERR_FORMAT;
        }
    }
    free(expected);
    return EVOKE_OK;
}

evoke_status
evoke_posting_extent_build_block_bounds(
    const evoke_index *index,
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    evoke_posting_block_bound **bounds_out,
    size_t *bound_count_out
)
{
    evoke_posting_block_record *records = NULL;
    evoke_posting_block_bound *bounds = NULL;
    size_t record_count = 0;
    size_t bytes;
    evoke_status status;

    if (index == NULL || extent == NULL || bounds_out == NULL ||
        bound_count_out == NULL || block_shift >= 32)
    {
        return EVOKE_ERR_INVALID;
    }
    *bounds_out = NULL;
    *bound_count_out = 0;
    if (evoke_posting_extent_validate_layout(index, extent) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->term_frequencies == NULL && extent->values == NULL) ||
        ((extent->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
          extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT) &&
         extent->data == NULL && extent->values == NULL) ||
        (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
         extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT) ||
        (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         index->doc_lengths == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (extent->len == 0)
    {
        return EVOKE_OK;
    }
    status = evoke_posting_extent_build_block_records(
        extent,
        block_shift,
        &records,
        &record_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (record_count >
            SIZE_MAX / sizeof(*bounds) ||
        !evoke_checked_mul_size(
            record_count,
            sizeof(*bounds),
            &bytes))
    {
        free(records);
        return EVOKE_ERR_RANGE;
    }
    bounds = calloc(1, bytes);
    if (bounds == NULL)
    {
        free(records);
        return EVOKE_ERR_NOMEM;
    }

    for (size_t record_index = 0;
         record_index < record_count;
         record_index++)
    {
        const evoke_posting_block_record *record =
            &records[record_index];
        evoke_posting_block_bound *bound = &bounds[record_index];

        bound->posting_offset = record->posting_offset;
        bound->posting_count = record->posting_count;
        bound->block_id = record->block_id;
        bound->first_document_id = record->first_document_id;
        bound->last_document_id = record->last_document_id;
        bound->min_term_frequency = record->min_term_frequency;
        bound->max_term_frequency = record->max_term_frequency;
        bound->min_impact = record->min_impact;
        bound->max_impact = record->max_impact;
        bound->kind = record->kind;
        if (record->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            uint64_t posting_end =
                record->posting_offset + record->posting_count;

            for (uint64_t posting_index = record->posting_offset;
                 posting_index < posting_end;
                 posting_index++)
            {
                uint32_t document_id =
                    evoke_posting_extent_document_id(
                        extent,
                        posting_index
                    );
                uint32_t document_length =
                    index->doc_lengths[document_id];

                if (posting_index == record->posting_offset)
                {
                    bound->min_document_length = document_length;
                    bound->max_document_length = document_length;
                }
                else
                {
                    if (document_length <
                        bound->min_document_length)
                    {
                        bound->min_document_length =
                            document_length;
                    }
                    if (document_length >
                        bound->max_document_length)
                    {
                        bound->max_document_length =
                            document_length;
                    }
                }
            }
        }
    }

    free(records);
    *bounds_out = bounds;
    *bound_count_out = record_count;
    return EVOKE_OK;
}

void
evoke_posting_block_bounds_free(evoke_posting_block_bound *bounds)
{
    free(bounds);
}

static float
evoke_posting_score_round_up(double value)
{
    float rounded;

    if (!(value > 0.0))
    {
        return 0.0f;
    }
    if (!isfinite(value) || value > FLT_MAX)
    {
        return INFINITY;
    }
    rounded = (float) value;
    if (!isfinite(rounded))
    {
        return INFINITY;
    }
    return nextafterf(rounded, INFINITY);
}

evoke_status
evoke_posting_block_score_upper_bound(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    uint32_t live_document_frequency,
    const evoke_posting_block_bound *bound,
    float query_weight,
    float *score_upper_bound_out
)
{
    double best = 0.0;

    if (index == NULL || stats == NULL || bound == NULL ||
        score_upper_bound_out == NULL || !isfinite(query_weight) ||
        bound->posting_count == 0 ||
        bound->first_document_id > bound->last_document_id ||
        bound->last_document_id >= index->num_docs)
    {
        return EVOKE_ERR_INVALID;
    }
    *score_upper_bound_out = 0.0f;
    if (bound->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
    {
        uint32_t term_frequencies[2];
        uint32_t document_lengths[2];
        double average_document_length;
        double idf;
        double nonoccurrence = 0.0;
        size_t tf_index;
        size_t length_index;

        if (!evoke_params_are_valid(&index->params) ||
            stats->document_count == 0 ||
            stats->total_document_length == 0 ||
            live_document_frequency == 0 ||
            (uint64_t) live_document_frequency > stats->document_count ||
            bound->min_term_frequency == 0 ||
            bound->min_term_frequency > bound->max_term_frequency ||
            bound->min_document_length > bound->max_document_length)
        {
            return EVOKE_ERR_FORMAT;
        }
        average_document_length =
            (double) stats->total_document_length /
            (double) stats->document_count;
        idf = evoke_score_idf(
            index->params.idf_method,
            (double) live_document_frequency,
            (double) stats->document_count
        );
        if (evoke_method_requires_nonoccurrence(index->params.method))
        {
            nonoccurrence = idf * evoke_score_tfc(
                index->params.method,
                0.0,
                0.0,
                average_document_length,
                index->params.k1,
                index->params.b,
                index->params.delta
            );
        }
        term_frequencies[0] = bound->min_term_frequency;
        term_frequencies[1] = bound->max_term_frequency;
        document_lengths[0] = bound->min_document_length;
        document_lengths[1] = bound->max_document_length;
        for (tf_index = 0; tf_index < 2; tf_index++)
        {
            for (length_index = 0; length_index < 2; length_index++)
            {
                double score = idf * evoke_score_tfc(
                    index->params.method,
                    term_frequencies[tf_index],
                    document_lengths[length_index],
                    average_document_length,
                    index->params.k1,
                    index->params.b,
                    index->params.delta
                );
                double contribution =
                    (double) query_weight * (score - nonoccurrence);

                if (contribution > best)
                {
                    best = contribution;
                }
            }
        }
    }
    else if (
        bound->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        bound->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
    {
        double low;
        double high;

        if (!isfinite(bound->min_impact) ||
            !isfinite(bound->max_impact) ||
            bound->min_impact > bound->max_impact)
        {
            return EVOKE_ERR_FORMAT;
        }
        low = (double) query_weight * (double) bound->min_impact;
        high = (double) query_weight * (double) bound->max_impact;
        best = fmax(0.0, fmax(low, high));
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }

    *score_upper_bound_out = evoke_posting_score_round_up(best);
    return EVOKE_OK;
}

static evoke_status
evoke_scores_from_ids_impl(
    const evoke_index *index,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    float *scores = NULL;
    uint32_t *filtered_ids = NULL;
    size_t filtered_len = 0;
    size_t i;
    size_t bytes;
    double nonoccurrence_sum = 0.0;

    if (index == NULL || scores_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (term_extents != NULL &&
        num_term_extent_lists < (size_t) index->vocab_size)
    {
        return EVOKE_ERR_INVALID;
    }

    *scores_out = NULL;
    if (index->num_docs == 0)
    {
        return EVOKE_OK;
    }
    if (!evoke_checked_mul_size(index->num_docs, sizeof(*scores), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    scores = calloc(index->num_docs, sizeof(*scores));
    if (scores == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    if (query_len > 0)
    {
        if (!evoke_checked_mul_size(query_len, sizeof(*filtered_ids), &bytes))
        {
            free(scores);
            return EVOKE_ERR_RANGE;
        }
        filtered_ids = malloc(bytes);
        if (filtered_ids == NULL)
        {
            free(scores);
            return EVOKE_ERR_NOMEM;
        }

        for (i = 0; i < query_len; i++)
        {
            if (query_ids[i] < index->vocab_size)
            {
                filtered_ids[filtered_len++] = query_ids[i];
            }
        }
    }

    if (filtered_len == 0 && index->has_empty_token)
    {
        uint32_t *resized = realloc(
            filtered_ids,
            sizeof(*filtered_ids)
        );

        if (resized == NULL)
        {
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        filtered_ids = resized;
        filtered_ids[0] = index->empty_token_id;
        filtered_len = 1;
    }

    if (term_extents == NULL)
    {
        for (i = 0; i < filtered_len; i++)
        {
            uint32_t token_id = filtered_ids[i];
            uint64_t start = index->indptr[token_id];
            uint64_t end = index->indptr[token_id + 1];
            uint64_t j;

            for (j = start; j < end; j++)
            {
                scores[index->indices[j]] += index->data[j];
            }
        }
    }
    else
    {
        for (i = 0; i < filtered_len; i++)
        {
            uint32_t token_id = filtered_ids[i];
            const evoke_term_extent_list *list = &term_extents[token_id];
            size_t extent_index;

            if (list->len > 0 && list->extents == NULL)
            {
                free(filtered_ids);
                free(scores);
                return EVOKE_ERR_INVALID;
            }

            for (extent_index = 0; extent_index < list->len; extent_index++)
            {
                const evoke_posting_extent *extent =
                    &list->extents[extent_index];
                uint64_t j;

                if (extent->len > 0 &&
                    extent->data == NULL &&
                    extent->values == NULL)
                {
                    free(filtered_ids);
                    free(scores);
                    return EVOKE_ERR_INVALID;
                }
                if (!evoke_posting_extent_layout_is_valid(index, extent))
                {
                    free(filtered_ids);
                    free(scores);
                    return EVOKE_ERR_INVALID;
                }

                if (extent->document_id_map == NULL)
                {
                    for (j = 0; j < extent->len; j++)
                    {
                        uint32_t document_id =
                            extent->document_id_base + extent->indices[j];

                        scores[document_id] += extent->data == NULL ?
                            extent->values[j].impact :
                            extent->data[j];
                    }
                }
                else
                {
                    for (j = 0; j < extent->len; j++)
                    {
                        uint32_t document_id =
                            extent->document_id_map[extent->indices[j]];

                        scores[document_id] += extent->data == NULL ?
                            extent->values[j].impact :
                            extent->data[j];
                    }
                }
            }
        }
    }

    evoke_apply_weight_mask_inplace(scores, weight_mask, index->num_docs);

    if (index->nonoccurrence != NULL)
    {
        for (i = 0; i < filtered_len; i++)
        {
            nonoccurrence_sum += (double) index->nonoccurrence[filtered_ids[i]];
        }
        evoke_add_constant_inplace(
            scores,
            (float) nonoccurrence_sum,
            index->num_docs
        );
    }

    free(filtered_ids);
    *scores_out = scores;
    return EVOKE_OK;
}

evoke_status
evoke_scores_from_ids(
    const evoke_index *index,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    return evoke_scores_from_ids_impl(
        index,
        NULL,
        0,
        query_ids,
        query_len,
        weight_mask,
        scores_out
    );
}

evoke_status
evoke_scores_from_ids_extents(
    const evoke_index *index,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    if (term_extents == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    return evoke_scores_from_ids_impl(
        index,
        term_extents,
        num_term_extent_lists,
        query_ids,
        query_len,
        weight_mask,
        scores_out
    );
}

#define EVOKE_TFC_CACHE_SIZE 256U

typedef struct evoke_tfc_cache
{
    uint64_t keys[EVOKE_TFC_CACHE_SIZE];
    double values[EVOKE_TFC_CACHE_SIZE];
    uint8_t occupied[EVOKE_TFC_CACHE_SIZE];
} evoke_tfc_cache;

static inline double
evoke_score_tfc_cached(
    const evoke_index *index,
    evoke_tfc_cache *cache,
    uint32_t term_frequency,
    uint32_t doc_length,
    double average_doc_length
)
{
    uint64_t key =
        ((uint64_t) term_frequency << 32) | (uint64_t) doc_length;
    size_t slot = (size_t) (
        (key * UINT64_C(11400714819323198485)) >>
        (64U - 8U)
    );

    if (cache->occupied[slot] != 0 && cache->keys[slot] == key)
    {
        return cache->values[slot];
    }

    cache->keys[slot] = key;
    cache->values[slot] = evoke_score_tfc(
        index->params.method,
        (double) term_frequency,
        (double) doc_length,
        average_doc_length,
        index->params.k1,
        index->params.b,
        index->params.delta
    );
    cache->occupied[slot] = 1;
    return cache->values[slot];
}

static inline void
evoke_accumulate_neutral_posting(
    const evoke_index *index,
    uint32_t doc_id,
    uint32_t term_frequency,
    double idf,
    double nonoccurrence,
    double average_doc_length,
    double query_weight,
    evoke_tfc_cache *tfc_cache,
    float *scores
)
{
    double tfc = evoke_score_tfc_cached(
        index,
        tfc_cache,
        term_frequency,
        index->doc_lengths[doc_id],
        average_doc_length
    );
    double score = idf * tfc;

    if (evoke_method_requires_nonoccurrence(index->params.method))
    {
        score -= nonoccurrence;
    }
    scores[doc_id] += (float) (query_weight * score);
}

static inline evoke_status
evoke_accumulate_neutral_extent(
    const evoke_index *index,
    const evoke_posting_extent *extent,
    double idf,
    double nonoccurrence,
    double average_doc_length,
    double query_weight,
    evoke_tfc_cache *tfc_cache,
    float *scores
)
{
    uint64_t posting_index;

    if (extent->len > 0 &&
        (extent->indices == NULL ||
         (extent->term_frequencies == NULL && extent->values == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }

    if (!evoke_posting_extent_layout_is_valid(index, extent))
    {
        return EVOKE_ERR_INVALID;
    }

    if (extent->document_id_map == NULL)
    {
        if (extent->term_frequencies != NULL)
        {
            for (posting_index = 0;
                 posting_index < extent->len;
                 posting_index++)
            {
                uint32_t doc_id =
                    extent->document_id_base +
                    extent->indices[posting_index];

                evoke_accumulate_neutral_posting(
                    index,
                    doc_id,
                    extent->term_frequencies[posting_index],
                    idf,
                    nonoccurrence,
                    average_doc_length,
                    query_weight,
                    tfc_cache,
                    scores
                );
            }
        }
        else
        {
            for (posting_index = 0;
                 posting_index < extent->len;
                 posting_index++)
            {
                uint32_t doc_id =
                    extent->document_id_base +
                    extent->indices[posting_index];

                evoke_accumulate_neutral_posting(
                    index,
                    doc_id,
                    extent->values[posting_index].term_frequency,
                    idf,
                    nonoccurrence,
                    average_doc_length,
                    query_weight,
                    tfc_cache,
                    scores
                );
            }
        }
    }
    else if (extent->term_frequencies != NULL)
    {
        for (posting_index = 0;
             posting_index < extent->len;
             posting_index++)
        {
            uint32_t doc_id = extent->document_id_map[
                extent->indices[posting_index]
            ];

            evoke_accumulate_neutral_posting(
                index,
                doc_id,
                extent->term_frequencies[posting_index],
                idf,
                nonoccurrence,
                average_doc_length,
                query_weight,
                tfc_cache,
                scores
            );
        }
    }
    else
    {
        for (posting_index = 0;
             posting_index < extent->len;
             posting_index++)
        {
            uint32_t doc_id = extent->document_id_map[
                extent->indices[posting_index]
            ];

            evoke_accumulate_neutral_posting(
                index,
                doc_id,
                extent->values[posting_index].term_frequency,
                idf,
                nonoccurrence,
                average_doc_length,
                query_weight,
                tfc_cache,
                scores
            );
        }
    }

    return EVOKE_OK;
}

evoke_status
evoke_scores_from_ids_neutral(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    float *scores = NULL;
    uint32_t *filtered_ids = NULL;
    size_t filtered_len = 0;
    size_t i;
    size_t bytes;
    double avg_doc_len;
    evoke_tfc_cache tfc_cache;

    if (index == NULL || stats == NULL || scores_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *scores_out = NULL;
    if (index->num_docs == 0)
    {
        return EVOKE_OK;
    }
    if (stats->document_count == 0 ||
        stats->doc_frequencies == NULL ||
        stats->vocab_size < (size_t) index->vocab_size ||
        index->doc_lengths == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (term_extents == NULL)
    {
        if (index->term_frequencies == NULL ||
            index->indices == NULL ||
            index->indptr == NULL)
        {
            return EVOKE_ERR_INVALID;
        }
    }
    else if (num_term_extent_lists < (size_t) index->vocab_size)
    {
        return EVOKE_ERR_INVALID;
    }

    if (!evoke_checked_mul_size(index->num_docs, sizeof(*scores), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    scores = calloc(index->num_docs, sizeof(*scores));
    if (scores == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    if (query_len > 0)
    {
        if (!evoke_checked_mul_size(
                query_len,
                sizeof(*filtered_ids),
                &bytes))
        {
            free(scores);
            return EVOKE_ERR_RANGE;
        }
        filtered_ids = malloc(bytes);
        if (filtered_ids == NULL)
        {
            free(scores);
            return EVOKE_ERR_NOMEM;
        }

        for (i = 0; i < query_len; i++)
        {
            if (query_ids[i] < index->vocab_size)
            {
                filtered_ids[filtered_len++] = query_ids[i];
            }
        }
    }

    if (filtered_len == 0 && index->has_empty_token)
    {
        uint32_t *resized = realloc(
            filtered_ids,
            sizeof(*filtered_ids)
        );

        if (resized == NULL)
        {
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        filtered_ids = resized;
        filtered_ids[0] = index->empty_token_id;
        filtered_len = 1;
    }

    avg_doc_len =
        (double) stats->total_document_length /
        (double) stats->document_count;
    memset(&tfc_cache, 0, sizeof(tfc_cache));

    for (i = 0; i < filtered_len; i++)
    {
        uint32_t token_id = filtered_ids[i];
        uint32_t df = stats->doc_frequencies[token_id];
        double idf;
        double nonocc = 0.0;
        evoke_status status;

        if (df == 0)
        {
            continue;
        }

        idf = evoke_score_idf(
            index->params.idf_method,
            (double) df,
            (double) stats->document_count
        );
        if (evoke_method_requires_nonoccurrence(index->params.method))
        {
            nonocc = idf * evoke_score_tfc(
                index->params.method,
                0.0,
                0.0,
                avg_doc_len,
                index->params.k1,
                index->params.b,
                index->params.delta
            );
        }

        if (term_extents == NULL)
        {
            uint64_t start = index->indptr[token_id];
            uint64_t end = index->indptr[token_id + 1];
            evoke_posting_extent extent = {
                .data = NULL,
                .indices = &index->indices[start],
                .term_frequencies = &index->term_frequencies[start],
                .document_id_map = NULL,
                .len = end - start,
                .document_id_base = 0,
                .local_document_count = index->num_docs
            };

            status = evoke_accumulate_neutral_extent(
                index,
                &extent,
                idf,
                nonocc,
                avg_doc_len,
                1.0,
                &tfc_cache,
                scores
            );
        }
        else
        {
            const evoke_term_extent_list *list = &term_extents[token_id];
            size_t extent_index;

            if (list->len > 0 && list->extents == NULL)
            {
                free(filtered_ids);
                free(scores);
                return EVOKE_ERR_INVALID;
            }
            status = EVOKE_OK;
            for (extent_index = 0;
                 extent_index < list->len && status == EVOKE_OK;
                 extent_index++)
            {
                status = evoke_accumulate_neutral_extent(
                    index,
                    &list->extents[extent_index],
                    idf,
                    nonocc,
                    avg_doc_len,
                    1.0,
                    &tfc_cache,
                    scores
                );
            }
        }
        if (status != EVOKE_OK)
        {
            free(filtered_ids);
            free(scores);
            return status;
        }
    }

    evoke_apply_weight_mask_inplace(scores, weight_mask, index->num_docs);

    if (evoke_method_requires_nonoccurrence(index->params.method))
    {
        double nonoccurrence_sum = 0.0;

        for (i = 0; i < filtered_len; i++)
        {
            uint32_t token_id = filtered_ids[i];
            uint32_t df = stats->doc_frequencies[token_id];
            double idf;
            double nonocc;

            if (df == 0)
            {
                continue;
            }

            idf = evoke_score_idf(
                index->params.idf_method,
                (double) df,
                (double) stats->document_count
            );
            nonocc = idf * evoke_score_tfc(
                index->params.method,
                0.0,
                0.0,
                avg_doc_len,
                index->params.k1,
                index->params.b,
                index->params.delta
            );
            nonoccurrence_sum += nonocc;
        }

        evoke_add_constant_inplace(
            scores,
            (float) nonoccurrence_sum,
            index->num_docs
        );
    }

    free(filtered_ids);
    *scores_out = scores;
    return EVOKE_OK;
}

static evoke_status
evoke_accumulate_impact_extent(
    const evoke_index *index,
    const evoke_posting_extent *extent,
    float query_weight,
    float *scores
)
{
    uint64_t i;

    if (extent->len > 0 &&
        extent->data == NULL &&
        extent->values == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (!evoke_posting_extent_layout_is_valid(index, extent))
    {
        return EVOKE_ERR_INVALID;
    }

    if (extent->document_id_map == NULL && extent->data != NULL)
    {
        for (i = 0; i < extent->len; i++)
        {
            uint32_t document_id =
                extent->document_id_base + extent->indices[i];

            scores[document_id] += query_weight * extent->data[i];
        }
    }
    else if (extent->document_id_map == NULL)
    {
        for (i = 0; i < extent->len; i++)
        {
            uint32_t document_id =
                extent->document_id_base + extent->indices[i];

            scores[document_id] +=
                query_weight * extent->values[i].impact;
        }
    }
    else if (extent->data != NULL)
    {
        for (i = 0; i < extent->len; i++)
        {
            uint32_t document_id =
                extent->document_id_map[extent->indices[i]];

            scores[document_id] += query_weight * extent->data[i];
        }
    }
    else
    {
        for (i = 0; i < extent->len; i++)
        {
            uint32_t document_id =
                extent->document_id_map[extent->indices[i]];

            scores[document_id] +=
                query_weight * extent->values[i].impact;
        }
    }
    return EVOKE_OK;
}

static uint32_t
evoke_extent_retired_posting_count(
    const evoke_posting_extent *extent,
    const uint32_t *retired_document_ids,
    size_t retired_document_count
)
{
    uint64_t posting_index = 0;
    size_t retired_index = 0;
    uint32_t count = 0;

    while (posting_index < extent->len &&
           retired_index < retired_document_count)
    {
        uint32_t document_id;
        uint32_t retired_id = retired_document_ids[retired_index];

        if (extent->document_id_map == NULL)
        {
            document_id =
                extent->document_id_base + extent->indices[posting_index];
        }
        else
        {
            document_id =
                extent->document_id_map[extent->indices[posting_index]];
        }
        if (document_id < retired_id)
        {
            posting_index++;
        }
        else if (document_id > retired_id)
        {
            retired_index++;
        }
        else
        {
            count++;
            posting_index++;
            retired_index++;
        }
    }
    return count;
}

static evoke_status
evoke_retired_document_ids_validate(
    const evoke_index *index,
    const uint32_t *retired_document_ids,
    size_t retired_document_count
)
{
    size_t i;

    if (retired_document_count > 0 && retired_document_ids == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (i = 0; i < retired_document_count; i++)
    {
        if (retired_document_ids[i] >= index->num_docs ||
            (i > 0 &&
             retired_document_ids[i - 1] >= retired_document_ids[i]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_term_live_document_frequency_unchecked(
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    uint32_t term_id,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    uint32_t *live_document_frequency_out
)
{
    const evoke_term_extent_list *list;
    uint32_t live_document_frequency;
    uint32_t retired_document_frequency = 0;

    if (stats == NULL || term_extents == NULL ||
        live_document_frequency_out == NULL ||
        stats->doc_frequencies == NULL ||
        term_id >= stats->vocab_size)
    {
        return EVOKE_ERR_INVALID;
    }
    list = &term_extents[term_id];
    if (list->len > 0 && list->extents == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    live_document_frequency = stats->doc_frequencies[term_id];
    for (size_t extent_index = 0;
         extent_index < list->len;
         extent_index++)
    {
        const evoke_posting_extent *extent =
            &list->extents[extent_index];
        uint32_t retired_in_extent;

        if (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
            extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
        {
            continue;
        }
        retired_in_extent = evoke_extent_retired_posting_count(
            extent,
            retired_document_ids,
            retired_document_count
        );
        if (UINT32_MAX - retired_document_frequency <
            retired_in_extent)
        {
            return EVOKE_ERR_RANGE;
        }
        retired_document_frequency += retired_in_extent;
    }
    if (retired_document_frequency > live_document_frequency)
    {
        return EVOKE_ERR_FORMAT;
    }
    live_document_frequency -= retired_document_frequency;
    if ((uint64_t) live_document_frequency > stats->document_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *live_document_frequency_out = live_document_frequency;
    return EVOKE_OK;
}

evoke_status
evoke_term_live_document_frequency(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    uint32_t term_id,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    uint32_t *live_document_frequency_out
)
{
    evoke_status status;

    if (index == NULL || stats == NULL || term_extents == NULL ||
        live_document_frequency_out == NULL ||
        term_id >= index->vocab_size ||
        stats->vocab_size < index->vocab_size)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_retired_document_ids_validate(
        index,
        retired_document_ids,
        retired_document_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_term_live_document_frequency_unchecked(
        stats,
        term_extents,
        term_id,
        retired_document_ids,
        retired_document_count,
        live_document_frequency_out
    );
}

evoke_status
evoke_scores_from_weighted_ids_mixed_retired(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    float *scores = NULL;
    uint32_t *filtered_ids = NULL;
    float *filtered_weights = NULL;
    size_t filtered_len = 0;
    size_t bytes;
    size_t i;
    double avg_doc_len;
    double nonoccurrence_sum = 0.0;
    evoke_tfc_cache tfc_cache;
    evoke_status retired_status;

    if (index == NULL || stats == NULL ||
        (index->vocab_size > 0 && term_extents == NULL) ||
        scores_out == NULL || (query_len > 0 && query_ids == NULL) ||
        num_term_extent_lists < (size_t) index->vocab_size)
    {
        return EVOKE_ERR_INVALID;
    }
    *scores_out = NULL;
    if (index->num_docs == 0)
    {
        return EVOKE_OK;
    }
    if (stats->doc_frequencies == NULL ||
        stats->vocab_size < (size_t) index->vocab_size ||
        index->doc_lengths == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    retired_status = evoke_retired_document_ids_validate(
        index,
        retired_document_ids,
        retired_document_count
    );
    if (retired_status != EVOKE_OK)
    {
        return retired_status;
    }
    if (!evoke_checked_mul_size(index->num_docs, sizeof(*scores), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    scores = calloc(index->num_docs, sizeof(*scores));
    if (scores == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    if (stats->document_count == 0)
    {
        if (retired_document_count == 0)
        {
            free(scores);
            return EVOKE_ERR_INVALID;
        }
        *scores_out = scores;
        return EVOKE_OK;
    }

    if (query_len > 0)
    {
        if (!evoke_checked_mul_size(
                query_len,
                sizeof(*filtered_ids),
                &bytes))
        {
            free(scores);
            return EVOKE_ERR_RANGE;
        }
        filtered_ids = malloc(bytes);
        if (filtered_ids == NULL)
        {
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        if (!evoke_checked_mul_size(
                query_len,
                sizeof(*filtered_weights),
                &bytes))
        {
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_RANGE;
        }
        filtered_weights = malloc(bytes);
        if (filtered_weights == NULL)
        {
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        for (i = 0; i < query_len; i++)
        {
            float query_weight =
                query_weights == NULL ? 1.0f : query_weights[i];

            if (!isfinite(query_weight))
            {
                free(filtered_weights);
                free(filtered_ids);
                free(scores);
                return EVOKE_ERR_INVALID;
            }
            if (query_ids[i] < index->vocab_size)
            {
                filtered_ids[filtered_len] = query_ids[i];
                filtered_weights[filtered_len] = query_weight;
                filtered_len++;
            }
        }
    }
    if (filtered_len == 0 && index->has_empty_token)
    {
        uint32_t *resized = realloc(
            filtered_ids,
            sizeof(*filtered_ids)
        );
        float *resized_weights;

        if (resized == NULL)
        {
            free(filtered_weights);
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        filtered_ids = resized;
        resized_weights = realloc(
            filtered_weights,
            sizeof(*filtered_weights)
        );
        if (resized_weights == NULL)
        {
            free(filtered_weights);
            free(filtered_ids);
            free(scores);
            return EVOKE_ERR_NOMEM;
        }
        filtered_weights = resized_weights;
        filtered_ids[0] = index->empty_token_id;
        filtered_weights[0] = 1.0f;
        filtered_len = 1;
    }

    avg_doc_len =
        (double) stats->total_document_length /
        (double) stats->document_count;
    memset(&tfc_cache, 0, sizeof(tfc_cache));

    for (i = 0; i < filtered_len; i++)
    {
        uint32_t token_id = filtered_ids[i];
        float query_weight = filtered_weights[i];
        const evoke_term_extent_list *list = &term_extents[token_id];
        uint32_t df;
        double idf = 0.0;
        double nonocc = 0.0;
        bool lexical_seen = false;
        bool lexical_stats_ready = false;
        size_t extent_index;
        evoke_status live_df_status;

        if (list->len > 0 && list->extents == NULL)
        {
            free(filtered_ids);
            free(filtered_weights);
            free(scores);
            return EVOKE_ERR_INVALID;
        }
        live_df_status =
            evoke_term_live_document_frequency_unchecked(
                stats,
                term_extents,
                token_id,
                retired_document_ids,
                retired_document_count,
                &df
            );
        if (live_df_status != EVOKE_OK)
        {
            free(filtered_ids);
            free(filtered_weights);
            free(scores);
            return live_df_status;
        }
        for (extent_index = 0;
             extent_index < list->len;
             extent_index++)
        {
            const evoke_posting_extent *extent =
                &list->extents[extent_index];
            evoke_status status;

            if (extent->kind ==
                EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                if (!lexical_stats_ready)
                {
                    if (df == 0)
                    {
                        lexical_stats_ready = true;
                        continue;
                    }
                    idf = evoke_score_idf(
                        index->params.idf_method,
                        (double) df,
                        (double) stats->document_count
                    );
                    if (evoke_method_requires_nonoccurrence(
                            index->params.method))
                    {
                        nonocc = idf * evoke_score_tfc(
                            index->params.method,
                            0.0,
                            0.0,
                            avg_doc_len,
                            index->params.k1,
                            index->params.b,
                            index->params.delta
                        );
                    }
                    lexical_stats_ready = true;
                }
                status = evoke_accumulate_neutral_extent(
                    index,
                    extent,
                    idf,
                    nonocc,
                    avg_doc_len,
                    (double) query_weight,
                    &tfc_cache,
                    scores
                );
                lexical_seen = true;
            }
            else if (extent->kind ==
                     EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                status = evoke_accumulate_impact_extent(
                    index,
                    extent,
                    query_weight,
                    scores
                );
            }
            else if (extent->kind ==
                     EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                if (!lexical_stats_ready)
                {
                    if (df == 0)
                    {
                        free(filtered_ids);
                        free(filtered_weights);
                        free(scores);
                        return EVOKE_ERR_FORMAT;
                    }
                    idf = evoke_score_idf(
                        index->params.idf_method,
                        (double) df,
                        (double) stats->document_count
                    );
                    if (evoke_method_requires_nonoccurrence(
                            index->params.method))
                    {
                        nonocc = idf * evoke_score_tfc(
                            index->params.method,
                            0.0,
                            0.0,
                            avg_doc_len,
                            index->params.k1,
                            index->params.b,
                            index->params.delta
                        );
                    }
                    lexical_stats_ready = true;
                }
                status = evoke_accumulate_impact_extent(
                    index,
                    extent,
                    query_weight,
                    scores
                );
                lexical_seen = true;
            }
            else
            {
                free(filtered_ids);
                free(filtered_weights);
                free(scores);
                return EVOKE_ERR_FORMAT;
            }
            if (status != EVOKE_OK)
            {
                free(filtered_ids);
                free(filtered_weights);
                free(scores);
                return status;
            }
        }
        if (lexical_seen &&
            evoke_method_requires_nonoccurrence(index->params.method))
        {
            nonoccurrence_sum += (double) query_weight * nonocc;
        }
    }

    evoke_apply_weight_mask_inplace(scores, weight_mask, index->num_docs);
    if (nonoccurrence_sum != 0.0)
    {
        evoke_add_constant_inplace(
            scores,
            (float) nonoccurrence_sum,
            index->num_docs
        );
    }
    for (i = 0; i < retired_document_count; i++)
    {
        scores[retired_document_ids[i]] = 0.0f;
    }

    free(filtered_ids);
    free(filtered_weights);
    *scores_out = scores;
    return EVOKE_OK;
}

evoke_status
evoke_scores_from_ids_mixed_retired(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    return evoke_scores_from_weighted_ids_mixed_retired(
        index,
        stats,
        term_extents,
        num_term_extent_lists,
        retired_document_ids,
        retired_document_count,
        query_ids,
        NULL,
        query_len,
        weight_mask,
        scores_out
    );
}

evoke_status
evoke_scores_from_ids_mixed(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    return evoke_scores_from_ids_mixed_retired(
        index,
        stats,
        term_extents,
        num_term_extent_lists,
        NULL,
        0,
        query_ids,
        query_len,
        weight_mask,
        scores_out
    );
}

evoke_status
evoke_scores_from_ids_exact_stats(
    const evoke_index *index,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
)
{
    evoke_corpus_stats stats;
    uint64_t total_document_length = 0;
    uint32_t i;

    if (index == NULL || scores_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *scores_out = NULL;
    if (index->num_docs == 0)
    {
        return EVOKE_OK;
    }
    if (index->doc_lengths == NULL || index->doc_frequencies == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    for (i = 0; i < index->num_docs; i++)
    {
        if (UINT64_MAX - total_document_length < index->doc_lengths[i])
        {
            return EVOKE_ERR_RANGE;
        }
        total_document_length += index->doc_lengths[i];
    }

    stats.document_count = index->num_docs;
    stats.total_document_length = total_document_length;
    stats.doc_frequencies = index->doc_frequencies;
    stats.vocab_size = index->vocab_size;
    return evoke_scores_from_ids_neutral(
        index,
        &stats,
        NULL,
        0,
        query_ids,
        query_len,
        weight_mask,
        scores_out
    );
}

static bool
evoke_heap_item_is_worse(
    const evoke_heap_item *lhs,
    const evoke_heap_item *rhs
)
{
    if (lhs->score < rhs->score)
    {
        return true;
    }
    if (lhs->score > rhs->score)
    {
        return false;
    }
    if (lhs->tie_break_key > rhs->tie_break_key)
    {
        return true;
    }
    if (lhs->tie_break_key < rhs->tie_break_key)
    {
        return false;
    }
    return lhs->doc_id > rhs->doc_id;
}

static bool
evoke_heap_item_is_better(
    const evoke_heap_item *lhs,
    const evoke_heap_item *rhs
)
{
    if (lhs->score > rhs->score)
    {
        return true;
    }
    if (lhs->score < rhs->score)
    {
        return false;
    }
    if (lhs->tie_break_key < rhs->tie_break_key)
    {
        return true;
    }
    if (lhs->tie_break_key > rhs->tie_break_key)
    {
        return false;
    }
    return lhs->doc_id < rhs->doc_id;
}

static bool
evoke_compact_topk_item_is_better(
    const evoke_compact_topk_item *lhs,
    const evoke_compact_topk_item *rhs
)
{
    if (lhs->score > rhs->score)
    {
        return true;
    }
    if (lhs->score < rhs->score)
    {
        return false;
    }
    return lhs->doc_id < rhs->doc_id;
}

static int
evoke_cmp_compact_topk_desc(const void *lhs, const void *rhs)
{
    const evoke_compact_topk_item *a = lhs;
    const evoke_compact_topk_item *b = rhs;

    if (a->score < b->score)
    {
        return 1;
    }
    if (a->score > b->score)
    {
        return -1;
    }
    if (a->doc_id < b->doc_id)
    {
        return -1;
    }
    if (a->doc_id > b->doc_id)
    {
        return 1;
    }
    return 0;
}

static void
evoke_heap_item_init(
    evoke_heap_item *item,
    float score,
    uint32_t doc_id,
    const uint64_t *tie_break_keys
)
{
    item->score = score;
    item->doc_id = doc_id;
    item->tie_break_key = tie_break_keys == NULL
        ? (uint64_t) doc_id
        : tie_break_keys[doc_id];
}

static void
evoke_heap_sift_up(
    evoke_heap_item *heap,
    size_t idx
)
{
    while (idx > 0)
    {
        size_t parent = (idx - 1) / 2;

        if (!evoke_heap_item_is_worse(&heap[idx], &heap[parent]))
        {
            break;
        }

        {
            evoke_heap_item tmp = heap[parent];
            heap[parent] = heap[idx];
            heap[idx] = tmp;
        }
        idx = parent;
    }
}

static void
evoke_heap_sift_down(
    evoke_heap_item *heap,
    size_t len,
    size_t idx
)
{
    while (true)
    {
        size_t left = idx * 2 + 1;
        size_t right = left + 1;
        size_t smallest = idx;

        if (left < len &&
            evoke_heap_item_is_worse(&heap[left], &heap[smallest]))
        {
            smallest = left;
        }
        if (right < len &&
            evoke_heap_item_is_worse(&heap[right], &heap[smallest]))
        {
            smallest = right;
        }
        if (smallest == idx)
        {
            break;
        }

        {
            evoke_heap_item tmp = heap[idx];
            heap[idx] = heap[smallest];
            heap[smallest] = tmp;
        }
        idx = smallest;
    }
}

static int
evoke_cmp_topk_desc(const void *lhs, const void *rhs)
{
    const evoke_heap_item *a = lhs;
    const evoke_heap_item *b = rhs;

    if (a->score < b->score)
    {
        return 1;
    }
    if (a->score > b->score)
    {
        return -1;
    }
    if (a->tie_break_key < b->tie_break_key)
    {
        return -1;
    }
    if (a->tie_break_key > b->tie_break_key)
    {
        return 1;
    }
    if (a->doc_id < b->doc_id)
    {
        return -1;
    }
    if (a->doc_id > b->doc_id)
    {
        return 1;
    }
    return 0;
}

static bool
evoke_use_buffered_topk(
    size_t universe,
    size_t k,
    bool sorted
)
{
    static const size_t buffered_topk_max_k = 4096;

    return sorted && k > 0 && k < universe && k <= buffered_topk_max_k;
}

static evoke_status
evoke_topk_result_from_items(
    const evoke_heap_item *items,
    size_t len,
    evoke_topk_result *result_out
)
{
    size_t bytes;
    size_t i;

    if (result_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    memset(result_out, 0, sizeof(*result_out));
    if (len == 0)
    {
        return EVOKE_OK;
    }

    if (!evoke_checked_mul_size(
            len,
            sizeof(*result_out->doc_ids),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    result_out->doc_ids = malloc(bytes);
    if (result_out->doc_ids == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    if (!evoke_checked_mul_size(
            len,
            sizeof(*result_out->scores),
            &bytes))
    {
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_RANGE;
    }
    result_out->scores = malloc(bytes);
    if (result_out->scores == NULL)
    {
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < len; i++)
    {
        result_out->doc_ids[i] = items[i].doc_id;
        result_out->scores[i] = items[i].score;
    }
    result_out->len = len;
    return EVOKE_OK;
}

evoke_status
evoke_topk_accumulator_init(
    evoke_topk_accumulator *accumulator,
    size_t capacity
)
{
    size_t bytes;

    if (accumulator == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(accumulator, 0, sizeof(*accumulator));
    if (capacity == 0)
    {
        return EVOKE_OK;
    }
    if (!evoke_checked_mul_size(
            capacity,
            sizeof(*accumulator->heap),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    accumulator->heap = malloc(bytes);
    if (accumulator->heap == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    accumulator->capacity = capacity;
    return EVOKE_OK;
}

evoke_status
evoke_topk_accumulator_offer(
    evoke_topk_accumulator *accumulator,
    float score,
    uint32_t doc_id,
    uint64_t tie_break_key
)
{
    evoke_heap_item item;

    if (accumulator == NULL || accumulator->finalized ||
        (accumulator->capacity > 0 && accumulator->heap == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    if (accumulator->capacity == 0)
    {
        return EVOKE_OK;
    }
    item.score = score;
    item.doc_id = doc_id;
    item.tie_break_key = tie_break_key;
    if (accumulator->len < accumulator->capacity)
    {
        accumulator->heap[accumulator->len] = item;
        evoke_heap_sift_up(accumulator->heap, accumulator->len);
        accumulator->len++;
    }
    else if (evoke_heap_item_is_better(
            &item,
            &accumulator->heap[0]))
    {
        accumulator->heap[0] = item;
        evoke_heap_sift_down(
            accumulator->heap,
            accumulator->len,
            0
        );
    }
    return EVOKE_OK;
}

evoke_status
evoke_topk_accumulator_finish(
    evoke_topk_accumulator *accumulator,
    bool sorted,
    evoke_topk_result *result_out
)
{
    if (accumulator == NULL || result_out == NULL ||
        accumulator->finalized ||
        (accumulator->capacity > 0 && accumulator->heap == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    accumulator->finalized = true;
    if (sorted && accumulator->len > 1)
    {
        qsort(
            accumulator->heap,
            accumulator->len,
            sizeof(*accumulator->heap),
            evoke_cmp_topk_desc
        );
    }
    return evoke_topk_result_from_items(
        accumulator->heap,
        accumulator->len,
        result_out
    );
}

void
evoke_topk_accumulator_free(evoke_topk_accumulator *accumulator)
{
    if (accumulator == NULL)
    {
        return;
    }
    free(accumulator->heap);
    memset(accumulator, 0, sizeof(*accumulator));
}

static evoke_status
evoke_topk_buffered(
    const float *scores,
    size_t num_scores,
    size_t k,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
)
{
    evoke_heap_item *items = NULL;
    size_t buffer_cap = 0;
    size_t buffer_len = 0;
    size_t bytes;
    size_t i;

    if (scores == NULL || result_out == NULL ||
        k == 0 || k >= num_scores)
    {
        return EVOKE_ERR_INVALID;
    }

    if (!evoke_checked_mul_size(k, 2, &buffer_cap))
    {
        return EVOKE_ERR_RANGE;
    }
    if (buffer_cap > num_scores)
    {
        buffer_cap = num_scores;
    }
    if (buffer_cap == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (!evoke_checked_mul_size(
            buffer_cap,
            sizeof(*items),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    items = malloc(bytes);
    if (items == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < k; i++)
    {
        evoke_heap_item_init(
            &items[buffer_len],
            scores[i],
            (uint32_t) i,
            tie_break_keys
        );
        buffer_len++;
    }
    qsort(items, buffer_len, sizeof(*items), evoke_cmp_topk_desc);

    for (i = k; i < num_scores; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[i],
            (uint32_t) i,
            tie_break_keys
        );
        if (!evoke_heap_item_is_better(&item, &items[k - 1]))
        {
            continue;
        }

        items[buffer_len] = item;
        buffer_len++;
        if (buffer_len == buffer_cap)
        {
            qsort(
                items,
                buffer_len,
                sizeof(*items),
                evoke_cmp_topk_desc
            );
            buffer_len = k;
        }
    }

    if (buffer_len > k)
    {
        qsort(items, buffer_len, sizeof(*items), evoke_cmp_topk_desc);
        buffer_len = k;
    }

    {
        evoke_status status = evoke_topk_result_from_items(
            items,
            buffer_len,
            result_out
        );

        free(items);
        return status;
    }
}

static evoke_status
evoke_topk_subset_buffered(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool positive_only,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
)
{
    evoke_heap_item *items = NULL;
    size_t heap_cap;
    size_t buffer_cap = 0;
    size_t buffer_len = 0;
    size_t bytes;
    size_t i;

    heap_cap = k < num_candidate_doc_ids ? k : num_candidate_doc_ids;
    if (heap_cap == 0)
    {
        memset(result_out, 0, sizeof(*result_out));
        return EVOKE_OK;
    }

    if (!evoke_checked_mul_size(heap_cap, 2, &buffer_cap))
    {
        return EVOKE_ERR_RANGE;
    }
    if (buffer_cap > num_candidate_doc_ids)
    {
        buffer_cap = num_candidate_doc_ids;
    }
    if (buffer_cap < heap_cap)
    {
        return EVOKE_ERR_RANGE;
    }
    if (!evoke_checked_mul_size(
            buffer_cap,
            sizeof(*items),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    items = malloc(bytes);
    if (items == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_candidate_doc_ids; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[candidate_doc_ids[i]],
            candidate_doc_ids[i],
            tie_break_keys
        );
        if (positive_only && item.score <= 0.0f)
        {
            continue;
        }

        items[buffer_len] = item;
        buffer_len++;
        if (buffer_len == heap_cap)
        {
            break;
        }
    }

    if (buffer_len == 0)
    {
        free(items);
        memset(result_out, 0, sizeof(*result_out));
        return EVOKE_OK;
    }

    qsort(items, buffer_len, sizeof(*items), evoke_cmp_topk_desc);

    for (i++; i < num_candidate_doc_ids; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[candidate_doc_ids[i]],
            candidate_doc_ids[i],
            tie_break_keys
        );
        if (positive_only && item.score <= 0.0f)
        {
            continue;
        }
        if (!evoke_heap_item_is_better(
                &item,
                &items[heap_cap - 1]))
        {
            continue;
        }

        items[buffer_len] = item;
        buffer_len++;
        if (buffer_len == buffer_cap)
        {
            qsort(
                items,
                buffer_len,
                sizeof(*items),
                evoke_cmp_topk_desc
            );
            buffer_len = heap_cap;
        }
    }

    if (buffer_len > heap_cap)
    {
        qsort(items, buffer_len, sizeof(*items), evoke_cmp_topk_desc);
        buffer_len = heap_cap;
    }

    {
        evoke_status status = evoke_topk_result_from_items(
            items,
            buffer_len,
            result_out
        );

        free(items);
        return status;
    }
}

static evoke_status
evoke_rank_subset_sorted_all(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    bool positive_only,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
)
{
    evoke_heap_item *items = NULL;
    size_t bytes;
    size_t len = 0;
    size_t i;

    if (!evoke_checked_mul_size(
            num_candidate_doc_ids,
            sizeof(*items),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    items = malloc(bytes);
    if (items == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_candidate_doc_ids; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[candidate_doc_ids[i]],
            candidate_doc_ids[i],
            tie_break_keys
        );
        if (positive_only && item.score <= 0.0f)
        {
            continue;
        }
        items[len++] = item;
    }

    if (len > 1)
    {
        qsort(items, len, sizeof(*items), evoke_cmp_topk_desc);
    }

    {
        evoke_status status = evoke_topk_result_from_items(
            items,
            len,
            result_out
        );

        free(items);
        return status;
    }
}

evoke_status
evoke_topk_with_tie_breaks(
    const float *scores,
    size_t num_scores,
    size_t k,
    bool sorted,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
)
{
    evoke_heap_item *heap = NULL;
    size_t heap_len = 0;
    size_t i;
    size_t bytes;

    if ((num_scores > 0 && scores == NULL) || result_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    memset(result_out, 0, sizeof(*result_out));

    if (k > num_scores)
    {
        return EVOKE_ERR_RANGE;
    }
    if (k == 0)
    {
        return EVOKE_OK;
    }
    if (evoke_use_buffered_topk(num_scores, k, sorted))
    {
        return evoke_topk_buffered(
            scores,
            num_scores,
            k,
            tie_break_keys,
            result_out
        );
    }

    if (!evoke_checked_mul_size(k, sizeof(*heap), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    heap = malloc(bytes);
    if (heap == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_scores; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[i],
            (uint32_t) i,
            tie_break_keys
        );
        if (heap_len < k)
        {
            heap[heap_len] = item;
            evoke_heap_sift_up(heap, heap_len);
            heap_len++;
        }
        else if (evoke_heap_item_is_better(&item, &heap[0]))
        {
            heap[0] = item;
            evoke_heap_sift_down(heap, heap_len, 0);
        }
    }

    if (sorted)
    {
        qsort(heap, heap_len, sizeof(*heap), evoke_cmp_topk_desc);
    }

    if (!evoke_checked_mul_size(k, sizeof(*result_out->doc_ids), &bytes))
    {
        free(heap);
        return EVOKE_ERR_RANGE;
    }
    result_out->doc_ids = malloc(bytes);
    if (result_out->doc_ids == NULL)
    {
        free(heap);
        return EVOKE_ERR_NOMEM;
    }

    if (!evoke_checked_mul_size(k, sizeof(*result_out->scores), &bytes))
    {
        free(heap);
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_RANGE;
    }
    result_out->scores = malloc(bytes);
    if (result_out->scores == NULL)
    {
        free(heap);
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < heap_len; i++)
    {
        result_out->doc_ids[i] = heap[i].doc_id;
        result_out->scores[i] = heap[i].score;
    }
    result_out->len = heap_len;

    free(heap);
    return EVOKE_OK;
}

evoke_status
evoke_topk(
    const float *scores,
    size_t num_scores,
    size_t k,
    bool sorted,
    evoke_topk_result *result_out
)
{
    return evoke_topk_with_tie_breaks(
        scores,
        num_scores,
        k,
        sorted,
        NULL,
        result_out
    );
}

typedef struct evoke_blockmax_contribution
{
    const evoke_posting_extent *extent;
    const evoke_posting_block_record *record;
    size_t next;
    double idf;
    double nonoccurrence;
    float query_weight;
} evoke_blockmax_contribution;

typedef struct evoke_blockmax_order
{
    uint32_t block_id;
    double upper_bound;
} evoke_blockmax_order;

static int
evoke_cmp_blockmax_order(const void *lhs, const void *rhs)
{
    const evoke_blockmax_order *left = lhs;
    const evoke_blockmax_order *right = rhs;

    if (left->upper_bound < right->upper_bound)
    {
        return 1;
    }
    if (left->upper_bound > right->upper_bound)
    {
        return -1;
    }
    if (left->block_id < right->block_id)
    {
        return -1;
    }
    if (left->block_id > right->block_id)
    {
        return 1;
    }
    return 0;
}

static bool
evoke_blockmax_document_is_retired(
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    uint32_t document_id
)
{
    size_t low = 0;
    size_t high = retired_document_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2;
        uint32_t retired_id = retired_document_ids[middle];

        if (retired_id < document_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < retired_document_count &&
        retired_document_ids[low] == document_id;
}

static bool
evoke_blockmax_document_is_allowed(
    const uint8_t *allowed_document_bitmap,
    uint32_t document_id
)
{
    if (allowed_document_bitmap == NULL)
    {
        return true;
    }
    return (allowed_document_bitmap[document_id >> 3] &
        (uint8_t) (UINT8_C(1) << (document_id & 7))) != 0;
}

static bool
evoke_blockmax_block_has_allowed_document(
    const uint8_t *allowed_document_bitmap,
    uint32_t first_document,
    uint32_t document_count
)
{
    uint64_t end_document;
    uint32_t first_byte;
    uint32_t last_byte;
    uint8_t first_mask;
    uint8_t last_mask;

    if (allowed_document_bitmap == NULL)
    {
        return true;
    }
    if (document_count == 0)
    {
        return false;
    }
    end_document = (uint64_t) first_document + document_count;
    first_byte = first_document >> 3;
    last_byte = (uint32_t) ((end_document - 1) >> 3);
    first_mask = (uint8_t) (UINT8_MAX << (first_document & 7));
    last_mask = (uint8_t) (UINT8_MAX >> (7 - ((end_document - 1) & 7)));
    if (first_byte == last_byte)
    {
        return (allowed_document_bitmap[first_byte] &
            first_mask & last_mask) != 0;
    }
    if ((allowed_document_bitmap[first_byte] & first_mask) != 0 ||
        (allowed_document_bitmap[last_byte] & last_mask) != 0)
    {
        return true;
    }
    for (uint32_t byte_index = first_byte + 1;
         byte_index < last_byte;
         byte_index++)
    {
        if (allowed_document_bitmap[byte_index] != 0)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_blockmax_use_allowed_block_lookup(
    size_t record_count,
    size_t allowed_block_count
)
{
    size_t search_steps = 0;
    size_t remaining = record_count;

    if (allowed_block_count == 0 || record_count <= 1 ||
        allowed_block_count >= record_count)
    {
        return false;
    }
    while (remaining > 0)
    {
        search_steps++;
        remaining >>= 1;
    }
    return allowed_block_count <= record_count / search_steps;
}

static const evoke_posting_block_record *
evoke_blockmax_find_block_record(
    const evoke_posting_extent *extent,
    uint32_t block_id
)
{
    size_t low = 0;
    size_t high = extent->block_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2;
        uint32_t candidate = extent->blocks[middle].block_id;

        if (candidate < block_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= extent->block_count ||
        extent->blocks[low].block_id != block_id)
    {
        return NULL;
    }
    return &extent->blocks[low];
}

static evoke_status
evoke_blockmax_bound_from_record(
    const evoke_index *index,
    const evoke_posting_extent *extent,
    const evoke_posting_block_record *record,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    evoke_posting_block_bound *bound_out
)
{
    uint64_t posting_end;
    uint64_t block_first;
    uint64_t block_end;

    if (index == NULL || extent == NULL || record == NULL ||
        document_blocks == NULL || bound_out == NULL ||
        block_shift == 0 || block_shift >= 32 ||
        extent->block_shift != block_shift ||
        record->kind != extent->kind ||
        record->block_id >= document_block_count ||
        record->posting_count == 0 ||
        record->posting_offset > UINT64_MAX - record->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    posting_end = record->posting_offset + record->posting_count;
    block_first = (uint64_t) record->block_id << block_shift;
    block_end = block_first + (UINT64_C(1) << block_shift);
    if (posting_end > extent->len ||
        record->first_document_id > record->last_document_id ||
        record->last_document_id >= index->num_docs ||
        record->first_document_id < block_first ||
        record->last_document_id >= block_end)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(bound_out, 0, sizeof(*bound_out));
    bound_out->posting_offset = record->posting_offset;
    bound_out->posting_count = record->posting_count;
    bound_out->block_id = record->block_id;
    bound_out->first_document_id = record->first_document_id;
    bound_out->last_document_id = record->last_document_id;
    bound_out->min_term_frequency = record->min_term_frequency;
    bound_out->max_term_frequency = record->max_term_frequency;
    bound_out->min_impact = record->min_impact;
    bound_out->max_impact = record->max_impact;
    bound_out->kind = record->kind;
    if (record->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
    {
        const evoke_document_block_extrema *document_block =
            &document_blocks[record->block_id];

        if (document_block->document_count == 0 ||
            document_block->min_document_length >
                document_block->max_document_length)
        {
            return EVOKE_ERR_FORMAT;
        }
        bound_out->min_document_length =
            document_block->min_document_length;
        bound_out->max_document_length =
            document_block->max_document_length;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_blockmax_accumulate_contribution(
    const evoke_index *index,
    const evoke_blockmax_contribution *contribution,
    uint32_t block_first,
    uint32_t block_document_count,
    double average_document_length,
    evoke_tfc_cache *tfc_cache,
    const uint8_t *allowed_document_bitmap,
    float *block_scores,
    uint64_t *postings_scored
)
{
    const evoke_posting_extent *extent;
    const evoke_posting_block_record *record;
    uint64_t posting_end;

    if (index == NULL || contribution == NULL ||
        tfc_cache == NULL || block_scores == NULL ||
        postings_scored == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    extent = contribution->extent;
    record = contribution->record;
    if (extent == NULL || record == NULL ||
        record->posting_offset > UINT64_MAX - record->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    posting_end = record->posting_offset + record->posting_count;
    if (posting_end > extent->len)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (uint64_t posting_index = record->posting_offset;
         posting_index < posting_end;
         posting_index++)
    {
        uint32_t document_id = evoke_posting_extent_document_id(
            extent,
            posting_index
        );
        uint32_t local_document_id;

        if (document_id < block_first ||
            document_id - block_first >= block_document_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        local_document_id = document_id - block_first;
        if (!evoke_blockmax_document_is_allowed(
                allowed_document_bitmap,
                document_id))
        {
            continue;
        }
        if (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            uint32_t term_frequency =
                extent->term_frequencies == NULL
                    ? extent->values[posting_index].term_frequency
                    : extent->term_frequencies[posting_index];
            double tfc = evoke_score_tfc_cached(
                index,
                tfc_cache,
                term_frequency,
                index->doc_lengths[document_id],
                average_document_length
            );
            double score = contribution->idf * tfc;

            if (evoke_method_requires_nonoccurrence(
                    index->params.method))
            {
                score -= contribution->nonoccurrence;
            }
            block_scores[local_document_id] += (float) (
                (double) contribution->query_weight * score
            );
        }
        else if (
            extent->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
            extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
        {
            float impact = extent->data == NULL
                ? extent->values[posting_index].impact
                : extent->data[posting_index];

            block_scores[local_document_id] +=
                contribution->query_weight * impact;
        }
        else
        {
            return EVOKE_ERR_FORMAT;
        }
        if (*postings_scored == UINT64_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        (*postings_scored)++;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax_internal(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint64_t *tie_break_keys,
    const uint32_t *tie_break_order,
    size_t tie_break_order_count,
    const uint8_t *allowed_document_bitmap,
    size_t allowed_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
)
{
    uint32_t *filtered_ids = NULL;
    float *filtered_weights = NULL;
    size_t filtered_len = 0;
    size_t contribution_capacity = 0;
    size_t contribution_count = 0;
    evoke_blockmax_contribution *contributions = NULL;
    uint32_t *allowed_block_ids = NULL;
    size_t allowed_block_count = 0;
    size_t *block_heads = NULL;
    size_t *block_tails = NULL;
    double *block_upper_bounds = NULL;
    evoke_blockmax_order *block_order = NULL;
    size_t block_order_count;
    evoke_heap_item *heap = NULL;
    size_t heap_len = 0;
    float *block_scores = NULL;
    uint64_t block_size;
    uint64_t expected_block_count;
    double average_document_length;
    double nonoccurrence_sum = 0.0;
    float base_score = 0.0f;
    evoke_tfc_cache tfc_cache;
    evoke_blockmax_stats blockmax_stats = {0};
    size_t bytes;
    evoke_status status = EVOKE_OK;

    if (result_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    if (blockmax_stats_out != NULL)
    {
        memset(blockmax_stats_out, 0, sizeof(*blockmax_stats_out));
    }
    if (index == NULL || stats == NULL ||
        (index->vocab_size > 0 && term_extents == NULL) ||
        num_term_extent_lists < (size_t) index->vocab_size ||
        (query_len > 0 && query_ids == NULL) ||
        index->doc_lengths == NULL ||
        stats->doc_frequencies == NULL ||
        stats->vocab_size < (size_t) index->vocab_size ||
        stats->document_count == 0 ||
        stats->total_document_length == 0 ||
        block_shift == 0 || block_shift >= 32 ||
        k > index->num_docs ||
        (allowed_document_bitmap != NULL &&
         (allowed_document_count > index->num_docs ||
          k > allowed_document_count)) ||
        (tie_break_order == NULL && tie_break_order_count != 0) ||
        tie_break_order_count > index->num_docs)
    {
        return EVOKE_ERR_INVALID;
    }
    if (index->num_docs == 0 || k == 0)
    {
        return EVOKE_OK;
    }
    status = evoke_retired_document_ids_validate(
        index,
        retired_document_ids,
        retired_document_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    block_size = UINT64_C(1) << block_shift;
    expected_block_count =
        ((uint64_t) index->num_docs + block_size - 1) >> block_shift;
    if (document_blocks == NULL ||
        expected_block_count != document_block_count)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t block_index = 0;
         block_index < document_block_count;
         block_index++)
    {
        uint64_t first_document = (uint64_t) block_index << block_shift;
        uint64_t block_documents = index->num_docs - first_document;
        const evoke_document_block_extrema *block =
            &document_blocks[block_index];

        if (block_documents > block_size)
        {
            block_documents = block_size;
        }
        if (block->document_count > block_documents ||
            (block->document_count == 0 &&
             (block->min_document_length != 0 ||
              block->max_document_length != 0)) ||
            (block->document_count > 0 &&
             block->min_document_length > block->max_document_length))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (allowed_document_bitmap != NULL)
    {
        if (!evoke_checked_mul_size(
                document_block_count,
                sizeof(*allowed_block_ids),
                &bytes))
        {
            return EVOKE_ERR_RANGE;
        }
        allowed_block_ids = malloc(bytes);
        if (allowed_block_ids == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        for (size_t block_index = 0;
             block_index < document_block_count;
             block_index++)
        {
            uint64_t first_document =
                (uint64_t) block_index << block_shift;
            uint64_t block_documents =
                (uint64_t) index->num_docs - first_document;

            if (block_documents > block_size)
            {
                block_documents = block_size;
            }
            if (evoke_blockmax_block_has_allowed_document(
                    allowed_document_bitmap,
                    (uint32_t) first_document,
                    (uint32_t) block_documents))
            {
                allowed_block_ids[allowed_block_count++] =
                    (uint32_t) block_index;
            }
        }
        if (allowed_document_count > 0 && allowed_block_count == 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    block_order_count = allowed_document_bitmap == NULL
        ? document_block_count
        : allowed_block_count;

    if (query_len > 0)
    {
        if (!evoke_checked_mul_size(
                query_len,
                sizeof(*filtered_ids),
                &bytes))
        {
            return EVOKE_ERR_RANGE;
        }
        filtered_ids = malloc(bytes);
        if (filtered_ids == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        if (!evoke_checked_mul_size(
                query_len,
                sizeof(*filtered_weights),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        filtered_weights = malloc(bytes);
        if (filtered_weights == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        for (size_t query_index = 0;
             query_index < query_len;
             query_index++)
        {
            float query_weight =
                query_weights == NULL ? 1.0f : query_weights[query_index];

            if (!isfinite(query_weight))
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }
            if (query_ids[query_index] < index->vocab_size)
            {
                filtered_ids[filtered_len] = query_ids[query_index];
                filtered_weights[filtered_len] = query_weight;
                filtered_len++;
            }
        }
    }
    if (filtered_len == 0 && index->has_empty_token)
    {
        uint32_t *resized_ids = realloc(
            filtered_ids,
            sizeof(*filtered_ids)
        );
        float *resized_weights;

        if (resized_ids == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        filtered_ids = resized_ids;
        resized_weights = realloc(
            filtered_weights,
            sizeof(*filtered_weights)
        );
        if (resized_weights == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        filtered_weights = resized_weights;
        filtered_ids[0] = index->empty_token_id;
        filtered_weights[0] = 1.0f;
        filtered_len = 1;
    }
    if (filtered_len == 0)
    {
        evoke_heap_item *items;
        size_t item_count = 0;

        if (!evoke_checked_mul_size(k, sizeof(*items), &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        items = malloc(bytes);
        if (items == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        if (tie_break_order != NULL)
        {
            for (size_t result_index = 0;
                 result_index < tie_break_order_count && item_count < k;
                 result_index++)
            {
                uint32_t document_id = tie_break_order[result_index];

                if (document_id >= index->num_docs)
                {
                    free(items);
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                if (!evoke_blockmax_document_is_allowed(
                        allowed_document_bitmap,
                        document_id))
                {
                    continue;
                }
                evoke_heap_item_init(
                    &items[item_count++],
                    0.0f,
                    document_id,
                    tie_break_keys
                );
            }
        }
        else if (tie_break_keys == NULL)
        {
            for (size_t result_index = 0;
                 result_index < index->num_docs && item_count < k;
                 result_index++)
            {
                if (!evoke_blockmax_document_is_allowed(
                        allowed_document_bitmap,
                        (uint32_t) result_index))
                {
                    continue;
                }
                evoke_heap_item_init(
                    &items[item_count++],
                    0.0f,
                    (uint32_t) result_index,
                    NULL
                );
            }
        }
        else
        {
            for (uint32_t document_id = 0;
                 document_id < index->num_docs;
                 document_id++)
            {
                evoke_heap_item item;

                if (!evoke_blockmax_document_is_allowed(
                        allowed_document_bitmap,
                        document_id))
                {
                    continue;
                }
                evoke_heap_item_init(
                    &item,
                    0.0f,
                    document_id,
                    tie_break_keys
                );
                if (item_count < k)
                {
                    items[item_count] = item;
                    evoke_heap_sift_up(items, item_count);
                    item_count++;
                }
                else if (evoke_heap_item_is_better(&item, &items[0]))
                {
                    items[0] = item;
                    evoke_heap_sift_down(items, item_count, 0);
                }
            }
            qsort(items, item_count, sizeof(*items), evoke_cmp_topk_desc);
        }
        if (item_count != k)
        {
            free(items);
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        blockmax_stats.zero_score_documents_considered =
            tie_break_order != NULL || tie_break_keys == NULL
                ? item_count
                : index->num_docs;
        status = evoke_topk_result_from_items(
            items,
            item_count,
            result_out
        );
        free(items);
        goto cleanup;
    }

    for (size_t query_index = 0;
         query_index < filtered_len;
         query_index++)
    {
        const evoke_term_extent_list *list =
            &term_extents[filtered_ids[query_index]];

        if (list->len > 0 && list->extents == NULL)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        for (size_t extent_index = 0;
             extent_index < list->len;
             extent_index++)
        {
            const evoke_posting_extent *extent =
                &list->extents[extent_index];
            size_t extent_capacity;

            status = evoke_posting_extent_validate_layout(index, extent);
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (extent->len > 0 &&
                (extent->blocks == NULL ||
                 extent->block_count == 0 ||
                 extent->block_shift != block_shift))
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }
            extent_capacity =
                evoke_blockmax_use_allowed_block_lookup(
                    extent->block_count,
                    allowed_block_count)
                    ? allowed_block_count
                    : extent->block_count;

            if (SIZE_MAX - contribution_capacity < extent_capacity)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            contribution_capacity += extent_capacity;
        }
    }
    if (contribution_capacity > 0)
    {
        if (!evoke_checked_mul_size(
                contribution_capacity,
                sizeof(*contributions),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        contributions = malloc(bytes);
        if (contributions == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }
    if (!evoke_checked_mul_size(
            document_block_count,
            sizeof(*block_heads),
            &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    block_heads = malloc(bytes);
    block_tails = malloc(bytes);
    if (block_heads == NULL || block_tails == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    if (!evoke_checked_mul_size(
            document_block_count,
            sizeof(*block_upper_bounds),
            &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    block_upper_bounds = calloc(
        document_block_count,
        sizeof(*block_upper_bounds)
    );
    if (block_upper_bounds == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t block_index = 0;
         block_index < document_block_count;
         block_index++)
    {
        block_heads[block_index] = SIZE_MAX;
        block_tails[block_index] = SIZE_MAX;
    }

    average_document_length =
        (double) stats->total_document_length /
        (double) stats->document_count;
    for (size_t query_index = 0;
         query_index < filtered_len;
         query_index++)
    {
        uint32_t token_id = filtered_ids[query_index];
        float query_weight = filtered_weights[query_index];
        const evoke_term_extent_list *list = &term_extents[token_id];
        uint32_t live_document_frequency;
        double idf = 0.0;
        double nonoccurrence = 0.0;
        bool lexical_seen = false;

        status = evoke_term_live_document_frequency_unchecked(
            stats,
            term_extents,
            token_id,
            retired_document_ids,
            retired_document_count,
            &live_document_frequency
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (live_document_frequency > 0)
        {
            idf = evoke_score_idf(
                index->params.idf_method,
                (double) live_document_frequency,
                (double) stats->document_count
            );
            if (evoke_method_requires_nonoccurrence(
                    index->params.method))
            {
                nonoccurrence = idf * evoke_score_tfc(
                    index->params.method,
                    0.0,
                    0.0,
                    average_document_length,
                    index->params.k1,
                    index->params.b,
                    index->params.delta
                );
            }
        }

        for (size_t extent_index = 0;
             extent_index < list->len;
             extent_index++)
        {
            const evoke_posting_extent *extent =
                &list->extents[extent_index];
            bool use_allowed_lookup;
            size_t candidate_record_count;

            if (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                if (live_document_frequency == 0)
                {
                    continue;
                }
                lexical_seen = true;
            }
            else if (extent->kind ==
                     EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                if (live_document_frequency == 0)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                lexical_seen = true;
            }
            else if (extent->kind !=
                     EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }

            use_allowed_lookup =
                evoke_blockmax_use_allowed_block_lookup(
                    extent->block_count,
                    allowed_block_count);
            candidate_record_count = use_allowed_lookup
                ? allowed_block_count
                : extent->block_count;

            for (size_t candidate_index = 0;
                 candidate_index < candidate_record_count;
                 candidate_index++)
            {
                const evoke_posting_block_record *record =
                    use_allowed_lookup
                        ? evoke_blockmax_find_block_record(
                            extent,
                            allowed_block_ids[candidate_index])
                        : &extent->blocks[candidate_index];
                evoke_posting_block_bound bound;
                float upper_bound = 0.0f;
                evoke_blockmax_contribution *contribution;
                size_t block_id;
                uint64_t block_first;
                uint64_t block_documents;

                if (record == NULL)
                {
                    continue;
                }
                block_id = record->block_id;
                block_first = (uint64_t) block_id << block_shift;
                block_documents =
                    (uint64_t) index->num_docs - block_first;

                if (block_documents > (UINT64_C(1) << block_shift))
                {
                    block_documents = UINT64_C(1) << block_shift;
                }

                status = evoke_blockmax_bound_from_record(
                    index,
                    extent,
                    record,
                    document_blocks,
                    document_block_count,
                    block_shift,
                    &bound
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if (!evoke_blockmax_block_has_allowed_document(
                        allowed_document_bitmap,
                        (uint32_t) block_first,
                        (uint32_t) block_documents))
                {
                    continue;
                }
                status = evoke_posting_block_score_upper_bound(
                    index,
                    stats,
                    live_document_frequency,
                    &bound,
                    query_weight,
                    &upper_bound
                );
                if (status != EVOKE_OK ||
                    contribution_count >= contribution_capacity)
                {
                    status = status == EVOKE_OK
                        ? EVOKE_ERR_FORMAT
                        : status;
                    goto cleanup;
                }
                contribution = &contributions[contribution_count];
                contribution->extent = extent;
                contribution->record = record;
                contribution->next = SIZE_MAX;
                contribution->idf = idf;
                contribution->nonoccurrence = nonoccurrence;
                contribution->query_weight = query_weight;
                if (block_heads[block_id] == SIZE_MAX)
                {
                    block_heads[block_id] = contribution_count;
                }
                else
                {
                    contributions[
                        block_tails[block_id]
                    ].next = contribution_count;
                }
                block_tails[block_id] = contribution_count;
                contribution_count++;
                block_upper_bounds[block_id] = nextafter(
                    block_upper_bounds[block_id] +
                        (double) upper_bound,
                    INFINITY
                );
            }
        }
        if (lexical_seen &&
            evoke_method_requires_nonoccurrence(index->params.method))
        {
            nonoccurrence_sum +=
                (double) query_weight * nonoccurrence;
        }
    }
    base_score = (float) nonoccurrence_sum;
    if (!isfinite(base_score))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }

    if (!evoke_checked_mul_size(
            block_order_count,
            sizeof(*block_order),
            &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    block_order = malloc(bytes);
    if (block_order == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t order_index = 0;
         order_index < block_order_count;
         order_index++)
    {
        size_t block_index = allowed_document_bitmap == NULL
            ? order_index
            : allowed_block_ids[order_index];
        double upper_bound = nextafter(
            block_upper_bounds[block_index] + (double) base_score,
            INFINITY
        );

        block_order[order_index].block_id = (uint32_t) block_index;
        block_order[order_index].upper_bound =
            fmax(0.0, upper_bound);
    }
    qsort(
        block_order,
        block_order_count,
        sizeof(*block_order),
        evoke_cmp_blockmax_order
    );

    if (!evoke_checked_mul_size(k, sizeof(*heap), &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    heap = malloc(bytes);
    if (heap == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    {
        uint64_t score_capacity =
            index->num_docs < block_size
                ? index->num_docs
                : block_size;

        if (score_capacity > SIZE_MAX ||
            !evoke_checked_mul_size(
                (size_t) score_capacity,
                sizeof(*block_scores),
                &bytes))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        block_scores = malloc(bytes);
        if (block_scores == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }
    memset(&tfc_cache, 0, sizeof(tfc_cache));

    for (size_t order_index = 0;
         order_index < block_order_count;
         order_index++)
    {
        uint32_t block_id = block_order[order_index].block_id;
        uint64_t first_document_u64 =
            (uint64_t) block_id << block_shift;
        uint64_t block_documents_u64 =
            (uint64_t) index->num_docs - first_document_u64;
        uint32_t first_document = (uint32_t) first_document_u64;
        uint32_t block_documents;
        size_t contribution_index;

        if (block_documents_u64 > block_size)
        {
            block_documents_u64 = block_size;
        }
        block_documents = (uint32_t) block_documents_u64;
        blockmax_stats.blocks_considered++;
        if (!evoke_blockmax_block_has_allowed_document(
                allowed_document_bitmap,
                first_document,
                block_documents))
        {
            blockmax_stats.blocks_skipped++;
            continue;
        }
        if (heap_len == k &&
            block_order[order_index].upper_bound <
                (double) heap[0].score)
        {
            blockmax_stats.blocks_skipped++;
            continue;
        }
        memset(
            block_scores,
            0,
            (size_t) block_documents * sizeof(*block_scores)
        );
        contribution_index = block_heads[block_id];
        while (contribution_index != SIZE_MAX)
        {
            status = evoke_blockmax_accumulate_contribution(
                index,
                &contributions[contribution_index],
                first_document,
                block_documents,
                average_document_length,
                &tfc_cache,
                allowed_document_bitmap,
                block_scores,
                &blockmax_stats.postings_scored
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            contribution_index =
                contributions[contribution_index].next;
        }

        for (uint32_t local_document = 0;
             local_document < block_documents;
             local_document++)
        {
            evoke_heap_item item;
            uint32_t document_id = first_document + local_document;

            if (!evoke_blockmax_document_is_allowed(
                    allowed_document_bitmap,
                    document_id))
            {
                continue;
            }

            evoke_heap_item_init(
                &item,
                evoke_blockmax_document_is_retired(
                    retired_document_ids,
                    retired_document_count,
                    document_id
                )
                    ? 0.0f
                    : block_scores[local_document] + base_score,
                document_id,
                tie_break_keys
            );
            if (heap_len < k)
            {
                heap[heap_len] = item;
                evoke_heap_sift_up(heap, heap_len);
                heap_len++;
            }
            else if (evoke_heap_item_is_better(&item, &heap[0]))
            {
                heap[0] = item;
                evoke_heap_sift_down(heap, heap_len, 0);
            }
        }
        blockmax_stats.blocks_scored++;
    }
    if (heap_len != k)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    qsort(heap, heap_len, sizeof(*heap), evoke_cmp_topk_desc);
    status = evoke_topk_result_from_items(heap, heap_len, result_out);

cleanup:
    free(block_scores);
    free(heap);
    free(block_order);
    free(block_upper_bounds);
    free(block_tails);
    free(block_heads);
    free(allowed_block_ids);
    free(contributions);
    free(filtered_weights);
    free(filtered_ids);
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    if (blockmax_stats_out != NULL)
    {
        *blockmax_stats_out = blockmax_stats;
    }
    return status;
}

evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint64_t *tie_break_keys,
    const uint32_t *tie_break_order,
    size_t tie_break_order_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
)
{
    return evoke_topk_from_weighted_ids_mixed_retired_blockmax_internal(
        index,
        stats,
        term_extents,
        num_term_extent_lists,
        document_blocks,
        document_block_count,
        block_shift,
        retired_document_ids,
        retired_document_count,
        tie_break_keys,
        tie_break_order,
        tie_break_order_count,
        NULL,
        index == NULL ? 0 : index->num_docs,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        blockmax_stats_out
    );
}

evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax_filtered_with_tie_breaks(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint64_t *tie_break_keys,
    const uint32_t *tie_break_order,
    size_t tie_break_order_count,
    const uint8_t *allowed_document_bitmap,
    size_t allowed_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
)
{
    if (allowed_document_bitmap == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_topk_from_weighted_ids_mixed_retired_blockmax_internal(
        index,
        stats,
        term_extents,
        num_term_extent_lists,
        document_blocks,
        document_block_count,
        block_shift,
        retired_document_ids,
        retired_document_count,
        tie_break_keys,
        tie_break_order,
        tie_break_order_count,
        allowed_document_bitmap,
        allowed_document_count,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        blockmax_stats_out
    );
}

evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
)
{
    return
        evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
            index,
            stats,
            term_extents,
            num_term_extent_lists,
            document_blocks,
            document_block_count,
            block_shift,
            retired_document_ids,
            retired_document_count,
            NULL,
            NULL,
            0,
            query_ids,
            query_weights,
            query_len,
            k,
            result_out,
            blockmax_stats_out
        );
}

static evoke_status
evoke_topk_subset_compact(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool positive_only,
    evoke_topk_result *result_out
)
{
    evoke_compact_topk_item *items = NULL;
    size_t result_capacity;
    size_t buffer_capacity;
    size_t buffer_len = 0;
    size_t bytes;
    size_t i;

    result_capacity = k < num_candidate_doc_ids
        ? k
        : num_candidate_doc_ids;
    if (result_capacity == 0)
    {
        memset(result_out, 0, sizeof(*result_out));
        return EVOKE_OK;
    }
    if (!evoke_checked_mul_size(
            result_capacity,
            2,
            &buffer_capacity))
    {
        return EVOKE_ERR_RANGE;
    }
    if (buffer_capacity > num_candidate_doc_ids)
    {
        buffer_capacity = num_candidate_doc_ids;
    }
    if (!evoke_checked_mul_size(
            buffer_capacity,
            sizeof(*items),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    items = malloc(bytes);
    if (items == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_candidate_doc_ids; i++)
    {
        evoke_compact_topk_item item;

        item.doc_id = candidate_doc_ids[i];
        item.score = scores[item.doc_id];
        if (positive_only && item.score <= 0.0f)
        {
            continue;
        }
        items[buffer_len++] = item;
        if (buffer_len == result_capacity)
        {
            break;
        }
    }
    if (buffer_len == 0)
    {
        free(items);
        memset(result_out, 0, sizeof(*result_out));
        return EVOKE_OK;
    }
    qsort(
        items,
        buffer_len,
        sizeof(*items),
        evoke_cmp_compact_topk_desc
    );

    for (i++; i < num_candidate_doc_ids; i++)
    {
        evoke_compact_topk_item item;

        item.doc_id = candidate_doc_ids[i];
        item.score = scores[item.doc_id];
        if ((positive_only && item.score <= 0.0f) ||
            !evoke_compact_topk_item_is_better(
                &item,
                &items[result_capacity - 1]))
        {
            continue;
        }
        items[buffer_len++] = item;
        if (buffer_len == buffer_capacity)
        {
            qsort(
                items,
                buffer_len,
                sizeof(*items),
                evoke_cmp_compact_topk_desc
            );
            buffer_len = result_capacity;
        }
    }
    if (buffer_len > result_capacity)
    {
        qsort(
            items,
            buffer_len,
            sizeof(*items),
            evoke_cmp_compact_topk_desc
        );
        buffer_len = result_capacity;
    }

    memset(result_out, 0, sizeof(*result_out));
    if (!evoke_checked_mul_size(
            buffer_len,
            sizeof(*result_out->doc_ids),
            &bytes))
    {
        free(items);
        return EVOKE_ERR_RANGE;
    }
    result_out->doc_ids = malloc(bytes);
    if (result_out->doc_ids == NULL)
    {
        free(items);
        return EVOKE_ERR_NOMEM;
    }
    if (!evoke_checked_mul_size(
            buffer_len,
            sizeof(*result_out->scores),
            &bytes))
    {
        free(items);
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_RANGE;
    }
    result_out->scores = malloc(bytes);
    if (result_out->scores == NULL)
    {
        free(items);
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_NOMEM;
    }
    for (i = 0; i < buffer_len; i++)
    {
        result_out->doc_ids[i] = items[i].doc_id;
        result_out->scores[i] = items[i].score;
    }
    result_out->len = buffer_len;
    free(items);
    return EVOKE_OK;
}

evoke_status
evoke_topk_subset_with_tie_breaks(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool sorted,
    bool positive_only,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
)
{
    evoke_heap_item *heap = NULL;
    size_t heap_len = 0;
    size_t heap_cap;
    size_t i;
    size_t bytes;

    if (scores == NULL || result_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (num_candidate_doc_ids > 0 && candidate_doc_ids == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    memset(result_out, 0, sizeof(*result_out));
    if (k == 0 || num_candidate_doc_ids == 0)
    {
        return EVOKE_OK;
    }
    if (sorted && tie_break_keys == NULL)
    {
        return evoke_topk_subset_compact(
            scores,
            candidate_doc_ids,
            num_candidate_doc_ids,
            k,
            positive_only,
            result_out
        );
    }
    if (sorted && k == num_candidate_doc_ids)
    {
        return evoke_rank_subset_sorted_all(
            scores,
            candidate_doc_ids,
            num_candidate_doc_ids,
            positive_only,
            tie_break_keys,
            result_out
        );
    }
    if (evoke_use_buffered_topk(
            num_candidate_doc_ids,
            k < num_candidate_doc_ids ? k : num_candidate_doc_ids,
            sorted))
    {
        return evoke_topk_subset_buffered(
            scores,
            candidate_doc_ids,
            num_candidate_doc_ids,
            k,
            positive_only,
            tie_break_keys,
            result_out
        );
    }

    heap_cap = k < num_candidate_doc_ids ? k : num_candidate_doc_ids;
    if (!evoke_checked_mul_size(heap_cap, sizeof(*heap), &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    heap = malloc(bytes);
    if (heap == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (i = 0; i < num_candidate_doc_ids; i++)
    {
        evoke_heap_item item;

        evoke_heap_item_init(
            &item,
            scores[candidate_doc_ids[i]],
            candidate_doc_ids[i],
            tie_break_keys
        );
        if (positive_only && item.score <= 0.0f)
        {
            continue;
        }

        if (heap_len < heap_cap)
        {
            heap[heap_len] = item;
            evoke_heap_sift_up(heap, heap_len);
            heap_len++;
        }
        else if (evoke_heap_item_is_better(&item, &heap[0]))
        {
            heap[0] = item;
            evoke_heap_sift_down(heap, heap_len, 0);
        }
    }

    if (sorted)
    {
        qsort(heap, heap_len, sizeof(*heap), evoke_cmp_topk_desc);
    }

    if (!evoke_checked_mul_size(heap_len, sizeof(*result_out->doc_ids), &bytes))
    {
        free(heap);
        return EVOKE_ERR_RANGE;
    }
    if (heap_len > 0)
    {
        result_out->doc_ids = malloc(bytes);
        if (result_out->doc_ids == NULL)
        {
            free(heap);
            return EVOKE_ERR_NOMEM;
        }
    }

    if (!evoke_checked_mul_size(heap_len, sizeof(*result_out->scores), &bytes))
    {
        free(heap);
        free(result_out->doc_ids);
        result_out->doc_ids = NULL;
        return EVOKE_ERR_RANGE;
    }
    if (heap_len > 0)
    {
        result_out->scores = malloc(bytes);
        if (result_out->scores == NULL)
        {
            free(heap);
            free(result_out->doc_ids);
            result_out->doc_ids = NULL;
            return EVOKE_ERR_NOMEM;
        }
    }

    for (i = 0; i < heap_len; i++)
    {
        result_out->doc_ids[i] = heap[i].doc_id;
        result_out->scores[i] = heap[i].score;
    }
    result_out->len = heap_len;

    free(heap);
    return EVOKE_OK;
}

evoke_status
evoke_topk_subset(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool sorted,
    bool positive_only,
    evoke_topk_result *result_out
)
{
    return evoke_topk_subset_with_tie_breaks(
        scores,
        candidate_doc_ids,
        num_candidate_doc_ids,
        k,
        sorted,
        positive_only,
        NULL,
        result_out
    );
}
