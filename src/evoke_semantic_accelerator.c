#include "evoke_semantic_accelerator.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SEMANTIC_ACCELERATOR_MAGIC UINT32_C(0x31414353)
#define EVOKE_SEMANTIC_ACCELERATOR_CHECKSUM_OFFSET 72U

typedef struct evoke_semantic_accelerator_query_term
{
    uint32_t term_id;
    float weight;
} evoke_semantic_accelerator_query_term;

typedef struct evoke_semantic_accelerator_ranked_cluster
{
    uint32_t cluster_index;
    float score;
} evoke_semantic_accelerator_ranked_cluster;

typedef struct evoke_semantic_accelerator_block_ref
{
    const evoke_semantic_accelerator_index *index;
    const evoke_semantic_accelerator_cluster *cluster;
} evoke_semantic_accelerator_block_ref;

typedef struct evoke_semantic_accelerator_ranked_block
{
    uint32_t block_id;
    float score;
} evoke_semantic_accelerator_ranked_block;

struct evoke_semantic_accelerator_query_scratch
{
    evoke_semantic_accelerator_query_term *query;
    evoke_semantic_accelerator_query_term *selected;
    evoke_semantic_accelerator_ranked_cluster *ranked;
    uint8_t *visited;
    evoke_topk_accumulator topk;
    evoke_semantic_accelerator_block_ref *block_refs;
    evoke_semantic_accelerator_ranked_block *ranked_blocks;
    uint32_t *block_ref_counts;
    uint32_t *block_ref_offsets;
    uint32_t *block_ref_cursors;
    float *block_scores;
    uint8_t *block_visited;
    evoke_topk_accumulator block_topk;
};

static void
evoke_semantic_accelerator_query_scratch_clear_blocks(
    evoke_semantic_accelerator_query_scratch *scratch
)
{
    if (scratch == NULL)
    {
        return;
    }
    evoke_topk_accumulator_free(&scratch->block_topk);
    free(scratch->block_visited);
    free(scratch->block_scores);
    free(scratch->block_ref_cursors);
    free(scratch->block_ref_offsets);
    free(scratch->block_ref_counts);
    free(scratch->ranked_blocks);
    free(scratch->block_refs);
    scratch->block_visited = NULL;
    scratch->block_scores = NULL;
    scratch->block_ref_cursors = NULL;
    scratch->block_ref_offsets = NULL;
    scratch->block_ref_counts = NULL;
    scratch->ranked_blocks = NULL;
    scratch->block_refs = NULL;
}

static void
evoke_semantic_accelerator_query_scratch_clear(
    evoke_semantic_accelerator_query_scratch *scratch
)
{
    if (scratch == NULL)
    {
        return;
    }
    evoke_semantic_accelerator_query_scratch_clear_blocks(scratch);
    evoke_topk_accumulator_free(&scratch->topk);
    free(scratch->visited);
    free(scratch->ranked);
    free(scratch->selected);
    free(scratch->query);
    scratch->visited = NULL;
    scratch->ranked = NULL;
    scratch->selected = NULL;
    scratch->query = NULL;
}

evoke_semantic_accelerator_query_scratch *
evoke_semantic_accelerator_query_scratch_create(void)
{
    return calloc(1, sizeof(evoke_semantic_accelerator_query_scratch));
}

void
evoke_semantic_accelerator_query_scratch_destroy(
    evoke_semantic_accelerator_query_scratch *scratch
)
{
    evoke_semantic_accelerator_query_scratch_clear(scratch);
    free(scratch);
}

static uint64_t
evoke_semantic_accelerator_memory_add(
    uint64_t bytes,
    uint64_t count,
    size_t element_size
)
{
    return evoke_u64_saturating_add(
        bytes,
        evoke_u64_saturating_mul(count, (uint64_t) element_size)
    );
}

static void
evoke_semantic_accelerator_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_semantic_accelerator_write_u32(uint8_t *bytes, uint32_t value)
{
    for (uint32_t index = 0; index < 4; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_accelerator_write_u64(uint8_t *bytes, uint64_t value)
{
    for (uint32_t index = 0; index < 8; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static uint16_t
evoke_semantic_accelerator_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] | ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_semantic_accelerator_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_semantic_accelerator_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (uint32_t index = 0; index < 8; index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static void
evoke_semantic_accelerator_write_float(uint8_t *bytes, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    evoke_semantic_accelerator_write_u32(bytes, bits);
}

static float
evoke_semantic_accelerator_read_float(const uint8_t *bytes)
{
    uint32_t bits = evoke_semantic_accelerator_read_u32(bytes);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t
evoke_semantic_accelerator_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_SEMANTIC_ACCELERATOR_CHECKSUM_OFFSET &&
            index < EVOKE_SEMANTIC_ACCELERATOR_CHECKSUM_OFFSET + 8U
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static bool
evoke_semantic_accelerator_add_size(
    size_t *value,
    uint64_t count,
    size_t width
)
{
    if (count > SIZE_MAX / width || *value > SIZE_MAX - count * width)
    {
        return false;
    }
    *value += (size_t) count * width;
    return true;
}

static int
evoke_semantic_accelerator_compare_query_id(
    const void *left,
    const void *right
)
{
    const evoke_semantic_accelerator_query_term *a = left;
    const evoke_semantic_accelerator_query_term *b = right;

    return (a->term_id > b->term_id) - (a->term_id < b->term_id);
}

static int
evoke_semantic_accelerator_compare_query_weight(
    const void *left,
    const void *right
)
{
    const evoke_semantic_accelerator_query_term *a = left;
    const evoke_semantic_accelerator_query_term *b = right;

    if (a->weight > b->weight)
    {
        return -1;
    }
    if (a->weight < b->weight)
    {
        return 1;
    }
    return (a->term_id > b->term_id) - (a->term_id < b->term_id);
}

static int
evoke_semantic_accelerator_compare_cluster_score(
    const void *left,
    const void *right
)
{
    const evoke_semantic_accelerator_ranked_cluster *a = left;
    const evoke_semantic_accelerator_ranked_cluster *b = right;

    if (a->score > b->score)
    {
        return -1;
    }
    if (a->score < b->score)
    {
        return 1;
    }
    return (a->cluster_index > b->cluster_index) -
        (a->cluster_index < b->cluster_index);
}

static int
evoke_semantic_accelerator_compare_block_score(
    const void *left,
    const void *right
)
{
    const evoke_semantic_accelerator_ranked_block *a = left;
    const evoke_semantic_accelerator_ranked_block *b = right;

    if (a->score > b->score)
    {
        return -1;
    }
    if (a->score < b->score)
    {
        return 1;
    }
    return (a->block_id > b->block_id) -
        (a->block_id < b->block_id);
}

static const evoke_semantic_accelerator_term *
evoke_semantic_accelerator_find_term(
    const evoke_semantic_accelerator_index *index,
    uint32_t term_id
)
{
    uint32_t low = 0;
    uint32_t high = index->term_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2U;
        uint32_t current = index->terms[middle].term_id;

        if (current < term_id)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    if (low < index->term_count && index->terms[low].term_id == term_id)
    {
        return &index->terms[low];
    }
    return NULL;
}

static float
evoke_semantic_accelerator_query_weight_for(
    const evoke_semantic_accelerator_query_term *query,
    size_t query_count,
    uint32_t term_id
)
{
    size_t low = 0;
    size_t high = query_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2U;

        if (query[middle].term_id < term_id)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    return low < query_count && query[low].term_id == term_id
        ? query[low].weight
        : 0.0f;
}

void
evoke_semantic_accelerator_index_init(
    evoke_semantic_accelerator_index *index
)
{
    if (index != NULL)
    {
        memset(index, 0, sizeof(*index));
    }
}

void
evoke_semantic_accelerator_index_free(
    evoke_semantic_accelerator_index *index
)
{
    if (index == NULL)
    {
        return;
    }
    free(index->terms);
    free(index->clusters);
    free(index->document_ids);
    free(index->summaries);
    evoke_semantic_accelerator_index_init(index);
}

evoke_status
evoke_semantic_accelerator_index_validate(
    const evoke_semantic_accelerator_index *index
)
{
    uint32_t cluster_cursor = 0;
    uint32_t document_cursor = 0;
    uint32_t summary_cursor = 0;

    if (index == NULL || index->source_root_checksum == 0 ||
        (index->term_count > 0 && index->terms == NULL) ||
        (index->cluster_count > 0 && index->clusters == NULL) ||
        (index->document_ref_count > 0 && index->document_ids == NULL) ||
        (index->summary_count > 0 && index->summaries == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_accelerator_term *term =
            &index->terms[term_index];

        if ((term_index > 0 &&
             index->terms[term_index - 1U].term_id >= term->term_id) ||
            term->first_cluster != cluster_cursor ||
            term->cluster_count >
                index->cluster_count - term->first_cluster)
        {
            return EVOKE_ERR_FORMAT;
        }
        cluster_cursor += term->cluster_count;
    }
    if (cluster_cursor != index->cluster_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t cluster_index = 0;
         cluster_index < index->cluster_count;
         cluster_index++)
    {
        const evoke_semantic_accelerator_cluster *cluster =
            &index->clusters[cluster_index];

        if (cluster->reserved != 0 || cluster->document_count == 0 ||
            cluster->first_document != document_cursor ||
            cluster->document_count >
                index->document_ref_count - cluster->first_document ||
            cluster->first_summary != summary_cursor ||
            cluster->summary_count >
                index->summary_count - cluster->first_summary ||
            !isfinite(cluster->quantum) ||
            (cluster->summary_count == 0 && cluster->quantum != 0.0f) ||
            (cluster->summary_count > 0 && cluster->quantum <= 0.0f))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t offset = 0;
             offset < cluster->document_count;
             offset++)
        {
            if (index->document_ids[cluster->first_document + offset] >=
                index->document_count)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t offset = 0;
             offset < cluster->summary_count;
             offset++)
        {
            const evoke_semantic_accelerator_summary *summary =
                &index->summaries[cluster->first_summary + offset];

            if (summary->quantized_impact == 0 ||
                summary->reserved[0] != 0 ||
                summary->reserved[1] != 0 ||
                summary->reserved[2] != 0 ||
                (offset > 0 &&
                 index->summaries[
                     cluster->first_summary + offset - 1U
                 ].term_id >= summary->term_id))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        document_cursor += cluster->document_count;
        summary_cursor += cluster->summary_count;
    }
    if (document_cursor != index->document_ref_count ||
        summary_cursor != index->summary_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

bool
evoke_semantic_accelerator_root_matches(
    const evoke_semantic_accelerator_index *index,
    uint64_t source_root_checksum
)
{
    return index != NULL && source_root_checksum != 0 &&
        index->source_root_checksum == source_root_checksum;
}

evoke_status
evoke_semantic_accelerator_index_build(
    uint64_t source_root_checksum,
    uint32_t document_count,
    const evoke_semantic_accelerator_term_input *terms,
    uint32_t term_count,
    evoke_semantic_accelerator_index *index_out
)
{
    uint64_t cluster_count = 0;
    uint64_t document_ref_count = 0;
    uint64_t summary_count = 0;
    evoke_semantic_accelerator_index index;
    uint32_t cluster_cursor = 0;
    uint32_t document_cursor = 0;
    uint32_t summary_cursor = 0;

    if (source_root_checksum == 0 || index_out == NULL ||
        (term_count > 0 && terms == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_accelerator_index_init(&index);
    for (uint32_t term_index = 0; term_index < term_count; term_index++)
    {
        if ((term_index > 0 &&
             terms[term_index - 1U].term_id >= terms[term_index].term_id) ||
            (terms[term_index].cluster_count > 0 &&
             terms[term_index].clusters == NULL))
        {
            return EVOKE_ERR_INVALID;
        }
        if (terms[term_index].cluster_count > UINT32_MAX - cluster_count)
        {
            return EVOKE_ERR_RANGE;
        }
        cluster_count += terms[term_index].cluster_count;
        for (uint32_t cluster_index = 0;
             cluster_index < terms[term_index].cluster_count;
             cluster_index++)
        {
            const evoke_semantic_accelerator_cluster_input *cluster =
                &terms[term_index].clusters[cluster_index];

            if (cluster->document_count == 0 ||
                cluster->document_ids == NULL ||
                cluster->summary_count > UINT16_MAX ||
                (cluster->summary_count > 0 && cluster->summary == NULL))
            {
                return EVOKE_ERR_INVALID;
            }
            if (cluster->document_count >
                    UINT32_MAX - document_ref_count ||
                cluster->summary_count > UINT32_MAX - summary_count)
            {
                return EVOKE_ERR_RANGE;
            }
            document_ref_count += cluster->document_count;
            summary_count += cluster->summary_count;
        }
    }
    if (cluster_count > UINT32_MAX ||
        document_ref_count > UINT32_MAX || summary_count > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    index.source_root_checksum = source_root_checksum;
    index.document_count = document_count;
    index.term_count = term_count;
    index.cluster_count = (uint32_t) cluster_count;
    index.document_ref_count = (uint32_t) document_ref_count;
    index.summary_count = (uint32_t) summary_count;
    index.terms = calloc(term_count, sizeof(*index.terms));
    index.clusters = calloc(
        index.cluster_count,
        sizeof(*index.clusters)
    );
    index.document_ids = malloc(
        (size_t) index.document_ref_count * sizeof(*index.document_ids)
    );
    index.summaries = calloc(
        index.summary_count,
        sizeof(*index.summaries)
    );
    if ((term_count > 0 && index.terms == NULL) ||
        (index.cluster_count > 0 && index.clusters == NULL) ||
        (index.document_ref_count > 0 && index.document_ids == NULL) ||
        (index.summary_count > 0 && index.summaries == NULL))
    {
        evoke_semantic_accelerator_index_free(&index);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_semantic_accelerator_term_input *input =
            &terms[term_index];
        evoke_semantic_accelerator_term *term = &index.terms[term_index];

        term->term_id = input->term_id;
        term->first_cluster = cluster_cursor;
        term->cluster_count = input->cluster_count;
        for (uint32_t input_cluster = 0;
             input_cluster < input->cluster_count;
             input_cluster++, cluster_cursor++)
        {
            const evoke_semantic_accelerator_cluster_input *source =
                &input->clusters[input_cluster];
            evoke_semantic_accelerator_cluster *cluster =
                &index.clusters[cluster_cursor];
            float maximum = 0.0f;

            cluster->first_document = document_cursor;
            cluster->document_count = source->document_count;
            cluster->first_summary = summary_cursor;
            cluster->summary_count = (uint16_t) source->summary_count;
            if (source->document_count > 0)
            {
                memcpy(
                    &index.document_ids[document_cursor],
                    source->document_ids,
                    (size_t) source->document_count *
                        sizeof(*index.document_ids)
                );
                document_cursor += source->document_count;
            }
            for (uint32_t offset = 0;
                 offset < source->summary_count;
                 offset++)
            {
                const evoke_semantic_accelerator_summary_input *summary =
                    &source->summary[offset];

                if (!isfinite(summary->max_impact) ||
                    summary->max_impact <= 0.0f ||
                    (offset > 0 &&
                     source->summary[offset - 1U].term_id >=
                        summary->term_id))
                {
                    evoke_semantic_accelerator_index_free(&index);
                    return EVOKE_ERR_INVALID;
                }
                maximum = fmaxf(maximum, summary->max_impact);
            }
            cluster->quantum = maximum > 0.0f
                ? maximum / 255.0f
                : 0.0f;
            if (maximum > 0.0f && cluster->quantum == 0.0f)
            {
                evoke_semantic_accelerator_index_free(&index);
                return EVOKE_ERR_RANGE;
            }
            for (uint32_t offset = 0;
                 offset < source->summary_count;
                 offset++, summary_cursor++)
            {
                const evoke_semantic_accelerator_summary_input *source_item =
                    &source->summary[offset];
                evoke_semantic_accelerator_summary *summary =
                    &index.summaries[summary_cursor];
                float quantized = ceilf(
                    source_item->max_impact / cluster->quantum
                );

                summary->term_id = source_item->term_id;
                summary->quantized_impact = quantized > 255.0f
                    ? UINT8_MAX
                    : (uint8_t) quantized;
            }
        }
    }
    if (evoke_semantic_accelerator_index_validate(&index) != EVOKE_OK)
    {
        evoke_semantic_accelerator_index_free(&index);
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_accelerator_index_free(index_out);
    *index_out = index;
    return EVOKE_OK;
}

static evoke_status
evoke_semantic_accelerator_layout(
    const evoke_semantic_accelerator_index *index,
    size_t *terms_offset,
    size_t *clusters_offset,
    size_t *documents_offset,
    size_t *summaries_offset,
    size_t *total_size
)
{
    size_t size = EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE;

    *terms_offset = size;
    if (!evoke_semantic_accelerator_add_size(
            &size,
            index->term_count,
            EVOKE_SEMANTIC_ACCELERATOR_TERM_SIZE
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    *clusters_offset = size;
    if (!evoke_semantic_accelerator_add_size(
            &size,
            index->cluster_count,
            EVOKE_SEMANTIC_ACCELERATOR_CLUSTER_SIZE
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    *documents_offset = size;
    if (!evoke_semantic_accelerator_add_size(
            &size,
            index->document_ref_count,
            sizeof(uint32_t)
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    *summaries_offset = size;
    if (!evoke_semantic_accelerator_add_size(
            &size,
            index->summary_count,
            EVOKE_SEMANTIC_ACCELERATOR_SUMMARY_SIZE
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    *total_size = size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_serialize(
    const evoke_semantic_accelerator_index *index,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    size_t terms_offset;
    size_t clusters_offset;
    size_t documents_offset;
    size_t summaries_offset;
    size_t total_size;
    uint8_t *bytes;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    if (evoke_semantic_accelerator_index_validate(index) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_accelerator_layout(
        index,
        &terms_offset,
        &clusters_offset,
        &documents_offset,
        &summaries_offset,
        &total_size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_semantic_accelerator_write_u32(
        bytes + 0,
        EVOKE_SEMANTIC_ACCELERATOR_MAGIC
    );
    evoke_semantic_accelerator_write_u16(
        bytes + 4,
        EVOKE_SEMANTIC_ACCELERATOR_FORMAT_VERSION
    );
    evoke_semantic_accelerator_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE
    );
    evoke_semantic_accelerator_write_u32(bytes + 8, index->document_count);
    evoke_semantic_accelerator_write_u32(bytes + 12, index->term_count);
    evoke_semantic_accelerator_write_u32(bytes + 16, index->cluster_count);
    evoke_semantic_accelerator_write_u32(
        bytes + 20,
        index->document_ref_count
    );
    evoke_semantic_accelerator_write_u32(bytes + 24, index->summary_count);
    evoke_semantic_accelerator_write_u64(bytes + 32, terms_offset);
    evoke_semantic_accelerator_write_u64(bytes + 40, clusters_offset);
    evoke_semantic_accelerator_write_u64(bytes + 48, documents_offset);
    evoke_semantic_accelerator_write_u64(bytes + 56, summaries_offset);
    evoke_semantic_accelerator_write_u64(bytes + 64, total_size);
    evoke_semantic_accelerator_write_u64(
        bytes + 80,
        index->source_root_checksum
    );
    for (uint32_t index_position = 0;
         index_position < index->term_count;
         index_position++)
    {
        const evoke_semantic_accelerator_term *term =
            &index->terms[index_position];
        uint8_t *target = bytes + terms_offset +
            (size_t) index_position * EVOKE_SEMANTIC_ACCELERATOR_TERM_SIZE;

        evoke_semantic_accelerator_write_u32(target + 0, term->term_id);
        evoke_semantic_accelerator_write_u32(
            target + 4,
            term->first_cluster
        );
        evoke_semantic_accelerator_write_u32(
            target + 8,
            term->cluster_count
        );
    }
    for (uint32_t index_position = 0;
         index_position < index->cluster_count;
         index_position++)
    {
        const evoke_semantic_accelerator_cluster *cluster =
            &index->clusters[index_position];
        uint8_t *target = bytes + clusters_offset +
            (size_t) index_position * EVOKE_SEMANTIC_ACCELERATOR_CLUSTER_SIZE;

        evoke_semantic_accelerator_write_u32(
            target + 0,
            cluster->first_document
        );
        evoke_semantic_accelerator_write_u32(
            target + 4,
            cluster->document_count
        );
        evoke_semantic_accelerator_write_u32(
            target + 8,
            cluster->first_summary
        );
        evoke_semantic_accelerator_write_u16(
            target + 12,
            cluster->summary_count
        );
        evoke_semantic_accelerator_write_u16(target + 14, 0);
        evoke_semantic_accelerator_write_float(
            target + 16,
            cluster->quantum
        );
    }
    for (uint32_t index_position = 0;
         index_position < index->document_ref_count;
         index_position++)
    {
        evoke_semantic_accelerator_write_u32(
            bytes + documents_offset +
                (size_t) index_position * sizeof(uint32_t),
            index->document_ids[index_position]
        );
    }
    for (uint32_t index_position = 0;
         index_position < index->summary_count;
         index_position++)
    {
        const evoke_semantic_accelerator_summary *summary =
            &index->summaries[index_position];
        uint8_t *target = bytes + summaries_offset +
            (size_t) index_position * EVOKE_SEMANTIC_ACCELERATOR_SUMMARY_SIZE;

        evoke_semantic_accelerator_write_u32(target + 0, summary->term_id);
        target[4] = summary->quantized_impact;
    }
    evoke_semantic_accelerator_write_u64(
        bytes + EVOKE_SEMANTIC_ACCELERATOR_CHECKSUM_OFFSET,
        evoke_semantic_accelerator_checksum(bytes, total_size)
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_index *index_out
)
{
    evoke_semantic_accelerator_index index;
    uint32_t document_count;
    uint32_t term_count;
    uint32_t cluster_count;
    uint32_t document_ref_count;
    uint32_t summary_count;
    uint64_t terms_offset;
    uint64_t clusters_offset;
    uint64_t documents_offset;
    uint64_t summaries_offset;
    uint64_t total_size;
    uint64_t checksum;
    uint64_t source_root_checksum;

    if (bytes == NULL || index_out == NULL ||
        size < EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE ||
        evoke_semantic_accelerator_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_ACCELERATOR_MAGIC ||
        evoke_semantic_accelerator_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_ACCELERATOR_FORMAT_VERSION ||
        evoke_semantic_accelerator_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    document_count = evoke_semantic_accelerator_read_u32(bytes + 8);
    term_count = evoke_semantic_accelerator_read_u32(bytes + 12);
    cluster_count = evoke_semantic_accelerator_read_u32(bytes + 16);
    document_ref_count = evoke_semantic_accelerator_read_u32(bytes + 20);
    summary_count = evoke_semantic_accelerator_read_u32(bytes + 24);
    terms_offset = evoke_semantic_accelerator_read_u64(bytes + 32);
    clusters_offset = evoke_semantic_accelerator_read_u64(bytes + 40);
    documents_offset = evoke_semantic_accelerator_read_u64(bytes + 48);
    summaries_offset = evoke_semantic_accelerator_read_u64(bytes + 56);
    total_size = evoke_semantic_accelerator_read_u64(bytes + 64);
    checksum = evoke_semantic_accelerator_read_u64(
        bytes + EVOKE_SEMANTIC_ACCELERATOR_CHECKSUM_OFFSET
    );
    source_root_checksum = evoke_semantic_accelerator_read_u64(bytes + 80);
    if (total_size != size || checksum !=
            evoke_semantic_accelerator_checksum(bytes, size) ||
        bytes[28] != 0 || bytes[29] != 0 ||
        bytes[30] != 0 || bytes[31] != 0 ||
        source_root_checksum == 0 ||
        terms_offset != EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE ||
        clusters_offset != terms_offset +
            (uint64_t) term_count * EVOKE_SEMANTIC_ACCELERATOR_TERM_SIZE ||
        documents_offset != clusters_offset +
            (uint64_t) cluster_count *
                EVOKE_SEMANTIC_ACCELERATOR_CLUSTER_SIZE ||
        summaries_offset != documents_offset +
            (uint64_t) document_ref_count * sizeof(uint32_t) ||
        total_size != summaries_offset +
            (uint64_t) summary_count *
                EVOKE_SEMANTIC_ACCELERATOR_SUMMARY_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_accelerator_index_init(&index);
    index.source_root_checksum = source_root_checksum;
    index.document_count = document_count;
    index.term_count = term_count;
    index.cluster_count = cluster_count;
    index.document_ref_count = document_ref_count;
    index.summary_count = summary_count;
    index.terms = calloc(term_count, sizeof(*index.terms));
    index.clusters = calloc(cluster_count, sizeof(*index.clusters));
    index.document_ids = malloc(
        (size_t) document_ref_count * sizeof(*index.document_ids)
    );
    index.summaries = calloc(summary_count, sizeof(*index.summaries));
    if ((term_count > 0 && index.terms == NULL) ||
        (cluster_count > 0 && index.clusters == NULL) ||
        (document_ref_count > 0 && index.document_ids == NULL) ||
        (summary_count > 0 && index.summaries == NULL))
    {
        evoke_semantic_accelerator_index_free(&index);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t position = 0; position < term_count; position++)
    {
        const uint8_t *source = bytes + terms_offset +
            (size_t) position * EVOKE_SEMANTIC_ACCELERATOR_TERM_SIZE;

        index.terms[position].term_id =
            evoke_semantic_accelerator_read_u32(source + 0);
        index.terms[position].first_cluster =
            evoke_semantic_accelerator_read_u32(source + 4);
        index.terms[position].cluster_count =
            evoke_semantic_accelerator_read_u32(source + 8);
    }
    for (uint32_t position = 0; position < cluster_count; position++)
    {
        const uint8_t *source = bytes + clusters_offset +
            (size_t) position * EVOKE_SEMANTIC_ACCELERATOR_CLUSTER_SIZE;
        evoke_semantic_accelerator_cluster *cluster =
            &index.clusters[position];

        cluster->first_document =
            evoke_semantic_accelerator_read_u32(source + 0);
        cluster->document_count =
            evoke_semantic_accelerator_read_u32(source + 4);
        cluster->first_summary =
            evoke_semantic_accelerator_read_u32(source + 8);
        cluster->summary_count =
            evoke_semantic_accelerator_read_u16(source + 12);
        cluster->reserved =
            evoke_semantic_accelerator_read_u16(source + 14);
        cluster->quantum =
            evoke_semantic_accelerator_read_float(source + 16);
    }
    for (uint32_t position = 0; position < document_ref_count; position++)
    {
        index.document_ids[position] = evoke_semantic_accelerator_read_u32(
            bytes + documents_offset + (size_t) position * sizeof(uint32_t)
        );
    }
    for (uint32_t position = 0; position < summary_count; position++)
    {
        const uint8_t *source = bytes + summaries_offset +
            (size_t) position * EVOKE_SEMANTIC_ACCELERATOR_SUMMARY_SIZE;

        index.summaries[position].term_id =
            evoke_semantic_accelerator_read_u32(source + 0);
        index.summaries[position].quantized_impact = source[4];
        memcpy(index.summaries[position].reserved, source + 5, 3);
    }
    if (evoke_semantic_accelerator_index_validate(&index) != EVOKE_OK)
    {
        evoke_semantic_accelerator_index_free(&index);
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_accelerator_index_free(index_out);
    *index_out = index;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_term_serialize(
    uint64_t source_root_checksum,
    uint32_t document_count,
    const evoke_semantic_accelerator_term_input *term,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_semantic_accelerator_index index;
    evoke_status status;

    if (term == NULL || bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    evoke_semantic_accelerator_index_init(&index);
    status = evoke_semantic_accelerator_index_build(
        source_root_checksum,
        document_count,
        term,
        1,
        &index
    );
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_accelerator_serialize(
            &index,
            bytes_out,
            size_out
        );
    }
    evoke_semantic_accelerator_index_free(&index);
    return status;
}

evoke_status
evoke_semantic_accelerator_term_deserialize(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_root_checksum,
    uint32_t expected_term_id,
    evoke_semantic_accelerator_index *index_out
)
{
    evoke_semantic_accelerator_index index;
    evoke_status status;

    if (expected_source_root_checksum == 0 || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_accelerator_index_init(&index);
    status = evoke_semantic_accelerator_deserialize(bytes, size, &index);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_semantic_accelerator_root_matches(
            &index,
            expected_source_root_checksum) ||
        index.term_count != 1 || index.terms[0].term_id != expected_term_id)
    {
        evoke_semantic_accelerator_index_free(&index);
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_accelerator_index_free(index_out);
    *index_out = index;
    return EVOKE_OK;
}

static float
evoke_semantic_accelerator_cluster_score(
    const evoke_semantic_accelerator_index *index,
    const evoke_semantic_accelerator_cluster *cluster,
    const evoke_semantic_accelerator_query_term *query,
    size_t query_count,
    evoke_semantic_accelerator_stats *stats
)
{
    double score = 0.0;

    for (uint32_t offset = 0; offset < cluster->summary_count; offset++)
    {
        const evoke_semantic_accelerator_summary *summary =
            &index->summaries[cluster->first_summary + offset];
        float weight = evoke_semantic_accelerator_query_weight_for(
            query,
            query_count,
            summary->term_id
        );

        score += (double) weight * cluster->quantum *
            summary->quantized_impact;
        stats->summary_entries_examined++;
    }
    return score > FLT_MAX ? INFINITY : (float) score;
}

static evoke_status
evoke_semantic_accelerator_topk_blocks(
    const evoke_semantic_accelerator_index *const *indexes,
    size_t index_count,
    const evoke_semantic_accelerator_query_term *query,
    size_t query_count,
    const evoke_semantic_accelerator_query_term *selected,
    size_t selected_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t original_query_count,
    size_t k,
    size_t candidate_capacity,
    const evoke_semantic_accelerator_options *options,
    evoke_semantic_accelerator_score_cb score_document,
    void *score_context,
    evoke_topk_result *result_out,
    evoke_semantic_accelerator_stats *stats,
    bool *supported_out,
    evoke_semantic_accelerator_query_scratch *scratch
)
{
    uint32_t documents_per_block;
    uint32_t block_count;
    uint64_t total_refs = 0;
    uint64_t scratch_bytes = 0;
    evoke_status status = EVOKE_OK;

    *supported_out = false;
    evoke_semantic_accelerator_query_scratch_clear_blocks(scratch);
    if (options->document_shift == 0 || options->document_shift > 30)
    {
        return EVOKE_OK;
    }
    documents_per_block = UINT32_C(1) << options->document_shift;
    block_count = (uint32_t) (
        ((uint64_t) indexes[0]->document_count + documents_per_block - 1U) /
        documents_per_block
    );
    scratch->block_ref_counts = calloc(
        block_count,
        sizeof(*scratch->block_ref_counts)
    );
    scratch->block_ref_offsets = calloc(
        (size_t) block_count + 1U,
        sizeof(*scratch->block_ref_offsets)
    );
    scratch->block_ref_cursors = calloc(
        block_count,
        sizeof(*scratch->block_ref_cursors)
    );
    scratch->block_scores = calloc(
        block_count,
        sizeof(*scratch->block_scores)
    );
    scratch->ranked_blocks = calloc(
        block_count,
        sizeof(*scratch->ranked_blocks)
    );
    if (scratch->block_ref_counts == NULL ||
        scratch->block_ref_offsets == NULL ||
        scratch->block_ref_cursors == NULL ||
        scratch->block_scores == NULL ||
        scratch->ranked_blocks == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t selected_index = 0;
         selected_index < selected_count;
         selected_index++)
    {
        const evoke_semantic_accelerator_index *index = NULL;
        const evoke_semantic_accelerator_term *term = NULL;

        if (selected[selected_index].weight == 0.0f)
        {
            continue;
        }
        for (size_t index_position = 0;
             index_position < index_count;
             index_position++)
        {
            term = evoke_semantic_accelerator_find_term(
                indexes[index_position],
                selected[selected_index].term_id
            );
            if (term != NULL)
            {
                index = indexes[index_position];
                break;
            }
        }
        if (term == NULL)
        {
            continue;
        }
        for (uint32_t offset = 0; offset < term->cluster_count; offset++)
        {
            const evoke_semantic_accelerator_cluster *cluster =
                &index->clusters[term->first_cluster + offset];
            uint32_t first_document;
            uint32_t last_document;
            uint32_t block;
            float score;

            if (cluster->document_count == 0)
            {
                goto cleanup;
            }
            first_document = index->document_ids[cluster->first_document];
            last_document = index->document_ids[
                cluster->first_document + cluster->document_count - 1U
            ];
            block = first_document >> options->document_shift;
            if (block >= block_count ||
                last_document >> options->document_shift != block ||
                scratch->block_ref_counts[block] == UINT32_MAX)
            {
                goto cleanup;
            }
            score = evoke_semantic_accelerator_cluster_score(
                index,
                cluster,
                query,
                query_count,
                stats
            );
            if (!isfinite(score))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            scratch->block_scores[block] =
                scratch->block_scores[block] > FLT_MAX - score
                ? FLT_MAX
                : scratch->block_scores[block] + score;
            scratch->block_ref_counts[block]++;
            total_refs++;
            if (total_refs > SIZE_MAX / sizeof(*scratch->block_refs))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
    }
    if (total_refs == 0)
    {
        goto cleanup;
    }
    scratch->block_refs = malloc(
        (size_t) total_refs * sizeof(*scratch->block_refs)
    );
    if (scratch->block_refs == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t block = 0; block < block_count; block++)
    {
        if (UINT32_MAX - scratch->block_ref_offsets[block] <
            scratch->block_ref_counts[block])
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        scratch->block_ref_offsets[block + 1U] =
            scratch->block_ref_offsets[block] +
            scratch->block_ref_counts[block];
        scratch->block_ref_cursors[block] =
            scratch->block_ref_offsets[block];
        scratch->ranked_blocks[block].block_id = block;
        scratch->ranked_blocks[block].score =
            scratch->block_scores[block];
    }
    for (size_t selected_index = 0;
         selected_index < selected_count;
         selected_index++)
    {
        const evoke_semantic_accelerator_index *index = NULL;
        const evoke_semantic_accelerator_term *term = NULL;

        if (selected[selected_index].weight == 0.0f)
        {
            continue;
        }
        for (size_t index_position = 0;
             index_position < index_count;
             index_position++)
        {
            term = evoke_semantic_accelerator_find_term(
                indexes[index_position],
                selected[selected_index].term_id
            );
            if (term != NULL)
            {
                index = indexes[index_position];
                break;
            }
        }
        if (term == NULL)
        {
            continue;
        }
        for (uint32_t offset = 0; offset < term->cluster_count; offset++)
        {
            const evoke_semantic_accelerator_cluster *cluster =
                &index->clusters[term->first_cluster + offset];
            uint32_t first_document =
                index->document_ids[cluster->first_document];
            uint32_t block = first_document >> options->document_shift;
            uint32_t cursor = scratch->block_ref_cursors[block]++;

            scratch->block_refs[cursor].index = index;
            scratch->block_refs[cursor].cluster = cluster;
        }
    }
    qsort(
        scratch->ranked_blocks,
        block_count,
        sizeof(*scratch->ranked_blocks),
        evoke_semantic_accelerator_compare_block_score);
    scratch->block_visited = calloc(
        ((size_t) indexes[0]->document_count + 7U) / 8U,
        1
    );
    if (scratch->block_visited == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    status = evoke_topk_accumulator_init(
        &scratch->block_topk,
        candidate_capacity
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    scratch_bytes = evoke_semantic_accelerator_memory_add(
        scratch_bytes,
        block_count,
        sizeof(*scratch->block_ref_counts) +
            sizeof(*scratch->block_ref_cursors) +
            sizeof(*scratch->block_scores) +
            sizeof(*scratch->ranked_blocks)
    );
    scratch_bytes = evoke_semantic_accelerator_memory_add(
        scratch_bytes,
        (uint64_t) block_count + 1U,
        sizeof(*scratch->block_ref_offsets)
    );
    scratch_bytes = evoke_semantic_accelerator_memory_add(
        scratch_bytes,
        total_refs,
        sizeof(*scratch->block_refs)
    );
    scratch_bytes = evoke_semantic_accelerator_memory_add(
        scratch_bytes,
        ((uint64_t) indexes[0]->document_count + 7U) / 8U,
        sizeof(*scratch->block_visited)
    );
    scratch_bytes = evoke_semantic_accelerator_memory_add(
        scratch_bytes,
        candidate_capacity,
        sizeof(*scratch->block_topk.heap)
    );
    stats->query_scratch_peak_bytes = scratch_bytes;
    for (uint32_t ranked_index = 0;
         ranked_index < block_count;
         ranked_index++)
    {
        uint32_t block =
            scratch->ranked_blocks[ranked_index].block_id;
        bool skip;

        if (scratch->block_ref_counts[block] == 0)
        {
            continue;
        }
        stats->clusters_considered +=
            scratch->block_ref_counts[block];
        skip = scratch->block_topk.len ==
                scratch->block_topk.capacity &&
            scratch->ranked_blocks[ranked_index].score <
                options->heap_factor * scratch->block_topk.heap[0].score;
        if (skip)
        {
            stats->clusters_skipped += scratch->block_ref_counts[block];
            continue;
        }
        stats->clusters_opened += scratch->block_ref_counts[block];
        for (uint32_t ref_index = scratch->block_ref_offsets[block];
             ref_index < scratch->block_ref_offsets[block + 1U];
             ref_index++)
        {
            const evoke_semantic_accelerator_block_ref *ref =
                &scratch->block_refs[ref_index];

            for (uint32_t document_offset = 0;
                 document_offset < ref->cluster->document_count;
                 document_offset++)
            {
                uint32_t document_id = ref->index->document_ids[
                    ref->cluster->first_document + document_offset
                ];
                uint8_t bit =
                    (uint8_t) (UINT8_C(1) << (document_id & 7U));
                uint8_t *slot =
                    &scratch->block_visited[document_id >> 3];
                float score;

                stats->document_refs_examined++;
                if ((*slot & bit) != 0)
                {
                    stats->duplicate_documents_skipped++;
                    continue;
                }
                *slot |= bit;
                status = score_document(
                    score_context,
                    document_id,
                    query_ids,
                    query_weights,
                    original_query_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if (!isfinite(score))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                stats->documents_scored++;
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &scratch->block_topk,
                        score,
                        document_id,
                        document_id
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
            }
        }
    }
    status = evoke_topk_accumulator_finish(
        &scratch->block_topk,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        uint64_t finish_bytes = evoke_semantic_accelerator_memory_add(
            scratch_bytes,
            result_out->len,
            sizeof(*result_out->doc_ids) + sizeof(*result_out->scores)
        );

        stats->query_scratch_peak_bytes = finish_bytes;
    }
    if (status == EVOKE_OK && result_out->len > k)
    {
        result_out->len = k;
    }
    if (status == EVOKE_OK)
    {
        stats->block_major_query = true;
        *supported_out = true;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_semantic_accelerator_query_scratch_clear_blocks(scratch);
    return status;
}

evoke_status
evoke_semantic_accelerator_topk_many_owned(
    const evoke_semantic_accelerator_index *const *indexes,
    size_t index_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_accelerator_options *options,
    evoke_semantic_accelerator_score_cb score_document,
    void *score_context,
    evoke_topk_result *result_out,
    evoke_semantic_accelerator_stats *stats_out,
    evoke_semantic_accelerator_query_scratch *scratch
)
{
    evoke_semantic_accelerator_stats stats;
    size_t unique_count = 0;
    size_t selected_count;
    size_t candidate_capacity;
    size_t visited_bytes;
    size_t ranked_capacity_peak = 0;
    uint64_t outer_scratch_bytes;
    uint32_t document_count;
    uint64_t source_root_checksum;
    evoke_status status = EVOKE_OK;

    memset(&stats, 0, sizeof(stats));
    if (scratch == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_accelerator_query_scratch_clear(scratch);
    if (indexes == NULL || index_count == 0 || indexes[0] == NULL ||
        result_out == NULL || stats_out == NULL ||
        options == NULL || score_document == NULL ||
        (query_count > 0 &&
         (query_ids == NULL || query_weights == NULL)) ||
        options->query_cut == 0 || options->candidate_multiplier == 0 ||
        !isfinite(options->heap_factor) ||
        options->heap_factor < 0.0f || options->heap_factor > 1.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    document_count = indexes[0]->document_count;
    source_root_checksum = indexes[0]->source_root_checksum;
    if (k > document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t index_position = 0;
         index_position < index_count;
         index_position++)
    {
        const evoke_semantic_accelerator_index *index =
            indexes[index_position];

        if (index == NULL ||
            index->document_count != document_count ||
            index->source_root_checksum != source_root_checksum ||
            evoke_semantic_accelerator_index_validate(index) != EVOKE_OK)
        {
            return EVOKE_ERR_INVALID;
        }
    }
    memset(result_out, 0, sizeof(*result_out));
    memset(stats_out, 0, sizeof(*stats_out));
    if (k == 0 || query_count == 0)
    {
        return EVOKE_OK;
    }
    if (query_count > SIZE_MAX / sizeof(*scratch->query) ||
        query_count > SIZE_MAX / sizeof(*scratch->selected))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    scratch->query = malloc(query_count * sizeof(*scratch->query));
    scratch->selected = malloc(
        query_count * sizeof(*scratch->selected)
    );
    if (scratch->query == NULL || scratch->selected == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    outer_scratch_bytes = evoke_semantic_accelerator_memory_add(
        0,
        query_count,
        sizeof(*scratch->query) + sizeof(*scratch->selected)
    );
    stats.query_scratch_peak_bytes = outer_scratch_bytes;
    for (size_t index_position = 0;
         index_position < query_count;
         index_position++)
    {
        if (!isfinite(query_weights[index_position]) ||
            query_weights[index_position] < 0.0f)
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        scratch->query[index_position].term_id = query_ids[index_position];
        scratch->query[index_position].weight = query_weights[index_position];
    }
    qsort(
        scratch->query,
        query_count,
        sizeof(*scratch->query),
        evoke_semantic_accelerator_compare_query_id
    );
    for (size_t index_position = 0;
         index_position < query_count;
         index_position++)
    {
        if (unique_count > 0 &&
            scratch->query[unique_count - 1U].term_id ==
                scratch->query[index_position].term_id)
        {
            scratch->query[unique_count - 1U].weight +=
                scratch->query[index_position].weight;
            if (!isfinite(scratch->query[unique_count - 1U].weight))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
        else
        {
            scratch->query[unique_count++] = scratch->query[index_position];
        }
    }
    memcpy(
        scratch->selected,
        scratch->query,
        unique_count * sizeof(*scratch->selected)
    );
    qsort(
        scratch->selected,
        unique_count,
        sizeof(*scratch->selected),
        evoke_semantic_accelerator_compare_query_weight
    );
    selected_count = unique_count < options->query_cut
        ? unique_count
        : options->query_cut;
    if (k > SIZE_MAX / options->candidate_multiplier)
    {
        candidate_capacity = document_count;
    }
    else
    {
        candidate_capacity = k * options->candidate_multiplier;
        if (candidate_capacity > document_count)
        {
            candidate_capacity = document_count;
        }
    }
    if (options->document_shift > 0)
    {
        bool block_major_supported = false;

        status = evoke_semantic_accelerator_topk_blocks(
            indexes,
            index_count,
            scratch->query,
            unique_count,
            scratch->selected,
            selected_count,
            query_ids,
            query_weights,
            query_count,
            k,
            candidate_capacity,
            options,
            score_document,
            score_context,
            result_out,
            &stats,
            &block_major_supported,
            scratch
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        stats.query_scratch_peak_bytes = evoke_u64_saturating_add(
            stats.query_scratch_peak_bytes,
            outer_scratch_bytes
        );
        if (block_major_supported)
        {
            *stats_out = stats;
            goto cleanup;
        }
        memset(&stats, 0, sizeof(stats));
        stats.query_scratch_peak_bytes = outer_scratch_bytes;
    }
    visited_bytes = ((size_t) document_count + 7U) / 8U;
    scratch->visited = calloc(visited_bytes, 1);
    if (visited_bytes > 0 && scratch->visited == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    status = evoke_topk_accumulator_init(&scratch->topk, candidate_capacity);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (size_t selected_index = 0;
         selected_index < selected_count;
         selected_index++)
    {
        const evoke_semantic_accelerator_index *index = NULL;
        const evoke_semantic_accelerator_term *term = NULL;
        evoke_semantic_accelerator_ranked_cluster *replacement;

        for (size_t index_position = 0;
             index_position < index_count;
             index_position++)
        {
            term = evoke_semantic_accelerator_find_term(
                indexes[index_position],
                scratch->selected[selected_index].term_id
            );
            if (term != NULL)
            {
                index = indexes[index_position];
                break;
            }
        }
        if (term == NULL ||
            scratch->selected[selected_index].weight == 0.0f)
        {
            continue;
        }
        replacement = realloc(
            scratch->ranked,
            (size_t) term->cluster_count * sizeof(*scratch->ranked)
        );
        if (term->cluster_count > 0 && replacement == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        scratch->ranked = replacement;
        ranked_capacity_peak = ranked_capacity_peak > term->cluster_count
            ? ranked_capacity_peak
            : term->cluster_count;
        for (uint32_t offset = 0; offset < term->cluster_count; offset++)
        {
            uint32_t cluster_index = term->first_cluster + offset;

            scratch->ranked[offset].cluster_index = cluster_index;
            scratch->ranked[offset].score =
                evoke_semantic_accelerator_cluster_score(
                index,
                &index->clusters[cluster_index],
                scratch->query,
                unique_count,
                &stats
            );
        }
        qsort(
            scratch->ranked,
            term->cluster_count,
            sizeof(*scratch->ranked),
            evoke_semantic_accelerator_compare_cluster_score
        );
        for (uint32_t offset = 0; offset < term->cluster_count; offset++)
        {
            const evoke_semantic_accelerator_ranked_cluster *candidate =
                &scratch->ranked[offset];
            const evoke_semantic_accelerator_cluster *cluster =
                &index->clusters[candidate->cluster_index];
            bool skip = scratch->topk.len == scratch->topk.capacity &&
                candidate->score <
                    options->heap_factor * scratch->topk.heap[0].score;

            stats.clusters_considered++;
            if (skip)
            {
                stats.clusters_skipped++;
                continue;
            }
            stats.clusters_opened++;
            for (uint32_t document_offset = 0;
                 document_offset < cluster->document_count;
                 document_offset++)
            {
                uint32_t document_id = index->document_ids[
                    cluster->first_document + document_offset
                ];
                uint8_t bit = (uint8_t) (UINT8_C(1) << (document_id & 7U));
                uint8_t *slot = &scratch->visited[document_id >> 3];
                float score;

                stats.document_refs_examined++;
                if ((*slot & bit) != 0)
                {
                    stats.duplicate_documents_skipped++;
                    continue;
                }
                *slot |= bit;
                status = score_document(
                    score_context,
                    document_id,
                    query_ids,
                    query_weights,
                    query_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if (!isfinite(score))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                stats.documents_scored++;
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &scratch->topk,
                        score,
                        document_id,
                        document_id
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
            }
        }
    }
    status = evoke_topk_accumulator_finish(
        &scratch->topk,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        uint64_t scratch_bytes = outer_scratch_bytes;

        scratch_bytes = evoke_semantic_accelerator_memory_add(
            scratch_bytes,
            visited_bytes,
            sizeof(*scratch->visited)
        );
        scratch_bytes = evoke_semantic_accelerator_memory_add(
            scratch_bytes,
            candidate_capacity,
            sizeof(*scratch->topk.heap)
        );
        scratch_bytes = evoke_semantic_accelerator_memory_add(
            scratch_bytes,
            ranked_capacity_peak,
            sizeof(*scratch->ranked)
        );
        scratch_bytes = evoke_semantic_accelerator_memory_add(
            scratch_bytes,
            result_out->len,
            sizeof(*result_out->doc_ids) + sizeof(*result_out->scores)
        );
        stats.query_scratch_peak_bytes = scratch_bytes;
        if (result_out->len > k)
        {
            result_out->len = k;
        }
        *stats_out = stats;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_semantic_accelerator_query_scratch_clear(scratch);
    return status;
}

evoke_status
evoke_semantic_accelerator_topk_many(
    const evoke_semantic_accelerator_index *const *indexes,
    size_t index_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_accelerator_options *options,
    evoke_semantic_accelerator_score_cb score_document,
    void *score_context,
    evoke_topk_result *result_out,
    evoke_semantic_accelerator_stats *stats_out
)
{
    evoke_semantic_accelerator_query_scratch *scratch =
        evoke_semantic_accelerator_query_scratch_create();
    evoke_status status;

    if (scratch == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    status = evoke_semantic_accelerator_topk_many_owned(
        indexes,
        index_count,
        query_ids,
        query_weights,
        query_count,
        k,
        options,
        score_document,
        score_context,
        result_out,
        stats_out,
        scratch
    );
    evoke_semantic_accelerator_query_scratch_destroy(scratch);
    return status;
}

evoke_status
evoke_semantic_accelerator_topk(
    const evoke_semantic_accelerator_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_accelerator_options *options,
    evoke_semantic_accelerator_score_cb score_document,
    void *score_context,
    evoke_topk_result *result_out,
    evoke_semantic_accelerator_stats *stats_out
)
{
    const evoke_semantic_accelerator_index *indexes[1] = {index};

    return evoke_semantic_accelerator_topk_many(
        indexes,
        1,
        query_ids,
        query_weights,
        query_count,
        k,
        options,
        score_document,
        score_context,
        result_out,
        stats_out
    );
}
