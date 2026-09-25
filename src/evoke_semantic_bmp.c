#include "evoke_semantic_bmp.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint16_t evoke_semantic_bmp_read_u16(const uint8_t *bytes);
static float evoke_semantic_bmp_read_float(const uint8_t *bytes);
static void evoke_semantic_bmp_write_u16(uint8_t *bytes, uint16_t value);
static void evoke_semantic_bmp_write_float(uint8_t *bytes, float value);

#define EVOKE_SEMANTIC_BMP_MAGIC UINT32_C(0x504D4232)
#define EVOKE_SEMANTIC_BMP_VERSION \
    ((uint16_t) EVOKE_SEMANTIC_BMP_FORMAT_VERSION)
#define EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET 112U
#define EVOKE_SEMANTIC_BMP_PACKED_MAGIC UINT32_C(0x334D4250)
#define EVOKE_SEMANTIC_BMP_PACKED_VERSION \
    ((uint16_t) EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION)
#define EVOKE_SEMANTIC_BMP_MAX_UNPRUNED_SUPERBLOCKS UINT32_C(64)

typedef struct evoke_semantic_bmp_layout
{
    size_t terms_offset;
    size_t super_refs_offset;
    size_t refs_offset;
    size_t blocks_offset;
    size_t records_offset;
    size_t impacts_offset;
    size_t total_size;
} evoke_semantic_bmp_layout;

typedef struct evoke_semantic_bmp_packed_layout
{
    size_t terms_offset;
    size_t super_refs_offset;
    size_t refs_offset;
    size_t block_membership_offset;
    size_t doc_deltas_offset;
    size_t impacts_offset;
    size_t total_size;
} evoke_semantic_bmp_packed_layout;

typedef struct evoke_semantic_bmp_build_posting
{
    uint32_t term_id;
    uint32_t document_id;
    uint32_t block_id;
    float impact;
} evoke_semantic_bmp_build_posting;

typedef struct evoke_semantic_bmp_build_ref
{
    uint32_t term_id;
    uint32_t block_id;
    uint32_t record_index;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_build_ref;

typedef struct evoke_semantic_bmp_query_term
{
    uint32_t term_id;
    float weight;
} evoke_semantic_bmp_query_term;

typedef struct evoke_semantic_bmp_ranked_block
{
    uint32_t block_id;
    float upper_bound;
} evoke_semantic_bmp_ranked_block;

size_t
evoke_semantic_bmp_impact_width(
    evoke_semantic_impact_precision impact_precision
)
{
    switch (impact_precision)
    {
        case EVOKE_SEMANTIC_IMPACT_PRECISION_F32:
            return sizeof(uint32_t);
        case EVOKE_SEMANTIC_IMPACT_PRECISION_FP16:
            return sizeof(uint16_t);
        case EVOKE_SEMANTIC_IMPACT_PRECISION_U8:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

static uint16_t
evoke_semantic_bmp_float_to_half(float value, bool *valid_out)
{
    uint32_t bits;
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    int32_t half_exponent;

    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & UINT32_C(0x8000);
    exponent = (bits >> 23) & UINT32_C(0xff);
    mantissa = bits & UINT32_C(0x7fffff);
    if (valid_out != NULL)
    {
        *valid_out = false;
    }
    if (exponent == UINT32_C(0xff))
    {
        return 0;
    }

    half_exponent = (int32_t) exponent - 127 + 15;
    if (half_exponent <= 0)
    {
        uint32_t shift;
        uint32_t half_mantissa;
        uint32_t remainder;
        uint32_t halfway;

        if (half_exponent < -10)
        {
            if (valid_out != NULL)
            {
                *valid_out = true;
            }
            return (uint16_t) sign;
        }
        mantissa |= UINT32_C(0x800000);
        shift = (uint32_t) (14 - half_exponent);
        half_mantissa = mantissa >> shift;
        remainder = mantissa & ((UINT32_C(1) << shift) - 1);
        halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1U) != 0))
        {
            half_mantissa++;
        }
        if (valid_out != NULL)
        {
            *valid_out = true;
        }
        return (uint16_t) (sign | half_mantissa);
    }
    if (half_exponent >= 31)
    {
        return 0;
    }

    {
        uint32_t half_mantissa = mantissa >> 13;
        uint32_t remainder = mantissa & UINT32_C(0x1fff);

        if (remainder > UINT32_C(0x1000) ||
            (remainder == UINT32_C(0x1000) &&
             (half_mantissa & 1U) != 0))
        {
            half_mantissa++;
            if (half_mantissa == UINT32_C(0x400))
            {
                half_mantissa = 0;
                half_exponent++;
                if (half_exponent >= 31)
                {
                    return 0;
                }
            }
        }
        if (valid_out != NULL)
        {
            *valid_out = true;
        }
        return (uint16_t) (
            sign | ((uint32_t) half_exponent << 10) | half_mantissa
        );
    }
}

static float
evoke_semantic_bmp_half_to_float(uint16_t half)
{
    uint32_t sign = ((uint32_t) half & UINT32_C(0x8000)) << 16;
    uint32_t exponent = ((uint32_t) half >> 10) & UINT32_C(0x1f);
    uint32_t mantissa = (uint32_t) half & UINT32_C(0x3ff);
    uint32_t bits;
    float value;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            int32_t normalized_exponent = -14;

            while ((mantissa & UINT32_C(0x400)) == 0)
            {
                mantissa <<= 1;
                normalized_exponent--;
            }
            mantissa &= UINT32_C(0x3ff);
            bits = sign |
                ((uint32_t) (normalized_exponent + 127) << 23) |
                (mantissa << 13);
        }
    }
    else if (exponent == UINT32_C(0x1f))
    {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint8_t
evoke_semantic_bmp_u8_code(
    float impact
)
{
    long code;

    /*
     * Use the frozen P2 document-impact range rather than a per-term scale.
     * This value-local code is stable across COW segments, merges,
     * compaction, and a full REINDEX.  Code zero remains reserved for an
     * absent posting.  Values above 2.0 are deliberately saturated by this
     * aggressive profile; custom models that need a wider range use fp16.
     */
    if (!isfinite(impact) || impact <= 0.0f)
    {
        return 0;
    }
    code = lround((double) impact * 255.0 / 2.0);
    if (code < 1)
    {
        code = 1;
    }
    if (code > UINT8_MAX)
    {
        code = UINT8_MAX;
    }
    return (uint8_t) code;
}

static float
evoke_semantic_bmp_u8_decode(uint8_t code)
{
    return (float) code * (2.0f / 255.0f);
}

evoke_status
evoke_semantic_impact_encode(
    uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float impact,
    float *quantized_out
)
{
    size_t impact_width = evoke_semantic_bmp_impact_width(
        impact_precision
    );
    float quantized;

    if (bytes == NULL || impact_width == 0 || size < impact_width ||
        !isfinite(impact) || impact == 0.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32)
    {
        evoke_semantic_bmp_write_float(bytes, impact);
        quantized = impact;
    }
    else if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_FP16)
    {
        bool valid;
        uint16_t half = evoke_semantic_bmp_float_to_half(impact, &valid);

        if (!valid)
        {
            return EVOKE_ERR_RANGE;
        }
        if ((half & UINT16_C(0x7fff)) == 0)
        {
            half = (half & UINT16_C(0x8000)) | UINT16_C(1);
        }
        evoke_semantic_bmp_write_u16(bytes, half);
        quantized = evoke_semantic_bmp_half_to_float(half);
    }
    else
    {
        uint8_t code = evoke_semantic_bmp_u8_code(impact);

        if (code == 0)
        {
            return EVOKE_ERR_RANGE;
        }
        bytes[0] = code;
        quantized = evoke_semantic_bmp_u8_decode(code);
    }
    if (!isfinite(quantized) || quantized == 0.0f)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (quantized_out != NULL)
    {
        *quantized_out = quantized;
    }
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float *impact_out
)
{
    size_t impact_width = evoke_semantic_bmp_impact_width(
        impact_precision
    );
    float impact;

    if (bytes == NULL || impact_out == NULL || impact_width == 0 ||
        size < impact_width)
    {
        return EVOKE_ERR_INVALID;
    }
    if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32)
    {
        impact = evoke_semantic_bmp_read_float(bytes);
    }
    else if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_FP16)
    {
        uint16_t half = evoke_semantic_bmp_read_u16(bytes);

        if ((half & UINT16_C(0x7fff)) == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        impact = evoke_semantic_bmp_half_to_float(half);
    }
    else
    {
        if (bytes[0] == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        impact = evoke_semantic_bmp_u8_decode(bytes[0]);
    }
    if (!isfinite(impact) || impact == 0.0f)
    {
        return EVOKE_ERR_FORMAT;
    }
    *impact_out = impact;
    return EVOKE_OK;
}

static int
evoke_semantic_bmp_compare_build_postings(
    const void *left,
    const void *right
)
{
    const evoke_semantic_bmp_build_posting *a = left;
    const evoke_semantic_bmp_build_posting *b = right;

    if (a->block_id != b->block_id)
    {
        return (a->block_id > b->block_id) -
            (a->block_id < b->block_id);
    }
    if (a->term_id != b->term_id)
    {
        return (a->term_id > b->term_id) -
            (a->term_id < b->term_id);
    }
    return (a->document_id > b->document_id) -
        (a->document_id < b->document_id);
}

static uint16_t
evoke_semantic_bmp_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] | (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_semantic_bmp_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_semantic_bmp_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (uint32_t index = 0; index < 8; index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static float
evoke_semantic_bmp_read_float(const uint8_t *bytes)
{
    uint32_t bits = evoke_semantic_bmp_read_u32(bytes);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
evoke_semantic_bmp_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_semantic_bmp_write_u32(uint8_t *bytes, uint32_t value)
{
    for (uint32_t index = 0; index < 4; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_bmp_write_u64(uint8_t *bytes, uint64_t value)
{
    for (uint32_t index = 0; index < 8; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_bmp_write_float(uint8_t *bytes, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    evoke_semantic_bmp_write_u32(bytes, bits);
}

static void
evoke_semantic_bmp_write_delta(
    uint8_t *bytes,
    size_t offset,
    uint32_t width,
    uint32_t value
)
{
    if (width == 1U)
    {
        bytes[offset] = (uint8_t) value;
    }
    else if (width == 2U)
    {
        evoke_semantic_bmp_write_u16(bytes + offset, (uint16_t) value);
    }
    else
    {
        evoke_semantic_bmp_write_u32(bytes + offset, value);
    }
}

static uint32_t
evoke_semantic_bmp_read_delta(
    const uint8_t *bytes,
    size_t offset,
    uint32_t width
)
{
    if (width == 1U)
    {
        return bytes[offset];
    }
    if (width == 2U)
    {
        return evoke_semantic_bmp_read_u16(bytes + offset);
    }
    return evoke_semantic_bmp_read_u32(bytes + offset);
}

static uint8_t
evoke_semantic_bmp_quantize_min(
    float value,
    float range_min,
    float range_max
)
{
    double normalized;

    if (range_min == range_max || value <= range_min)
    {
        return 0;
    }
    if (value >= range_max)
    {
        return UINT8_MAX;
    }
    normalized = ((double) value - range_min) /
        ((double) range_max - range_min) * UINT8_MAX;
    return (uint8_t) floor(normalized);
}

static uint8_t
evoke_semantic_bmp_quantize_max(
    float value,
    float range_min,
    float range_max
)
{
    double normalized;

    if (range_min == range_max || value <= range_min)
    {
        return 0;
    }
    if (value >= range_max)
    {
        return UINT8_MAX;
    }
    normalized = ((double) value - range_min) /
        ((double) range_max - range_min) * UINT8_MAX;
    return (uint8_t) ceil(normalized);
}

static float
evoke_semantic_bmp_dequantize_min(
    uint8_t quantized,
    float range_min,
    float range_max
)
{
    double value;

    if (range_min == range_max || quantized == 0)
    {
        return range_min;
    }
    if (quantized == UINT8_MAX)
    {
        return range_max;
    }
    value = (double) range_min +
        ((double) range_max - range_min) * quantized / UINT8_MAX;
    return nextafterf((float) value, -INFINITY);
}

static float
evoke_semantic_bmp_dequantize_max(
    uint8_t quantized,
    float range_min,
    float range_max
)
{
    double value;

    if (range_min == range_max || quantized == 0)
    {
        return range_min;
    }
    if (quantized == UINT8_MAX)
    {
        return range_max;
    }
    value = (double) range_min +
        ((double) range_max - range_min) * quantized / UINT8_MAX;
    return nextafterf((float) value, INFINITY);
}

static uint64_t
evoke_semantic_bmp_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value = index >= EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET &&
            index < EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET + sizeof(uint64_t)
            ? 0
            : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static bool
evoke_semantic_bmp_checked_add(
    size_t left,
    size_t right,
    size_t *result_out
)
{
    if (result_out == NULL || right > SIZE_MAX - left)
    {
        return false;
    }
    *result_out = left + right;
    return true;
}

static bool
evoke_semantic_bmp_checked_section(
    size_t offset,
    uint64_t count,
    size_t width,
    size_t *next_out
)
{
    if (count > SIZE_MAX / width)
    {
        return false;
    }
    return evoke_semantic_bmp_checked_add(
        offset,
        (size_t) count * width,
        next_out
    );
}

static evoke_status
evoke_semantic_bmp_layout_build(
    uint32_t term_count,
    uint32_t super_ref_count,
    uint32_t ref_count,
    uint32_t active_block_count,
    uint32_t record_count,
    uint64_t posting_count,
    evoke_semantic_bmp_layout *layout_out
)
{
    evoke_semantic_bmp_layout layout;

    if (layout_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&layout, 0, sizeof(layout));
    layout.terms_offset = EVOKE_SEMANTIC_BMP_HEADER_SIZE;
    if (!evoke_semantic_bmp_checked_section(
            layout.terms_offset,
            term_count,
            EVOKE_SEMANTIC_BMP_TERM_SIZE,
            &layout.super_refs_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.super_refs_offset,
            super_ref_count,
            EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE,
            &layout.refs_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.refs_offset,
            ref_count,
            EVOKE_SEMANTIC_BMP_REF_SIZE,
            &layout.blocks_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.blocks_offset,
            active_block_count,
            EVOKE_SEMANTIC_BMP_BLOCK_SIZE,
            &layout.records_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.records_offset,
            record_count,
            EVOKE_SEMANTIC_BMP_RECORD_SIZE,
            &layout.impacts_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.impacts_offset,
            posting_count,
            sizeof(uint32_t),
            &layout.total_size))
    {
        return EVOKE_ERR_RANGE;
    }
    *layout_out = layout;
    return EVOKE_OK;
}

static evoke_status
evoke_semantic_bmp_packed_layout_build(
    uint32_t term_count,
    uint32_t super_ref_bytes,
    uint32_t ref_bytes,
    uint32_t block_membership_bytes,
    uint32_t doc_delta_bytes,
    uint64_t posting_count,
    evoke_semantic_impact_precision impact_precision,
    evoke_semantic_bmp_packed_layout *layout_out
)
{
    evoke_semantic_bmp_packed_layout layout;
    size_t impact_width;

    if (layout_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&layout, 0, sizeof(layout));
    impact_width = evoke_semantic_bmp_impact_width(impact_precision);
    if (impact_width == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    layout.terms_offset = EVOKE_SEMANTIC_BMP_HEADER_SIZE;
    if (!evoke_semantic_bmp_checked_section(
            layout.terms_offset,
            term_count,
            EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE,
            &layout.super_refs_offset) ||
        !evoke_semantic_bmp_checked_add(
            layout.super_refs_offset,
            super_ref_bytes,
            &layout.refs_offset) ||
        !evoke_semantic_bmp_checked_add(
            layout.refs_offset,
            ref_bytes,
            &layout.block_membership_offset) ||
        !evoke_semantic_bmp_checked_add(
            layout.block_membership_offset,
            block_membership_bytes,
            &layout.doc_deltas_offset) ||
        !evoke_semantic_bmp_checked_add(
            layout.doc_deltas_offset,
            doc_delta_bytes,
            &layout.impacts_offset) ||
        !evoke_semantic_bmp_checked_section(
            layout.impacts_offset,
            posting_count,
            impact_width,
            &layout.total_size))
    {
        return EVOKE_ERR_RANGE;
    }
    *layout_out = layout;
    return EVOKE_OK;
}

static int
evoke_semantic_bmp_compare_build_refs(
    const void *left,
    const void *right
)
{
    const evoke_semantic_bmp_build_ref *a = left;
    const evoke_semantic_bmp_build_ref *b = right;

    if (a->term_id != b->term_id)
    {
        return (a->term_id > b->term_id) -
            (a->term_id < b->term_id);
    }
    return (a->block_id > b->block_id) -
        (a->block_id < b->block_id);
}

static int
evoke_semantic_bmp_compare_query_terms(
    const void *left,
    const void *right
)
{
    const evoke_semantic_bmp_query_term *a = left;
    const evoke_semantic_bmp_query_term *b = right;

    return (a->term_id > b->term_id) -
        (a->term_id < b->term_id);
}

static int
evoke_semantic_bmp_compare_ranked_blocks(
    const void *left,
    const void *right
)
{
    const evoke_semantic_bmp_ranked_block *a = left;
    const evoke_semantic_bmp_ranked_block *b = right;

    if (a->upper_bound < b->upper_bound)
    {
        return 1;
    }
    if (a->upper_bound > b->upper_bound)
    {
        return -1;
    }
    return (a->block_id > b->block_id) -
        (a->block_id < b->block_id);
}

static const evoke_semantic_bmp_term *
evoke_semantic_bmp_find_term(
    const evoke_semantic_bmp_index *index,
    uint32_t term_id
)
{
    uint32_t low = 0;
    uint32_t high = index->term_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;
        const evoke_semantic_bmp_term *term = &index->terms[middle];

        if (term->term_id < term_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < index->term_count && index->terms[low].term_id == term_id
        ? &index->terms[low]
        : NULL;
}

static const evoke_semantic_bmp_query_term *
evoke_semantic_bmp_find_query_term(
    const evoke_semantic_bmp_query_term *terms,
    size_t term_count,
    uint32_t term_id
)
{
    size_t low = 0;
    size_t high = term_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2;

        if (terms[middle].term_id < term_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < term_count && terms[low].term_id == term_id
        ? &terms[low]
        : NULL;
}

static const evoke_semantic_bmp_super_ref *
evoke_semantic_bmp_find_super_ref(
    const evoke_semantic_bmp_index *index,
    const evoke_semantic_bmp_term *term,
    uint32_t superblock_id
)
{
    uint32_t low = 0;
    uint32_t high = term->super_ref_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;
        const evoke_semantic_bmp_super_ref *ref =
            &index->super_refs[term->first_super_ref + middle];

        if (ref->superblock_id < superblock_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= term->super_ref_count)
    {
        return NULL;
    }
    return index->super_refs[
        term->first_super_ref + low
    ].superblock_id == superblock_id
        ? &index->super_refs[term->first_super_ref + low]
        : NULL;
}

static const evoke_semantic_bmp_block *
evoke_semantic_bmp_find_block(
    const evoke_semantic_bmp_index *index,
    uint32_t block_id
)
{
    uint32_t low = 0;
    uint32_t high = index->active_block_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;

        if (index->blocks[middle].block_id < block_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < index->active_block_count &&
        index->blocks[low].block_id == block_id
        ? &index->blocks[low]
        : NULL;
}

float
evoke_semantic_bmp_contribution_bound(
    float weight,
    float min_impact,
    float max_impact
)
{
    float first = weight * min_impact;
    float second = weight * max_impact;
    float bound = first > second ? first : second;

    return bound > 0.0f ? nextafterf(bound, INFINITY) : 0.0f;
}

void
evoke_semantic_bmp_index_init(evoke_semantic_bmp_index *index)
{
    if (index != NULL)
    {
        memset(index, 0, sizeof(*index));
    }
}

void
evoke_semantic_bmp_index_free(evoke_semantic_bmp_index *index)
{
    if (index == NULL)
    {
        return;
    }
    free(index->terms);
    free(index->super_refs);
    free(index->refs);
    free(index->blocks);
    free(index->records);
    free(index->impacts);
    evoke_semantic_bmp_index_init(index);
}

static evoke_status
evoke_semantic_bmp_build_super_refs(evoke_semantic_bmp_index *index)
{
    uint32_t super_ref_count = 0;
    uint32_t super_ref_cursor = 0;

    if (index == NULL || index->block_shift >=
            EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT ||
        index->ref_count == 0 || index->refs == NULL ||
        index->term_count == 0 || index->terms == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    index->superblock_shift = EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT;
    index->superblock_count = (index->document_count +
        (UINT32_C(1) << index->superblock_shift) - UINT32_C(1)) >>
        index->superblock_shift;
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index->terms[term_index];
        uint32_t previous_superblock = UINT32_MAX;

        for (uint32_t offset = 0; offset < term->ref_count; offset++)
        {
            uint32_t superblock_id =
                index->refs[term->first_ref + offset].block_id >>
                (index->superblock_shift - index->block_shift);

            if (superblock_id != previous_superblock)
            {
                if (super_ref_count == UINT32_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                super_ref_count++;
                previous_superblock = superblock_id;
            }
        }
    }
    index->super_refs = calloc(
        super_ref_count,
        sizeof(*index->super_refs)
    );
    if (super_ref_count > 0 && index->super_refs == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    index->super_ref_count = super_ref_count;
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        evoke_semantic_bmp_term *term = &index->terms[term_index];
        uint32_t offset = 0;

        term->first_super_ref = super_ref_cursor;
        while (offset < term->ref_count)
        {
            uint32_t first_offset = offset;
            const evoke_semantic_bmp_ref *first =
                &index->refs[term->first_ref + offset];
            uint32_t superblock_id = first->block_id >>
                (index->superblock_shift - index->block_shift);
            float min_impact = first->min_impact;
            float max_impact = first->max_impact;
            evoke_semantic_bmp_super_ref *super_ref;

            offset++;
            while (offset < term->ref_count)
            {
                const evoke_semantic_bmp_ref *ref =
                    &index->refs[term->first_ref + offset];
                uint32_t current_superblock = ref->block_id >>
                    (index->superblock_shift - index->block_shift);

                if (current_superblock != superblock_id)
                {
                    break;
                }
                if (ref->min_impact < min_impact)
                {
                    min_impact = ref->min_impact;
                }
                if (ref->max_impact > max_impact)
                {
                    max_impact = ref->max_impact;
                }
                offset++;
            }
            if (offset - first_offset > UINT16_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            super_ref = &index->super_refs[super_ref_cursor++];
            super_ref->superblock_id = superblock_id;
            super_ref->first_ref = term->first_ref + first_offset;
            super_ref->ref_count = (uint16_t) (offset - first_offset);
            super_ref->min_impact = min_impact;
            super_ref->max_impact = max_impact;
            term->super_ref_count++;
        }
    }
    return super_ref_cursor == index->super_ref_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_semantic_bmp_index_build(
    uint32_t document_count,
    const evoke_semantic_bmp_posting *postings,
    size_t posting_count,
    evoke_semantic_bmp_index *index_out
)
{
    evoke_semantic_bmp_index index;
    evoke_semantic_bmp_build_posting *sorted = NULL;
    evoke_semantic_bmp_build_ref *build_refs = NULL;
    uint32_t block_count;
    uint32_t record_count = 0;
    uint32_t active_block_count = 0;
    uint32_t term_count = 0;
    uint32_t record_index = 0;
    uint32_t term_index = 0;
    size_t cursor = 0;
    evoke_status status = EVOKE_OK;

    if (index_out == NULL || document_count == 0 ||
        (posting_count > 0 && postings == NULL) ||
        posting_count > UINT32_MAX ||
        posting_count > SIZE_MAX / sizeof(*sorted))
    {
        return EVOKE_ERR_INVALID;
    }
    block_count = (document_count +
        (UINT32_C(1) << EVOKE_SEMANTIC_BMP_BLOCK_SHIFT) - 1) >>
        EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
    evoke_semantic_bmp_index_init(&index);
    index.document_count = document_count;
    index.block_shift = EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
    index.block_count = block_count;
    index.posting_count = posting_count;

    if (posting_count > 0)
    {
        sorted = malloc(posting_count * sizeof(*sorted));
        build_refs = malloc(posting_count * sizeof(*build_refs));
        index.records = calloc(posting_count, sizeof(*index.records));
        index.refs = calloc(posting_count, sizeof(*index.refs));
        index.impacts = malloc(posting_count * sizeof(*index.impacts));
    }
    if (posting_count > 0)
    {
        index.blocks = calloc(posting_count, sizeof(*index.blocks));
    }
    if ((posting_count > 0 &&
         (sorted == NULL || build_refs == NULL || index.records == NULL ||
          index.refs == NULL || index.impacts == NULL)) ||
        (posting_count > 0 && index.blocks == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (size_t posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        if (postings[posting_index].document_id >= document_count ||
            !isfinite(postings[posting_index].impact) ||
            postings[posting_index].impact == 0.0f)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        sorted[posting_index].term_id = postings[posting_index].term_id;
        sorted[posting_index].document_id =
            postings[posting_index].document_id;
        sorted[posting_index].block_id =
            postings[posting_index].document_id >>
            EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
        sorted[posting_index].impact = postings[posting_index].impact;
    }
    qsort(
        sorted,
        posting_count,
        sizeof(*sorted),
        evoke_semantic_bmp_compare_build_postings
    );

    while (cursor < posting_count)
    {
        size_t start = cursor;
        uint32_t block_id = sorted[cursor].block_id;
        uint32_t term_id = sorted[cursor].term_id;
        uint16_t document_mask = 0;
        float min_impact = sorted[cursor].impact;
        float max_impact = sorted[cursor].impact;

        while (cursor < posting_count &&
               sorted[cursor].block_id == block_id &&
               sorted[cursor].term_id == term_id)
        {
            uint32_t local_document = sorted[cursor].document_id &
                ((UINT32_C(1) << EVOKE_SEMANTIC_BMP_BLOCK_SHIFT) - 1);
            uint16_t bit = (uint16_t) (UINT16_C(1) << local_document);

            if ((document_mask & bit) != 0)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            document_mask |= bit;
            if (sorted[cursor].impact < min_impact)
            {
                min_impact = sorted[cursor].impact;
            }
            if (sorted[cursor].impact > max_impact)
            {
                max_impact = sorted[cursor].impact;
            }
            index.impacts[cursor] = sorted[cursor].impact;
            cursor++;
        }
        index.records[record_index].term_id = term_id;
        index.records[record_index].first_impact = (uint32_t) start;
        index.records[record_index].document_mask = document_mask;
        index.records[record_index].impact_count =
            (uint16_t) (cursor - start);
        build_refs[record_index].term_id = term_id;
        build_refs[record_index].block_id = block_id;
        build_refs[record_index].record_index = record_index;
        build_refs[record_index].min_impact = min_impact;
        build_refs[record_index].max_impact = max_impact;
        if (active_block_count == 0 ||
            index.blocks[active_block_count - 1].block_id != block_id)
        {
            index.blocks[active_block_count].block_id = block_id;
            index.blocks[active_block_count].first_record = record_index;
            active_block_count++;
        }
        index.blocks[active_block_count - 1].record_count++;
        record_index++;
    }
    record_count = record_index;
    index.active_block_count = active_block_count;
    index.record_count = record_count;
    index.ref_count = record_count;

    qsort(
        build_refs,
        record_count,
        sizeof(*build_refs),
        evoke_semantic_bmp_compare_build_refs
    );
    for (uint32_t ref_index = 0; ref_index < record_count; ref_index++)
    {
        if (ref_index == 0 ||
            build_refs[ref_index - 1].term_id !=
                build_refs[ref_index].term_id)
        {
            term_count++;
        }
    }
    index.terms = calloc(term_count, sizeof(*index.terms));
    if (term_count > 0 && index.terms == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    index.term_count = term_count;
    for (uint32_t ref_index = 0; ref_index < record_count; ref_index++)
    {
        const evoke_semantic_bmp_build_ref *source =
            &build_refs[ref_index];
        evoke_semantic_bmp_term *term;

        if (ref_index == 0 ||
            build_refs[ref_index - 1].term_id != source->term_id)
        {
            term = &index.terms[term_index++];
            term->term_id = source->term_id;
            term->first_ref = ref_index;
            term->min_impact = source->min_impact;
            term->max_impact = source->max_impact;
        }
        else
        {
            term = &index.terms[term_index - 1];
            if (source->min_impact < term->min_impact)
            {
                term->min_impact = source->min_impact;
            }
            if (source->max_impact > term->max_impact)
            {
                term->max_impact = source->max_impact;
            }
        }
        term->ref_count++;
        index.refs[ref_index].block_id = source->block_id;
        index.refs[ref_index].record_index = source->record_index;
        index.refs[ref_index].min_impact = source->min_impact;
        index.refs[ref_index].max_impact = source->max_impact;
    }

    status = evoke_semantic_bmp_build_super_refs(&index);
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_bmp_index_validate(&index);
    }
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_index_init(&index);
    }

cleanup:
    free(sorted);
    free(build_refs);
    evoke_semantic_bmp_index_free(&index);
    return status;
}

static evoke_status
evoke_semantic_bmp_run_document_id(
    const evoke_semantic_bmp_run *run,
    uint64_t posting_index,
    uint32_t document_count,
    uint32_t *document_id_out
)
{
    uint32_t local_document_id;
    uint64_t global_document_id;

    if (run == NULL || document_id_out == NULL ||
        posting_index >= run->posting_count)
    {
        return EVOKE_ERR_INVALID;
    }
    local_document_id = run->local_document_ids[posting_index];
    if (local_document_id >= run->local_document_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    global_document_id = run->document_id_map == NULL
        ? (uint64_t) run->document_id_base + local_document_id
        : run->document_id_map[local_document_id];
    if (global_document_id >= document_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *document_id_out = (uint32_t) global_document_id;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_index_build_runs(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_bmp_index *index_out
)
{
    evoke_semantic_bmp_index index;
    uint32_t *block_record_counts = NULL;
    uint32_t *block_posting_counts = NULL;
    uint32_t *block_record_cursors = NULL;
    uint32_t *block_posting_cursors = NULL;
    uint32_t block_count;
    uint32_t active_block_count = 0;
    uint32_t record_count = 0;
    uint64_t posting_count = 0;
    uint32_t record_prefix = 0;
    uint32_t posting_prefix = 0;
    uint32_t active_block_index = 0;
    uint32_t ref_cursor = 0;
    evoke_status status = EVOKE_OK;

    if (document_count == 0 || index_out == NULL ||
        (run_count > 0 && runs == NULL) || run_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    block_count = (document_count +
        (UINT32_C(1) << EVOKE_SEMANTIC_BMP_BLOCK_SHIFT) - 1) >>
        EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
    evoke_semantic_bmp_index_init(&index);
    index.document_count = document_count;
    index.block_shift = EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
    index.block_count = block_count;

    block_record_counts = calloc(
        block_count,
        sizeof(*block_record_counts)
    );
    block_posting_counts = calloc(
        block_count,
        sizeof(*block_posting_counts)
    );
    block_record_cursors = calloc(
        block_count,
        sizeof(*block_record_cursors)
    );
    block_posting_cursors = calloc(
        block_count,
        sizeof(*block_posting_cursors)
    );
    if (block_record_counts == NULL || block_posting_counts == NULL ||
        block_record_cursors == NULL || block_posting_cursors == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (size_t run_index = 0; run_index < run_count; run_index++)
    {
        const evoke_semantic_bmp_run *run = &runs[run_index];
        uint32_t previous_document_id = 0;
        uint32_t previous_block_id = UINT32_MAX;

        if (run->posting_count == 0 ||
            run->local_document_ids == NULL || run->values == NULL ||
            run->local_document_count == 0 ||
            (run_index > 0 && runs[run_index - 1].term_id >= run->term_id) ||
            run->posting_count > UINT32_MAX - posting_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        for (uint64_t posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            uint32_t document_id;
            uint32_t block_id;
            float impact = run->values[posting_index].impact;

            status = evoke_semantic_bmp_run_document_id(
                run,
                posting_index,
                document_count,
                &document_id
            );
            if (status != EVOKE_OK || !isfinite(impact) || impact == 0.0f ||
                (posting_index > 0 && document_id <= previous_document_id))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            block_id = document_id >> EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
            if (block_posting_counts[block_id] == UINT32_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            block_posting_counts[block_id]++;
            if (block_id != previous_block_id)
            {
                if (block_record_counts[block_id] == UINT32_MAX ||
                    record_count == UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                block_record_counts[block_id]++;
                record_count++;
                previous_block_id = block_id;
            }
            previous_document_id = document_id;
        }
        posting_count += run->posting_count;
    }
    for (uint32_t block_id = 0; block_id < block_count; block_id++)
    {
        if (block_record_counts[block_id] > 0)
        {
            active_block_count++;
        }
    }
    index.active_block_count = active_block_count;
    index.term_count = (uint32_t) run_count;
    index.ref_count = record_count;
    index.record_count = record_count;
    index.posting_count = posting_count;
    index.terms = calloc(index.term_count, sizeof(*index.terms));
    index.refs = calloc(index.ref_count, sizeof(*index.refs));
    index.blocks = calloc(
        index.active_block_count,
        sizeof(*index.blocks)
    );
    index.records = calloc(index.record_count, sizeof(*index.records));
    index.impacts = malloc(
        (size_t) index.posting_count * sizeof(*index.impacts)
    );
    if ((index.term_count > 0 && index.terms == NULL) ||
        (index.ref_count > 0 && index.refs == NULL) ||
        (index.active_block_count > 0 && index.blocks == NULL) ||
        (index.record_count > 0 && index.records == NULL) ||
        (index.posting_count > 0 && index.impacts == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t block_id = 0; block_id < block_count; block_id++)
    {
        block_record_cursors[block_id] = record_prefix;
        block_posting_cursors[block_id] = posting_prefix;
        if (block_record_counts[block_id] > 0)
        {
            evoke_semantic_bmp_block *block =
                &index.blocks[active_block_index++];

            block->block_id = block_id;
            block->first_record = record_prefix;
            block->record_count = block_record_counts[block_id];
        }
        record_prefix += block_record_counts[block_id];
        posting_prefix += block_posting_counts[block_id];
    }
    if (record_prefix != record_count || posting_prefix != posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }

    for (size_t run_index = 0; run_index < run_count; run_index++)
    {
        const evoke_semantic_bmp_run *run = &runs[run_index];
        evoke_semantic_bmp_term *term = &index.terms[run_index];
        uint64_t posting_index = 0;

        term->term_id = run->term_id;
        term->first_ref = ref_cursor;
        while (posting_index < run->posting_count)
        {
            uint32_t document_id;
            uint32_t block_id;
            uint32_t record_index;
            uint32_t first_impact;
            uint16_t document_mask = 0;
            uint16_t impact_count = 0;
            float min_impact;
            float max_impact;

            status = evoke_semantic_bmp_run_document_id(
                run,
                posting_index,
                document_count,
                &document_id
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            block_id = document_id >> EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
            record_index = block_record_cursors[block_id]++;
            first_impact = block_posting_cursors[block_id];
            min_impact = run->values[posting_index].impact;
            max_impact = min_impact;
            while (posting_index < run->posting_count)
            {
                uint32_t current_document_id;
                uint32_t current_block_id;
                uint32_t local_document_id;
                float impact;

                status = evoke_semantic_bmp_run_document_id(
                    run,
                    posting_index,
                    document_count,
                    &current_document_id
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                current_block_id = current_document_id >>
                    EVOKE_SEMANTIC_BMP_BLOCK_SHIFT;
                if (current_block_id != block_id)
                {
                    break;
                }
                local_document_id = current_document_id &
                    ((UINT32_C(1) << EVOKE_SEMANTIC_BMP_BLOCK_SHIFT) - 1);
                impact = run->values[posting_index].impact;
                document_mask |= (uint16_t)
                    (UINT16_C(1) << local_document_id);
                index.impacts[block_posting_cursors[block_id]++] = impact;
                if (impact < min_impact)
                {
                    min_impact = impact;
                }
                if (impact > max_impact)
                {
                    max_impact = impact;
                }
                impact_count++;
                posting_index++;
            }
            index.records[record_index].term_id = run->term_id;
            index.records[record_index].first_impact = first_impact;
            index.records[record_index].document_mask = document_mask;
            index.records[record_index].impact_count = impact_count;
            index.refs[ref_cursor].block_id = block_id;
            index.refs[ref_cursor].record_index = record_index;
            index.refs[ref_cursor].min_impact = min_impact;
            index.refs[ref_cursor].max_impact = max_impact;
            if (term->ref_count == 0)
            {
                term->min_impact = min_impact;
                term->max_impact = max_impact;
            }
            else
            {
                if (min_impact < term->min_impact)
                {
                    term->min_impact = min_impact;
                }
                if (max_impact > term->max_impact)
                {
                    term->max_impact = max_impact;
                }
            }
            ref_cursor++;
            term->ref_count++;
        }
    }
    status = evoke_semantic_bmp_build_super_refs(&index);
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_bmp_index_validate(&index);
    }
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_index_init(&index);
    }

cleanup:
    free(block_record_counts);
    free(block_posting_counts);
    free(block_record_cursors);
    free(block_posting_cursors);
    evoke_semantic_bmp_index_free(&index);
    return status;
}

evoke_status
evoke_semantic_bmp_index_validate(const evoke_semantic_bmp_index *index)
{
    uint64_t posting_cursor = 0;
    uint32_t record_cursor = 0;
    uint32_t ref_cursor = 0;
    uint32_t super_ref_cursor = 0;

    if (index == NULL || index->document_count == 0 ||
        index->block_shift != EVOKE_SEMANTIC_BMP_BLOCK_SHIFT ||
        index->block_count !=
            (index->document_count +
             (UINT32_C(1) << index->block_shift) - 1) >>
                index->block_shift ||
        index->superblock_shift !=
            EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT ||
        index->superblock_count !=
            (index->document_count +
             (UINT32_C(1) << index->superblock_shift) - 1) >>
                index->superblock_shift ||
        (index->term_count > 0 && index->terms == NULL) ||
        (index->super_ref_count > 0 && index->super_refs == NULL) ||
        (index->ref_count > 0 && index->refs == NULL) ||
        (index->active_block_count > 0 && index->blocks == NULL) ||
        (index->record_count > 0 && index->records == NULL) ||
        (index->posting_count > 0 && index->impacts == NULL) ||
        index->active_block_count > index->block_count ||
        index->ref_count != index->record_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index->terms[term_index];

        if (term->ref_count == 0 ||
            term->first_ref != ref_cursor ||
            term->ref_count > index->ref_count - term->first_ref ||
            term->super_ref_count == 0 ||
            term->first_super_ref != super_ref_cursor ||
            term->super_ref_count >
                index->super_ref_count - term->first_super_ref ||
            !isfinite(term->min_impact) ||
            !isfinite(term->max_impact) ||
            term->min_impact > term->max_impact ||
            (term_index > 0 &&
             index->terms[term_index - 1].term_id >= term->term_id))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t offset = 0; offset < term->ref_count; offset++)
        {
            const evoke_semantic_bmp_ref *ref =
                &index->refs[term->first_ref + offset];
            const evoke_semantic_bmp_block *block;

            if (ref->block_id >= index->block_count ||
                ref->record_index >= index->record_count ||
                !isfinite(ref->min_impact) ||
                !isfinite(ref->max_impact) ||
                ref->min_impact > ref->max_impact ||
                (offset > 0 &&
                 index->refs[term->first_ref + offset - 1].block_id >=
                    ref->block_id))
            {
                return EVOKE_ERR_FORMAT;
            }
            block = evoke_semantic_bmp_find_block(index, ref->block_id);
            if (block == NULL ||
                ref->record_index < block->first_record ||
                ref->record_index >=
                    block->first_record + block->record_count ||
                index->records[ref->record_index].term_id != term->term_id)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t offset = 0;
             offset < term->super_ref_count;
             offset++)
        {
            const evoke_semantic_bmp_super_ref *super_ref =
                &index->super_refs[term->first_super_ref + offset];
            uint32_t first_term_ref = super_ref->first_ref -
                term->first_ref;

            if (super_ref->superblock_id >= index->superblock_count ||
                super_ref->ref_count == 0 ||
                super_ref->reserved != 0 ||
                super_ref->first_ref < term->first_ref ||
                first_term_ref > term->ref_count ||
                super_ref->ref_count > term->ref_count - first_term_ref ||
                !isfinite(super_ref->min_impact) ||
                !isfinite(super_ref->max_impact) ||
                super_ref->min_impact > super_ref->max_impact ||
                (offset > 0 &&
                 index->super_refs[
                     term->first_super_ref + offset - 1
                 ].superblock_id >= super_ref->superblock_id))
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint16_t child = 0;
                 child < super_ref->ref_count;
                 child++)
            {
                const evoke_semantic_bmp_ref *ref =
                    &index->refs[super_ref->first_ref + child];

                if ((ref->block_id >>
                     (index->superblock_shift - index->block_shift)) !=
                    super_ref->superblock_id)
                {
                    return EVOKE_ERR_FORMAT;
                }
            }
        }
        ref_cursor += term->ref_count;
        super_ref_cursor += term->super_ref_count;
    }
    for (uint32_t block_index = 0;
         block_index < index->active_block_count;
         block_index++)
    {
        const evoke_semantic_bmp_block *block =
            &index->blocks[block_index];

        if (block->block_id >= index->block_count ||
            block->record_count == 0 ||
            block->first_record != record_cursor ||
            block->record_count > index->record_count - record_cursor)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (block_index > 0 &&
            index->blocks[block_index - 1].block_id >= block->block_id)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t offset = 0; offset < block->record_count; offset++)
        {
            const evoke_semantic_bmp_record *record =
                &index->records[record_cursor + offset];
            uint16_t mask = record->document_mask;
            uint16_t bit_count = 0;

            while (mask != 0)
            {
                bit_count += mask & UINT16_C(1);
                mask >>= 1;
            }
            if (record->impact_count == 0 ||
                bit_count != record->impact_count ||
                record->first_impact != posting_cursor ||
                record->impact_count >
                    index->posting_count - posting_cursor ||
                (offset > 0 &&
                 index->records[record_cursor + offset - 1].term_id >=
                    record->term_id))
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint16_t impact_index = 0;
                 impact_index < record->impact_count;
                 impact_index++)
            {
                if (!isfinite(index->impacts[posting_cursor + impact_index]) ||
                    index->impacts[posting_cursor + impact_index] == 0.0f)
                {
                    return EVOKE_ERR_FORMAT;
                }
            }
            posting_cursor += record->impact_count;
        }
        record_cursor += block->record_count;
    }
    return ref_cursor == index->ref_count &&
        super_ref_cursor == index->super_ref_count &&
        record_cursor == index->record_count &&
        posting_cursor == index->posting_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_semantic_bmp_serialized_size(
    const evoke_semantic_bmp_index *index,
    size_t *size_out
)
{
    evoke_semantic_bmp_layout layout;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    status = evoke_semantic_bmp_index_validate(index);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_semantic_bmp_layout_build(
        index->term_count,
        index->super_ref_count,
        index->ref_count,
        index->active_block_count,
        index->record_count,
        index->posting_count,
        &layout
    );
    if (status == EVOKE_OK)
    {
        *size_out = layout.total_size;
    }
    return status;
}

evoke_status
evoke_semantic_bmp_serialize_into(
    const evoke_semantic_bmp_index *index,
    uint8_t *bytes,
    size_t size
)
{
    evoke_semantic_bmp_layout layout;
    evoke_status status;

    if (bytes == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_bmp_index_validate(index);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_semantic_bmp_layout_build(
        index->term_count,
        index->super_ref_count,
        index->ref_count,
        index->active_block_count,
        index->record_count,
        index->posting_count,
        &layout
    );
    if (status != EVOKE_OK || size != layout.total_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    memset(bytes, 0, size);

    evoke_semantic_bmp_write_u32(bytes + 0, EVOKE_SEMANTIC_BMP_MAGIC);
    evoke_semantic_bmp_write_u16(bytes + 4, EVOKE_SEMANTIC_BMP_VERSION);
    evoke_semantic_bmp_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_BMP_HEADER_SIZE
    );
    evoke_semantic_bmp_write_u32(bytes + 8, index->document_count);
    evoke_semantic_bmp_write_u32(bytes + 12, index->block_shift);
    evoke_semantic_bmp_write_u32(bytes + 16, index->block_count);
    evoke_semantic_bmp_write_u32(
        bytes + 20,
        index->active_block_count
    );
    evoke_semantic_bmp_write_u32(bytes + 24, index->term_count);
    evoke_semantic_bmp_write_u32(bytes + 28, index->ref_count);
    evoke_semantic_bmp_write_u32(bytes + 32, index->record_count);
    evoke_semantic_bmp_write_u32(bytes + 36, index->superblock_shift);
    evoke_semantic_bmp_write_u64(bytes + 40, index->posting_count);
    evoke_semantic_bmp_write_u32(bytes + 48, index->superblock_count);
    evoke_semantic_bmp_write_u32(bytes + 52, index->super_ref_count);
    evoke_semantic_bmp_write_u64(bytes + 56, layout.terms_offset);
    evoke_semantic_bmp_write_u64(bytes + 64, layout.super_refs_offset);
    evoke_semantic_bmp_write_u64(bytes + 72, layout.refs_offset);
    evoke_semantic_bmp_write_u64(bytes + 80, layout.blocks_offset);
    evoke_semantic_bmp_write_u64(bytes + 88, layout.records_offset);
    evoke_semantic_bmp_write_u64(bytes + 96, layout.impacts_offset);
    evoke_semantic_bmp_write_u64(bytes + 104, layout.total_size);

    for (uint32_t index_position = 0;
         index_position < index->term_count;
         index_position++)
    {
        const evoke_semantic_bmp_term *term =
            &index->terms[index_position];
        uint8_t *destination = bytes + layout.terms_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_TERM_SIZE;

        evoke_semantic_bmp_write_u32(destination + 0, term->term_id);
        evoke_semantic_bmp_write_u32(destination + 4, term->first_ref);
        evoke_semantic_bmp_write_u32(destination + 8, term->ref_count);
        evoke_semantic_bmp_write_u32(
            destination + 12,
            term->first_super_ref
        );
        evoke_semantic_bmp_write_u32(
            destination + 16,
            term->super_ref_count
        );
        evoke_semantic_bmp_write_float(
            destination + 20,
            term->min_impact
        );
        evoke_semantic_bmp_write_float(
            destination + 24,
            term->max_impact
        );
    }
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index->terms[term_index];

        for (uint32_t offset = 0;
             offset < term->super_ref_count;
             offset++)
        {
            uint32_t index_position = term->first_super_ref + offset;
            const evoke_semantic_bmp_super_ref *ref =
                &index->super_refs[index_position];
            uint8_t *destination = bytes + layout.super_refs_offset +
                (size_t) index_position *
                    EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE;

            evoke_semantic_bmp_write_u32(
                destination + 0,
                ref->superblock_id
            );
            evoke_semantic_bmp_write_u32(
                destination + 4,
                ref->first_ref
            );
            evoke_semantic_bmp_write_u16(
                destination + 8,
                ref->ref_count
            );
            destination[10] = evoke_semantic_bmp_quantize_min(
                ref->min_impact,
                term->min_impact,
                term->max_impact
            );
            destination[11] = evoke_semantic_bmp_quantize_max(
                ref->max_impact,
                term->min_impact,
                term->max_impact
            );
        }
    }
    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index->terms[term_index];

        for (uint32_t offset = 0; offset < term->ref_count; offset++)
        {
            uint32_t index_position = term->first_ref + offset;
            const evoke_semantic_bmp_ref *ref =
                &index->refs[index_position];
            uint8_t *destination = bytes + layout.refs_offset +
                (size_t) index_position * EVOKE_SEMANTIC_BMP_REF_SIZE;

            evoke_semantic_bmp_write_u32(
                destination + 0,
                ref->block_id
            );
            destination[4] = evoke_semantic_bmp_quantize_min(
                ref->min_impact,
                term->min_impact,
                term->max_impact
            );
            destination[5] = evoke_semantic_bmp_quantize_max(
                ref->max_impact,
                term->min_impact,
                term->max_impact
            );
        }
    }
    for (uint32_t index_position = 0;
         index_position < index->active_block_count;
         index_position++)
    {
        const evoke_semantic_bmp_block *block =
            &index->blocks[index_position];
        uint8_t *destination = bytes + layout.blocks_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_BLOCK_SIZE;

        evoke_semantic_bmp_write_u32(destination + 0, block->block_id);
        evoke_semantic_bmp_write_u32(
            destination + 4,
            block->first_record
        );
        evoke_semantic_bmp_write_u32(
            destination + 8,
            block->record_count
        );
    }
    for (uint32_t index_position = 0;
         index_position < index->record_count;
         index_position++)
    {
        const evoke_semantic_bmp_record *record =
            &index->records[index_position];
        uint8_t *destination = bytes + layout.records_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_RECORD_SIZE;

        evoke_semantic_bmp_write_u32(destination + 0, record->term_id);
        evoke_semantic_bmp_write_u32(
            destination + 4,
            record->first_impact
        );
        evoke_semantic_bmp_write_u16(
            destination + 8,
            record->document_mask
        );
        evoke_semantic_bmp_write_u16(
            destination + 10,
            record->impact_count
        );
    }
    for (uint64_t index_position = 0;
         index_position < index->posting_count;
         index_position++)
    {
        evoke_semantic_bmp_write_float(
            bytes + layout.impacts_offset +
                (size_t) index_position * sizeof(uint32_t),
            index->impacts[index_position]
        );
    }
    evoke_semantic_bmp_write_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET,
        evoke_semantic_bmp_checksum(bytes, layout.total_size)
    );
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_serialize(
    const evoke_semantic_bmp_index *index,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t size;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_semantic_bmp_serialized_size(index, &size);
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = malloc(size);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    status = evoke_semantic_bmp_serialize_into(index, bytes, size);
    if (status != EVOKE_OK)
    {
        free(bytes);
        return status;
    }
    *bytes_out = bytes;
    *size_out = size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_disk_header *header_out
)
{
    evoke_semantic_bmp_disk_header header;
    evoke_semantic_bmp_layout layout;
    evoke_status status;

    if (bytes == NULL || header_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_BMP_MAGIC ||
        evoke_semantic_bmp_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_BMP_VERSION ||
        evoke_semantic_bmp_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_BMP_HEADER_SIZE ||
        evoke_semantic_bmp_read_u32(bytes + 12) !=
            EVOKE_SEMANTIC_BMP_BLOCK_SHIFT ||
        evoke_semantic_bmp_read_u32(bytes + 36) !=
            EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT ||
        evoke_semantic_bmp_read_u64(bytes + 120) != 0 ||
        evoke_semantic_bmp_read_u64(bytes + 128) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&header, 0, sizeof(header));
    header.document_count = evoke_semantic_bmp_read_u32(bytes + 8);
    header.block_shift = evoke_semantic_bmp_read_u32(bytes + 12);
    header.block_count = evoke_semantic_bmp_read_u32(bytes + 16);
    header.active_block_count = evoke_semantic_bmp_read_u32(bytes + 20);
    header.term_count = evoke_semantic_bmp_read_u32(bytes + 24);
    header.ref_count = evoke_semantic_bmp_read_u32(bytes + 28);
    header.record_count = evoke_semantic_bmp_read_u32(bytes + 32);
    header.superblock_shift = evoke_semantic_bmp_read_u32(bytes + 36);
    header.posting_count = evoke_semantic_bmp_read_u64(bytes + 40);
    header.superblock_count = evoke_semantic_bmp_read_u32(bytes + 48);
    header.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 52);
    header.terms_offset = evoke_semantic_bmp_read_u64(bytes + 56);
    header.super_refs_offset = evoke_semantic_bmp_read_u64(bytes + 64);
    header.refs_offset = evoke_semantic_bmp_read_u64(bytes + 72);
    header.blocks_offset = evoke_semantic_bmp_read_u64(bytes + 80);
    header.records_offset = evoke_semantic_bmp_read_u64(bytes + 88);
    header.impacts_offset = evoke_semantic_bmp_read_u64(bytes + 96);
    header.total_size = evoke_semantic_bmp_read_u64(bytes + 104);
    header.checksum = evoke_semantic_bmp_read_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET
    );
    if (header.document_count == 0 || header.block_count == 0 ||
        header.superblock_count == 0 ||
        header.active_block_count == 0 || header.term_count == 0 ||
        header.super_ref_count == 0 || header.ref_count == 0 ||
        header.record_count == 0 ||
        header.posting_count == 0 || header.checksum == 0 ||
        header.block_count !=
            ((header.document_count - UINT32_C(1)) >>
             header.block_shift) + UINT32_C(1) ||
        header.superblock_count !=
            ((header.document_count - UINT32_C(1)) >>
             header.superblock_shift) + UINT32_C(1) ||
        header.active_block_count > header.block_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_semantic_bmp_layout_build(
        header.term_count,
        header.super_ref_count,
        header.ref_count,
        header.active_block_count,
        header.record_count,
        header.posting_count,
        &layout
    );
    if (status != EVOKE_OK ||
        header.terms_offset != layout.terms_offset ||
        header.super_refs_offset != layout.super_refs_offset ||
        header.refs_offset != layout.refs_offset ||
        header.blocks_offset != layout.blocks_offset ||
        header.records_offset != layout.records_offset ||
        header.impacts_offset != layout.impacts_offset ||
        header.total_size != layout.total_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_term_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_term *term_out
)
{
    evoke_semantic_bmp_term term;

    if (bytes == NULL || term_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_TERM_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&term, 0, sizeof(term));
    term.term_id = evoke_semantic_bmp_read_u32(bytes + 0);
    term.first_ref = evoke_semantic_bmp_read_u32(bytes + 4);
    term.ref_count = evoke_semantic_bmp_read_u32(bytes + 8);
    term.first_super_ref = evoke_semantic_bmp_read_u32(bytes + 12);
    term.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 16);
    term.min_impact = evoke_semantic_bmp_read_float(bytes + 20);
    term.max_impact = evoke_semantic_bmp_read_float(bytes + 24);
    if (term.ref_count == 0 || term.super_ref_count == 0 ||
        !isfinite(term.min_impact) ||
        !isfinite(term.max_impact) ||
        term.min_impact > term.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *term_out = term;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_super_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_super_ref *ref_out
)
{
    evoke_semantic_bmp_super_ref ref;

    if (bytes == NULL || ref_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE ||
        !isfinite(term_min_impact) || !isfinite(term_max_impact) ||
        term_min_impact > term_max_impact)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u32(bytes + 12) != 0 ||
        bytes[10] > bytes[11])
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&ref, 0, sizeof(ref));
    ref.superblock_id = evoke_semantic_bmp_read_u32(bytes + 0);
    ref.first_ref = evoke_semantic_bmp_read_u32(bytes + 4);
    ref.ref_count = evoke_semantic_bmp_read_u16(bytes + 8);
    ref.min_impact = evoke_semantic_bmp_dequantize_min(
        bytes[10],
        term_min_impact,
        term_max_impact
    );
    ref.max_impact = evoke_semantic_bmp_dequantize_max(
        bytes[11],
        term_min_impact,
        term_max_impact
    );
    if (ref.ref_count == 0 || !isfinite(ref.min_impact) ||
        !isfinite(ref.max_impact) || ref.min_impact > ref.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_ref *ref_out
)
{
    evoke_semantic_bmp_ref ref;

    if (bytes == NULL || ref_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_REF_SIZE ||
        !isfinite(term_min_impact) || !isfinite(term_max_impact) ||
        term_min_impact > term_max_impact)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u16(bytes + 6) != 0 ||
        bytes[4] > bytes[5])
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&ref, 0, sizeof(ref));
    ref.block_id = evoke_semantic_bmp_read_u32(bytes + 0);
    ref.record_index = UINT32_MAX;
    ref.min_impact = evoke_semantic_bmp_dequantize_min(
        bytes[4],
        term_min_impact,
        term_max_impact
    );
    ref.max_impact = evoke_semantic_bmp_dequantize_max(
        bytes[5],
        term_min_impact,
        term_max_impact
    );
    if (!isfinite(ref.min_impact) || !isfinite(ref.max_impact) ||
        ref.min_impact > ref.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_block_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_block *block_out
)
{
    evoke_semantic_bmp_block block;

    if (bytes == NULL || block_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_BLOCK_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    block.block_id = evoke_semantic_bmp_read_u32(bytes + 0);
    block.first_record = evoke_semantic_bmp_read_u32(bytes + 4);
    block.record_count = evoke_semantic_bmp_read_u32(bytes + 8);
    if (block.record_count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    *block_out = block;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_record_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_record *record_out
)
{
    evoke_semantic_bmp_record record;
    uint16_t mask;
    uint16_t count = 0;

    if (bytes == NULL || record_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_RECORD_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    record.term_id = evoke_semantic_bmp_read_u32(bytes + 0);
    record.first_impact = evoke_semantic_bmp_read_u32(bytes + 4);
    record.document_mask = evoke_semantic_bmp_read_u16(bytes + 8);
    record.impact_count = evoke_semantic_bmp_read_u16(bytes + 10);
    mask = record.document_mask;
    while (mask != 0)
    {
        count += mask & UINT16_C(1);
        mask >>= 1;
    }
    if (record.document_mask == 0 || record.impact_count == 0 ||
        record.impact_count != count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *record_out = record;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_impacts_decode(
    const uint8_t *bytes,
    size_t size,
    uint16_t impact_count,
    float *impacts_out
)
{
    size_t required_size;

    if (bytes == NULL || impacts_out == NULL || impact_count == 0 ||
        impact_count > (UINT16_C(1) << EVOKE_SEMANTIC_BMP_BLOCK_SHIFT))
    {
        return EVOKE_ERR_INVALID;
    }
    required_size = (size_t) impact_count * sizeof(uint32_t);
    if (size < required_size)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint16_t index = 0; index < impact_count; index++)
    {
        impacts_out[index] = evoke_semantic_bmp_read_float(
            bytes + (size_t) index * sizeof(uint32_t)
        );
        if (!isfinite(impacts_out[index]) || impacts_out[index] == 0.0f)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_impact_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float term_min_impact,
    float term_max_impact,
    float *impact_out
)
{
    if (bytes == NULL || impact_out == NULL ||
        !isfinite(term_min_impact) ||
        !isfinite(term_max_impact) || term_min_impact > term_max_impact ||
        (term_min_impact == 0.0f && term_max_impact == 0.0f))
    {
        return EVOKE_ERR_INVALID;
    }
    if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_U8 &&
        term_min_impact <= 0.0f)
    {
        return EVOKE_ERR_FORMAT;
    }
    return evoke_semantic_impact_decode(
        bytes,
        size,
        impact_precision,
        impact_out
    );
}

evoke_status
evoke_semantic_bmp_packed_impacts_decode(
    const uint8_t *bytes,
    size_t size,
    uint16_t impact_count,
    evoke_semantic_impact_precision impact_precision,
    float term_min_impact,
    float term_max_impact,
    float *impacts_out
)
{
    size_t impact_width = evoke_semantic_bmp_impact_width(
        impact_precision
    );
    size_t required_size;

    if (bytes == NULL || impacts_out == NULL || impact_count == 0 ||
        impact_count > EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS ||
        impact_width == 0 || !isfinite(term_min_impact) ||
        !isfinite(term_max_impact) || term_min_impact > term_max_impact ||
        (term_min_impact == 0.0f && term_max_impact == 0.0f) ||
        impact_count > SIZE_MAX / impact_width)
    {
        return EVOKE_ERR_INVALID;
    }
    required_size = (size_t) impact_count * impact_width;
    if (size < required_size)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint16_t index = 0; index < impact_count; index++)
    {
        evoke_status status = evoke_semantic_bmp_packed_impact_decode(
            bytes + (size_t) index * impact_width,
            size - (size_t) index * impact_width,
            impact_precision,
            term_min_impact,
            term_max_impact,
            &impacts_out[index]
        );

        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_index *index_out
)
{
    evoke_semantic_bmp_index index;
    evoke_semantic_bmp_layout layout;
    uint64_t expected_checksum;
    evoke_status status;

    if (bytes == NULL || index_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_BMP_MAGIC ||
        evoke_semantic_bmp_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_BMP_VERSION ||
        evoke_semantic_bmp_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_BMP_HEADER_SIZE ||
        evoke_semantic_bmp_read_u32(bytes + 12) !=
            EVOKE_SEMANTIC_BMP_BLOCK_SHIFT)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_semantic_bmp_layout_build(
        evoke_semantic_bmp_read_u32(bytes + 24),
        evoke_semantic_bmp_read_u32(bytes + 52),
        evoke_semantic_bmp_read_u32(bytes + 28),
        evoke_semantic_bmp_read_u32(bytes + 20),
        evoke_semantic_bmp_read_u32(bytes + 32),
        evoke_semantic_bmp_read_u64(bytes + 40),
        &layout
    );
    if (status != EVOKE_OK || layout.total_size != size ||
        evoke_semantic_bmp_read_u32(bytes + 36) !=
            EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT ||
        evoke_semantic_bmp_read_u64(bytes + 56) !=
            layout.terms_offset ||
        evoke_semantic_bmp_read_u64(bytes + 64) !=
            layout.super_refs_offset ||
        evoke_semantic_bmp_read_u64(bytes + 72) != layout.refs_offset ||
        evoke_semantic_bmp_read_u64(bytes + 80) != layout.blocks_offset ||
        evoke_semantic_bmp_read_u64(bytes + 88) != layout.records_offset ||
        evoke_semantic_bmp_read_u64(bytes + 96) != layout.impacts_offset ||
        evoke_semantic_bmp_read_u64(bytes + 104) != layout.total_size ||
        evoke_semantic_bmp_read_u64(bytes + 120) != 0 ||
        evoke_semantic_bmp_read_u64(bytes + 128) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_checksum = evoke_semantic_bmp_read_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET
    );
    if (expected_checksum != evoke_semantic_bmp_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_semantic_bmp_index_init(&index);
    index.document_count = evoke_semantic_bmp_read_u32(bytes + 8);
    index.block_shift = evoke_semantic_bmp_read_u32(bytes + 12);
    index.block_count = evoke_semantic_bmp_read_u32(bytes + 16);
    index.active_block_count = evoke_semantic_bmp_read_u32(bytes + 20);
    index.term_count = evoke_semantic_bmp_read_u32(bytes + 24);
    index.ref_count = evoke_semantic_bmp_read_u32(bytes + 28);
    index.record_count = evoke_semantic_bmp_read_u32(bytes + 32);
    index.superblock_shift = evoke_semantic_bmp_read_u32(bytes + 36);
    index.posting_count = evoke_semantic_bmp_read_u64(bytes + 40);
    index.superblock_count = evoke_semantic_bmp_read_u32(bytes + 48);
    index.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 52);
    if (index.posting_count > SIZE_MAX / sizeof(*index.impacts))
    {
        return EVOKE_ERR_RANGE;
    }
    index.terms = calloc(index.term_count, sizeof(*index.terms));
    index.super_refs = calloc(
        index.super_ref_count,
        sizeof(*index.super_refs)
    );
    index.refs = calloc(index.ref_count, sizeof(*index.refs));
    index.blocks = calloc(
        index.active_block_count,
        sizeof(*index.blocks)
    );
    index.records = calloc(index.record_count, sizeof(*index.records));
    index.impacts = malloc(
        (size_t) index.posting_count * sizeof(*index.impacts)
    );
    if ((index.term_count > 0 && index.terms == NULL) ||
        (index.super_ref_count > 0 && index.super_refs == NULL) ||
        (index.ref_count > 0 && index.refs == NULL) ||
        (index.active_block_count > 0 && index.blocks == NULL) ||
        (index.record_count > 0 && index.records == NULL) ||
        (index.posting_count > 0 && index.impacts == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t index_position = 0;
         index_position < index.term_count;
         index_position++)
    {
        const uint8_t *source = bytes + layout.terms_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_TERM_SIZE;

        index.terms[index_position].term_id =
            evoke_semantic_bmp_read_u32(source + 0);
        index.terms[index_position].first_ref =
            evoke_semantic_bmp_read_u32(source + 4);
        index.terms[index_position].ref_count =
            evoke_semantic_bmp_read_u32(source + 8);
        index.terms[index_position].first_super_ref =
            evoke_semantic_bmp_read_u32(source + 12);
        index.terms[index_position].super_ref_count =
            evoke_semantic_bmp_read_u32(source + 16);
        index.terms[index_position].min_impact =
            evoke_semantic_bmp_read_float(source + 20);
        index.terms[index_position].max_impact =
            evoke_semantic_bmp_read_float(source + 24);
    }
    for (uint32_t term_index = 0;
         term_index < index.term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index.terms[term_index];

        if (term->first_ref > index.ref_count ||
            term->ref_count > index.ref_count - term->first_ref ||
            term->first_super_ref > index.super_ref_count ||
            term->super_ref_count >
                index.super_ref_count - term->first_super_ref)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        for (uint32_t offset = 0;
             offset < term->super_ref_count;
             offset++)
        {
            uint32_t index_position = term->first_super_ref + offset;
            const uint8_t *source = bytes + layout.super_refs_offset +
                (size_t) index_position *
                    EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE;

            status = evoke_semantic_bmp_super_ref_decode(
                source,
                EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE,
                term->min_impact,
                term->max_impact,
                &index.super_refs[index_position]
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        for (uint32_t offset = 0; offset < term->ref_count; offset++)
        {
            uint32_t index_position = term->first_ref + offset;
            const uint8_t *source = bytes + layout.refs_offset +
                (size_t) index_position * EVOKE_SEMANTIC_BMP_REF_SIZE;

            if (evoke_semantic_bmp_read_u16(source + 6) != 0)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            index.refs[index_position].block_id =
                evoke_semantic_bmp_read_u32(source + 0);
            index.refs[index_position].min_impact =
                evoke_semantic_bmp_dequantize_min(
                    source[4],
                    term->min_impact,
                    term->max_impact
                );
            index.refs[index_position].max_impact =
                evoke_semantic_bmp_dequantize_max(
                    source[5],
                    term->min_impact,
                    term->max_impact
                );
        }
    }
    for (uint32_t index_position = 0;
         index_position < index.active_block_count;
         index_position++)
    {
        const uint8_t *source = bytes + layout.blocks_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_BLOCK_SIZE;

        index.blocks[index_position].block_id =
            evoke_semantic_bmp_read_u32(source + 0);
        index.blocks[index_position].first_record =
            evoke_semantic_bmp_read_u32(source + 4);
        index.blocks[index_position].record_count =
            evoke_semantic_bmp_read_u32(source + 8);
    }
    for (uint32_t index_position = 0;
         index_position < index.record_count;
         index_position++)
    {
        const uint8_t *source = bytes + layout.records_offset +
            (size_t) index_position * EVOKE_SEMANTIC_BMP_RECORD_SIZE;

        index.records[index_position].term_id =
            evoke_semantic_bmp_read_u32(source + 0);
        index.records[index_position].first_impact =
            evoke_semantic_bmp_read_u32(source + 4);
        index.records[index_position].document_mask =
            evoke_semantic_bmp_read_u16(source + 8);
        index.records[index_position].impact_count =
            evoke_semantic_bmp_read_u16(source + 10);
    }
    for (uint64_t index_position = 0;
         index_position < index.posting_count;
         index_position++)
    {
        index.impacts[index_position] = evoke_semantic_bmp_read_float(
            bytes + layout.impacts_offset +
                (size_t) index_position * sizeof(uint32_t)
        );
    }
    for (uint32_t term_index = 0;
         term_index < index.term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &index.terms[term_index];

        for (uint32_t offset = 0; offset < term->ref_count; offset++)
        {
            evoke_semantic_bmp_ref *ref =
                &index.refs[term->first_ref + offset];
            const evoke_semantic_bmp_block *block =
                evoke_semantic_bmp_find_block(&index, ref->block_id);
            uint32_t low;
            uint32_t high;

            if (block == NULL)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            low = block->first_record;
            high = low + block->record_count;
            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2;

                if (index.records[middle].term_id < term->term_id)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            if (low >= block->first_record + block->record_count ||
                index.records[low].term_id != term->term_id)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            ref->record_index = low;
        }
    }
    status = evoke_semantic_bmp_index_validate(&index);
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_index_init(&index);
    }

cleanup:
    evoke_semantic_bmp_index_free(&index);
    return status;
}

evoke_status
evoke_semantic_bmp_topk(
    const evoke_semantic_bmp_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    evoke_semantic_bmp_query_term *query_terms = NULL;
    evoke_semantic_bmp_ranked_block *ranked_superblocks = NULL;
    float *super_upper_bounds = NULL;
    float *block_scores = NULL;
    evoke_topk_accumulator accumulator;
    evoke_semantic_bmp_stats stats;
    size_t unique_query_count = 0;
    size_t ranked_superblock_count = 0;
    evoke_status status;

    memset(&accumulator, 0, sizeof(accumulator));
    memset(&stats, 0, sizeof(stats));
    if (result_out == NULL || stats_out == NULL || index == NULL ||
        (query_count > 0 && query_ids == NULL) ||
        k > index->document_count ||
        query_count > SIZE_MAX / sizeof(*query_terms))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    memset(stats_out, 0, sizeof(*stats_out));
    if (index->document_count == 0 ||
        index->block_shift != EVOKE_SEMANTIC_BMP_BLOCK_SHIFT ||
        index->superblock_shift != EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT ||
        index->terms == NULL || index->super_refs == NULL ||
        index->refs == NULL || index->blocks == NULL ||
        index->records == NULL || index->impacts == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (k == 0 || query_count == 0)
    {
        return EVOKE_OK;
    }

    query_terms = malloc(query_count * sizeof(*query_terms));
    super_upper_bounds = calloc(
        index->superblock_count,
        sizeof(*super_upper_bounds)
    );
    ranked_superblocks = malloc(
        (size_t) index->superblock_count * sizeof(*ranked_superblocks)
    );
    block_scores = calloc(
        UINT32_C(1) << index->block_shift,
        sizeof(*block_scores)
    );
    if (query_terms == NULL || super_upper_bounds == NULL ||
        ranked_superblocks == NULL || block_scores == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t query_index = 0;
         query_index < query_count;
         query_index++)
    {
        float weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(weight))
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        query_terms[query_index].term_id = query_ids[query_index];
        query_terms[query_index].weight = weight;
    }
    qsort(
        query_terms,
        query_count,
        sizeof(*query_terms),
        evoke_semantic_bmp_compare_query_terms
    );
    for (size_t query_index = 0;
         query_index < query_count;
         query_index++)
    {
        if (unique_query_count > 0 &&
            query_terms[unique_query_count - 1].term_id ==
                query_terms[query_index].term_id)
        {
            query_terms[unique_query_count - 1].weight +=
                query_terms[query_index].weight;
            if (!isfinite(query_terms[unique_query_count - 1].weight))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
        else
        {
            query_terms[unique_query_count++] = query_terms[query_index];
        }
    }

    for (size_t query_index = 0;
         query_index < unique_query_count;
         query_index++)
    {
        const evoke_semantic_bmp_query_term *query =
            &query_terms[query_index];
        const evoke_semantic_bmp_term *term =
            evoke_semantic_bmp_find_term(index, query->term_id);

        if (term == NULL || query->weight == 0.0f)
        {
            continue;
        }
        for (uint32_t offset = 0;
             offset < term->super_ref_count;
             offset++)
        {
            const evoke_semantic_bmp_super_ref *ref =
                &index->super_refs[term->first_super_ref + offset];
            float contribution = evoke_semantic_bmp_contribution_bound(
                query->weight,
                ref->min_impact,
                ref->max_impact
            );
            double next = nextafter(
                (double) super_upper_bounds[ref->superblock_id] +
                    contribution,
                INFINITY
            );

            super_upper_bounds[ref->superblock_id] = next > FLT_MAX
                ? INFINITY
                : nextafterf((float) next, INFINITY);
            stats.super_bound_entries_visited++;
        }
    }
    for (uint32_t superblock_id = 0;
         superblock_id < index->superblock_count;
         superblock_id++)
    {
        if (super_upper_bounds[superblock_id] > 0.0f)
        {
            ranked_superblocks[ranked_superblock_count].block_id =
                superblock_id;
            ranked_superblocks[ranked_superblock_count].upper_bound =
                super_upper_bounds[superblock_id];
            ranked_superblock_count++;
        }
    }
    stats.superblocks_with_positive_bound = ranked_superblock_count;
    qsort(
        ranked_superblocks,
        ranked_superblock_count,
        sizeof(*ranked_superblocks),
        evoke_semantic_bmp_compare_ranked_blocks
    );
    status = evoke_topk_accumulator_init(&accumulator, k);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }

    for (size_t super_rank = 0;
         super_rank < ranked_superblock_count;
         super_rank++)
    {
        const evoke_semantic_bmp_ranked_block *ranked_super =
            &ranked_superblocks[super_rank];
        const uint32_t child_shift = index->superblock_shift -
            index->block_shift;
        const uint32_t child_count = UINT32_C(1) << child_shift;
        const uint32_t first_block = ranked_super->block_id << child_shift;
        evoke_semantic_bmp_ranked_block ranked_children[
            UINT32_C(1) << (
                EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT -
                EVOKE_SEMANTIC_BMP_BLOCK_SHIFT
            )
        ];
        float child_bounds[
            UINT32_C(1) << (
                EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT -
                EVOKE_SEMANTIC_BMP_BLOCK_SHIFT
            )
        ];
        size_t ranked_child_count = 0;

        if (accumulator.len == accumulator.capacity &&
            ranked_super->upper_bound < accumulator.heap[0].score)
        {
            stats.superblocks_skipped =
                ranked_superblock_count - super_rank;
            break;
        }
        memset(child_bounds, 0, sizeof(child_bounds));
        for (size_t query_index = 0;
             query_index < unique_query_count;
             query_index++)
        {
            const evoke_semantic_bmp_query_term *query =
                &query_terms[query_index];
            const evoke_semantic_bmp_term *term =
                evoke_semantic_bmp_find_term(index, query->term_id);
            const evoke_semantic_bmp_super_ref *super_ref;

            if (term == NULL || query->weight == 0.0f)
            {
                continue;
            }
            super_ref = evoke_semantic_bmp_find_super_ref(
                index,
                term,
                ranked_super->block_id
            );
            if (super_ref == NULL)
            {
                continue;
            }
            for (uint16_t child = 0;
                 child < super_ref->ref_count;
                 child++)
            {
                const evoke_semantic_bmp_ref *ref =
                    &index->refs[super_ref->first_ref + child];
                uint32_t local_block = ref->block_id - first_block;
                float contribution =
                    evoke_semantic_bmp_contribution_bound(
                        query->weight,
                        ref->min_impact,
                        ref->max_impact
                    );
                double next;

                if (local_block >= child_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                next = nextafter(
                    (double) child_bounds[local_block] + contribution,
                    INFINITY
                );
                child_bounds[local_block] = next > FLT_MAX
                    ? INFINITY
                    : nextafterf((float) next, INFINITY);
                stats.bound_entries_visited++;
            }
        }
        for (uint32_t child = 0; child < child_count; child++)
        {
            if (child_bounds[child] > 0.0f)
            {
                ranked_children[ranked_child_count].block_id =
                    first_block + child;
                ranked_children[ranked_child_count].upper_bound =
                    child_bounds[child];
                ranked_child_count++;
            }
        }
        stats.blocks_with_positive_bound += ranked_child_count;
        qsort(
            ranked_children,
            ranked_child_count,
            sizeof(*ranked_children),
            evoke_semantic_bmp_compare_ranked_blocks
        );
        for (size_t child_rank = 0;
             child_rank < ranked_child_count;
             child_rank++)
        {
            const evoke_semantic_bmp_ranked_block *ranked =
                &ranked_children[child_rank];
            const evoke_semantic_bmp_block *block =
                evoke_semantic_bmp_find_block(index, ranked->block_id);
            uint32_t first_document = ranked->block_id <<
                index->block_shift;
            uint32_t document_limit = first_document +
                (UINT32_C(1) << index->block_shift);

            if (block == NULL)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            if (accumulator.len == accumulator.capacity &&
                ranked->upper_bound < accumulator.heap[0].score)
            {
                stats.blocks_skipped +=
                    ranked_child_count - child_rank;
                break;
            }
            if (document_limit > index->document_count)
            {
                document_limit = index->document_count;
            }
            memset(
                block_scores,
                0,
                (UINT32_C(1) << index->block_shift) *
                    sizeof(*block_scores)
            );
            for (uint32_t offset = 0;
                 offset < block->record_count;
                 offset++)
            {
                const evoke_semantic_bmp_record *record =
                    &index->records[block->first_record + offset];
                const evoke_semantic_bmp_query_term *query =
                    evoke_semantic_bmp_find_query_term(
                        query_terms,
                        unique_query_count,
                        record->term_id
                    );
                uint16_t mask;
                uint16_t impact_index = 0;

                stats.forward_records_examined++;
                if (query == NULL || query->weight == 0.0f)
                {
                    continue;
                }
                mask = record->document_mask;
                for (uint32_t local_document = 0;
                     mask != 0;
                     local_document++, mask >>= 1)
                {
                    if ((mask & UINT16_C(1)) == 0)
                    {
                        continue;
                    }
                    block_scores[local_document] += query->weight *
                        index->impacts[
                            record->first_impact + impact_index
                        ];
                    impact_index++;
                    stats.postings_examined++;
                }
            }
            for (uint32_t document_id = first_document;
                 document_id < document_limit;
                 document_id++)
            {
                float score = block_scores[document_id - first_document];

                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &accumulator,
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
            stats.blocks_scored++;
        }
        stats.superblocks_scored++;
    }
    status = evoke_topk_accumulator_finish(
        &accumulator,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *stats_out = stats;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_topk_accumulator_free(&accumulator);
    free(block_scores);
    free(ranked_superblocks);
    free(super_upper_bounds);
    free(query_terms);
    return status;
}

static uint32_t
evoke_semantic_bmp_mask_count(uint64_t mask)
{
    return (uint32_t) __builtin_popcountll((unsigned long long) mask);
}

static uint32_t
evoke_semantic_bmp_packed_membership_bytes(
    uint32_t block_count,
    uint32_t ref_count
)
{
    uint32_t bitmap_bytes = (block_count + UINT32_C(7)) >> 3U;

    /* Keep the derived directory below one byte per exact fine reference. */
    return bitmap_bytes != 0 && bitmap_bytes <= ref_count
        ? bitmap_bytes
        : 0;
}

static uint32_t
evoke_semantic_bmp_bytes_popcount(const uint8_t *bytes, uint32_t size)
{
    uint32_t count = 0;

    for (uint32_t index = 0; index < size; index++)
    {
        count += (uint32_t) __builtin_popcount((unsigned) bytes[index]);
    }
    return count;
}

static const evoke_semantic_bmp_packed_term *
evoke_semantic_bmp_packed_find_term(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t term_id
)
{
    uint32_t low = 0;
    uint32_t high = index->term_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2U;

        if (index->terms[middle].term_id < term_id)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    return low < index->term_count && index->terms[low].term_id == term_id
        ? &index->terms[low]
        : NULL;
}

void
evoke_semantic_bmp_packed_index_init(
    evoke_semantic_bmp_packed_index *index
)
{
    if (index != NULL)
    {
        memset(index, 0, sizeof(*index));
        index->impact_precision = EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
}

void
evoke_semantic_bmp_packed_index_free(
    evoke_semantic_bmp_packed_index *index
)
{
    if (index == NULL)
    {
        return;
    }
    free(index->terms);
    free(index->super_refs);
    free(index->refs);
    free(index->block_membership);
    free(index->doc_deltas);
    free(index->impacts);
    evoke_semantic_bmp_packed_index_init(index);
}

static evoke_status
evoke_semantic_bmp_packed_apply_precision(
    evoke_semantic_bmp_packed_index *index,
    evoke_semantic_impact_precision impact_precision
)
{
    uint64_t impact_cursor = 0;
    evoke_status status;

    if (index == NULL ||
        evoke_semantic_bmp_impact_width(impact_precision) == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    index->impact_precision = impact_precision;
    if (impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32)
    {
        return evoke_semantic_bmp_packed_index_validate(index);
    }

    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        evoke_semantic_bmp_packed_term *term = &index->terms[term_index];
        float source_minimum = term->min_impact;
        float source_maximum = term->max_impact;
        float term_minimum = 0.0f;
        float term_maximum = 0.0f;

        if (!isfinite(source_minimum) || !isfinite(source_maximum) ||
            source_minimum > source_maximum ||
            (source_minimum == 0.0f && source_maximum == 0.0f) ||
            term->posting_count > index->posting_count - impact_cursor)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t posting_index = 0;
             posting_index < term->posting_count;
             posting_index++)
        {
            uint64_t position = impact_cursor + posting_index;
            uint8_t encoded[sizeof(uint32_t)];
            float impact;

            status = evoke_semantic_impact_encode(
                encoded,
                sizeof(encoded),
                impact_precision,
                index->impacts[position],
                &impact
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            index->impacts[position] = impact;
            if (posting_index == 0 || impact < term_minimum)
            {
                term_minimum = impact;
            }
            if (posting_index == 0 || impact > term_maximum)
            {
                term_maximum = impact;
            }
        }
        term->min_impact = term_minimum;
        term->max_impact = term_maximum;

        for (uint32_t super_offset = 0;
             super_offset < term->super_ref_count;
             super_offset++)
        {
            uint32_t super_index = term->first_super_ref + super_offset;
            uint8_t *super_source = index->super_refs +
                (size_t) super_index *
                    EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
            uint32_t first_ref = evoke_semantic_bmp_read_u32(
                super_source + 4
            );
            uint32_t first_impact = evoke_semantic_bmp_read_u32(
                super_source + 8
            );
            uint32_t ref_count = evoke_semantic_bmp_read_u16(
                super_source + 12
            );
            uint32_t block_impact = first_impact;
            float super_minimum = 0.0f;
            float super_maximum = 0.0f;

            if (first_ref < term->first_ref ||
                first_ref - term->first_ref > term->ref_count ||
                ref_count > term->ref_count -
                    (first_ref - term->first_ref) ||
                first_impact < impact_cursor ||
                first_impact - impact_cursor > term->posting_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint32_t ref_offset = 0;
                 ref_offset < ref_count;
                 ref_offset++)
            {
                uint8_t *ref_source = index->refs +
                    (size_t) (first_ref + ref_offset) *
                        EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
                uint32_t block_count = evoke_semantic_bmp_mask_count(
                    evoke_semantic_bmp_read_u64(ref_source + 1)
                );
                float block_minimum = 0.0f;
                float block_maximum = 0.0f;

                if (block_count == 0 ||
                    block_impact > index->posting_count - block_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                for (uint32_t local = 0; local < block_count; local++)
                {
                    float impact = index->impacts[block_impact + local];

                    if (local == 0 || impact < block_minimum)
                    {
                        block_minimum = impact;
                    }
                    if (local == 0 || impact > block_maximum)
                    {
                        block_maximum = impact;
                    }
                }
                ref_source[9] = evoke_semantic_bmp_quantize_min(
                    block_minimum,
                    term_minimum,
                    term_maximum
                );
                ref_source[10] = evoke_semantic_bmp_quantize_max(
                    block_maximum,
                    term_minimum,
                    term_maximum
                );
                if (ref_offset == 0 || block_minimum < super_minimum)
                {
                    super_minimum = block_minimum;
                }
                if (ref_offset == 0 || block_maximum > super_maximum)
                {
                    super_maximum = block_maximum;
                }
                block_impact += block_count;
            }
            super_source[14] = evoke_semantic_bmp_quantize_min(
                super_minimum,
                term_minimum,
                term_maximum
            );
            super_source[15] = evoke_semantic_bmp_quantize_max(
                super_maximum,
                term_minimum,
                term_maximum
            );
        }
        impact_cursor += term->posting_count;
    }
    if (impact_cursor != index->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    return evoke_semantic_bmp_packed_index_validate(index);
}

evoke_status
evoke_semantic_bmp_packed_index_build(
    const evoke_semantic_bmp_index *source,
    evoke_semantic_bmp_packed_index *index_out
)
{
    evoke_semantic_bmp_run *runs = NULL;
    uint32_t *document_ids = NULL;
    evoke_posting_value *values = NULL;
    uint64_t posting_cursor = 0;
    evoke_status status;

    if (source == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_bmp_index_validate(source);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (source->posting_count > SIZE_MAX / sizeof(*document_ids) ||
        source->posting_count > SIZE_MAX / sizeof(*values))
    {
        return EVOKE_ERR_RANGE;
    }
    if (source->term_count > 0)
    {
        runs = calloc(source->term_count, sizeof(*runs));
    }
    if (source->posting_count > 0)
    {
        document_ids = malloc(
            (size_t) source->posting_count * sizeof(*document_ids)
        );
        values = calloc(
            (size_t) source->posting_count,
            sizeof(*values)
        );
    }
    if ((source->term_count > 0 && runs == NULL) ||
        (source->posting_count > 0 &&
         (document_ids == NULL || values == NULL)))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t term_index = 0;
         term_index < source->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_term *term = &source->terms[term_index];
        uint64_t term_first = posting_cursor;

        for (uint32_t ref_offset = 0;
             ref_offset < term->ref_count;
             ref_offset++)
        {
            const evoke_semantic_bmp_ref *ref = &source->refs[
                term->first_ref + ref_offset
            ];
            const evoke_semantic_bmp_record *record =
                &source->records[ref->record_index];
            uint32_t local_impact = 0;

            for (uint32_t local_document = 0;
                 local_document < UINT32_C(16);
                 local_document++)
            {
                if ((record->document_mask &
                     (UINT64_C(1) << local_document)) == 0)
                {
                    continue;
                }
                if (posting_cursor >= source->posting_count ||
                    local_impact >= record->impact_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                document_ids[posting_cursor] = (ref->block_id <<
                    EVOKE_SEMANTIC_BMP_BLOCK_SHIFT) + local_document;
                values[posting_cursor].impact = source->impacts[
                    record->first_impact + local_impact
                ];
                posting_cursor++;
                local_impact++;
            }
            if (local_impact != record->impact_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
        }
        runs[term_index].term_id = term->term_id;
        runs[term_index].posting_count = posting_cursor - term_first;
        runs[term_index].local_document_ids =
            document_ids + (size_t) term_first;
        runs[term_index].values = values + (size_t) term_first;
        runs[term_index].document_id_base = 0;
        runs[term_index].local_document_count = source->document_count;
    }
    if (posting_cursor != source->posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    status = evoke_semantic_bmp_packed_index_build_runs(
        source->document_count,
        runs,
        source->term_count,
        index_out
    );

cleanup:
    free(values);
    free(document_ids);
    free(runs);
    return status;
}

evoke_status
evoke_semantic_bmp_packed_index_build_runs(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_bmp_packed_index *index_out
)
{
    evoke_semantic_bmp_packed_index index;
    uint64_t posting_count = 0;
    uint32_t ref_count = 0;
    uint32_t super_ref_count = 0;
    uint32_t ref_cursor = 0;
    uint32_t super_cursor = 0;
    uint32_t impact_cursor = 0;
    size_t doc_cursor = 0;
    evoke_status status = EVOKE_OK;

    if (document_count == 0 || index_out == NULL ||
        (run_count > 0 && runs == NULL) || run_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_bmp_packed_index_init(&index);
    index.document_count = document_count;
    index.block_count = (document_count +
        (UINT32_C(1) << EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) - 1U) >>
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
    index.superblock_count = (document_count +
        (UINT32_C(1) << EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT) - 1U) >>
        EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT;
    index.term_count = (uint32_t) run_count;
    if (run_count > 0)
    {
        index.terms = calloc(run_count, sizeof(*index.terms));
    }
    if (run_count > 0 && index.terms == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    /*
     * The sizing pass is term-local. It never constructs an expanded BMP or
     * a corpus-sized document-id copy; each run supplies its final frame size
     * before the next run is examined.
     */
    for (size_t run_index = 0; run_index < run_count; run_index++)
    {
        const evoke_semantic_bmp_run *run = &runs[run_index];
        evoke_semantic_bmp_packed_term *term = &index.terms[run_index];
        uint32_t previous_document = 0;
        uint32_t previous_block = UINT32_MAX;
        uint32_t previous_superblock = UINT32_MAX;
        uint32_t max_delta = 0;
        float min_impact = 0.0f;
        float max_impact = 0.0f;

        if (run->posting_count == 0 ||
            run->posting_count > UINT32_MAX ||
            run->local_document_ids == NULL || run->values == NULL ||
            run->local_document_count == 0 ||
            (run_index > 0 && runs[run_index - 1].term_id >= run->term_id) ||
            posting_count > UINT32_MAX - run->posting_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        term->term_id = run->term_id;
        term->first_ref = ref_count;
        term->first_super_ref = super_ref_count;
        term->first_doc_byte = index.doc_delta_bytes;
        term->posting_count = (uint32_t) run->posting_count;
        for (uint64_t posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            uint32_t document_id;
            uint32_t block_id;
            uint32_t superblock_id;
            float impact = run->values[posting_index].impact;

            status = evoke_semantic_bmp_run_document_id(
                run,
                posting_index,
                document_count,
                &document_id
            );
            if (status != EVOKE_OK || !isfinite(impact) || impact == 0.0f ||
                (posting_index > 0 && document_id <= previous_document))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            if (posting_index == 0)
            {
                term->first_document = document_id;
                min_impact = impact;
                max_impact = impact;
            }
            else
            {
                uint32_t delta = document_id - previous_document;

                if (delta > max_delta)
                {
                    max_delta = delta;
                }
                if (impact < min_impact)
                {
                    min_impact = impact;
                }
                if (impact > max_impact)
                {
                    max_impact = impact;
                }
            }
            block_id = document_id >>
                EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
            superblock_id = document_id >>
                EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT;
            if (block_id != previous_block)
            {
                if (ref_count == UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                ref_count++;
                term->ref_count++;
                previous_block = block_id;
            }
            if (superblock_id != previous_superblock)
            {
                if (super_ref_count == UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                super_ref_count++;
                term->super_ref_count++;
                previous_superblock = superblock_id;
            }
            previous_document = document_id;
        }
        term->first_block_membership_byte =
            index.block_membership_bytes;
        term->block_membership_bytes =
            evoke_semantic_bmp_packed_membership_bytes(
                index.block_count,
                term->ref_count
            );
        if (term->block_membership_bytes > UINT32_MAX -
                index.block_membership_bytes)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        index.block_membership_bytes += term->block_membership_bytes;
        term->doc_delta_width = term->posting_count == 1U
            ? 0U
            : max_delta <= UINT8_MAX
                ? 1U
                : max_delta <= UINT16_MAX
                    ? 2U
                    : 4U;
        term->min_impact = min_impact;
        term->max_impact = max_impact;
        if (term->posting_count > 1U)
        {
            uint64_t bytes = (uint64_t) (term->posting_count - 1U) *
                term->doc_delta_width;

            if (bytes > UINT32_MAX - index.doc_delta_bytes)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            index.doc_delta_bytes += (uint32_t) bytes;
        }
        posting_count += run->posting_count;
    }

    index.posting_count = posting_count;
    index.ref_count = ref_count;
    index.record_count = ref_count;
    index.super_ref_count = super_ref_count;
    if (super_ref_count >
            UINT32_MAX / EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        ref_count > UINT32_MAX / EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE)
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    index.super_ref_bytes = super_ref_count *
        EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
    index.ref_bytes = ref_count * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
    index.super_refs = malloc(index.super_ref_bytes);
    index.refs = malloc(index.ref_bytes);
    index.block_membership = calloc(index.block_membership_bytes, 1U);
    index.doc_deltas = malloc(index.doc_delta_bytes);
    if (posting_count > 0)
    {
        index.impacts = malloc(
            (size_t) posting_count * sizeof(*index.impacts)
        );
    }
    if ((index.super_ref_bytes > 0 && index.super_refs == NULL) ||
        (index.ref_bytes > 0 && index.refs == NULL) ||
        (index.block_membership_bytes > 0 &&
         index.block_membership == NULL) ||
        (index.doc_delta_bytes > 0 && index.doc_deltas == NULL) ||
        (posting_count > 0 && index.impacts == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (size_t run_index = 0; run_index < run_count; run_index++)
    {
        const evoke_semantic_bmp_run *run = &runs[run_index];
        const evoke_semantic_bmp_packed_term *term =
            &index.terms[run_index];
        uint64_t posting_index = 0;
        uint32_t previous_document = 0;

        if (doc_cursor != term->first_doc_byte ||
            ref_cursor != term->first_ref ||
            super_cursor != term->first_super_ref)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        while (posting_index < run->posting_count)
        {
            uint32_t first_document;
            uint32_t superblock_id;
            uint32_t super_first_ref = ref_cursor;
            uint32_t super_first_impact = impact_cursor;
            float super_min = 0.0f;
            float super_max = 0.0f;
            bool has_super_impact = false;

            status = evoke_semantic_bmp_run_document_id(
                run,
                posting_index,
                document_count,
                &first_document
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            superblock_id = first_document >>
                EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT;
            while (posting_index < run->posting_count)
            {
                uint32_t document_id;
                uint32_t block_id;
                uint64_t document_mask = 0;
                float block_min = 0.0f;
                float block_max = 0.0f;
                bool has_block_impact = false;
                uint8_t *ref_destination;

                status = evoke_semantic_bmp_run_document_id(
                    run,
                    posting_index,
                    document_count,
                    &document_id
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if ((document_id >>
                     EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT) !=
                    superblock_id)
                {
                    break;
                }
                block_id = document_id >>
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
                while (posting_index < run->posting_count)
                {
                    uint32_t current_document;
                    uint32_t local_document;
                    float impact;

                    status = evoke_semantic_bmp_run_document_id(
                        run,
                        posting_index,
                        document_count,
                        &current_document
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                    if ((current_document >>
                         EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) != block_id)
                    {
                        break;
                    }
                    local_document = current_document & UINT32_C(63);
                    impact = run->values[posting_index].impact;
                    document_mask |= UINT64_C(1) << local_document;
                    if (!has_block_impact)
                    {
                        block_min = impact;
                        block_max = impact;
                        has_block_impact = true;
                    }
                    else
                    {
                        if (impact < block_min)
                        {
                            block_min = impact;
                        }
                        if (impact > block_max)
                        {
                            block_max = impact;
                        }
                    }
                    if (impact_cursor >= posting_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    index.impacts[impact_cursor++] = impact;
                    if (posting_index > 0)
                    {
                        uint32_t delta = current_document -
                            previous_document;

                        if (doc_cursor > index.doc_delta_bytes ||
                            term->doc_delta_width >
                                index.doc_delta_bytes - doc_cursor)
                        {
                            status = EVOKE_ERR_FORMAT;
                            goto cleanup;
                        }
                        evoke_semantic_bmp_write_delta(
                            index.doc_deltas,
                            doc_cursor,
                            term->doc_delta_width,
                            delta
                        );
                        doc_cursor += term->doc_delta_width;
                    }
                    previous_document = current_document;
                    posting_index++;
                }
                if (!has_block_impact || ref_cursor >= ref_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                ref_destination = index.refs + (size_t) ref_cursor *
                    EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
                ref_destination[0] = (uint8_t) (block_id & UINT32_C(15));
                evoke_semantic_bmp_write_u64(
                    ref_destination + 1,
                    document_mask
                );
                ref_destination[9] = evoke_semantic_bmp_quantize_min(
                    block_min,
                    term->min_impact,
                    term->max_impact
                );
                ref_destination[10] = evoke_semantic_bmp_quantize_max(
                    block_max,
                    term->min_impact,
                    term->max_impact
                );
                if (term->block_membership_bytes != 0)
                {
                    index.block_membership[
                        term->first_block_membership_byte +
                            (block_id >> 3U)
                    ] |= (uint8_t) (
                        UINT8_C(1) << (block_id & UINT32_C(7))
                    );
                }
                ref_cursor++;
                if (!has_super_impact)
                {
                    super_min = block_min;
                    super_max = block_max;
                    has_super_impact = true;
                }
                else
                {
                    if (block_min < super_min)
                    {
                        super_min = block_min;
                    }
                    if (block_max > super_max)
                    {
                        super_max = block_max;
                    }
                }
            }
            if (!has_super_impact || super_cursor >= super_ref_count ||
                ref_cursor - super_first_ref == 0 ||
                ref_cursor - super_first_ref >
                    (UINT32_C(1) <<
                     (EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT -
                      EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT)))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            {
                uint8_t *super_destination = index.super_refs +
                    (size_t) super_cursor *
                        EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;

                evoke_semantic_bmp_write_u32(
                    super_destination + 0,
                    superblock_id
                );
                evoke_semantic_bmp_write_u32(
                    super_destination + 4,
                    super_first_ref
                );
                evoke_semantic_bmp_write_u32(
                    super_destination + 8,
                    super_first_impact
                );
                evoke_semantic_bmp_write_u16(
                    super_destination + 12,
                    (uint16_t) (ref_cursor - super_first_ref)
                );
                super_destination[14] =
                    evoke_semantic_bmp_quantize_min(
                        super_min,
                        term->min_impact,
                        term->max_impact
                    );
                super_destination[15] =
                    evoke_semantic_bmp_quantize_max(
                        super_max,
                        term->min_impact,
                        term->max_impact
                    );
            }
            super_cursor++;
        }
    }
    if (ref_cursor != ref_count || super_cursor != super_ref_count ||
        impact_cursor != posting_count ||
        doc_cursor != index.doc_delta_bytes)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    status = evoke_semantic_bmp_packed_index_validate(&index);
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_packed_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_packed_index_init(&index);
    }

cleanup:
    evoke_semantic_bmp_packed_index_free(&index);
    return status;
}

evoke_status
evoke_semantic_bmp_packed_index_build_runs_with_precision(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_impact_precision impact_precision,
    evoke_semantic_bmp_packed_index *index_out
)
{
    evoke_semantic_bmp_packed_index index;
    evoke_status status;

    if (index_out == NULL ||
        evoke_semantic_bmp_impact_width(impact_precision) == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_bmp_packed_index_init(&index);
    status = evoke_semantic_bmp_packed_index_build_runs(
        document_count,
        runs,
        run_count,
        &index
    );
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_bmp_packed_apply_precision(
            &index,
            impact_precision
        );
    }
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_packed_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_packed_index_init(&index);
    }
    evoke_semantic_bmp_packed_index_free(&index);
    return status;
}

static uint32_t
evoke_semantic_bmp_packed_term_ref_limit(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t term_index
)
{
    return term_index + 1U < index->term_count
        ? index->terms[term_index + 1U].first_ref
        : index->ref_count;
}

static uint32_t
evoke_semantic_bmp_packed_term_super_limit(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t term_index
)
{
    return term_index + 1U < index->term_count
        ? index->terms[term_index + 1U].first_super_ref
        : index->super_ref_count;
}

evoke_status
evoke_semantic_bmp_packed_index_validate(
    const evoke_semantic_bmp_packed_index *index
)
{
    uint32_t expected_ref = 0;
    uint32_t expected_super_ref = 0;
    uint32_t expected_block_membership_byte = 0;
    uint32_t expected_doc_byte = 0;
    uint64_t expected_impact = 0;

    if (index == NULL || index->document_count == 0 ||
        evoke_semantic_bmp_impact_width(index->impact_precision) == 0 ||
        index->block_count != (index->document_count + 63U) / 64U ||
        index->superblock_count !=
            (index->document_count + 1023U) / 1024U ||
        index->posting_count > UINT32_MAX ||
        index->record_count != index->ref_count ||
        index->super_ref_count >
            UINT32_MAX / EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        index->ref_count >
            UINT32_MAX / EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE ||
        index->super_ref_bytes != index->super_ref_count *
            EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        index->ref_bytes != index->ref_count *
            EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE ||
        (index->term_count > 0 && index->terms == NULL) ||
        (index->super_ref_bytes > 0 && index->super_refs == NULL) ||
        (index->ref_bytes > 0 && index->refs == NULL) ||
        (index->block_membership_bytes > 0 &&
         index->block_membership == NULL) ||
        (index->doc_delta_bytes > 0 && index->doc_deltas == NULL) ||
        (index->posting_count > 0 && index->impacts == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint64_t posting_index = 0;
         posting_index < index->posting_count;
         posting_index++)
    {
        if (!isfinite(index->impacts[posting_index]) ||
            index->impacts[posting_index] == 0.0f)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_packed_term *term =
            &index->terms[term_index];
        uint32_t ref_limit = evoke_semantic_bmp_packed_term_ref_limit(
            index,
            term_index
        );
        uint32_t super_limit = evoke_semantic_bmp_packed_term_super_limit(
            index,
            term_index
        );
        uint32_t doc_limit = term_index + 1U < index->term_count
            ? index->terms[term_index + 1U].first_doc_byte
            : index->doc_delta_bytes;
        uint32_t membership_limit = term_index + 1U < index->term_count
            ? index->terms[
                term_index + 1U
            ].first_block_membership_byte
            : index->block_membership_bytes;
        uint32_t previous_superblock = 0;
        uint64_t term_first_impact = expected_impact;

        if ((term_index > 0 &&
             term->term_id <= index->terms[term_index - 1U].term_id) ||
            term->first_ref != expected_ref ||
            term->first_super_ref != expected_super_ref ||
            term->first_block_membership_byte !=
                expected_block_membership_byte ||
            term->first_doc_byte != expected_doc_byte ||
            !isfinite(term->min_impact) ||
            !isfinite(term->max_impact) ||
            term->min_impact > term->max_impact ||
            term->first_ref > ref_limit ||
            term->ref_count > ref_limit - term->first_ref ||
            term->first_ref + term->ref_count != ref_limit ||
            term->first_super_ref > super_limit ||
            term->super_ref_count >
                super_limit - term->first_super_ref ||
            term->first_super_ref + term->super_ref_count != super_limit)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (term->first_block_membership_byte > membership_limit ||
            term->block_membership_bytes > membership_limit -
                term->first_block_membership_byte ||
            term->first_block_membership_byte +
                term->block_membership_bytes != membership_limit ||
            term->block_membership_bytes !=
                evoke_semantic_bmp_packed_membership_bytes(
                    index->block_count,
                    term->ref_count
                ))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (term->posting_count == 0 ||
            term->first_document >= index->document_count ||
            (term->posting_count == 1 && term->doc_delta_width != 0) ||
            (term->posting_count > 1 &&
             term->doc_delta_width != 1 &&
             term->doc_delta_width != 2 &&
             term->doc_delta_width != 4))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t offset_index = 0;
             offset_index < term->super_ref_count;
             offset_index++)
        {
            const uint8_t *super_source = index->super_refs +
                (size_t) expected_super_ref *
                    EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
            uint32_t first_ref;
            uint32_t first_impact;
            uint32_t child_count;
            uint32_t superblock_id;
            uint32_t previous_local_block = UINT32_MAX;

            superblock_id = evoke_semantic_bmp_read_u32(
                super_source + 0
            );
            first_ref = evoke_semantic_bmp_read_u32(
                super_source + 4
            );
            first_impact = evoke_semantic_bmp_read_u32(
                super_source + 8
            );
            child_count = evoke_semantic_bmp_read_u16(
                super_source + 12
            );
            if ((offset_index > 0 && superblock_id <= previous_superblock) ||
                superblock_id >= index->superblock_count ||
                first_ref != expected_ref ||
                first_impact != expected_impact ||
                child_count == 0 || child_count > ref_limit - expected_ref ||
                super_source[14] > super_source[15])
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_superblock = superblock_id;
            for (uint32_t child = 0; child < child_count; child++)
            {
                const uint8_t *ref_source = index->refs +
                    (size_t) expected_ref *
                        EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
                uint32_t local_block = ref_source[0];
                uint32_t block_id;
                uint32_t impact_count;
                uint64_t mask = evoke_semantic_bmp_read_u64(
                    ref_source + 1
                );

                if (local_block >= UINT32_C(16) ||
                    (child > 0 && local_block <= previous_local_block) ||
                    (superblock_id << 4U) + local_block >=
                        index->block_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                previous_local_block = local_block;
                block_id = (superblock_id << 4U) + local_block;
                impact_count = evoke_semantic_bmp_mask_count(mask);
                if (mask == 0 || ref_source[9] > ref_source[10] ||
                    impact_count > index->posting_count ||
                    expected_impact >
                        index->posting_count - impact_count ||
                    (block_id + 1U == index->block_count &&
                     (index->document_count & UINT32_C(63)) != 0 &&
                     (mask >> (index->document_count & UINT32_C(63))) != 0))
                {
                    return EVOKE_ERR_FORMAT;
                }
                if (term->block_membership_bytes != 0 &&
                    (index->block_membership[
                        term->first_block_membership_byte +
                            (block_id >> 3U)
                    ] & (uint8_t) (
                        UINT8_C(1) << (block_id & UINT32_C(7))
                    )) == 0)
                {
                    return EVOKE_ERR_FORMAT;
                }
                expected_impact += impact_count;
                expected_ref++;
            }
            expected_super_ref++;
        }
        if (expected_ref != ref_limit || expected_super_ref != super_limit)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (term->block_membership_bytes != 0 &&
            evoke_semantic_bmp_bytes_popcount(
                index->block_membership +
                    term->first_block_membership_byte,
                term->block_membership_bytes
            ) != term->ref_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (expected_impact - term_first_impact != term->posting_count ||
            term->first_doc_byte > doc_limit ||
            (uint64_t) (term->posting_count - 1U) *
                term->doc_delta_width !=
                (uint64_t) doc_limit - term->first_doc_byte)
        {
            return EVOKE_ERR_FORMAT;
        }
        {
            size_t doc_offset = term->first_doc_byte;
            uint32_t previous_document = 0;

            for (uint32_t posting_index = 0;
                 posting_index < term->posting_count;
                 posting_index++)
            {
                uint32_t document_id;

                if (posting_index == 0)
                {
                    document_id = term->first_document;
                }
                else
                {
                    uint32_t delta;

                    if (doc_offset > doc_limit ||
                        term->doc_delta_width > doc_limit - doc_offset)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    delta = evoke_semantic_bmp_read_delta(
                        index->doc_deltas,
                        doc_offset,
                        term->doc_delta_width
                    );
                    doc_offset += term->doc_delta_width;
                    if (delta == 0 || delta >
                            UINT32_MAX - previous_document)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    document_id = previous_document + delta;
                }
                if (document_id >= index->document_count ||
                    (posting_index > 0 && document_id <= previous_document))
                {
                    return EVOKE_ERR_FORMAT;
                }
                previous_document = document_id;
            }
            if (doc_offset != doc_limit)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        expected_doc_byte = doc_limit;
        expected_block_membership_byte = membership_limit;
    }
    return expected_ref == index->ref_count &&
        expected_super_ref == index->super_ref_count &&
        expected_block_membership_byte ==
            index->block_membership_bytes &&
        expected_doc_byte == index->doc_delta_bytes &&
        expected_impact == index->posting_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_semantic_bmp_packed_serialized_size(
    const evoke_semantic_bmp_packed_index *index,
    size_t *size_out
)
{
    evoke_semantic_bmp_packed_layout layout;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_packed_index_validate(index) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_semantic_bmp_packed_layout_build(
        index->term_count,
        index->super_ref_bytes,
        index->ref_bytes,
        index->block_membership_bytes,
        index->doc_delta_bytes,
        index->posting_count,
        index->impact_precision,
        &layout
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    *size_out = layout.total_size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_serialize(
    const evoke_semantic_bmp_packed_index *index,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_semantic_bmp_packed_layout layout;
    uint8_t *bytes;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_semantic_bmp_packed_index_validate(index);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_semantic_bmp_packed_layout_build(
        index->term_count,
        index->super_ref_bytes,
        index->ref_bytes,
        index->block_membership_bytes,
        index->doc_delta_bytes,
        index->posting_count,
        index->impact_precision,
        &layout
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = calloc(layout.total_size, 1U);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    status = evoke_semantic_bmp_packed_serialize_into(
        index,
        bytes,
        layout.total_size
    );
    if (status != EVOKE_OK)
    {
        free(bytes);
        return status;
    }
    *bytes_out = bytes;
    *size_out = layout.total_size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_serialize_into(
    const evoke_semantic_bmp_packed_index *index,
    uint8_t *bytes,
    size_t size
)
{
    evoke_semantic_bmp_packed_layout layout;
    evoke_status status;

    if (bytes == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_bmp_packed_index_validate(index);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_semantic_bmp_packed_layout_build(
        index->term_count,
        index->super_ref_bytes,
        index->ref_bytes,
        index->block_membership_bytes,
        index->doc_delta_bytes,
        index->posting_count,
        index->impact_precision,
        &layout
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (size != layout.total_size)
    {
        return EVOKE_ERR_RANGE;
    }
    memset(bytes, 0, size);

    evoke_semantic_bmp_write_u32(
        bytes + 0,
        EVOKE_SEMANTIC_BMP_PACKED_MAGIC
    );
    evoke_semantic_bmp_write_u16(
        bytes + 4,
        EVOKE_SEMANTIC_BMP_PACKED_VERSION
    );
    evoke_semantic_bmp_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_BMP_HEADER_SIZE
    );
    evoke_semantic_bmp_write_u32(bytes + 8, index->document_count);
    evoke_semantic_bmp_write_u32(bytes + 12, index->block_count);
    evoke_semantic_bmp_write_u32(bytes + 16, index->superblock_count);
    evoke_semantic_bmp_write_u32(bytes + 20, index->term_count);
    evoke_semantic_bmp_write_u32(bytes + 24, index->ref_count);
    evoke_semantic_bmp_write_u32(bytes + 28, index->super_ref_count);
    evoke_semantic_bmp_write_u32(bytes + 32, index->record_count);
    evoke_semantic_bmp_write_u32(bytes + 36, index->super_ref_bytes);
    evoke_semantic_bmp_write_u32(bytes + 40, index->ref_bytes);
    evoke_semantic_bmp_write_u32(bytes + 44, index->doc_delta_bytes);
    evoke_semantic_bmp_write_u64(bytes + 48, index->posting_count);
    evoke_semantic_bmp_write_u64(bytes + 56, layout.terms_offset);
    evoke_semantic_bmp_write_u64(bytes + 64, layout.super_refs_offset);
    evoke_semantic_bmp_write_u64(bytes + 72, layout.refs_offset);
    evoke_semantic_bmp_write_u64(bytes + 80, layout.doc_deltas_offset);
    evoke_semantic_bmp_write_u64(bytes + 88, layout.impacts_offset);
    evoke_semantic_bmp_write_u64(bytes + 96, layout.total_size);
    evoke_semantic_bmp_write_u32(
        bytes + 104,
        index->block_membership_bytes
    );
    evoke_semantic_bmp_write_u32(
        bytes + 108,
        index->impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32
            ? 0
            : (uint32_t) index->impact_precision
    );
    evoke_semantic_bmp_write_u64(
        bytes + 120,
        layout.block_membership_offset
    );

    for (uint32_t term_index = 0;
         term_index < index->term_count;
         term_index++)
    {
        const evoke_semantic_bmp_packed_term *term =
            &index->terms[term_index];
        uint8_t *destination = bytes + layout.terms_offset +
            (size_t) term_index * EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE;

        evoke_semantic_bmp_write_u32(destination + 0, term->term_id);
        evoke_semantic_bmp_write_u32(destination + 4, term->first_ref);
        evoke_semantic_bmp_write_u32(destination + 8, term->ref_count);
        evoke_semantic_bmp_write_u32(
            destination + 12,
            term->first_super_ref
        );
        evoke_semantic_bmp_write_u32(
            destination + 16,
            term->super_ref_count
        );
        evoke_semantic_bmp_write_u32(
            destination + 20,
            term->first_doc_byte
        );
        evoke_semantic_bmp_write_u32(
            destination + 24,
            term->posting_count
        );
        evoke_semantic_bmp_write_u32(
            destination + 28,
            term->first_document
        );
        evoke_semantic_bmp_write_u32(
            destination + 32,
            term->doc_delta_width
        );
        evoke_semantic_bmp_write_float(
            destination + 36,
            term->min_impact
        );
        evoke_semantic_bmp_write_float(
            destination + 40,
            term->max_impact
        );
        evoke_semantic_bmp_write_u32(
            destination + 44,
            term->first_block_membership_byte
        );
        evoke_semantic_bmp_write_u32(
            destination + 48,
            term->block_membership_bytes
        );
    }
    if (index->super_ref_bytes > 0)
    {
        memcpy(
            bytes + layout.super_refs_offset,
            index->super_refs,
            index->super_ref_bytes
        );
    }
    if (index->ref_bytes > 0)
    {
        memcpy(bytes + layout.refs_offset, index->refs, index->ref_bytes);
    }
    if (index->block_membership_bytes > 0)
    {
        memcpy(
            bytes + layout.block_membership_offset,
            index->block_membership,
            index->block_membership_bytes
        );
    }
    if (index->doc_delta_bytes > 0)
    {
        memcpy(
            bytes + layout.doc_deltas_offset,
            index->doc_deltas,
            index->doc_delta_bytes
        );
    }
    {
        size_t impact_width = evoke_semantic_bmp_impact_width(
            index->impact_precision
        );

        for (uint64_t posting_index = 0;
             posting_index < index->posting_count;
             posting_index++)
        {
            float quantized;
            evoke_status status = evoke_semantic_impact_encode(
                bytes + layout.impacts_offset +
                    (size_t) posting_index * impact_width,
                impact_width,
                index->impact_precision,
                index->impacts[posting_index],
                &quantized
            );

            if (status != EVOKE_OK ||
                quantized != index->impacts[posting_index])
            {
                return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
            }
        }
    }
    evoke_semantic_bmp_write_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET,
        evoke_semantic_bmp_checksum(bytes, layout.total_size)
    );
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_disk_header *header_out
)
{
    evoke_semantic_bmp_packed_disk_header header;
    evoke_semantic_bmp_packed_layout layout;
    evoke_status status;

    if (bytes == NULL || header_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_BMP_PACKED_MAGIC ||
        evoke_semantic_bmp_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_BMP_PACKED_VERSION ||
        evoke_semantic_bmp_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_BMP_HEADER_SIZE ||
        evoke_semantic_bmp_read_u64(bytes + 128) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&header, 0, sizeof(header));
    header.document_count = evoke_semantic_bmp_read_u32(bytes + 8);
    header.block_count = evoke_semantic_bmp_read_u32(bytes + 12);
    header.superblock_count = evoke_semantic_bmp_read_u32(bytes + 16);
    header.term_count = evoke_semantic_bmp_read_u32(bytes + 20);
    header.ref_count = evoke_semantic_bmp_read_u32(bytes + 24);
    header.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 28);
    header.record_count = evoke_semantic_bmp_read_u32(bytes + 32);
    header.super_ref_bytes = evoke_semantic_bmp_read_u32(bytes + 36);
    header.ref_bytes = evoke_semantic_bmp_read_u32(bytes + 40);
    header.doc_delta_bytes = evoke_semantic_bmp_read_u32(bytes + 44);
    header.block_membership_bytes =
        evoke_semantic_bmp_read_u32(bytes + 104);
    header.impact_precision =
        (evoke_semantic_impact_precision)
            evoke_semantic_bmp_read_u32(bytes + 108);
    if (header.impact_precision == 0)
    {
        header.impact_precision = EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
    header.posting_count = evoke_semantic_bmp_read_u64(bytes + 48);
    header.terms_offset = evoke_semantic_bmp_read_u64(bytes + 56);
    header.super_refs_offset = evoke_semantic_bmp_read_u64(bytes + 64);
    header.refs_offset = evoke_semantic_bmp_read_u64(bytes + 72);
    header.block_membership_offset =
        evoke_semantic_bmp_read_u64(bytes + 120);
    header.doc_deltas_offset = evoke_semantic_bmp_read_u64(bytes + 80);
    header.impacts_offset = evoke_semantic_bmp_read_u64(bytes + 88);
    header.total_size = evoke_semantic_bmp_read_u64(bytes + 96);
    header.checksum = evoke_semantic_bmp_read_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET
    );
    if (header.document_count == 0 || header.block_count == 0 ||
        header.superblock_count == 0 || header.term_count == 0 ||
        header.ref_count == 0 || header.super_ref_count == 0 ||
        header.record_count != header.ref_count ||
        evoke_semantic_bmp_impact_width(header.impact_precision) == 0 ||
        header.super_ref_bytes !=
            (uint64_t) header.super_ref_count *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        header.ref_bytes !=
            (uint64_t) header.ref_count *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE ||
        header.posting_count == 0 || header.posting_count > UINT32_MAX ||
        header.checksum == 0 ||
        header.block_count !=
            ((header.document_count - UINT32_C(1)) >>
             EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT) + UINT32_C(1) ||
        header.superblock_count !=
            ((header.document_count - UINT32_C(1)) >>
             EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT) + UINT32_C(1))
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_semantic_bmp_packed_layout_build(
        header.term_count,
        header.super_ref_bytes,
        header.ref_bytes,
        header.block_membership_bytes,
        header.doc_delta_bytes,
        header.posting_count,
        header.impact_precision,
        &layout
    );
    if (status != EVOKE_OK ||
        header.terms_offset != layout.terms_offset ||
        header.super_refs_offset != layout.super_refs_offset ||
        header.refs_offset != layout.refs_offset ||
        header.block_membership_offset !=
            layout.block_membership_offset ||
        header.doc_deltas_offset != layout.doc_deltas_offset ||
        header.impacts_offset != layout.impacts_offset ||
        header.total_size != layout.total_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_term_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_term *term_out
)
{
    evoke_semantic_bmp_packed_term term;

    if (bytes == NULL || term_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&term, 0, sizeof(term));
    term.term_id = evoke_semantic_bmp_read_u32(bytes + 0);
    term.first_ref = evoke_semantic_bmp_read_u32(bytes + 4);
    term.ref_count = evoke_semantic_bmp_read_u32(bytes + 8);
    term.first_super_ref = evoke_semantic_bmp_read_u32(bytes + 12);
    term.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 16);
    term.first_doc_byte = evoke_semantic_bmp_read_u32(bytes + 20);
    term.posting_count = evoke_semantic_bmp_read_u32(bytes + 24);
    term.first_document = evoke_semantic_bmp_read_u32(bytes + 28);
    term.doc_delta_width = evoke_semantic_bmp_read_u32(bytes + 32);
    term.min_impact = evoke_semantic_bmp_read_float(bytes + 36);
    term.max_impact = evoke_semantic_bmp_read_float(bytes + 40);
    term.first_block_membership_byte =
        evoke_semantic_bmp_read_u32(bytes + 44);
    term.block_membership_bytes =
        evoke_semantic_bmp_read_u32(bytes + 48);
    if (term.ref_count == 0 || term.super_ref_count == 0 ||
        term.posting_count == 0 ||
        (term.posting_count == 1 && term.doc_delta_width != 0) ||
        (term.posting_count > 1 && term.doc_delta_width != 1 &&
         term.doc_delta_width != 2 && term.doc_delta_width != 4) ||
        !isfinite(term.min_impact) || !isfinite(term.max_impact) ||
        term.min_impact > term.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *term_out = term;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_super_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_packed_super_ref *ref_out
)
{
    evoke_semantic_bmp_packed_super_ref ref;

    if (bytes == NULL || ref_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE ||
        !isfinite(term_min_impact) || !isfinite(term_max_impact) ||
        term_min_impact > term_max_impact)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&ref, 0, sizeof(ref));
    ref.superblock_id = evoke_semantic_bmp_read_u32(bytes + 0);
    ref.first_ref = evoke_semantic_bmp_read_u32(bytes + 4);
    ref.first_impact = evoke_semantic_bmp_read_u32(bytes + 8);
    ref.ref_count = evoke_semantic_bmp_read_u16(bytes + 12);
    ref.min_impact = evoke_semantic_bmp_dequantize_min(
        bytes[14],
        term_min_impact,
        term_max_impact
    );
    ref.max_impact = evoke_semantic_bmp_dequantize_max(
        bytes[15],
        term_min_impact,
        term_max_impact
    );
    if (ref.ref_count == 0 || ref.ref_count > UINT16_C(16) ||
        bytes[14] > bytes[15] || !isfinite(ref.min_impact) ||
        !isfinite(ref.max_impact) || ref.min_impact > ref.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_packed_ref *ref_out
)
{
    evoke_semantic_bmp_packed_ref ref;

    if (bytes == NULL || ref_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE ||
        !isfinite(term_min_impact) || !isfinite(term_max_impact) ||
        term_min_impact > term_max_impact)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&ref, 0, sizeof(ref));
    ref.local_block_id = bytes[0];
    ref.document_mask = evoke_semantic_bmp_read_u64(bytes + 1);
    ref.min_impact = evoke_semantic_bmp_dequantize_min(
        bytes[9],
        term_min_impact,
        term_max_impact
    );
    ref.max_impact = evoke_semantic_bmp_dequantize_max(
        bytes[10],
        term_min_impact,
        term_max_impact
    );
    if (ref.local_block_id >= UINT8_C(16) || ref.document_mask == 0 ||
        bytes[9] > bytes[10] || !isfinite(ref.min_impact) ||
        !isfinite(ref.max_impact) || ref.min_impact > ref.max_impact)
    {
        return EVOKE_ERR_FORMAT;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_bmp_packed_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_index *index_out
)
{
    evoke_semantic_bmp_packed_index index;
    evoke_semantic_bmp_packed_layout layout;
    evoke_semantic_impact_precision impact_precision;
    uint64_t expected_checksum;
    evoke_status status;

    if (bytes == NULL || index_out == NULL ||
        size < EVOKE_SEMANTIC_BMP_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_bmp_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_BMP_PACKED_MAGIC ||
        evoke_semantic_bmp_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_BMP_PACKED_VERSION ||
        evoke_semantic_bmp_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_BMP_HEADER_SIZE ||
        evoke_semantic_bmp_read_u64(bytes + 128) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    impact_precision = (evoke_semantic_impact_precision)
        evoke_semantic_bmp_read_u32(bytes + 108);
    if (impact_precision == 0)
    {
        impact_precision = EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
    status = evoke_semantic_bmp_packed_layout_build(
        evoke_semantic_bmp_read_u32(bytes + 20),
        evoke_semantic_bmp_read_u32(bytes + 36),
        evoke_semantic_bmp_read_u32(bytes + 40),
        evoke_semantic_bmp_read_u32(bytes + 104),
        evoke_semantic_bmp_read_u32(bytes + 44),
        evoke_semantic_bmp_read_u64(bytes + 48),
        impact_precision,
        &layout
    );
    if (status != EVOKE_OK || layout.total_size != size ||
        evoke_semantic_bmp_read_u64(bytes + 56) !=
            layout.terms_offset ||
        evoke_semantic_bmp_read_u64(bytes + 64) !=
            layout.super_refs_offset ||
        evoke_semantic_bmp_read_u64(bytes + 72) != layout.refs_offset ||
        evoke_semantic_bmp_read_u64(bytes + 120) !=
            layout.block_membership_offset ||
        evoke_semantic_bmp_read_u64(bytes + 80) !=
            layout.doc_deltas_offset ||
        evoke_semantic_bmp_read_u64(bytes + 88) !=
            layout.impacts_offset ||
        evoke_semantic_bmp_read_u64(bytes + 96) != layout.total_size)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_checksum = evoke_semantic_bmp_read_u64(
        bytes + EVOKE_SEMANTIC_BMP_CHECKSUM_OFFSET
    );
    if (expected_checksum != evoke_semantic_bmp_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_semantic_bmp_packed_index_init(&index);
    index.document_count = evoke_semantic_bmp_read_u32(bytes + 8);
    index.block_count = evoke_semantic_bmp_read_u32(bytes + 12);
    index.superblock_count = evoke_semantic_bmp_read_u32(bytes + 16);
    index.term_count = evoke_semantic_bmp_read_u32(bytes + 20);
    index.ref_count = evoke_semantic_bmp_read_u32(bytes + 24);
    index.super_ref_count = evoke_semantic_bmp_read_u32(bytes + 28);
    index.record_count = evoke_semantic_bmp_read_u32(bytes + 32);
    index.super_ref_bytes = evoke_semantic_bmp_read_u32(bytes + 36);
    index.ref_bytes = evoke_semantic_bmp_read_u32(bytes + 40);
    index.doc_delta_bytes = evoke_semantic_bmp_read_u32(bytes + 44);
    index.block_membership_bytes =
        evoke_semantic_bmp_read_u32(bytes + 104);
    index.posting_count = evoke_semantic_bmp_read_u64(bytes + 48);
    index.impact_precision = impact_precision;
    if (index.posting_count > UINT32_MAX ||
        index.posting_count > SIZE_MAX / sizeof(*index.impacts))
    {
        return EVOKE_ERR_RANGE;
    }

    index.terms = calloc(index.term_count, sizeof(*index.terms));
    index.super_refs = malloc(index.super_ref_bytes);
    index.refs = malloc(index.ref_bytes);
    index.block_membership = malloc(index.block_membership_bytes);
    index.doc_deltas = malloc(index.doc_delta_bytes);
    index.impacts = malloc(
        (size_t) index.posting_count * sizeof(*index.impacts)
    );
    if ((index.term_count > 0 && index.terms == NULL) ||
        (index.super_ref_bytes > 0 && index.super_refs == NULL) ||
        (index.ref_bytes > 0 && index.refs == NULL) ||
        (index.block_membership_bytes > 0 &&
         index.block_membership == NULL) ||
        (index.doc_delta_bytes > 0 && index.doc_deltas == NULL) ||
        (index.posting_count > 0 && index.impacts == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t term_index = 0;
         term_index < index.term_count;
         term_index++)
    {
        const uint8_t *source = bytes + layout.terms_offset +
            (size_t) term_index * EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE;
        evoke_semantic_bmp_packed_term *term = &index.terms[term_index];

        term->term_id = evoke_semantic_bmp_read_u32(source + 0);
        term->first_ref = evoke_semantic_bmp_read_u32(source + 4);
        term->ref_count = evoke_semantic_bmp_read_u32(source + 8);
        term->first_super_ref =
            evoke_semantic_bmp_read_u32(source + 12);
        term->super_ref_count = evoke_semantic_bmp_read_u32(source + 16);
        term->first_doc_byte = evoke_semantic_bmp_read_u32(source + 20);
        term->posting_count = evoke_semantic_bmp_read_u32(source + 24);
        term->first_document = evoke_semantic_bmp_read_u32(source + 28);
        term->doc_delta_width = evoke_semantic_bmp_read_u32(source + 32);
        term->min_impact = evoke_semantic_bmp_read_float(source + 36);
        term->max_impact = evoke_semantic_bmp_read_float(source + 40);
        term->first_block_membership_byte =
            evoke_semantic_bmp_read_u32(source + 44);
        term->block_membership_bytes =
            evoke_semantic_bmp_read_u32(source + 48);
    }
    if (index.super_ref_bytes > 0)
    {
        memcpy(
            index.super_refs,
            bytes + layout.super_refs_offset,
            index.super_ref_bytes
        );
    }
    if (index.ref_bytes > 0)
    {
        memcpy(index.refs, bytes + layout.refs_offset, index.ref_bytes);
    }
    if (index.block_membership_bytes > 0)
    {
        memcpy(
            index.block_membership,
            bytes + layout.block_membership_offset,
            index.block_membership_bytes
        );
    }
    if (index.doc_delta_bytes > 0)
    {
        memcpy(
            index.doc_deltas,
            bytes + layout.doc_deltas_offset,
            index.doc_delta_bytes
        );
    }
    {
        size_t impact_width = evoke_semantic_bmp_impact_width(
            impact_precision
        );

        for (uint64_t posting_index = 0;
             posting_index < index.posting_count;
             posting_index++)
        {
            status = evoke_semantic_impact_decode(
                bytes + layout.impacts_offset +
                    (size_t) posting_index * impact_width,
                impact_width,
                impact_precision,
                &index.impacts[posting_index]
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
    }
    status = evoke_semantic_bmp_packed_index_validate(&index);
    if (status == EVOKE_OK)
    {
        evoke_semantic_bmp_packed_index_free(index_out);
        *index_out = index;
        evoke_semantic_bmp_packed_index_init(&index);
    }

cleanup:
    evoke_semantic_bmp_packed_index_free(&index);
    return status;
}

evoke_status
evoke_semantic_bmp_packed_term_materialize(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t term_index,
    uint32_t *document_ids_out,
    evoke_posting_value *values_out,
    size_t capacity
)
{
    const evoke_semantic_bmp_packed_term *term;
    const uint8_t *super_source;
    uint32_t impact_offset;
    uint32_t document_id;
    size_t doc_offset;
    size_t doc_limit;

    if (index == NULL || document_ids_out == NULL || values_out == NULL ||
        term_index >= index->term_count)
    {
        return EVOKE_ERR_INVALID;
    }
    term = &index->terms[term_index];
    if (capacity < term->posting_count || term->super_ref_count == 0 ||
        term->first_super_ref >= index->super_ref_count ||
        term->first_doc_byte > index->doc_delta_bytes)
    {
        return EVOKE_ERR_RANGE;
    }
    super_source = index->super_refs +
        (size_t) term->first_super_ref *
            EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
    impact_offset = evoke_semantic_bmp_read_u32(super_source + 8);
    doc_offset = term->first_doc_byte;
    doc_limit = term_index + 1U < index->term_count
        ? index->terms[term_index + 1U].first_doc_byte
        : index->doc_delta_bytes;
    if (doc_offset > doc_limit || doc_limit > index->doc_delta_bytes ||
        impact_offset > index->posting_count ||
        term->posting_count > index->posting_count - impact_offset ||
        (uint64_t) (term->posting_count - 1U) *
            term->doc_delta_width != (uint64_t) doc_limit - doc_offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    document_id = term->first_document;
    for (uint32_t posting_index = 0;
         posting_index < term->posting_count;
         posting_index++)
    {
        if (posting_index > 0)
        {
            uint32_t delta = evoke_semantic_bmp_read_delta(
                index->doc_deltas,
                doc_offset,
                term->doc_delta_width
            );

            doc_offset += term->doc_delta_width;
            if (delta == 0 || document_id >= index->document_count ||
                delta > index->document_count - 1U - document_id)
            {
                return EVOKE_ERR_FORMAT;
            }
            document_id += delta;
        }
        if (document_id >= index->document_count ||
            !isfinite(index->impacts[impact_offset + posting_index]) ||
            index->impacts[impact_offset + posting_index] == 0.0f)
        {
            return EVOKE_ERR_FORMAT;
        }
        document_ids_out[posting_index] = document_id;
        values_out[posting_index].impact =
            index->impacts[impact_offset + posting_index];
    }
    return doc_offset == doc_limit ? EVOKE_OK : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_semantic_bmp_packed_taat_topk(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    evoke_topk_accumulator accumulator;
    evoke_semantic_bmp_stats stats;
    float *scores = NULL;
    evoke_status status = EVOKE_OK;

    memset(&accumulator, 0, sizeof(accumulator));
    memset(&stats, 0, sizeof(stats));
    if (index == NULL || result_out == NULL || stats_out == NULL ||
        (query_count > 0 && query_ids == NULL) ||
        k > index->document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    memset(stats_out, 0, sizeof(*stats_out));
    if (index->document_count == 0 ||
        (index->super_ref_bytes > 0 && index->super_refs == NULL) ||
        (index->ref_bytes > 0 && index->refs == NULL) ||
        (index->doc_delta_bytes > 0 && index->doc_deltas == NULL) ||
        (index->posting_count > 0 && index->impacts == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (k == 0 || query_count == 0)
    {
        return EVOKE_OK;
    }
    scores = calloc(index->document_count, sizeof(*scores));
    if (scores == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    for (size_t query_index = 0;
         query_index < query_count;
         query_index++)
    {
        const evoke_semantic_bmp_packed_term *term;
        float weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(weight))
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        if (weight == 0.0f)
        {
            continue;
        }
        term = evoke_semantic_bmp_packed_find_term(
            index,
            query_ids[query_index]
        );
        if (term == NULL)
        {
            continue;
        }
        if (term->super_ref_count == 0 ||
            term->first_super_ref >= index->super_ref_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        {
            const uint8_t *super_source = index->super_refs +
                (size_t) term->first_super_ref *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
            uint32_t impact_offset = evoke_semantic_bmp_read_u32(
                super_source + 8
            );
            size_t doc_offset = term->first_doc_byte;
            size_t doc_limit = term + 1U <
                    index->terms + index->term_count
                ? term[1].first_doc_byte
                : index->doc_delta_bytes;
            uint32_t document_id;

            if (term->posting_count == 0 ||
                (term->posting_count == 1 && term->doc_delta_width != 0) ||
                (term->posting_count > 1 &&
                 term->doc_delta_width != 1 &&
                 term->doc_delta_width != 2 &&
                 term->doc_delta_width != 4) ||
                doc_offset > doc_limit ||
                doc_limit > index->doc_delta_bytes ||
                (uint64_t) (term->posting_count - 1U) *
                    term->doc_delta_width !=
                    (uint64_t) doc_limit - doc_offset ||
                impact_offset > index->posting_count ||
                term->posting_count >
                    index->posting_count - impact_offset)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            document_id = term->first_document;
            if (document_id >= index->document_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            scores[document_id] += weight * index->impacts[impact_offset];
            if (term->doc_delta_width == 1U)
            {
                for (uint32_t posting_index = 1;
                     posting_index < term->posting_count;
                     posting_index++)
                {
                    uint32_t delta = index->doc_deltas[doc_offset++];

                    if (delta == 0 || document_id >= index->document_count ||
                        delta > index->document_count - 1U - document_id)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    document_id += delta;
                    scores[document_id] += weight *
                        index->impacts[impact_offset + posting_index];
                }
            }
            else if (term->doc_delta_width == 2U)
            {
                for (uint32_t posting_index = 1;
                     posting_index < term->posting_count;
                     posting_index++)
                {
                    uint32_t delta = evoke_semantic_bmp_read_u16(
                        index->doc_deltas + doc_offset
                    );

                    doc_offset += 2U;
                    if (delta == 0 || document_id >= index->document_count ||
                        delta > index->document_count - 1U - document_id)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    document_id += delta;
                    scores[document_id] += weight *
                        index->impacts[impact_offset + posting_index];
                }
            }
            else if (term->doc_delta_width == 4U)
            {
                for (uint32_t posting_index = 1;
                     posting_index < term->posting_count;
                     posting_index++)
                {
                    uint32_t delta = evoke_semantic_bmp_read_u32(
                        index->doc_deltas + doc_offset
                    );

                    doc_offset += 4U;
                    if (delta == 0 || document_id >= index->document_count ||
                        delta > index->document_count - 1U - document_id)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    document_id += delta;
                    scores[document_id] += weight *
                        index->impacts[impact_offset + posting_index];
                }
            }
            stats.postings_examined += term->posting_count;
            if (doc_offset != doc_limit)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
        }
    }
    status = evoke_topk_accumulator_init(&accumulator, k);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (uint32_t document_id = 0;
         document_id < index->document_count;
         document_id++)
    {
        if (scores[document_id] <= 0.0f)
        {
            continue;
        }
        status = evoke_topk_accumulator_offer(
            &accumulator,
            scores[document_id],
            document_id,
            document_id
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_topk_accumulator_finish(
        &accumulator,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *stats_out = stats;
    }

cleanup:
    evoke_topk_accumulator_free(&accumulator);
    free(scores);
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    return status;
}

typedef struct evoke_semantic_bmp_packed_score_ref
{
    uint32_t query_index;
    uint32_t impact_offset;
    uint32_t next_index;
    uint64_t document_mask;
} evoke_semantic_bmp_packed_score_ref;

static int
evoke_semantic_bmp_compare_seeds(const void *left, const void *right)
{
    const evoke_semantic_bmp_seed *a = left;
    const evoke_semantic_bmp_seed *b = right;

    return (a->document_id > b->document_id) -
        (a->document_id < b->document_id);
}

static bool
evoke_semantic_bmp_seed_contains(
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    uint32_t document_id
)
{
    size_t low = 0;
    size_t high = seed_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2U;

        if (seeds[middle].document_id < document_id)
        {
            low = middle + 1U;
        }
        else
        {
            high = middle;
        }
    }
    return low < seed_count && seeds[low].document_id == document_id;
}

static bool
evoke_semantic_bmp_bound_is_prunable(
    float upper_bound,
    float threshold,
    float max_boundary_error
)
{
    double relaxed_threshold;

    if (max_boundary_error <= 0.0f)
    {
        return upper_bound < threshold;
    }
    relaxed_threshold = (double) threshold + max_boundary_error;
    return (double) upper_bound <= relaxed_threshold;
}

static void
evoke_semantic_bmp_note_approximate_skip(
    evoke_semantic_bmp_stats *stats,
    float upper_bound,
    float threshold
)
{
    if (upper_bound < threshold)
    {
        return;
    }
    stats->approximate_prune_events++;
    if (upper_bound > stats->max_approximate_skipped_bound)
    {
        stats->max_approximate_skipped_bound = upper_bound;
    }
}

static float
evoke_semantic_bmp_norm_envelope(
    float current_bound,
    double query_l2,
    double query_linf,
    float max_document_l1,
    float max_document_l2,
    evoke_semantic_bmp_stats *stats
)
{
    double l2_bound = query_l2 * max_document_l2;
    double l1_bound = query_linf * max_document_l1;
    double norm_bound = fmin(l2_bound, l1_bound);
    float conservative;

    if (!isfinite(norm_bound) || norm_bound > FLT_MAX)
    {
        return current_bound;
    }
    conservative = nextafterf((float) nextafter(norm_bound, INFINITY),
                              INFINITY);
    if (conservative < current_bound)
    {
        stats->norm_bound_reductions++;
        return conservative;
    }
    return current_bound;
}

static evoke_status
evoke_semantic_bmp_packed_score_block(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t block_id,
    const evoke_semantic_bmp_query_term *query_terms,
    const evoke_semantic_bmp_packed_score_ref *score_refs,
    uint32_t first_score_ref,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    evoke_topk_accumulator *accumulator,
    evoke_semantic_bmp_stats *stats
)
{
    float block_scores[
        UINT32_C(1) << EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT
    ] = {0};
    uint32_t first_document = block_id <<
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
    uint32_t document_limit = first_document +
        (UINT32_C(1) << EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT);

    if (document_limit > index->document_count)
    {
        document_limit = index->document_count;
    }
    for (uint32_t score_ref_index = first_score_ref;
         score_ref_index != UINT32_MAX;
         score_ref_index = score_refs[score_ref_index].next_index)
    {
        const evoke_semantic_bmp_packed_score_ref *score_ref =
            &score_refs[score_ref_index];
        const evoke_semantic_bmp_query_term *query =
            &query_terms[score_ref->query_index];
        uint64_t mask = score_ref->document_mask;
        uint32_t impact_offset = score_ref->impact_offset;
        uint32_t count = evoke_semantic_bmp_mask_count(mask);

        if (impact_offset > index->posting_count ||
            count > index->posting_count - impact_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        stats->forward_records_examined++;
        if (query->weight != 0.0f)
        {
            uint32_t local_impact = 0;

            for (uint32_t local_document = 0;
                 mask != 0;
                 local_document++, mask >>= 1)
            {
                if ((mask & UINT64_C(1)) == 0)
                {
                    continue;
                }
                block_scores[local_document] += query->weight *
                    index->impacts[impact_offset + local_impact];
                local_impact++;
                stats->postings_examined++;
            }
        }
    }
    for (uint32_t document_id = first_document;
         document_id < document_limit;
         document_id++)
    {
        float score = block_scores[document_id - first_document];
        evoke_status status;

        if (score <= 0.0f)
        {
            continue;
        }
        if (evoke_semantic_bmp_seed_contains(
                seeds,
                seed_count,
                document_id
            ))
        {
            continue;
        }
        status = evoke_topk_accumulator_offer(
            accumulator,
            score,
            document_id,
            document_id
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    stats->blocks_scored++;
    return EVOKE_OK;
}

static evoke_status
evoke_semantic_bmp_packed_topk_internal(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    float max_boundary_error,
    const evoke_semantic_bmp_norm_bounds *norm_bounds,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    evoke_semantic_bmp_query_term *query_terms = NULL;
    evoke_semantic_bmp_seed *ordered_seeds = NULL;
    evoke_semantic_bmp_packed_score_ref *score_refs = NULL;
    evoke_semantic_bmp_ranked_block *ranked_superblocks = NULL;
    float *super_upper_bounds = NULL;
    evoke_topk_accumulator accumulator;
    evoke_semantic_bmp_stats stats;
    size_t unique_query_count = 0;
    size_t ranked_superblock_count = 0;
    double query_l2_squared = 0.0;
    double query_l2 = 0.0;
    double query_linf = 0.0;
    bool fallback_to_taat = false;
    evoke_status status = EVOKE_OK;

    memset(&accumulator, 0, sizeof(accumulator));
    memset(&stats, 0, sizeof(stats));
    if (index == NULL || result_out == NULL || stats_out == NULL ||
        (query_count > 0 && query_ids == NULL) ||
        (seed_count > 0 && seeds == NULL) ||
        !isfinite(max_boundary_error) || max_boundary_error < 0.0f ||
        (norm_bounds != NULL &&
         (norm_bounds->block_count != index->block_count ||
          norm_bounds->superblock_count != index->superblock_count ||
          norm_bounds->block_max_l1 == NULL ||
          norm_bounds->block_max_l2 == NULL ||
          norm_bounds->superblock_max_l1 == NULL ||
          norm_bounds->superblock_max_l2 == NULL)) ||
        seed_count > SIZE_MAX / sizeof(*ordered_seeds) ||
        query_count > SIZE_MAX / sizeof(*query_terms) ||
        k > index->document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    memset(stats_out, 0, sizeof(*stats_out));
    if (index->document_count == 0 ||
        (index->super_ref_bytes > 0 && index->super_refs == NULL) ||
        (index->ref_bytes > 0 && index->refs == NULL) ||
        (index->posting_count > 0 && index->impacts == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (k == 0 || query_count == 0)
    {
        return EVOKE_OK;
    }
    if (seed_count > 0)
    {
        ordered_seeds = malloc(seed_count * sizeof(*ordered_seeds));
        if (ordered_seeds == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        memcpy(ordered_seeds, seeds, seed_count * sizeof(*ordered_seeds));
        qsort(
            ordered_seeds,
            seed_count,
            sizeof(*ordered_seeds),
            evoke_semantic_bmp_compare_seeds
        );
        for (size_t seed_index = 0;
             seed_index < seed_count;
             seed_index++)
        {
            if (ordered_seeds[seed_index].document_id >=
                    index->document_count ||
                !isfinite(ordered_seeds[seed_index].score) ||
                ordered_seeds[seed_index].score <= 0.0f ||
                (seed_index > 0 &&
                 ordered_seeds[seed_index - 1U].document_id ==
                    ordered_seeds[seed_index].document_id))
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }
        }
    }
    query_terms = malloc(query_count * sizeof(*query_terms));
    super_upper_bounds = calloc(
        index->superblock_count,
        sizeof(*super_upper_bounds)
    );
    ranked_superblocks = malloc(
        (size_t) index->superblock_count * sizeof(*ranked_superblocks)
    );
    if (query_terms == NULL || super_upper_bounds == NULL ||
        ranked_superblocks == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t query_index = 0; query_index < query_count; query_index++)
    {
        float weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(weight))
        {
            status = EVOKE_ERR_INVALID;
            goto cleanup;
        }
        query_terms[query_index].term_id = query_ids[query_index];
        query_terms[query_index].weight = weight;
    }
    qsort(
        query_terms,
        query_count,
        sizeof(*query_terms),
        evoke_semantic_bmp_compare_query_terms
    );
    for (size_t query_index = 0; query_index < query_count; query_index++)
    {
        if (unique_query_count > 0 &&
            query_terms[unique_query_count - 1U].term_id ==
                query_terms[query_index].term_id)
        {
            query_terms[unique_query_count - 1U].weight +=
                query_terms[query_index].weight;
            if (!isfinite(query_terms[unique_query_count - 1U].weight))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
        else
        {
            query_terms[unique_query_count++] = query_terms[query_index];
        }
    }
    for (size_t query_index = 0;
         query_index < unique_query_count;
         query_index++)
    {
        double absolute_weight = fabs((double) query_terms[query_index].weight);

        query_l2_squared += absolute_weight * absolute_weight;
        query_linf = fmax(query_linf, absolute_weight);
    }
    query_l2 = sqrt(query_l2_squared);
    if (unique_query_count > SIZE_MAX / 16U / sizeof(*score_refs))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    score_refs = malloc(
        unique_query_count * 16U * sizeof(*score_refs)
    );
    if (unique_query_count > 0 && score_refs == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (size_t query_index = 0;
         query_index < unique_query_count;
         query_index++)
    {
        const evoke_semantic_bmp_query_term *query =
            &query_terms[query_index];
        const evoke_semantic_bmp_packed_term *term =
            evoke_semantic_bmp_packed_find_term(index, query->term_id);

        if (term == NULL || query->weight == 0.0f)
        {
            continue;
        }
        if (term->first_super_ref > index->super_ref_count ||
            term->super_ref_count >
                index->super_ref_count - term->first_super_ref)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        for (uint32_t ref_index = 0;
             ref_index < term->super_ref_count;
             ref_index++)
        {
            const uint8_t *super_source = index->super_refs +
                (size_t) (term->first_super_ref + ref_index) *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
            uint32_t superblock_id;
            float min_impact;
            float max_impact;
            float contribution;
            double next;

            superblock_id = evoke_semantic_bmp_read_u32(
                super_source + 0
            );
            if (superblock_id >= index->superblock_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            min_impact = evoke_semantic_bmp_dequantize_min(
                super_source[14],
                term->min_impact,
                term->max_impact
            );
            max_impact = evoke_semantic_bmp_dequantize_max(
                super_source[15],
                term->min_impact,
                term->max_impact
            );
            contribution = evoke_semantic_bmp_contribution_bound(
                query->weight,
                min_impact,
                max_impact
            );
            next = nextafter(
                (double) super_upper_bounds[superblock_id] +
                    contribution,
                INFINITY
            );
            super_upper_bounds[superblock_id] = next > FLT_MAX
                ? INFINITY
                : nextafterf((float) next, INFINITY);
            stats.super_bound_entries_visited++;
        }
    }
    for (uint32_t superblock_id = 0;
         superblock_id < index->superblock_count;
         superblock_id++)
    {
        if (norm_bounds != NULL)
        {
            float max_l1 = norm_bounds->superblock_max_l1[superblock_id];
            float max_l2 = norm_bounds->superblock_max_l2[superblock_id];

            if (!isfinite(max_l1) || max_l1 < 0.0f ||
                !isfinite(max_l2) || max_l2 < 0.0f)
            {
                status = EVOKE_ERR_INVALID;
                goto cleanup;
            }
            super_upper_bounds[superblock_id] =
                evoke_semantic_bmp_norm_envelope(
                    super_upper_bounds[superblock_id],
                    query_l2,
                    query_linf,
                    max_l1,
                    max_l2,
                    &stats
                );
        }
        if (super_upper_bounds[superblock_id] > 0.0f)
        {
            ranked_superblocks[ranked_superblock_count].block_id =
                superblock_id;
            ranked_superblocks[ranked_superblock_count].upper_bound =
                super_upper_bounds[superblock_id];
            ranked_superblock_count++;
        }
    }
    stats.superblocks_with_positive_bound = ranked_superblock_count;
    qsort(
        ranked_superblocks,
        ranked_superblock_count,
        sizeof(*ranked_superblocks),
        evoke_semantic_bmp_compare_ranked_blocks
    );
    status = evoke_topk_accumulator_init(&accumulator, k);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (size_t seed_index = 0;
         seed_index < seed_count;
         seed_index++)
    {
        status = evoke_topk_accumulator_offer(
            &accumulator,
            ordered_seeds[seed_index].score,
            ordered_seeds[seed_index].document_id,
            ordered_seeds[seed_index].document_id
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        stats.seed_documents_offered++;
    }
    for (size_t super_rank = 0;
         super_rank < ranked_superblock_count;
         super_rank++)
    {
        const evoke_semantic_bmp_ranked_block *ranked_super =
            &ranked_superblocks[super_rank];
        evoke_semantic_bmp_ranked_block ranked_children[16];
        float child_bounds[16] = {0};
        uint32_t score_ref_heads[16];
        uint32_t score_ref_tails[16];
        uint32_t score_ref_count = 0;
        size_t ranked_child_count = 0;
        uint32_t superblock_id = ranked_super->block_id;

        memset(score_ref_heads, UINT8_MAX, sizeof(score_ref_heads));
        memset(score_ref_tails, UINT8_MAX, sizeof(score_ref_tails));

        if (super_rank >= EVOKE_SEMANTIC_BMP_MAX_UNPRUNED_SUPERBLOCKS &&
            accumulator.len == accumulator.capacity &&
            !evoke_semantic_bmp_bound_is_prunable(
                ranked_super->upper_bound,
                accumulator.heap[0].score,
                max_boundary_error
            ))
        {
            fallback_to_taat = true;
            break;
        }
        if (accumulator.len == accumulator.capacity &&
            evoke_semantic_bmp_bound_is_prunable(
                ranked_super->upper_bound,
                accumulator.heap[0].score,
                max_boundary_error
            ))
        {
            evoke_semantic_bmp_note_approximate_skip(
                &stats,
                ranked_super->upper_bound,
                accumulator.heap[0].score
            );
            stats.superblocks_skipped =
                ranked_superblock_count - super_rank;
            break;
        }
        for (size_t query_index = 0;
             query_index < unique_query_count;
             query_index++)
        {
            const evoke_semantic_bmp_query_term *query =
                &query_terms[query_index];
            const evoke_semantic_bmp_packed_term *term =
                evoke_semantic_bmp_packed_find_term(index, query->term_id);
            uint32_t low = 0;
            uint32_t high;
            uint32_t child_count;
            const uint8_t *super_source;
            uint32_t first_ref;
            uint32_t impact_offset;
            uint32_t previous_local_block = UINT32_MAX;

            if (term == NULL || query->weight == 0.0f)
            {
                continue;
            }
            high = term->super_ref_count;
            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2U;
                size_t middle_offset =
                    (size_t) (term->first_super_ref + middle) *
                    EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
                uint32_t middle_superblock = evoke_semantic_bmp_read_u32(
                    index->super_refs + middle_offset
                );

                if (middle_superblock < superblock_id)
                {
                    low = middle + 1U;
                }
                else
                {
                    high = middle;
                }
            }
            if (low >= term->super_ref_count)
            {
                continue;
            }
            super_source = index->super_refs +
                (size_t) (term->first_super_ref + low) *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE;
            if (evoke_semantic_bmp_read_u32(
                    super_source + 0
                ) != superblock_id)
            {
                continue;
            }
            first_ref = evoke_semantic_bmp_read_u32(
                super_source + 4
            );
            impact_offset = evoke_semantic_bmp_read_u32(super_source + 8);
            child_count = evoke_semantic_bmp_read_u16(
                super_source + 12
            );
            if (first_ref > index->ref_count ||
                child_count > index->ref_count - first_ref ||
                impact_offset > index->posting_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            for (uint32_t child = 0; child < child_count; child++)
            {
                const uint8_t *ref_source = index->refs +
                    (size_t) (first_ref + child) *
                    EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE;
                uint32_t local_block = ref_source[0];
                uint64_t document_mask;
                uint32_t impact_count;
                evoke_semantic_bmp_packed_score_ref *score_ref;
                float min_impact;
                float max_impact;
                float contribution;
                double next;

                if (local_block >= UINT32_C(16) ||
                    (child > 0 && local_block <= previous_local_block))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                previous_local_block = local_block;
                document_mask = evoke_semantic_bmp_read_u64(
                    ref_source + 1
                );
                impact_count = evoke_semantic_bmp_mask_count(document_mask);
                if (document_mask == 0 ||
                    impact_offset > index->posting_count ||
                    impact_count > index->posting_count - impact_offset)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                min_impact = evoke_semantic_bmp_dequantize_min(
                    ref_source[9],
                    term->min_impact,
                    term->max_impact
                );
                max_impact = evoke_semantic_bmp_dequantize_max(
                    ref_source[10],
                    term->min_impact,
                    term->max_impact
                );
                contribution = evoke_semantic_bmp_contribution_bound(
                    query->weight,
                    min_impact,
                    max_impact
                );
                next = nextafter(
                    (double) child_bounds[local_block] +
                        contribution,
                    INFINITY
                );
                child_bounds[local_block] = next > FLT_MAX
                    ? INFINITY
                    : nextafterf((float) next, INFINITY);
                if (score_ref_count >= unique_query_count * 16U)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                score_ref = &score_refs[score_ref_count];
                score_ref->query_index = (uint32_t) query_index;
                score_ref->impact_offset = impact_offset;
                score_ref->document_mask = document_mask;
                score_ref->next_index = UINT32_MAX;
                if (score_ref_heads[local_block] == UINT32_MAX)
                {
                    score_ref_heads[local_block] = score_ref_count;
                }
                else
                {
                    score_refs[
                        score_ref_tails[local_block]
                    ].next_index = score_ref_count;
                }
                score_ref_tails[local_block] = score_ref_count;
                score_ref_count++;
                impact_offset += impact_count;
                stats.bound_entries_visited++;
            }
        }
        for (uint32_t child = 0; child < 16U; child++)
        {
            uint32_t block_id = (superblock_id << 4U) + child;

            if (block_id < index->block_count && child_bounds[child] > 0.0f)
            {
                if (norm_bounds != NULL)
                {
                    float max_l1 = norm_bounds->block_max_l1[block_id];
                    float max_l2 = norm_bounds->block_max_l2[block_id];

                    if (!isfinite(max_l1) || max_l1 < 0.0f ||
                        !isfinite(max_l2) || max_l2 < 0.0f)
                    {
                        status = EVOKE_ERR_INVALID;
                        goto cleanup;
                    }
                    child_bounds[child] =
                        evoke_semantic_bmp_norm_envelope(
                            child_bounds[child],
                            query_l2,
                            query_linf,
                            max_l1,
                            max_l2,
                            &stats
                        );
                }
                ranked_children[ranked_child_count].block_id = block_id;
                ranked_children[ranked_child_count].upper_bound =
                    child_bounds[child];
                ranked_child_count++;
            }
        }
        stats.blocks_with_positive_bound += ranked_child_count;
        qsort(
            ranked_children,
            ranked_child_count,
            sizeof(*ranked_children),
            evoke_semantic_bmp_compare_ranked_blocks
        );
        for (size_t child_rank = 0;
             child_rank < ranked_child_count;
             child_rank++)
        {
            const evoke_semantic_bmp_ranked_block *ranked =
                &ranked_children[child_rank];

            if (accumulator.len == accumulator.capacity &&
                evoke_semantic_bmp_bound_is_prunable(
                    ranked->upper_bound,
                    accumulator.heap[0].score,
                    max_boundary_error
                ))
            {
                evoke_semantic_bmp_note_approximate_skip(
                    &stats,
                    ranked->upper_bound,
                    accumulator.heap[0].score
                );
                stats.blocks_skipped +=
                    ranked_child_count - child_rank;
                break;
            }
            status = evoke_semantic_bmp_packed_score_block(
                index,
                ranked->block_id,
                query_terms,
                score_refs,
                score_ref_heads[ranked->block_id & UINT32_C(15)],
                ordered_seeds,
                seed_count,
                &accumulator,
                &stats
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        stats.superblocks_scored++;
    }
    if (fallback_to_taat)
    {
        evoke_topk_accumulator_free(&accumulator);
        free(super_upper_bounds);
        free(ranked_superblocks);
        free(score_refs);
        free(query_terms);
        free(ordered_seeds);
        status = evoke_semantic_bmp_packed_taat_topk(
            index,
            query_ids,
            query_weights,
            query_count,
            k,
            result_out,
            stats_out
        );
        if (status == EVOKE_OK)
        {
            stats_out->adaptive_fallbacks = 1;
        }
        return status;
    }
    if (accumulator.len == accumulator.capacity)
    {
        stats.final_kth_score = accumulator.heap[0].score;
    }
    status = evoke_topk_accumulator_finish(
        &accumulator,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *stats_out = stats;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_topk_accumulator_free(&accumulator);
    free(super_upper_bounds);
    free(ranked_superblocks);
    free(score_refs);
    free(query_terms);
    free(ordered_seeds);
    return status;
}

evoke_status
evoke_semantic_bmp_packed_topk(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    return evoke_semantic_bmp_packed_topk_internal(
        index,
        query_ids,
        query_weights,
        query_count,
        k,
        NULL,
        0,
        0.0f,
        NULL,
        result_out,
        stats_out
    );
}

evoke_status
evoke_semantic_bmp_packed_topk_seeded(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    return evoke_semantic_bmp_packed_topk_internal(
        index,
        query_ids,
        query_weights,
        query_count,
        k,
        seeds,
        seed_count,
        0.0f,
        NULL,
        result_out,
        stats_out
    );
}

evoke_status
evoke_semantic_bmp_packed_topk_seeded_bounded(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    float max_boundary_error,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    return evoke_semantic_bmp_packed_topk_internal(
        index,
        query_ids,
        query_weights,
        query_count,
        k,
        seeds,
        seed_count,
        max_boundary_error,
        NULL,
        result_out,
        stats_out
    );
}

evoke_status
evoke_semantic_bmp_packed_topk_norm(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_norm_bounds *norm_bounds,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
)
{
    return evoke_semantic_bmp_packed_topk_internal(
        index,
        query_ids,
        query_weights,
        query_count,
        k,
        NULL,
        0,
        0.0f,
        norm_bounds,
        result_out,
        stats_out
    );
}
