#include "evoke_scope.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SCOPE_MAGIC UINT32_C(0x50435332)
#define EVOKE_SCOPE_VERSION UINT16_C(EVOKE_SCOPE_CURRENT_VERSION)
#define EVOKE_SCOPE_CHECKSUM_OFFSET 72U

static void
evoke_scope_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) (value & UINT16_C(0xff));
    bytes[1] = (uint8_t) ((value >> 8) & UINT16_C(0xff));
}

static void
evoke_scope_write_u32(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8U));
    }
}

static void
evoke_scope_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8U));
    }
}

static uint16_t
evoke_scope_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_scope_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_scope_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8U);
    }
    return value;
}

static uint64_t
evoke_scope_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_SCOPE_CHECKSUM_OFFSET &&
            index < EVOKE_SCOPE_CHECKSUM_OFFSET + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static bool
evoke_scope_add_size(size_t *total, size_t value)
{
    if (*total > SIZE_MAX - value)
    {
        return false;
    }
    *total += value;
    return true;
}

static uint8_t
evoke_scope_ascii_lower(uint8_t value)
{
    if (value >= (uint8_t) 'A' && value <= (uint8_t) 'Z')
    {
        return (uint8_t) (value + ((uint8_t) 'a' - (uint8_t) 'A'));
    }
    return value;
}

uint32_t
evoke_scope_ascii_gram_bit(uint8_t first, uint8_t second, uint8_t third)
{
    uint32_t hash = UINT32_C(2166136261);
    const uint8_t gram[3] = {
        evoke_scope_ascii_lower(first),
        evoke_scope_ascii_lower(second),
        evoke_scope_ascii_lower(third)
    };

    for (size_t index = 0; index < sizeof(gram); index++)
    {
        hash ^= gram[index];
        hash *= UINT32_C(16777619);
    }
    return hash % (EVOKE_SCOPE_GRAM_FILTER_BYTES * 8U);
}

bool
evoke_scope_ascii_ilike_contains_pattern(
    const uint8_t *pattern,
    size_t pattern_size,
    const uint8_t **literal_out,
    size_t *literal_size_out
)
{
    if (pattern == NULL || pattern_size < 2 || literal_out == NULL ||
        literal_size_out == NULL || pattern[0] != (uint8_t) '%' ||
        pattern[pattern_size - 1U] != (uint8_t) '%')
    {
        return false;
    }
    for (size_t offset = 1; offset + 1U < pattern_size; offset++)
    {
        uint8_t value = pattern[offset];

        if (value >= UINT8_C(0x80) || value == (uint8_t) '%' ||
            value == (uint8_t) '_' || value == (uint8_t) '\\')
        {
            return false;
        }
    }
    *literal_out = pattern + 1U;
    *literal_size_out = pattern_size - 2U;
    return true;
}

bool
evoke_scope_ascii_case_insensitive_contains(
    const uint8_t *value,
    size_t value_size,
    const uint8_t *literal,
    size_t literal_size,
    bool *matches_out
)
{
    const size_t high_bits =
        (SIZE_MAX / UINT8_MAX) * (size_t) UINT8_C(0x80);
    uint8_t first_lower;
    uint8_t first_upper;

    if ((value == NULL && value_size != 0) ||
        (literal == NULL && literal_size != 0) || matches_out == NULL)
    {
        return false;
    }
    for (size_t offset = 0; offset + sizeof(size_t) <= value_size;
         offset += sizeof(size_t))
    {
        size_t word;

        memcpy(&word, value + offset, sizeof(word));
        if ((word & high_bits) != 0)
        {
            return false;
        }
    }
    for (size_t offset = value_size - value_size % sizeof(size_t);
         offset < value_size;
         offset++)
    {
        if (value[offset] >= UINT8_C(0x80))
        {
            return false;
        }
    }
    for (size_t offset = 0; offset < literal_size; offset++)
    {
        if (literal[offset] >= UINT8_C(0x80))
        {
            return false;
        }
    }
    *matches_out = literal_size == 0;
    if (literal_size == 0 || literal_size > value_size)
    {
        return true;
    }
    first_lower = evoke_scope_ascii_lower(literal[0]);
    first_upper = first_lower >= (uint8_t) 'a' &&
            first_lower <= (uint8_t) 'z'
        ? (uint8_t) (first_lower - ((uint8_t) 'a' - (uint8_t) 'A'))
        : first_lower;
    for (const uint8_t *cursor = value;
         (size_t) (cursor - value) <= value_size - literal_size;)
    {
        size_t remaining = value_size - (size_t) (cursor - value);
        size_t searchable = remaining - literal_size + 1U;
        const uint8_t *lower_match = memchr(
            cursor,
            first_lower,
            searchable
        );
        const uint8_t *upper_match = first_upper == first_lower
            ? NULL
            : memchr(cursor, first_upper, searchable);
        const uint8_t *start;
        bool matches = true;

        if (lower_match == NULL)
        {
            start = upper_match;
        }
        else if (upper_match == NULL || lower_match < upper_match)
        {
            start = lower_match;
        }
        else
        {
            start = upper_match;
        }
        if (start == NULL)
        {
            break;
        }
        for (size_t offset = 1; offset < literal_size; offset++)
        {
            if (evoke_scope_ascii_lower(start[offset]) !=
                evoke_scope_ascii_lower(literal[offset]))
            {
                matches = false;
                break;
            }
        }
        if (matches)
        {
            *matches_out = true;
            return true;
        }
        cursor = start + 1U;
    }
    *matches_out = false;
    return true;
}

bool
evoke_scope_header_is_current(const evoke_scope_header *header)
{
    return header != NULL &&
        header->version == EVOKE_SCOPE_CURRENT_VERSION &&
        header->gram_block_values == EVOKE_SCOPE_GRAM_BLOCK_VALUES &&
        header->gram_filter_bytes == EVOKE_SCOPE_GRAM_FILTER_BYTES;
}

static void
evoke_scope_add_value_grams(
    uint8_t *filter,
    const uint8_t *value,
    uint32_t value_size,
    bool collation_folded,
    uint32_t *value_mask_out
)
{
    bool ascii_only = true;
    uint32_t value_mask = 0;

    if (value_mask_out == NULL)
    {
        return;
    }
    *value_mask_out = 0;
    if (filter == NULL || value == NULL || value_size < 3)
    {
        return;
    }
    for (uint32_t offset = 0; offset < value_size; offset++)
    {
        if (value[offset] >= UINT8_C(0x80))
        {
            ascii_only = false;
            break;
        }
    }
    if (!ascii_only && !collation_folded)
    {
        /*
         * PostgreSQL UTF-8 ILIKE lowercases complete strings.  Some Unicode
         * characters, such as the Kelvin sign, can therefore match ASCII.
         * Saturating this block preserves the no-false-negative contract
         * without baking a PostgreSQL collation into the durable codec.
         */
        memset(filter, UINT8_MAX, EVOKE_SCOPE_GRAM_FILTER_BYTES);
        *value_mask_out = UINT32_MAX;
        return;
    }
    for (uint32_t offset = 0; offset + 2U < value_size; offset++)
    {
        uint32_t bit;
        bit = evoke_scope_ascii_gram_bit(
            value[offset],
            value[offset + 1U],
            value[offset + 2U]
        );
        filter[bit / 8U] |= (uint8_t) (UINT8_C(1) << (bit % 8U));
        value_mask |= UINT32_C(1) << (bit % 32U);
    }
    *value_mask_out = value_mask;
}

void
evoke_scope_writer_init(evoke_scope_writer *writer)
{
    if (writer != NULL)
    {
        memset(writer, 0, sizeof(*writer));
    }
}

void
evoke_scope_writer_free(evoke_scope_writer *writer)
{
    if (writer == NULL)
    {
        return;
    }
    free(writer->bytes);
    evoke_scope_writer_init(writer);
}

static evoke_status
evoke_scope_writer_close_value(evoke_scope_writer *writer)
{
    uint8_t *entry;

    if (!writer->value_open)
    {
        return EVOKE_OK;
    }
    if (writer->current_document_count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    entry = writer->bytes + EVOKE_SCOPE_HEADER_SIZE + writer->column_bytes +
        (size_t) (writer->next_value - 1U) * EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    evoke_scope_write_u32(entry + 24, writer->current_document_count);
    writer->value_open = false;
    writer->current_document_count = 0;
    writer->previous_document = 0;
    return EVOKE_OK;
}

evoke_status
evoke_scope_writer_begin(
    evoke_scope_writer *writer,
    uint64_t source_authority_checksum,
    uint32_t document_count,
    const evoke_scope_column *columns,
    uint32_t column_count,
    uint32_t value_count,
    size_t value_dictionary_size,
    uint64_t posting_count,
    size_t maximum_size
)
{
    size_t value_bytes;
    size_t gram_filter_size;
    size_t total_size;
    size_t dictionary_size = value_dictionary_size;
    size_t dictionary_cursor = 0;
    size_t postings_size;

    if (writer == NULL || writer->bytes != NULL ||
        source_authority_checksum == 0 || document_count == 0 ||
        columns == NULL || column_count == 0 ||
        (value_count == 0 && posting_count != 0) ||
        posting_count > SIZE_MAX / sizeof(uint32_t))
    {
        return EVOKE_ERR_INVALID;
    }
    writer->column_bytes =
        (size_t) column_count * EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    value_bytes = (size_t) value_count * EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    if (writer->column_bytes / EVOKE_SCOPE_COLUMN_ENTRY_SIZE != column_count ||
        (value_count > 0 &&
         value_bytes / EVOKE_SCOPE_VALUE_ENTRY_SIZE != value_count))
    {
        return EVOKE_ERR_RANGE;
    }
    gram_filter_size = value_count == 0
        ? 0
        : ((size_t) value_count + EVOKE_SCOPE_GRAM_BLOCK_VALUES - 1U) /
            EVOKE_SCOPE_GRAM_BLOCK_VALUES;
    if (gram_filter_size > SIZE_MAX / EVOKE_SCOPE_GRAM_FILTER_BYTES)
    {
        return EVOKE_ERR_RANGE;
    }
    gram_filter_size *= EVOKE_SCOPE_GRAM_FILTER_BYTES;
    for (uint32_t column = 0; column < column_count; column++)
    {
        const evoke_scope_column *input = &columns[column];

        if (input->index_attribute == 0 || input->heap_attribute <= 0 ||
            input->type_oid == 0 || input->name == NULL ||
            input->name_size == 0 || input->first_value > value_count ||
            input->value_count > value_count - input->first_value ||
            (column > 0 &&
             columns[column - 1U].index_attribute >=
                input->index_attribute) ||
            (column == 0 && input->first_value != 0) ||
            (column > 0 && input->first_value !=
                columns[column - 1U].first_value +
                    columns[column - 1U].value_count) ||
            (column + 1U == column_count &&
             input->first_value + input->value_count != value_count) ||
            !evoke_scope_add_size(&dictionary_size, input->name_size))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (dictionary_size > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    postings_size = (size_t) posting_count * sizeof(uint32_t);
    total_size = EVOKE_SCOPE_HEADER_SIZE;
    if (!evoke_scope_add_size(&total_size, writer->column_bytes) ||
        !evoke_scope_add_size(&total_size, value_bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    writer->dictionary_offset = total_size;
    if (!evoke_scope_add_size(&total_size, dictionary_size))
    {
        return EVOKE_ERR_RANGE;
    }
    writer->gram_filter_offset = total_size;
    if (!evoke_scope_add_size(&total_size, gram_filter_size))
    {
        return EVOKE_ERR_RANGE;
    }
    writer->postings_offset = total_size;
    if (!evoke_scope_add_size(&total_size, postings_size))
    {
        return EVOKE_ERR_RANGE;
    }
    writer->total_size = total_size;
    if (total_size > maximum_size)
    {
        return EVOKE_ERR_NOMEM;
    }
    writer->bytes = calloc(total_size, 1);
    if (writer->bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    writer->dictionary_size = dictionary_size;
    writer->postings_size = postings_size;
    writer->document_count = document_count;
    writer->column_count = column_count;
    writer->value_count = value_count;
    evoke_scope_write_u32(writer->bytes + 0, EVOKE_SCOPE_MAGIC);
    evoke_scope_write_u16(writer->bytes + 4, EVOKE_SCOPE_VERSION);
    evoke_scope_write_u16(writer->bytes + 6, EVOKE_SCOPE_HEADER_SIZE);
    evoke_scope_write_u16(
        writer->bytes + 8,
        EVOKE_SCOPE_COLUMN_ENTRY_SIZE
    );
    evoke_scope_write_u16(writer->bytes + 10, EVOKE_SCOPE_VALUE_ENTRY_SIZE);
    evoke_scope_write_u16(writer->bytes + 12, EVOKE_SCOPE_GRAM_BLOCK_VALUES);
    evoke_scope_write_u16(writer->bytes + 14, EVOKE_SCOPE_GRAM_FILTER_BYTES);
    evoke_scope_write_u64(writer->bytes + 16, source_authority_checksum);
    evoke_scope_write_u32(writer->bytes + 24, document_count);
    evoke_scope_write_u32(writer->bytes + 28, column_count);
    evoke_scope_write_u32(writer->bytes + 32, value_count);
    evoke_scope_write_u64(
        writer->bytes + 40,
        writer->dictionary_offset
    );
    evoke_scope_write_u64(writer->bytes + 48, writer->dictionary_size);
    evoke_scope_write_u64(writer->bytes + 56, writer->postings_offset);
    evoke_scope_write_u64(writer->bytes + 64, writer->postings_size);
    for (uint32_t column = 0; column < column_count; column++)
    {
        const evoke_scope_column *input = &columns[column];
        uint8_t *entry = writer->bytes + EVOKE_SCOPE_HEADER_SIZE +
            (size_t) column * EVOKE_SCOPE_COLUMN_ENTRY_SIZE;

        evoke_scope_write_u16(entry + 0, input->index_attribute);
        evoke_scope_write_u16(entry + 2, (uint16_t) input->heap_attribute);
        evoke_scope_write_u32(entry + 4, input->type_oid);
        evoke_scope_write_u32(entry + 8, input->element_type_oid);
        evoke_scope_write_u32(entry + 12, input->collation_oid);
        evoke_scope_write_u32(entry + 16, (uint32_t) input->type_modifier);
        evoke_scope_write_u32(entry + 20, (uint32_t) dictionary_cursor);
        evoke_scope_write_u32(entry + 24, input->name_size);
        evoke_scope_write_u32(entry + 28, input->first_value);
        evoke_scope_write_u32(entry + 32, input->value_count);
        memcpy(
            writer->bytes + writer->dictionary_offset + dictionary_cursor,
            input->name,
            input->name_size
        );
        dictionary_cursor += input->name_size;
    }
    writer->dictionary_cursor = dictionary_cursor;
    return EVOKE_OK;
}

evoke_status
evoke_scope_writer_start_value(
    evoke_scope_writer *writer,
    const evoke_scope_value *value
)
{
    evoke_status status;
    uint8_t *entry;
    const uint8_t *gram_value;
    uint32_t gram_value_size;
    uint8_t *column_entry;
    uint32_t first_value;
    uint32_t column_value_count;
    uint32_t value_gram_mask = 0;

    if (writer == NULL || writer->bytes == NULL || value == NULL ||
        writer->next_value >= writer->value_count ||
        value->column_index >= writer->column_count ||
        (value->value_size > 0 && value->value == NULL) ||
        (value->gram_value_size > 0 && value->gram_value == NULL) ||
        (value->kind != EVOKE_SCOPE_VALUE_SCALAR &&
         value->kind != EVOKE_SCOPE_VALUE_ARRAY_ELEMENT) ||
        writer->dictionary_cursor > writer->dictionary_size ||
        value->value_size >
            writer->dictionary_size - writer->dictionary_cursor)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_scope_writer_close_value(writer);
    if (status != EVOKE_OK)
    {
        return status;
    }
    column_entry = writer->bytes + EVOKE_SCOPE_HEADER_SIZE +
        (size_t) value->column_index * EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    first_value = evoke_scope_read_u32(column_entry + 28);
    column_value_count = evoke_scope_read_u32(column_entry + 32);
    if (writer->next_value < first_value ||
        writer->next_value - first_value >= column_value_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (writer->next_value > 0)
    {
        uint8_t *previous_entry = writer->bytes + EVOKE_SCOPE_HEADER_SIZE +
            writer->column_bytes +
            (size_t) (writer->next_value - 1U) *
                EVOKE_SCOPE_VALUE_ENTRY_SIZE;
        uint32_t previous_column = evoke_scope_read_u32(previous_entry + 0);
        uint8_t previous_kind = previous_entry[4];
        uint32_t previous_offset = evoke_scope_read_u32(previous_entry + 8);
        uint32_t previous_size = evoke_scope_read_u32(previous_entry + 12);
        const uint8_t *previous_value = writer->bytes +
            writer->dictionary_offset + previous_offset;
        size_t common = previous_size < value->value_size
            ? previous_size
            : value->value_size;
        int comparison = 0;

        if (previous_column == value->column_index &&
            previous_kind == value->kind)
        {
            comparison = common == 0
                ? 0
                : memcmp(previous_value, value->value, common);
            if (comparison == 0)
            {
                comparison = previous_size < value->value_size
                    ? -1
                    : previous_size > value->value_size;
            }
        }
        if (previous_column > value->column_index ||
            (previous_column == value->column_index &&
             previous_kind > value->kind) ||
            (previous_column == value->column_index &&
             previous_kind == value->kind && comparison >= 0))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    entry = writer->bytes + EVOKE_SCOPE_HEADER_SIZE + writer->column_bytes +
        (size_t) writer->next_value * EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    evoke_scope_write_u32(entry + 0, value->column_index);
    entry[4] = value->kind;
    evoke_scope_write_u32(
        entry + 8,
        (uint32_t) writer->dictionary_cursor
    );
    evoke_scope_write_u32(entry + 12, value->value_size);
    evoke_scope_write_u64(entry + 16, writer->postings_cursor);
    if (value->value_size > 0)
    {
        memcpy(
            writer->bytes + writer->dictionary_offset +
                writer->dictionary_cursor,
            value->value,
            value->value_size
        );
        gram_value = value->gram_value == NULL
            ? value->value
            : value->gram_value;
        gram_value_size = value->gram_value == NULL
            ? value->value_size
            : value->gram_value_size;
        evoke_scope_add_value_grams(
            writer->bytes + writer->gram_filter_offset +
                ((size_t) writer->next_value /
                    EVOKE_SCOPE_GRAM_BLOCK_VALUES) *
                    EVOKE_SCOPE_GRAM_FILTER_BYTES,
            gram_value,
            gram_value_size,
            value->gram_value != NULL,
            &value_gram_mask
        );
    }
    evoke_scope_write_u32(entry + 28, value_gram_mask);
    writer->dictionary_cursor += value->value_size;
    writer->next_value++;
    writer->value_open = true;
    return EVOKE_OK;
}

evoke_status
evoke_scope_writer_append_document(
    evoke_scope_writer *writer,
    uint32_t document
)
{
    if (writer == NULL || writer->bytes == NULL || !writer->value_open ||
        document >= writer->document_count ||
        (writer->current_document_count > 0 &&
         writer->previous_document >= document) ||
        writer->postings_cursor > writer->postings_size ||
        writer->postings_size - writer->postings_cursor < sizeof(uint32_t))
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_scope_write_u32(
        writer->bytes + writer->postings_offset + writer->postings_cursor,
        document
    );
    writer->postings_cursor += sizeof(uint32_t);
    writer->previous_document = document;
    writer->current_document_count++;
    return EVOKE_OK;
}

evoke_status
evoke_scope_writer_finish(
    evoke_scope_writer *writer,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_status status;

    if (writer == NULL || writer->bytes == NULL || bytes_out == NULL ||
        size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_scope_writer_close_value(writer);
    if (status != EVOKE_OK || writer->next_value != writer->value_count ||
        writer->dictionary_cursor != writer->dictionary_size ||
        writer->postings_cursor != writer->postings_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    evoke_scope_write_u64(
        writer->bytes + EVOKE_SCOPE_CHECKSUM_OFFSET,
        evoke_scope_checksum(writer->bytes, writer->total_size)
    );
    *bytes_out = writer->bytes;
    *size_out = writer->total_size;
    writer->bytes = NULL;
    evoke_scope_writer_init(writer);
    return EVOKE_OK;
}

evoke_status
evoke_scope_serialize(
    uint64_t source_authority_checksum,
    uint32_t document_count,
    const evoke_scope_column *columns,
    uint32_t column_count,
    const evoke_scope_value *values,
    uint32_t value_count,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_scope_writer writer;
    size_t value_dictionary_size = 0;
    uint64_t posting_count = 0;
    evoke_status status;

    if (source_authority_checksum == 0 || document_count == 0 ||
        columns == NULL || column_count == 0 ||
        (value_count > 0 && values == NULL) ||
        bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    for (uint32_t value = 0; value < value_count; value++)
    {
        const evoke_scope_value *input = &values[value];

        if ((input->value_size > 0 && input->value == NULL) ||
            input->documents == NULL || input->document_count == 0 ||
            !evoke_scope_add_size(
                &value_dictionary_size,
                input->value_size
            ) ||
            posting_count >
                UINT64_MAX - (uint64_t) input->document_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        posting_count += input->document_count;
    }
    evoke_scope_writer_init(&writer);
    status = evoke_scope_writer_begin(
        &writer,
        source_authority_checksum,
        document_count,
        columns,
        column_count,
        value_count,
        value_dictionary_size,
        posting_count,
        SIZE_MAX
    );
    for (uint32_t value = 0;
         status == EVOKE_OK && value < value_count;
         value++)
    {
        const evoke_scope_value *input = &values[value];

        status = evoke_scope_writer_start_value(&writer, input);
        for (uint32_t document = 0;
             status == EVOKE_OK && document < input->document_count;
             document++)
        {
            status = evoke_scope_writer_append_document(
                &writer,
                input->documents[document]
            );
        }
    }
    if (status == EVOKE_OK)
    {
        status = evoke_scope_writer_finish(&writer, bytes_out, size_out);
    }
    evoke_scope_writer_free(&writer);
    return status;
}

evoke_status
evoke_scope_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_scope_header *header_out
)
{
    evoke_scope_header header;
    uint16_t version;
    uint64_t column_bytes;
    uint64_t value_bytes;
    uint64_t metadata_end;
    uint64_t gram_block_count;
    uint64_t expected_gram_filter_size;

    if (bytes == NULL || header_out == NULL ||
        size < EVOKE_SCOPE_HEADER_SIZE || object_size < size)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&header, 0, sizeof(header));
    version = evoke_scope_read_u16(bytes + 4);
    if (evoke_scope_read_u32(bytes + 0) != EVOKE_SCOPE_MAGIC ||
        version != EVOKE_SCOPE_VERSION ||
        evoke_scope_read_u16(bytes + 6) != EVOKE_SCOPE_HEADER_SIZE ||
        evoke_scope_read_u16(bytes + 8) != EVOKE_SCOPE_COLUMN_ENTRY_SIZE ||
        evoke_scope_read_u16(bytes + 10) != EVOKE_SCOPE_VALUE_ENTRY_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    header.version = version;
    header.gram_block_values = evoke_scope_read_u16(bytes + 12);
    header.gram_filter_bytes = evoke_scope_read_u16(bytes + 14);
    header.source_authority_checksum = evoke_scope_read_u64(bytes + 16);
    header.document_count = evoke_scope_read_u32(bytes + 24);
    header.column_count = evoke_scope_read_u32(bytes + 28);
    header.value_count = evoke_scope_read_u32(bytes + 32);
    header.dictionary_offset = evoke_scope_read_u64(bytes + 40);
    header.dictionary_size = evoke_scope_read_u64(bytes + 48);
    header.postings_offset = evoke_scope_read_u64(bytes + 56);
    header.postings_size = evoke_scope_read_u64(bytes + 64);
    header.object_checksum = evoke_scope_read_u64(bytes + 72);
    header.total_size = object_size;
    column_bytes = (uint64_t) header.column_count *
        EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    value_bytes = (uint64_t) header.value_count *
        EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    metadata_end = EVOKE_SCOPE_HEADER_SIZE + column_bytes + value_bytes;
    gram_block_count = header.value_count == 0
        ? 0
        : ((uint64_t) header.value_count +
            EVOKE_SCOPE_GRAM_BLOCK_VALUES - 1U) /
            EVOKE_SCOPE_GRAM_BLOCK_VALUES;
    expected_gram_filter_size =
        gram_block_count * EVOKE_SCOPE_GRAM_FILTER_BYTES;
    header.gram_filter_offset =
        header.dictionary_offset + header.dictionary_size;
    header.gram_filter_size = expected_gram_filter_size;
    if (header.source_authority_checksum == 0 ||
        header.source_authority_checksum !=
            expected_source_authority_checksum ||
        header.document_count == 0 || header.column_count == 0 ||
        metadata_end > object_size ||
        header.dictionary_offset != metadata_end ||
        header.dictionary_size > object_size - metadata_end ||
        header.gram_filter_offset < header.dictionary_offset ||
        header.gram_filter_offset > object_size ||
        header.gram_filter_size >
            object_size - header.gram_filter_offset ||
        header.postings_offset !=
            header.gram_filter_offset + header.gram_filter_size ||
        header.postings_size > object_size - header.postings_offset ||
        header.postings_offset + header.postings_size != object_size ||
        header.object_checksum == 0 || evoke_scope_read_u32(bytes + 36) != 0 ||
        header.gram_block_values != EVOKE_SCOPE_GRAM_BLOCK_VALUES ||
        header.gram_filter_bytes != EVOKE_SCOPE_GRAM_FILTER_BYTES)
    {
        return EVOKE_ERR_FORMAT;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_scope_column_entry_deserialize(
    const uint8_t *bytes,
    size_t size,
    const evoke_scope_header *header,
    uint32_t column_index,
    evoke_scope_column_entry *entry_out
)
{
    const uint8_t *entry;

    if (bytes == NULL || header == NULL || entry_out == NULL ||
        column_index >= header->column_count ||
        size < (size_t) header->column_count *
            EVOKE_SCOPE_COLUMN_ENTRY_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    entry = bytes + (size_t) column_index *
        EVOKE_SCOPE_COLUMN_ENTRY_SIZE;
    memset(entry_out, 0, sizeof(*entry_out));
    entry_out->index_attribute = evoke_scope_read_u16(entry + 0);
    entry_out->heap_attribute = (int16_t) evoke_scope_read_u16(entry + 2);
    entry_out->type_oid = evoke_scope_read_u32(entry + 4);
    entry_out->element_type_oid = evoke_scope_read_u32(entry + 8);
    entry_out->collation_oid = evoke_scope_read_u32(entry + 12);
    entry_out->type_modifier = (int32_t) evoke_scope_read_u32(entry + 16);
    entry_out->name_offset = evoke_scope_read_u32(entry + 20);
    entry_out->name_size = evoke_scope_read_u32(entry + 24);
    entry_out->first_value = evoke_scope_read_u32(entry + 28);
    entry_out->value_count = evoke_scope_read_u32(entry + 32);
    if (entry_out->index_attribute == 0 ||
        entry_out->heap_attribute <= 0 || entry_out->type_oid == 0 ||
        entry_out->name_size == 0 ||
        entry_out->name_offset > header->dictionary_size ||
        entry_out->name_size >
            header->dictionary_size - entry_out->name_offset ||
        entry_out->first_value > header->value_count ||
        entry_out->value_count >
            header->value_count - entry_out->first_value ||
        evoke_scope_read_u32(entry + 36) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_scope_value_entry_deserialize_one(
    const uint8_t *bytes,
    size_t size,
    const evoke_scope_header *header,
    uint32_t value_index,
    evoke_scope_value_entry *entry_out
)
{
    const uint8_t *entry;
    uint64_t posting_bytes;

    if (bytes == NULL || header == NULL || entry_out == NULL ||
        value_index >= header->value_count ||
        size < EVOKE_SCOPE_VALUE_ENTRY_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    entry = bytes;
    memset(entry_out, 0, sizeof(*entry_out));
    entry_out->column_index = evoke_scope_read_u32(entry + 0);
    entry_out->kind = entry[4];
    entry_out->value_offset = evoke_scope_read_u32(entry + 8);
    entry_out->value_size = evoke_scope_read_u32(entry + 12);
    entry_out->postings_offset = evoke_scope_read_u64(entry + 16);
    entry_out->document_count = evoke_scope_read_u32(entry + 24);
    entry_out->gram_mask = evoke_scope_read_u32(entry + 28);
    posting_bytes = (uint64_t) entry_out->document_count * sizeof(uint32_t);
    if (entry_out->column_index >= header->column_count ||
        (entry_out->kind != EVOKE_SCOPE_VALUE_SCALAR &&
         entry_out->kind != EVOKE_SCOPE_VALUE_ARRAY_ELEMENT) ||
        entry_out->document_count == 0 ||
        entry_out->value_offset > header->dictionary_size ||
        entry_out->value_size >
            header->dictionary_size - entry_out->value_offset ||
        entry_out->postings_offset > header->postings_size ||
        posting_bytes > header->postings_size - entry_out->postings_offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_scope_value_entry_deserialize(
    const uint8_t *bytes,
    size_t size,
    const evoke_scope_header *header,
    uint32_t value_index,
    evoke_scope_value_entry *entry_out
)
{
    size_t required_size;

    if (header == NULL || value_index >= header->value_count)
    {
        return EVOKE_ERR_INVALID;
    }
    required_size = (size_t) header->value_count *
        EVOKE_SCOPE_VALUE_ENTRY_SIZE;
    if ((header->value_count > 0 &&
         required_size / EVOKE_SCOPE_VALUE_ENTRY_SIZE !=
            header->value_count) ||
        bytes == NULL || size < required_size)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_scope_value_entry_deserialize_one(
        bytes + (size_t) value_index * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
        EVOKE_SCOPE_VALUE_ENTRY_SIZE,
        header,
        value_index,
        entry_out
    );
}

evoke_status
evoke_scope_validate(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum
)
{
    evoke_scope_header header;
    evoke_scope_column_entry previous_column;
    evoke_scope_value_entry previous_value;
    bool have_previous_value = false;
    uint64_t expected_dictionary_offset = 0;
    uint64_t expected_postings_offset = 0;
    uint32_t expected_first_value = 0;
    evoke_status status;

    status = evoke_scope_header_deserialize(
        bytes,
        size,
        size,
        expected_source_authority_checksum,
        &header
    );
    if (status != EVOKE_OK ||
        evoke_scope_read_u64(bytes + 72) != evoke_scope_checksum(bytes, size))
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    memset(&previous_column, 0, sizeof(previous_column));
    for (uint32_t column = 0; column < header.column_count; column++)
    {
        evoke_scope_column_entry entry;

        status = evoke_scope_column_entry_deserialize(
            bytes + EVOKE_SCOPE_HEADER_SIZE,
            (size_t) header.column_count * EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
            &header,
            column,
            &entry
        );
        if (status != EVOKE_OK ||
            (column > 0 &&
             previous_column.index_attribute >= entry.index_attribute) ||
            entry.name_offset != expected_dictionary_offset ||
            entry.first_value != expected_first_value)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        expected_dictionary_offset += entry.name_size;
        expected_first_value += entry.value_count;
        previous_column = entry;
    }
    if (expected_first_value != header.value_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&previous_value, 0, sizeof(previous_value));
    for (uint32_t value = 0; value < header.value_count; value++)
    {
        evoke_scope_value_entry entry;
        const uint8_t *entry_value;
        const uint8_t *previous_entry_value;

        status = evoke_scope_value_entry_deserialize(
            bytes + EVOKE_SCOPE_HEADER_SIZE +
                (size_t) header.column_count *
                    EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
            (size_t) header.value_count * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
            &header,
            value,
            &entry
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (entry.value_offset != expected_dictionary_offset ||
            entry.postings_offset != expected_postings_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        expected_dictionary_offset += entry.value_size;
        expected_postings_offset +=
            (uint64_t) entry.document_count * sizeof(uint32_t);
        if (entry.column_index >= header.column_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        entry_value = bytes + header.dictionary_offset + entry.value_offset;
        if (have_previous_value &&
            previous_value.column_index == entry.column_index &&
            previous_value.kind == entry.kind)
        {
            size_t common = previous_value.value_size < entry.value_size
                ? previous_value.value_size
                : entry.value_size;
            int comparison;

            previous_entry_value = bytes + header.dictionary_offset +
                previous_value.value_offset;
            comparison = memcmp(previous_entry_value, entry_value, common);
            if (comparison > 0 ||
                (comparison == 0 &&
                 previous_value.value_size >= entry.value_size))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        else if (have_previous_value &&
            (previous_value.column_index > entry.column_index ||
             (previous_value.column_index == entry.column_index &&
              previous_value.kind > entry.kind)))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t document = 0;
             document < entry.document_count;
             document++)
        {
            uint32_t document_id = evoke_scope_read_u32(
                bytes + header.postings_offset + entry.postings_offset +
                    (size_t) document * sizeof(uint32_t)
            );

            if (document_id >= header.document_count ||
                (document > 0 &&
                 evoke_scope_read_u32(
                    bytes + header.postings_offset +
                        entry.postings_offset +
                        (size_t) (document - 1U) * sizeof(uint32_t)
                 ) >= document_id))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        previous_value = entry;
        have_previous_value = true;
    }
    if (expected_dictionary_offset != header.dictionary_size ||
        expected_postings_offset != header.postings_size)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t column = 0; column < header.column_count; column++)
    {
        evoke_scope_column_entry entry;

        status = evoke_scope_column_entry_deserialize(
            bytes + EVOKE_SCOPE_HEADER_SIZE,
            (size_t) header.column_count * EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
            &header,
            column,
            &entry
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        for (uint32_t value = entry.first_value;
             value < entry.first_value + entry.value_count;
             value++)
        {
            evoke_scope_value_entry value_entry;

            status = evoke_scope_value_entry_deserialize(
                bytes + EVOKE_SCOPE_HEADER_SIZE +
                    (size_t) header.column_count *
                        EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
                (size_t) header.value_count *
                    EVOKE_SCOPE_VALUE_ENTRY_SIZE,
                &header,
                value,
                &value_entry
            );
            if (status != EVOKE_OK || value_entry.column_index != column)
            {
                return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
            }
        }
    }
    return EVOKE_OK;
}
