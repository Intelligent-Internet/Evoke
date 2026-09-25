#include "evoke_document_tid_lookup.h"

#include <stdlib.h>
#include <string.h>

#define EVOKE_DOCUMENT_TID_LOOKUP_MAGIC UINT32_C(0x4c544934)
#define EVOKE_DOCUMENT_TID_LOOKUP_VERSION UINT16_C(1)
#define EVOKE_DOCUMENT_TID_LOOKUP_CHECKSUM_OFFSET 32U

static void
evoke_document_tid_lookup_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) (value & UINT16_C(0xff));
    bytes[1] = (uint8_t) ((value >> 8) & UINT16_C(0xff));
}

static void
evoke_document_tid_lookup_write_u32(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (
            (value >> (index * 8)) & UINT32_C(0xff)
        );
    }
}

static void
evoke_document_tid_lookup_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (
            (value >> (index * 8)) & UINT64_C(0xff)
        );
    }
}

static uint16_t
evoke_document_tid_lookup_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_document_tid_lookup_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_document_tid_lookup_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static uint64_t
evoke_document_tid_lookup_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_DOCUMENT_TID_LOOKUP_CHECKSUM_OFFSET &&
            index < EVOKE_DOCUMENT_TID_LOOKUP_CHECKSUM_OFFSET +
                sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

evoke_status
evoke_document_tid_lookup_serialize(
    uint64_t source_authority_checksum,
    const uint64_t *tid_keys,
    const uint32_t *document_slots,
    uint32_t entry_count,
    uint32_t document_slot_count,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    size_t key_bytes;
    size_t slot_bytes;
    size_t total_size;
    size_t slots_offset;
    uint8_t *bytes;

    if (source_authority_checksum == 0 || tid_keys == NULL ||
        document_slots == NULL || entry_count == 0 ||
        document_slot_count < entry_count || bytes_out == NULL ||
        size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    key_bytes = (size_t) entry_count * sizeof(uint64_t);
    slot_bytes = (size_t) entry_count * sizeof(uint32_t);
    if (key_bytes / sizeof(uint64_t) != entry_count ||
        slot_bytes / sizeof(uint32_t) != entry_count ||
        key_bytes > SIZE_MAX - EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE ||
        slot_bytes > SIZE_MAX - EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE -
            key_bytes)
    {
        return EVOKE_ERR_RANGE;
    }
    slots_offset = EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE + key_bytes;
    total_size = slots_offset + slot_bytes;
    if (slots_offset > UINT32_MAX || total_size > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    for (uint32_t index = 0; index < entry_count; index++)
    {
        if ((index > 0 && tid_keys[index - 1U] >= tid_keys[index]) ||
            document_slots[index] >= document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    bytes = calloc(1, total_size);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_document_tid_lookup_write_u32(
        bytes + 0,
        EVOKE_DOCUMENT_TID_LOOKUP_MAGIC
    );
    evoke_document_tid_lookup_write_u16(
        bytes + 4,
        EVOKE_DOCUMENT_TID_LOOKUP_VERSION
    );
    evoke_document_tid_lookup_write_u16(
        bytes + 6,
        EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE
    );
    evoke_document_tid_lookup_write_u32(bytes + 8, entry_count);
    evoke_document_tid_lookup_write_u32(
        bytes + 12,
        document_slot_count
    );
    evoke_document_tid_lookup_write_u64(
        bytes + 16,
        source_authority_checksum
    );
    evoke_document_tid_lookup_write_u64(bytes + 24, total_size);
    evoke_document_tid_lookup_write_u32(
        bytes + 40,
        EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE
    );
    evoke_document_tid_lookup_write_u32(bytes + 44, slots_offset);
    for (uint32_t index = 0; index < entry_count; index++)
    {
        evoke_document_tid_lookup_write_u64(
            bytes + EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE +
                (size_t) index * sizeof(uint64_t),
            tid_keys[index]
        );
        evoke_document_tid_lookup_write_u32(
            bytes + slots_offset + (size_t) index * sizeof(uint32_t),
            document_slots[index]
        );
    }
    evoke_document_tid_lookup_write_u64(
        bytes + EVOKE_DOCUMENT_TID_LOOKUP_CHECKSUM_OFFSET,
        evoke_document_tid_lookup_checksum(bytes, total_size)
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_document_tid_lookup_open(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum,
    uint32_t expected_entry_count,
    uint32_t expected_document_slot_count,
    evoke_document_tid_lookup_view *view_out
)
{
    uint32_t entry_count;
    uint32_t document_slot_count;
    size_t keys_offset;
    size_t slots_offset;
    size_t expected_size;
    uint64_t source_authority_checksum;

    if (bytes == NULL || view_out == NULL ||
        size < EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(view_out, 0, sizeof(*view_out));
    entry_count = evoke_document_tid_lookup_read_u32(bytes + 8);
    document_slot_count = evoke_document_tid_lookup_read_u32(bytes + 12);
    source_authority_checksum =
        evoke_document_tid_lookup_read_u64(bytes + 16);
    keys_offset = evoke_document_tid_lookup_read_u32(bytes + 40);
    slots_offset = evoke_document_tid_lookup_read_u32(bytes + 44);
    if (evoke_document_tid_lookup_read_u32(bytes + 0) !=
            EVOKE_DOCUMENT_TID_LOOKUP_MAGIC ||
        evoke_document_tid_lookup_read_u16(bytes + 4) !=
            EVOKE_DOCUMENT_TID_LOOKUP_VERSION ||
        evoke_document_tid_lookup_read_u16(bytes + 6) !=
            EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE ||
        entry_count == 0 ||
        (expected_entry_count != 0 && entry_count != expected_entry_count) ||
        document_slot_count < entry_count ||
        (expected_document_slot_count != 0 &&
         document_slot_count != expected_document_slot_count) ||
        source_authority_checksum == 0 ||
        (expected_source_authority_checksum != 0 &&
         source_authority_checksum != expected_source_authority_checksum) ||
        keys_offset != EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE ||
        slots_offset != keys_offset +
            (size_t) entry_count * sizeof(uint64_t) ||
        slots_offset < keys_offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_size = slots_offset +
        (size_t) entry_count * sizeof(uint32_t);
    if (expected_size < slots_offset || expected_size != size ||
        evoke_document_tid_lookup_read_u64(bytes + 24) != size ||
        evoke_document_tid_lookup_read_u64(bytes + 32) !=
            evoke_document_tid_lookup_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0; index < entry_count; index++)
    {
        uint64_t key = evoke_document_tid_lookup_read_u64(
            bytes + keys_offset + (size_t) index * sizeof(uint64_t)
        );
        uint32_t slot = evoke_document_tid_lookup_read_u32(
            bytes + slots_offset + (size_t) index * sizeof(uint32_t)
        );
        uint64_t previous = index == 0
            ? 0
            : evoke_document_tid_lookup_read_u64(
                  bytes + keys_offset +
                      (size_t) (index - 1U) * sizeof(uint64_t)
              );

        if ((index > 0 && previous >= key) ||
            slot >= document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    view_out->bytes = bytes;
    view_out->size = size;
    view_out->keys_offset = keys_offset;
    view_out->slots_offset = slots_offset;
    view_out->source_authority_checksum = source_authority_checksum;
    view_out->entry_count = entry_count;
    view_out->document_slot_count = document_slot_count;
    return EVOKE_OK;
}

evoke_status
evoke_document_tid_lookup_entry(
    const evoke_document_tid_lookup_view *view,
    uint32_t entry_index,
    uint64_t *tid_key_out,
    uint32_t *document_slot_out
)
{
    if (view == NULL || view->bytes == NULL ||
        entry_index >= view->entry_count || tid_key_out == NULL ||
        document_slot_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *tid_key_out = evoke_document_tid_lookup_read_u64(
        view->bytes + view->keys_offset +
            (size_t) entry_index * sizeof(uint64_t)
    );
    *document_slot_out = evoke_document_tid_lookup_read_u32(
        view->bytes + view->slots_offset +
            (size_t) entry_index * sizeof(uint32_t)
    );
    return EVOKE_OK;
}

bool
evoke_document_tid_lookup_find(
    const evoke_document_tid_lookup_view *view,
    uint64_t tid_key,
    uint32_t *document_slot_out
)
{
    size_t low = 0;
    size_t high;

    if (view == NULL || view->bytes == NULL || document_slot_out == NULL)
    {
        return false;
    }
    high = view->entry_count;
    while (low < high)
    {
        size_t middle = low + (high - low) / 2;
        uint64_t candidate = evoke_document_tid_lookup_read_u64(
            view->bytes + view->keys_offset +
                middle * sizeof(uint64_t)
        );

        if (candidate < tid_key)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= view->entry_count ||
        evoke_document_tid_lookup_read_u64(
            view->bytes + view->keys_offset + low * sizeof(uint64_t)
        ) != tid_key)
    {
        return false;
    }
    *document_slot_out = evoke_document_tid_lookup_read_u32(
        view->bytes + view->slots_offset + low * sizeof(uint32_t)
    );
    return true;
}
