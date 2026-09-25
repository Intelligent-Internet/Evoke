#ifndef EVOKE_DOCUMENT_TID_LOOKUP_H
#define EVOKE_DOCUMENT_TID_LOOKUP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

#define EVOKE_DOCUMENT_TID_LOOKUP_HEADER_SIZE 48U

typedef struct evoke_document_tid_lookup_view
{
    const uint8_t *bytes;
    size_t size;
    size_t keys_offset;
    size_t slots_offset;
    uint64_t source_authority_checksum;
    uint32_t entry_count;
    uint32_t document_slot_count;
} evoke_document_tid_lookup_view;

evoke_status evoke_document_tid_lookup_serialize(
    uint64_t source_authority_checksum,
    const uint64_t *tid_keys,
    const uint32_t *document_slots,
    uint32_t entry_count,
    uint32_t document_slot_count,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_document_tid_lookup_open(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum,
    uint32_t expected_entry_count,
    uint32_t expected_document_slot_count,
    evoke_document_tid_lookup_view *view_out
);

evoke_status evoke_document_tid_lookup_entry(
    const evoke_document_tid_lookup_view *view,
    uint32_t entry_index,
    uint64_t *tid_key_out,
    uint32_t *document_slot_out
);

bool evoke_document_tid_lookup_find(
    const evoke_document_tid_lookup_view *view,
    uint64_t tid_key,
    uint32_t *document_slot_out
);

#endif
