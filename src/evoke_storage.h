#ifndef EVOKE_STORAGE_H
#define EVOKE_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

#define EVOKE_HEADER_SIZE 48U
#define EVOKE_STORAGE_CURRENT_VERSION 2U

typedef evoke_status (*evoke_stream_write_cb)(
    void *ctx,
    const uint8_t *bytes,
    size_t len
);

evoke_status evoke_serialized_index_size(
    const evoke_index *index,
    size_t *len_out
);

evoke_status evoke_serialize_index(
    const evoke_index *index,
    uint8_t **bytes_out,
    size_t *len_out
);

evoke_status evoke_serialize_index_stream(
    const evoke_index *index,
    evoke_stream_write_cb write_cb,
    void *write_ctx,
    size_t *len_out
);

evoke_status evoke_deserialize_index(
    const uint8_t *bytes,
    size_t len,
    evoke_index *index_out
);

evoke_status evoke_peek_serialized_meta(
    const uint8_t *bytes,
    size_t len,
    uint32_t *num_docs_out,
    uint32_t *vocab_size_out,
    uint64_t *data_len_out
);

#endif
