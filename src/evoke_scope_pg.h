#ifndef EVOKE_SCOPE_PG_H
#define EVOKE_SCOPE_PG_H

#include "postgres.h"

#include "utils/rel.h"

#include "evoke_core.h"

evoke_status evoke_scope_build_for_index(
    Relation index_relation,
    const ItemPointerData *document_tids,
    uint32 document_count,
    uint64 source_authority_checksum,
    size_t maximum_size,
    size_t *required_size_out,
    uint8 **bytes_out,
    size_t *size_out
);

#endif
