#ifndef EVOKE_AM_META_H
#define EVOKE_AM_META_H

#include "postgres.h"

#include "access/genam.h"
#include "storage/buf.h"
#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_semantic.h"
#include "evoke_segments.h"

#define EVOKE_AM_MAGIC UINT32_C(0x53323542)
#define EVOKE_AM_VERSION 3
#define EVOKE_AM_PAGE_META 1

#define EVOKE_AM_FLAG_STALE UINT16_C(0x0001)
#define EVOKE_AM_FLAG_CORRUPT UINT16_C(0x0002)
#define EVOKE_AM_FLAG_REBUILD_REQUIRED UINT16_C(0x0004)
#define EVOKE_AM_FLAG_SEMANTIC_QUARANTINE UINT16_C(0x0008)
#define EVOKE_AM_STORAGE_CONVERGENT_SEGMENTS 3

typedef struct evoke_am_meta_page
{
    uint32 magic;
    uint16 version;
    uint16 page_kind;
    uint16 flags;
    uint16 cache_epoch;
    Oid source_type;
    uint32 num_docs;
    uint64 tid_bytes_len;
    uint64 index_bytes_len;
    uint32 delta_record_count;
    uint64 delta_bytes_len;
    uint32 pending_write_tuples;
    uint32 pending_delete_tuples;
    uint64 rebuild_count;
    uint32 storage_version;
    uint32 active_generation;
    BlockNumber active_start_blkno;
    uint32 active_data_pages;
    BlockNumber delta_start_blkno;
    uint32 delta_data_pages;
    BlockNumber semantic_start_blkno;
    uint32 semantic_data_pages;
    uint32 semantic_record_count;
    uint64 semantic_bytes_len;
    char semantic_signature[EVOKE_AM_SEMANTIC_SIGNATURE_LEN + 1];
    uint8 segment_read_root_bytes[
        EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE
    ];
} evoke_am_meta_page;

typedef struct evoke_am_payload_health_state
{
    bool corrupt;
    bool rebuild_required;
    const char *status;
    const char *reason;
    uint64 expected_bytes;
    uint64 capacity_bytes;
} evoke_am_payload_health_state;

typedef struct evoke_am_convergent_mutation_debt
{
    uint32 upserts;
    uint32 retirements;
    uint32 records;
    uint64 bytes;
} evoke_am_convergent_mutation_debt;

bool evoke_am_meta_uses_convergent_segment_storage(
    const evoke_am_meta_page *meta
);
bool evoke_am_generation_identity_matches(
    const evoke_am_meta_page *cached,
    const evoke_am_meta_page *current
);
void evoke_am_require_convergent_segment_storage(
    const evoke_am_meta_page *meta
);
evoke_status evoke_am_segment_read_root_from_meta(
    const evoke_am_meta_page *meta,
    evoke_segment_read_root *root_out
);
void evoke_am_read_meta(
    Relation indexRelation,
    evoke_am_meta_page *meta_out
);
bool evoke_am_try_read_current_meta(
    Relation indexRelation,
    evoke_am_meta_page *meta_out
);
void evoke_am_note_maintenance_activity(
    Relation indexRelation,
    uint32 pending_write_add,
    uint32 pending_delete_add
);
void evoke_am_mark_stale(Relation indexRelation);
void evoke_am_get_stats(
    Relation indexRelation,
    IndexBulkDeleteResult *stats
);
void evoke_am_lock_generation_barrier(Relation indexRelation);
void evoke_am_unlock_generation_barrier(Relation indexRelation);
void evoke_am_mark_buffer_dirty_with_wal(
    Relation indexRelation,
    Buffer buffer
);
void evoke_am_write_page_at(
    Relation indexRelation,
    BlockNumber blkno,
    const void *contents,
    Size len
);
void evoke_am_write_init_page_at(
    Relation indexRelation,
    BlockNumber blkno,
    const void *contents,
    Size len
);
void evoke_am_write_new_page(
    Relation indexRelation,
    const void *contents,
    Size len
);
uint64 evoke_am_next_rebuild_count(Relation indexRelation);
uint16 evoke_am_next_cache_epoch(Relation indexRelation);
void evoke_am_publish_rebuild_meta(
    Relation indexRelation,
    ForkNumber fork_number,
    const evoke_segment_read_root *root,
    Oid source_type,
    uint32 num_docs,
    uint16 cache_epoch,
    uint16 flags,
    uint64 rebuild_count
);
BlockNumber evoke_am_relation_nblocks(Relation indexRelation);
void evoke_am_payload_health(
    Relation indexRelation,
    const evoke_am_meta_page *meta,
    evoke_am_payload_health_state *health_out
);
uint64 evoke_am_segment_object_bytes_add(uint64 left, uint64 right);
uint64 evoke_am_convergent_object_bytes(
    Relation indexRelation,
    const evoke_am_meta_page *meta
);
void evoke_am_convergent_mutation_debt_read(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_convergent_mutation_debt *debt_out
);
bool evoke_am_meta_has_pending_maintenance(
    const evoke_am_meta_page *meta
);

#endif
