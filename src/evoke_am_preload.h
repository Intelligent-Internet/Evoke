#ifndef EVOKE_AM_PRELOAD_H
#define EVOKE_AM_PRELOAD_H

#include "postgres.h"

#include "storage/lwlock.h"
#include "utils/rel.h"

#include "evoke_am_meta.h"

#define EVOKE_AM_UNIFIED_WARM_MARKER_MAGIC UINT32_C(0x4957514d)
#define EVOKE_AM_UNIFIED_WARM_MARKER_VERSION UINT16_C(2)
#define EVOKE_AM_UNIFIED_WARM_QUERY_METADATA_COMPLETE UINT16_C(0x0001)
#define EVOKE_AM_UNIFIED_WARM_ACCELERATOR_AUTHORITY UINT16_C(0x0002)

typedef struct evoke_am_unified_warm_marker
{
    uint32 magic;
    uint16 version;
    uint16 flags;
    uint64 pages_warmed;
    uint64 authority_id;
    evoke_segment_object_ref authority;
} evoke_am_unified_warm_marker;

typedef enum evoke_am_preload_kind
{
    EVOKE_AM_PRELOAD_UNIFIED_WARM = 4,
    EVOKE_AM_PRELOAD_HOT_FOLD = 5,
    EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS = 6,
    EVOKE_AM_PRELOAD_VALIDATED_PAGES = 7,
    EVOKE_AM_PRELOAD_RESIDENT_FOLD = 8,
    EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP = 9
} evoke_am_preload_kind;

typedef enum evoke_am_preload_reserve_result
{
    EVOKE_AM_PRELOAD_RESERVE_FAILED = 0,
    EVOKE_AM_PRELOAD_RESERVE_READY,
    EVOKE_AM_PRELOAD_RESERVE_LOADING,
    EVOKE_AM_PRELOAD_RESERVE_NEW
} evoke_am_preload_reserve_result;

typedef enum evoke_am_preload_admission_state
{
    EVOKE_AM_PRELOAD_ADMISSION_UNAVAILABLE = 0,
    EVOKE_AM_PRELOAD_ADMISSION_RESIDENT,
    EVOKE_AM_PRELOAD_ADMISSION_LOADING,
    EVOKE_AM_PRELOAD_ADMISSION_ZERO_PAYLOAD,
    EVOKE_AM_PRELOAD_ADMISSION_OVERSIZED,
    EVOKE_AM_PRELOAD_ADMISSION_ADMISSIBLE,
    EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SLOT,
    EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SPACE
} evoke_am_preload_admission_state;

/*
 * These handles are values, not registry authority. Callers must use the
 * typed accessors and release/commit/abort functions below.
 */
typedef struct evoke_am_preload_lease
{
    uintptr_t private_token;
    const void *private_payload;
    Size private_payload_size;
} evoke_am_preload_lease;

typedef struct evoke_am_preload_reservation
{
    uintptr_t private_token;
    void *private_payload;
    Size private_payload_size;
} evoke_am_preload_reservation;

typedef struct evoke_am_preload_status
{
    bool available;
    bool resident;
    bool loading;
    bool hot_fold_current;
    bool hot_fold_loading;
    bool resident_fold_current;
    bool resident_fold_loading;
    bool candidate_fits_empty;
    bool has_free_slot;
    bool has_reusable_slot;
    bool has_evictable_relation;
    Size arena_size;
    Size used;
    Size free_bytes;
    Size resident_fold_bytes;
    Size candidate_bytes;
    Size reusable_bytes;
    uint32 entry_capacity;
    uint32 hash_capacity;
    uint32 entries;
    uint32 ready_entries;
    uint32 obsolete_entries;
    uint32 reusable_entries;
    uint32 refcounted_entries;
    uint32 unified_warm_entries;
    uint32 hot_fold_entries;
    uint32 resident_fold_entries;
    uint32 document_length_entries;
    uint32 document_tid_lookup_entries;
    uint32 validated_page_entries;
    uint64 relation_entry_evictions;
    uint64 relation_entry_eviction_bytes;
    uint64 access_clock;
    evoke_am_preload_admission_state admission_state;
} evoke_am_preload_status;

void evoke_am_preload_define_gucs(void);
int evoke_am_preload_configured_mb(void);
int evoke_am_preload_test_registry_fill(void);

Size evoke_am_preload_shmem_size(void);
void evoke_am_preload_shmem_startup(LWLock *lock);

bool evoke_am_preload_available(void);
bool evoke_am_preload_cache_available(void);
bool evoke_am_preload_exact_state(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    bool *resident_out,
    bool *loading_out
);
void evoke_am_preload_retire_obsolete(
    Relation index_relation,
    const evoke_am_meta_page *meta
);
void evoke_am_preload_rekey_generation(
    Relation index_relation,
    const evoke_am_meta_page *old_meta,
    const evoke_am_meta_page *current_meta
);
void evoke_am_preload_retire_exact(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind
);

bool evoke_am_preload_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    evoke_am_preload_lease *lease_out
);
const void *evoke_am_preload_lease_payload(
    const evoke_am_preload_lease *lease
);
void *evoke_am_preload_lease_mutable_payload(
    evoke_am_preload_lease *lease
);
Size evoke_am_preload_lease_payload_size(
    const evoke_am_preload_lease *lease
);
void evoke_am_preload_lease_release(evoke_am_preload_lease *lease);
void evoke_am_preload_lease_retire(evoke_am_preload_lease *lease);

evoke_am_preload_reserve_result evoke_am_preload_reserve(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    Size payload_size,
    evoke_am_preload_reservation *reservation_out
);
void *evoke_am_preload_reservation_payload(
    evoke_am_preload_reservation *reservation
);
bool evoke_am_preload_reservation_commit(
    evoke_am_preload_reservation *reservation
);
void evoke_am_preload_reservation_abort(
    evoke_am_preload_reservation *reservation
);
void evoke_am_preload_abort_active_publication(void);

int evoke_am_preload_clear(void);
void evoke_am_preload_status_snapshot(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind requested_kind,
    Size candidate_bytes,
    evoke_am_preload_status *status_out
);
const char *evoke_am_preload_admission_state_name(
    evoke_am_preload_admission_state state
);

#endif
