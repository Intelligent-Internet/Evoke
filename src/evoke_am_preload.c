#include "postgres.h"

#include <sys/mman.h>
#include <unistd.h>

#include "miscadmin.h"
#include "port/atomics.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "utils/guc.h"

#include "evoke_am_hot_fold.h"
#include "evoke_am_options.h"
#include "evoke_am_preload.h"

#define EVOKE_AM_PRELOAD_MAGIC UINT32_C(0x50323542)
#define EVOKE_AM_PRELOAD_VERSION 32
#define EVOKE_AM_PRELOAD_MIN_ENTRIES 1024
#define EVOKE_AM_PRELOAD_MAX_ENTRIES 65536
#define EVOKE_AM_PRELOAD_ENTRY_TARGET_BYTES (4 * 1024 * 1024)
#define EVOKE_AM_PRELOAD_MAX_MB 1048576

typedef struct evoke_am_preload_entry
{
    bool in_use;
    bool ready;
    bool obsolete;
    uint8 generation_kind;
    int auto_preload_priority;
    Oid database_oid;
    Oid index_oid;
    RelFileLocator locator;
    evoke_am_meta_page meta;
    Size offset;
    Size mapped_size;
    Size allocation_size;
    uint64 key_hash;
    pg_atomic_uint32 refcount;
    pg_atomic_uint64 last_access_counter;
} evoke_am_preload_entry;

typedef struct evoke_am_preload_hash_slot
{
    uint64 key_hash;
    uint32 entry_slot_plus_one;
} evoke_am_preload_hash_slot;

typedef struct evoke_am_preload_control
{
    uint32 magic;
    uint32 version;
    Size arena_size;
    Size used;
    uint64 relation_entry_evictions;
    uint64 relation_entry_eviction_bytes;
    pg_atomic_uint64 access_clock;
    uint32 entry_capacity;
    uint32 hash_capacity;
    evoke_am_preload_entry entries[FLEXIBLE_ARRAY_MEMBER];
} evoke_am_preload_control;

static bool evoke_am_preload_gucs_initialized = false;
static int evoke_shared_runtime_size_mb = 0;
static int evoke_test_shared_preload_registry_capacity = 0;
static int evoke_test_shared_preload_registry_fill = 0;
static evoke_am_preload_control *evoke_preload = NULL;
static LWLock *evoke_preload_lock = NULL;
static uintptr_t evoke_active_publication_token = 0;

static bool
evoke_am_preload_checked_add(Size left, Size right, Size *result_out)
{
    if (left > SIZE_MAX - right)
    {
        return false;
    }
    *result_out = left + right;
    return true;
}

static bool
evoke_am_preload_checked_mul(Size left, Size right, Size *result_out)
{
    if (left == 0 || right == 0)
    {
        *result_out = 0;
        return true;
    }
    if (left > SIZE_MAX / right)
    {
        return false;
    }
    *result_out = left * right;
    return true;
}

static bool
evoke_am_preload_kind_valid(evoke_am_preload_kind kind)
{
    return kind == EVOKE_AM_PRELOAD_UNIFIED_WARM ||
        kind == EVOKE_AM_PRELOAD_HOT_FOLD ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS ||
        kind == EVOKE_AM_PRELOAD_VALIDATED_PAGES ||
        kind == EVOKE_AM_PRELOAD_RESIDENT_FOLD ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP;
}

static bool
evoke_am_preload_kind_rekey_safe(evoke_am_preload_kind kind)
{
    /*
     * Query metadata carries and validates serving authority in its payload,
     * so it can survive manifest-only successors. Resident folds remain tied
     * to an exact immutable generation.
     */
    return kind == EVOKE_AM_PRELOAD_UNIFIED_WARM ||
        kind == EVOKE_AM_PRELOAD_HOT_FOLD ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS ||
        kind == EVOKE_AM_PRELOAD_VALIDATED_PAGES ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP;
}

static bool
evoke_am_preload_kind_uses_base_identity(
    evoke_am_preload_kind kind
)
{
    return kind == EVOKE_AM_PRELOAD_UNIFIED_WARM ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS ||
        kind == EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP;
}

static bool
evoke_am_preload_payload_size_valid(
    evoke_am_preload_kind kind,
    Size payload_size
)
{
    if (kind == EVOKE_AM_PRELOAD_UNIFIED_WARM)
    {
        return payload_size == sizeof(evoke_am_unified_warm_marker);
    }
    if (kind == EVOKE_AM_PRELOAD_HOT_FOLD)
    {
        return payload_size >= sizeof(evoke_am_hot_fold_header);
    }
    if (kind == EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS)
    {
        return payload_size >= sizeof(uint32) * 4 &&
            payload_size % sizeof(uint32) == 0;
    }
    if (kind == EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP)
    {
        return payload_size >= sizeof(uint32) * 4;
    }
    if (kind == EVOKE_AM_PRELOAD_RESIDENT_FOLD)
    {
        return payload_size > 0;
    }
    return kind == EVOKE_AM_PRELOAD_VALIDATED_PAGES &&
        payload_size >= sizeof(pg_atomic_uint64) &&
        payload_size % sizeof(pg_atomic_uint64) == 0;
}

static Size
evoke_am_preload_arena_size(void)
{
    uint64 bytes;

    if (evoke_shared_runtime_size_mb <= 0)
    {
        return 0;
    }
    bytes = (uint64) evoke_shared_runtime_size_mb * 1024ULL *
        1024ULL;
    if (bytes > (uint64) SIZE_MAX)
    {
        ereport(ERROR, (errmsg("evoke shared runtime arena is too large")));
    }
    return (Size) bytes;
}

static uint32
evoke_am_preload_entry_capacity_for_arena(Size arena_size)
{
    uint64 capacity = EVOKE_AM_PRELOAD_MIN_ENTRIES;
    uint64 arena_capacity;

    if (evoke_test_shared_preload_registry_capacity > 0)
    {
        return (uint32) evoke_test_shared_preload_registry_capacity;
    }
    arena_capacity = (uint64) arena_size /
        EVOKE_AM_PRELOAD_ENTRY_TARGET_BYTES;
    if (arena_capacity > capacity)
    {
        capacity = arena_capacity;
    }
    if (capacity > EVOKE_AM_PRELOAD_MAX_ENTRIES)
    {
        capacity = EVOKE_AM_PRELOAD_MAX_ENTRIES;
    }
    return (uint32) capacity;
}

static uint32
evoke_am_preload_entry_capacity(void)
{
    if (evoke_preload == NULL || evoke_preload->entry_capacity == 0)
    {
        return 0;
    }
    return evoke_preload->entry_capacity;
}

static uint32
evoke_am_preload_hash_capacity_for_entries(uint32 entry_capacity)
{
    uint32 capacity = 1;
    uint64 target = (uint64) entry_capacity * 2;

    while ((uint64) capacity < target)
    {
        if (capacity > UINT32_MAX / 2)
        {
            ereport(ERROR, (errmsg("evoke shared preload hash is too large")));
        }
        capacity *= 2;
    }
    return capacity;
}

static Size
evoke_am_preload_control_size(uint32 entry_capacity)
{
    Size entries_size;
    Size hash_size;
    Size control_size;
    uint32 hash_capacity;

    if (!evoke_am_preload_checked_mul(
            entry_capacity,
            sizeof(evoke_am_preload_entry),
            &entries_size) ||
        !evoke_am_preload_checked_add(
            offsetof(evoke_am_preload_control, entries),
            entries_size,
            &control_size))
    {
        ereport(ERROR, (errmsg("evoke shared preload registry is too large")));
    }
    control_size = MAXALIGN(control_size);
    hash_capacity =
        evoke_am_preload_hash_capacity_for_entries(entry_capacity);
    if (!evoke_am_preload_checked_mul(
            hash_capacity,
            sizeof(evoke_am_preload_hash_slot),
            &hash_size) ||
        !evoke_am_preload_checked_add(
            control_size,
            hash_size,
            &control_size))
    {
        ereport(ERROR, (errmsg("evoke shared preload hash is too large")));
    }
    return MAXALIGN(control_size);
}

Size
evoke_am_preload_shmem_size(void)
{
    Size arena_size = evoke_am_preload_arena_size();
    uint32 entry_capacity =
        evoke_am_preload_entry_capacity_for_arena(arena_size);

    return evoke_am_preload_control_size(entry_capacity) + arena_size;
}

bool
evoke_am_preload_available(void)
{
    return evoke_preload != NULL &&
        evoke_preload_lock != NULL &&
        evoke_preload->magic == EVOKE_AM_PRELOAD_MAGIC &&
        evoke_preload->version == EVOKE_AM_PRELOAD_VERSION &&
        evoke_preload->entry_capacity > 0 &&
        evoke_preload->hash_capacity >= evoke_preload->entry_capacity * 2 &&
        (evoke_preload->hash_capacity &
         (evoke_preload->hash_capacity - 1)) == 0;
}

bool
evoke_am_preload_cache_available(void)
{
    return evoke_am_preload_available() && evoke_preload->arena_size > 0;
}

static char *
evoke_am_preload_arena_base(void)
{
    if (!evoke_am_preload_available())
    {
        return NULL;
    }
    return ((char *) evoke_preload) +
        evoke_am_preload_control_size(evoke_preload->entry_capacity);
}

static evoke_am_preload_hash_slot *
evoke_am_preload_hash_slots(void)
{
    Size entries_size;
    Size offset;

    if (!evoke_am_preload_available())
    {
        return NULL;
    }
    if (!evoke_am_preload_checked_mul(
            evoke_preload->entry_capacity,
            sizeof(evoke_am_preload_entry),
            &entries_size) ||
        !evoke_am_preload_checked_add(
            offsetof(evoke_am_preload_control, entries),
            entries_size,
            &offset))
    {
        ereport(ERROR, (errmsg("evoke shared preload hash is too large")));
    }
    return (evoke_am_preload_hash_slot *) (
        ((char *) evoke_preload) + MAXALIGN(offset)
    );
}

static uint64
evoke_am_preload_hash_mix(uint64 hash, uint64 value)
{
    return hash ^ (
        value +
        UINT64_C(0x9e3779b97f4a7c15) +
        (hash << 6) +
        (hash >> 2)
    );
}

static bool
evoke_am_preload_base_identity_matches(
    const evoke_am_meta_page *left,
    const evoke_am_meta_page *right
)
{
    return left != NULL && right != NULL &&
        left->magic == right->magic &&
        left->version == right->version &&
        left->page_kind == right->page_kind &&
        left->cache_epoch == right->cache_epoch &&
        left->storage_version == right->storage_version &&
        left->source_type == right->source_type;
}

static uint64
evoke_am_preload_entry_key_hash(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind
)
{
    const unsigned char *signature;
    uint64 hash = UINT64_C(0xcbf29ce484222325);
    size_t index;

    hash = evoke_am_preload_hash_mix(hash, MyDatabaseId);
    hash = evoke_am_preload_hash_mix(
        hash,
        RelationGetRelid(index_relation)
    );
    hash = evoke_am_preload_hash_mix(
        hash,
        index_relation->rd_locator.spcOid
    );
    hash = evoke_am_preload_hash_mix(
        hash,
        index_relation->rd_locator.dbOid
    );
    hash = evoke_am_preload_hash_mix(
        hash,
        index_relation->rd_locator.relNumber
    );
    hash = evoke_am_preload_hash_mix(hash, kind);
    hash = evoke_am_preload_hash_mix(hash, meta->magic);
    hash = evoke_am_preload_hash_mix(hash, meta->version);
    hash = evoke_am_preload_hash_mix(hash, meta->page_kind);
    hash = evoke_am_preload_hash_mix(hash, meta->cache_epoch);
    hash = evoke_am_preload_hash_mix(hash, meta->storage_version);
    hash = evoke_am_preload_hash_mix(hash, meta->source_type);
    if (!evoke_am_meta_uses_convergent_segment_storage(meta))
    {
        ereport(
            ERROR,
            (errmsg("evoke shared residency requires a convergent root"))
        );
    }
    if (!evoke_am_preload_kind_uses_base_identity(kind))
    {
        signature =
            (const unsigned char *) meta->segment_read_root_bytes;
        for (index = 0;
             index < sizeof(meta->segment_read_root_bytes);
             index++)
        {
            hash = evoke_am_preload_hash_mix(hash, signature[index]);
        }
    }
    /* Base-keyed query metadata authenticates authority in its payload. */
    return hash == 0 ? UINT64_C(1) : hash;
}

static bool
evoke_am_preload_entry_has_block(const evoke_am_preload_entry *entry)
{
    if (entry == NULL ||
        !evoke_am_preload_cache_available() ||
        entry->mapped_size == 0 ||
        entry->allocation_size == 0 ||
        entry->mapped_size > entry->allocation_size ||
        entry->offset > evoke_preload->arena_size ||
        entry->allocation_size >
            evoke_preload->arena_size - entry->offset)
    {
        return false;
    }
    return true;
}

static void
evoke_am_preload_entry_reset(evoke_am_preload_entry *entry)
{
    if (entry == NULL)
    {
        return;
    }
    memset(entry, 0, sizeof(*entry));
    pg_atomic_init_u32(&entry->refcount, 0);
    pg_atomic_init_u64(&entry->last_access_counter, 0);
}

static void
evoke_am_preload_try_rewind_locked(void)
{
    for (;;)
    {
        Size new_used = 0;
        int tail_slot = -1;
        uint32 index;

        for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
        {
            evoke_am_preload_entry *entry = &evoke_preload->entries[index];
            Size end;

            if (entry->allocation_size == 0)
            {
                continue;
            }
            if (!evoke_am_preload_entry_has_block(entry))
            {
                if (!entry->in_use)
                {
                    evoke_am_preload_entry_reset(entry);
                }
                continue;
            }
            end = entry->offset + entry->allocation_size;
            if (end > new_used)
            {
                new_used = end;
                tail_slot = (int) index;
            }
        }
        if (tail_slot >= 0 && !evoke_preload->entries[tail_slot].in_use)
        {
            evoke_am_preload_entry_reset(&evoke_preload->entries[tail_slot]);
            continue;
        }
        evoke_preload->used = new_used;
        break;
    }
}

static void
evoke_am_preload_touch_entry_locked(evoke_am_preload_entry *entry)
{
    uint64 access_counter;

    if (entry == NULL)
    {
        return;
    }
    access_counter = pg_atomic_fetch_add_u64(
        &evoke_preload->access_clock,
        1
    ) + 1;
    pg_atomic_write_u64(&entry->last_access_counter, access_counter);
}

static void
evoke_am_preload_hash_remove_locked(const evoke_am_preload_entry *entry);

static void
evoke_am_preload_retire_entry_locked(evoke_am_preload_entry *entry)
{
    if (entry == NULL || !entry->in_use)
    {
        return;
    }
    evoke_am_preload_hash_remove_locked(entry);
    entry->ready = false;
    if (pg_atomic_read_u32(&entry->refcount) == 0)
    {
        entry->in_use = false;
        entry->obsolete = false;
    }
    else
    {
        entry->obsolete = true;
    }
}

static bool
evoke_am_preload_entry_same_generation(
    const evoke_am_preload_entry *entry,
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind
)
{
    if (entry == NULL || !entry->in_use || entry->obsolete ||
        entry->generation_kind != kind ||
        !evoke_am_preload_kind_valid(kind) ||
        !evoke_am_preload_payload_size_valid(kind, entry->mapped_size))
    {
        return false;
    }
    if (entry->database_oid != MyDatabaseId ||
        entry->index_oid != RelationGetRelid(index_relation) ||
        !RelFileLocatorEquals(entry->locator, index_relation->rd_locator))
    {
        return false;
    }
    if (evoke_am_preload_kind_uses_base_identity(kind))
    {
        return evoke_am_preload_base_identity_matches(&entry->meta, meta);
    }
    return evoke_am_generation_identity_matches(&entry->meta, meta);
}

static bool
evoke_am_preload_entry_matches_current_kind(
    const evoke_am_preload_entry *entry,
    Relation index_relation,
    const evoke_am_meta_page *meta
)
{
    if (entry == NULL)
    {
        return false;
    }
    return evoke_am_preload_entry_same_generation(
        entry,
        index_relation,
        meta,
        (evoke_am_preload_kind) entry->generation_kind
    );
}

static int
evoke_am_preload_hash_lookup_locked(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind
)
{
    evoke_am_preload_hash_slot *hash_slots =
        evoke_am_preload_hash_slots();
    uint32 hash_capacity = evoke_preload->hash_capacity;
    uint64 key_hash;
    uint32 probe;

    if (hash_slots == NULL || hash_capacity == 0)
    {
        return -1;
    }
    key_hash = evoke_am_preload_entry_key_hash(index_relation, meta, kind);
    for (probe = 0; probe < hash_capacity; probe++)
    {
        uint32 bucket = (uint32) (
            (key_hash + probe) & (hash_capacity - 1)
        );
        evoke_am_preload_hash_slot *hash_slot = &hash_slots[bucket];
        uint32 entry_slot_plus_one = hash_slot->entry_slot_plus_one;

        if (entry_slot_plus_one == 0)
        {
            return -1;
        }
        if (entry_slot_plus_one > evoke_preload->entry_capacity)
        {
            ereport(ERROR, (errmsg("evoke shared preload hash is corrupt")));
        }
        if (hash_slot->key_hash == key_hash &&
            evoke_am_preload_entry_same_generation(
                &evoke_preload->entries[entry_slot_plus_one - 1],
                index_relation,
                meta,
                kind))
        {
            return (int) entry_slot_plus_one - 1;
        }
    }
    return -1;
}

static void
evoke_am_preload_hash_insert_locked(int entry_slot)
{
    evoke_am_preload_hash_slot *hash_slots;
    evoke_am_preload_entry *entry;
    uint64 key_hash;
    uint32 hash_capacity;
    uint32 probe;

    if (entry_slot < 0 ||
        (uint32) entry_slot >= evoke_preload->entry_capacity)
    {
        ereport(ERROR, (errmsg("evoke shared preload slot is invalid")));
    }
    hash_slots = evoke_am_preload_hash_slots();
    hash_capacity = evoke_preload->hash_capacity;
    entry = &evoke_preload->entries[entry_slot];
    key_hash = entry->key_hash;
    if (hash_slots == NULL || hash_capacity == 0 || key_hash == 0)
    {
        ereport(
            ERROR,
            (errmsg("evoke shared preload hash is unavailable"))
        );
    }
    for (probe = 0; probe < hash_capacity; probe++)
    {
        uint32 bucket = (uint32) (
            (key_hash + probe) & (hash_capacity - 1)
        );
        evoke_am_preload_hash_slot *hash_slot = &hash_slots[bucket];

        if (hash_slot->entry_slot_plus_one != 0)
        {
            continue;
        }
        hash_slot->key_hash = key_hash;
        hash_slot->entry_slot_plus_one = (uint32) entry_slot + 1;
        return;
    }
    ereport(ERROR, (errmsg("evoke shared preload hash is full")));
}

static void
evoke_am_preload_hash_remove_locked(const evoke_am_preload_entry *entry)
{
    evoke_am_preload_hash_slot *hash_slots;
    uint32 entry_slot;
    uint32 hash_capacity;
    uint32 mask;
    uint32 probe;

    if (entry == NULL || entry->key_hash == 0)
    {
        return;
    }
    entry_slot = (uint32) (entry - evoke_preload->entries);
    if (entry_slot >= evoke_preload->entry_capacity)
    {
        ereport(ERROR, (errmsg("evoke shared preload slot is invalid")));
    }
    hash_slots = evoke_am_preload_hash_slots();
    hash_capacity = evoke_preload->hash_capacity;
    mask = hash_capacity - 1;
    for (probe = 0; probe < hash_capacity; probe++)
    {
        uint32 bucket = (uint32) ((entry->key_hash + probe) & mask);
        evoke_am_preload_hash_slot *hash_slot = &hash_slots[bucket];

        if (hash_slot->entry_slot_plus_one == 0)
        {
            return;
        }
        if (hash_slot->entry_slot_plus_one == entry_slot + 1 &&
            hash_slot->key_hash == entry->key_hash)
        {
            uint32 hole = bucket;
            uint32 scan = (hole + 1) & mask;

            while (hash_slots[scan].entry_slot_plus_one != 0)
            {
                uint32 home = (uint32) (
                    hash_slots[scan].key_hash & mask
                );
                uint32 scan_distance = (scan - home) & mask;
                uint32 hole_distance = (hole - home) & mask;

                if (scan_distance > hole_distance)
                {
                    hash_slots[hole] = hash_slots[scan];
                    hole = scan;
                }
                scan = (scan + 1) & mask;
            }
            memset(&hash_slots[hole], 0, sizeof(hash_slots[hole]));
            return;
        }
    }
}

static bool
evoke_am_preload_try_acquire_ref(evoke_am_preload_entry *entry)
{
    uint32 expected = pg_atomic_read_u32(&entry->refcount);

    for (;;)
    {
        if (expected == UINT32_MAX)
        {
            return false;
        }
        if (pg_atomic_compare_exchange_u32(
                &entry->refcount,
                &expected,
                expected + 1))
        {
            return true;
        }
    }
}

static void
evoke_am_preload_release_token(uintptr_t token)
{
    evoke_am_preload_entry *entry;
    uint32 expected;
    uint32 slot;

    if (token == 0 || !evoke_am_preload_available())
    {
        return;
    }
    slot = (uint32) (token - 1);
    if (slot >= evoke_am_preload_entry_capacity())
    {
        return;
    }
    entry = &evoke_preload->entries[slot];
    expected = pg_atomic_read_u32(&entry->refcount);
    for (;;)
    {
        if (expected == 0)
        {
            return;
        }
        if (pg_atomic_compare_exchange_u32(
                &entry->refcount,
                &expected,
                expected - 1))
        {
            break;
        }
    }
    if (expected > 1)
    {
        return;
    }
    LWLockAcquire(evoke_preload_lock, LW_SHARED);
    if (pg_atomic_read_u32(&entry->refcount) != 0 || !entry->obsolete)
    {
        LWLockRelease(evoke_preload_lock);
        return;
    }
    LWLockRelease(evoke_preload_lock);

    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    if (pg_atomic_read_u32(&entry->refcount) == 0 && entry->obsolete)
    {
        entry->in_use = false;
        entry->ready = false;
        entry->obsolete = false;
    }
    evoke_am_preload_try_rewind_locked();
    LWLockRelease(evoke_preload_lock);
}

static void
evoke_am_preload_retire_obsolete_locked(
    Relation index_relation,
    const evoke_am_meta_page *meta
)
{
    Oid index_oid = RelationGetRelid(index_relation);
    uint32 index;

    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];

        if (!entry->in_use ||
            entry->database_oid != MyDatabaseId ||
            entry->index_oid != index_oid)
        {
            continue;
        }
        if (!evoke_am_preload_entry_matches_current_kind(
                entry,
                index_relation,
                meta))
        {
            evoke_am_preload_retire_entry_locked(entry);
        }
    }
    evoke_am_preload_try_rewind_locked();
}

void
evoke_am_preload_retire_obsolete(
    Relation index_relation,
    const evoke_am_meta_page *meta
)
{
    if (!evoke_am_preload_available())
    {
        return;
    }
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    evoke_am_preload_retire_obsolete_locked(index_relation, meta);
    LWLockRelease(evoke_preload_lock);
}

void
evoke_am_preload_rekey_generation(
    Relation index_relation,
    const evoke_am_meta_page *old_meta,
    const evoke_am_meta_page *current_meta
)
{
    Oid index_oid;
    uint32 index;

    if (!evoke_am_preload_available() || index_relation == NULL ||
        old_meta == NULL || current_meta == NULL ||
        !evoke_am_meta_uses_convergent_segment_storage(old_meta) ||
        !evoke_am_meta_uses_convergent_segment_storage(current_meta))
    {
        return;
    }
    index_oid = RelationGetRelid(index_relation);
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];
        evoke_am_preload_kind kind;
        int current_slot;

        if (!entry->in_use || entry->obsolete ||
            entry->database_oid != MyDatabaseId ||
            entry->index_oid != index_oid ||
            !RelFileLocatorEquals(
                entry->locator,
                index_relation->rd_locator))
        {
            continue;
        }
        kind = (evoke_am_preload_kind) entry->generation_kind;
        if (!evoke_am_preload_entry_same_generation(
                entry,
                index_relation,
                old_meta,
                kind))
        {
            continue;
        }
        if (!evoke_am_preload_kind_rekey_safe(kind))
        {
            evoke_am_preload_retire_entry_locked(entry);
            continue;
        }
        current_slot = evoke_am_preload_hash_lookup_locked(
            index_relation,
            current_meta,
            kind
        );
        if (current_slot >= 0 && current_slot != (int) index)
        {
            evoke_am_preload_retire_entry_locked(entry);
            continue;
        }
        evoke_am_preload_hash_remove_locked(entry);
        entry->meta = *current_meta;
        entry->key_hash = evoke_am_preload_entry_key_hash(
            index_relation,
            current_meta,
            kind
        );
        evoke_am_preload_hash_insert_locked((int) index);
    }
    evoke_am_preload_retire_obsolete_locked(index_relation, current_meta);
    LWLockRelease(evoke_preload_lock);
}

void
evoke_am_preload_retire_exact(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind
)
{
    int slot;

    if (!evoke_am_preload_available())
    {
        return;
    }
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    slot = evoke_am_preload_hash_lookup_locked(index_relation, meta, kind);
    if (slot >= 0)
    {
        evoke_am_preload_retire_entry_locked(&evoke_preload->entries[slot]);
        evoke_am_preload_try_rewind_locked();
    }
    LWLockRelease(evoke_preload_lock);
}

static bool
evoke_am_preload_exact_state_locked(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    bool *resident_out,
    bool *loading_out
)
{
    bool resident = false;
    bool loading = false;
    int slot;

    slot = evoke_am_preload_hash_lookup_locked(index_relation, meta, kind);
    if (slot >= 0)
    {
        resident = evoke_preload->entries[slot].ready;
        loading = !resident;
    }
    if (resident_out != NULL)
    {
        *resident_out = resident;
    }
    if (loading_out != NULL)
    {
        *loading_out = loading;
    }
    return resident || loading;
}

bool
evoke_am_preload_exact_state(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    bool *resident_out,
    bool *loading_out
)
{
    bool found;

    if (resident_out != NULL)
    {
        *resident_out = false;
    }
    if (loading_out != NULL)
    {
        *loading_out = false;
    }
    if (!evoke_am_preload_available())
    {
        return false;
    }
    LWLockAcquire(evoke_preload_lock, LW_SHARED);
    found = evoke_am_preload_exact_state_locked(
        index_relation,
        meta,
        kind,
        resident_out,
        loading_out
    );
    LWLockRelease(evoke_preload_lock);
    return found;
}

bool
evoke_am_preload_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    evoke_am_preload_lease *lease_out
)
{
    evoke_am_preload_entry *entry;
    char *arena_base;
    int slot;

    if (lease_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke preload lease output is required")));
    }
    memset(lease_out, 0, sizeof(*lease_out));
    if (!evoke_am_preload_available())
    {
        return false;
    }
    LWLockAcquire(evoke_preload_lock, LW_SHARED);
    slot = evoke_am_preload_hash_lookup_locked(index_relation, meta, kind);
    if (slot < 0)
    {
        LWLockRelease(evoke_preload_lock);
        return false;
    }
    entry = &evoke_preload->entries[slot];
    if (!entry->ready || !evoke_am_preload_try_acquire_ref(entry))
    {
        LWLockRelease(evoke_preload_lock);
        return false;
    }
    evoke_am_preload_touch_entry_locked(entry);
    arena_base = evoke_am_preload_arena_base();
    if (arena_base == NULL || !evoke_am_preload_entry_has_block(entry))
    {
        LWLockRelease(evoke_preload_lock);
        evoke_am_preload_release_token((uintptr_t) slot + 1);
        return false;
    }
    lease_out->private_token = (uintptr_t) slot + 1;
    lease_out->private_payload = arena_base + entry->offset;
    lease_out->private_payload_size = entry->mapped_size;
    LWLockRelease(evoke_preload_lock);
    return true;
}

const void *
evoke_am_preload_lease_payload(const evoke_am_preload_lease *lease)
{
    return lease == NULL ? NULL : lease->private_payload;
}

void *
evoke_am_preload_lease_mutable_payload(evoke_am_preload_lease *lease)
{
    return lease == NULL ? NULL : (void *) lease->private_payload;
}

Size
evoke_am_preload_lease_payload_size(const evoke_am_preload_lease *lease)
{
    return lease == NULL ? 0 : lease->private_payload_size;
}

void
evoke_am_preload_lease_release(evoke_am_preload_lease *lease)
{
    uintptr_t token;

    if (lease == NULL)
    {
        return;
    }
    token = lease->private_token;
    memset(lease, 0, sizeof(*lease));
    evoke_am_preload_release_token(token);
}

void
evoke_am_preload_lease_retire(evoke_am_preload_lease *lease)
{
    evoke_am_preload_entry *entry;
    uintptr_t token;
    uint32 slot;

    if (lease == NULL)
    {
        return;
    }
    token = lease->private_token;
    memset(lease, 0, sizeof(*lease));
    if (token == 0 || !evoke_am_preload_available())
    {
        return;
    }
    slot = (uint32) (token - 1);
    if (slot >= evoke_am_preload_entry_capacity())
    {
        evoke_am_preload_release_token(token);
        return;
    }

    /* The held reference prevents this slot from being reused mid-retire. */
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    entry = &evoke_preload->entries[slot];
    if (entry->in_use && pg_atomic_read_u32(&entry->refcount) > 0)
    {
        evoke_am_preload_retire_entry_locked(entry);
    }
    LWLockRelease(evoke_preload_lock);
    evoke_am_preload_release_token(token);
}

static bool
evoke_am_preload_entry_relation_evictable(
    evoke_am_preload_entry *entry,
    Relation index_relation,
    const evoke_am_meta_page *meta,
    bool allow_preload_entries,
    evoke_am_preload_kind requested_kind
)
{
    if (entry == NULL || !entry->in_use || entry->obsolete ||
        !entry->ready ||
        !evoke_am_preload_kind_valid(
            (evoke_am_preload_kind) entry->generation_kind) ||
        pg_atomic_read_u32(&entry->refcount) != 0 ||
        entry->database_oid != MyDatabaseId)
    {
        return false;
    }
    if (!allow_preload_entries && entry->auto_preload_priority > 0)
    {
        return false;
    }
    if (evoke_am_preload_entry_same_generation(
            entry,
            index_relation,
            meta,
            (evoke_am_preload_kind) entry->generation_kind))
    {
        /*
         * The resident fold serves the unfiltered peak route while the
         * reverse TID directory serves exact filtered membership. They are
         * complementary views of one root, not interchangeable cache space.
         * Letting either evict the other makes background preload turn a
         * previously warm product route into an O(N) fallback.
         */
        return entry->generation_kind != EVOKE_AM_PRELOAD_UNIFIED_WARM &&
            entry->generation_kind != EVOKE_AM_PRELOAD_RESIDENT_FOLD &&
            entry->generation_kind !=
                EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP &&
            entry->generation_kind != requested_kind;
    }
    if (allow_preload_entries &&
        entry->auto_preload_priority >
            evoke_am_auto_preload_priority(index_relation))
    {
        /* A lower-priority preload must not displace a preferred relation. */
        return false;
    }
    return true;
}

static bool
evoke_am_preload_has_evictable_relation_locked(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    Size payload_size,
    bool allow_preload_entries,
    evoke_am_preload_kind requested_kind
)
{
    Size used = MAXALIGN(evoke_preload->used);
    uint32 index;

    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];
        Size end;

        if (!evoke_am_preload_entry_relation_evictable(
                entry,
                index_relation,
                meta,
                allow_preload_entries,
                requested_kind))
        {
            continue;
        }
        end = entry->offset + entry->allocation_size;
        if (entry->allocation_size >= payload_size || end == used)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_am_preload_evict_relation_entry_locked(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    Size payload_size,
    bool allow_preload_entries,
    evoke_am_preload_kind requested_kind
)
{
    Size used = MAXALIGN(evoke_preload->used);
    int best_slot = -1;
    uint32 index;

    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];
        evoke_am_preload_entry *best;
        bool entry_fits;
        bool best_fits;
        Size end;

        if (!evoke_am_preload_entry_relation_evictable(
                entry,
                index_relation,
                meta,
                allow_preload_entries,
                requested_kind))
        {
            continue;
        }
        end = entry->offset + entry->allocation_size;
        entry_fits = entry->allocation_size >= payload_size || end == used;
        if (!entry_fits)
        {
            continue;
        }
        if (best_slot < 0)
        {
            best_slot = (int) index;
            continue;
        }
        best = &evoke_preload->entries[best_slot];
        if ((entry->auto_preload_priority > 0) !=
            (best->auto_preload_priority > 0))
        {
            if (entry->auto_preload_priority == 0)
            {
                best_slot = (int) index;
            }
            continue;
        }
        if (entry->auto_preload_priority != best->auto_preload_priority)
        {
            if (entry->auto_preload_priority < best->auto_preload_priority)
            {
                best_slot = (int) index;
            }
            continue;
        }
        best_fits = best->allocation_size >= payload_size;
        if (entry->allocation_size >= payload_size && !best_fits)
        {
            best_slot = (int) index;
            continue;
        }
        if ((entry->allocation_size >= payload_size) == best_fits &&
            entry->allocation_size != best->allocation_size)
        {
            if (entry->allocation_size < best->allocation_size)
            {
                best_slot = (int) index;
            }
            continue;
        }
        if ((entry->allocation_size >= payload_size) == best_fits &&
            pg_atomic_read_u64(&entry->last_access_counter) !=
                pg_atomic_read_u64(&best->last_access_counter) &&
            pg_atomic_read_u64(&entry->last_access_counter) <
                pg_atomic_read_u64(&best->last_access_counter))
        {
            best_slot = (int) index;
        }
    }
    if (best_slot < 0)
    {
        return false;
    }
    if (evoke_preload->relation_entry_evictions < UINT64_MAX)
    {
        evoke_preload->relation_entry_evictions++;
    }
    if (UINT64_MAX - evoke_preload->relation_entry_eviction_bytes >=
        evoke_preload->entries[best_slot].mapped_size)
    {
        evoke_preload->relation_entry_eviction_bytes +=
            evoke_preload->entries[best_slot].mapped_size;
    }
    else
    {
        evoke_preload->relation_entry_eviction_bytes = UINT64_MAX;
    }
    evoke_am_preload_retire_entry_locked(&evoke_preload->entries[best_slot]);
    evoke_am_preload_try_rewind_locked();
    return true;
}

static bool
evoke_am_preload_publish_token(uintptr_t token, bool ready)
{
    evoke_am_preload_entry *entry;
    bool published = false;
    uint32 slot;

    if (token == 0 || !evoke_am_preload_available())
    {
        return false;
    }
    slot = (uint32) (token - 1);
    if (slot >= evoke_am_preload_entry_capacity())
    {
        return false;
    }
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    entry = &evoke_preload->entries[slot];
    if (ready && entry->in_use && !entry->obsolete &&
        pg_atomic_read_u32(&entry->refcount) > 0)
    {
        entry->ready = true;
        entry->obsolete = false;
        published = true;
    }
    else
    {
        evoke_am_preload_retire_entry_locked(entry);
        evoke_am_preload_try_rewind_locked();
    }
    LWLockRelease(evoke_preload_lock);
    return published;
}

evoke_am_preload_reserve_result
evoke_am_preload_reserve(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind kind,
    Size payload_size,
    evoke_am_preload_reservation *reservation_out
)
{
    Size offset;
    Size allocation_size;
    Size end;
    int free_slot = -1;
    int reusable_slot = -1;
    int existing_slot;
    uint32 index;
    uint32 eviction_attempts = 0;
    uint32 eviction_limit;

    if (reservation_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke preload reservation is required")));
    }
    memset(reservation_out, 0, sizeof(*reservation_out));
    if (!evoke_am_preload_cache_available() ||
        !evoke_am_preload_payload_size_valid(kind, payload_size))
    {
        return EVOKE_AM_PRELOAD_RESERVE_FAILED;
    }
    if (evoke_active_publication_token != 0)
    {
        ereport(ERROR, (errmsg("evoke preload publication is already active")));
    }
    eviction_limit = evoke_am_preload_entry_capacity();
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
retry_allocate:
    free_slot = -1;
    reusable_slot = -1;
    evoke_am_preload_retire_obsolete_locked(index_relation, meta);
    existing_slot = evoke_am_preload_hash_lookup_locked(
        index_relation,
        meta,
        kind
    );
    if (existing_slot >= 0)
    {
        bool ready = evoke_preload->entries[existing_slot].ready;

        LWLockRelease(evoke_preload_lock);
        return ready
            ? EVOKE_AM_PRELOAD_RESERVE_READY
            : EVOKE_AM_PRELOAD_RESERVE_LOADING;
    }
    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];

        if (!entry->in_use && !evoke_am_preload_entry_has_block(entry) &&
            free_slot < 0)
        {
            free_slot = (int) index;
        }
        if (!entry->in_use && evoke_am_preload_entry_has_block(entry) &&
            entry->allocation_size >= payload_size &&
            (reusable_slot < 0 ||
             entry->allocation_size <
                evoke_preload->entries[reusable_slot].allocation_size))
        {
            reusable_slot = (int) index;
        }
    }
    if (reusable_slot >= 0)
    {
        free_slot = reusable_slot;
        offset = evoke_preload->entries[free_slot].offset;
        allocation_size =
            evoke_preload->entries[free_slot].allocation_size;
    }
    else
    {
        offset = MAXALIGN(evoke_preload->used);
        allocation_size = MAXALIGN(payload_size);
        if (free_slot < 0 ||
            !evoke_am_preload_checked_add(
                offset,
                allocation_size,
                &end) ||
            end > evoke_preload->arena_size)
        {
            if (eviction_attempts < eviction_limit &&
                evoke_am_preload_evict_relation_entry_locked(
                    index_relation,
                    meta,
                    payload_size,
                    true,
                    kind))
            {
                eviction_attempts++;
                goto retry_allocate;
            }
            LWLockRelease(evoke_preload_lock);
            return EVOKE_AM_PRELOAD_RESERVE_FAILED;
        }
        evoke_preload->used = MAXALIGN(end);
    }
    if (free_slot < 0)
    {
        LWLockRelease(evoke_preload_lock);
        return EVOKE_AM_PRELOAD_RESERVE_FAILED;
    }
    evoke_am_preload_entry_reset(&evoke_preload->entries[free_slot]);
    evoke_preload->entries[free_slot].in_use = true;
    evoke_preload->entries[free_slot].generation_kind = kind;
    evoke_preload->entries[free_slot].auto_preload_priority =
        evoke_am_auto_preload_priority(index_relation);
    evoke_preload->entries[free_slot].database_oid = MyDatabaseId;
    evoke_preload->entries[free_slot].index_oid =
        RelationGetRelid(index_relation);
    evoke_preload->entries[free_slot].locator = index_relation->rd_locator;
    evoke_preload->entries[free_slot].meta = *meta;
    evoke_preload->entries[free_slot].offset = offset;
    evoke_preload->entries[free_slot].mapped_size = payload_size;
    evoke_preload->entries[free_slot].allocation_size = allocation_size;
    evoke_preload->entries[free_slot].key_hash =
        evoke_am_preload_entry_key_hash(index_relation, meta, kind);
    pg_atomic_write_u32(&evoke_preload->entries[free_slot].refcount, 1);
    evoke_am_preload_touch_entry_locked(&evoke_preload->entries[free_slot]);
    evoke_am_preload_hash_insert_locked(free_slot);
    reservation_out->private_token = (uintptr_t) free_slot + 1;
    reservation_out->private_payload =
        evoke_am_preload_arena_base() + offset;
    reservation_out->private_payload_size = payload_size;
    evoke_active_publication_token = reservation_out->private_token;
    LWLockRelease(evoke_preload_lock);
    return EVOKE_AM_PRELOAD_RESERVE_NEW;
}

void *
evoke_am_preload_reservation_payload(
    evoke_am_preload_reservation *reservation
)
{
    return reservation == NULL ? NULL : reservation->private_payload;
}

bool
evoke_am_preload_reservation_commit(
    evoke_am_preload_reservation *reservation
)
{
    bool published;
    uintptr_t token;

    if (reservation == NULL || reservation->private_token == 0)
    {
        return false;
    }
    token = reservation->private_token;
    published = evoke_am_preload_publish_token(token, true);
    evoke_am_preload_release_token(token);
    memset(reservation, 0, sizeof(*reservation));
    if (evoke_active_publication_token == token)
    {
        evoke_active_publication_token = 0;
    }
    return published;
}

void
evoke_am_preload_reservation_abort(
    evoke_am_preload_reservation *reservation
)
{
    uintptr_t token;

    if (reservation == NULL || reservation->private_token == 0)
    {
        return;
    }
    token = reservation->private_token;
    evoke_am_preload_publish_token(token, false);
    evoke_am_preload_release_token(token);
    memset(reservation, 0, sizeof(*reservation));
    if (evoke_active_publication_token == token)
    {
        evoke_active_publication_token = 0;
    }
}

void
evoke_am_preload_abort_active_publication(void)
{
    evoke_am_preload_reservation reservation = {0};

    reservation.private_token = evoke_active_publication_token;
    evoke_am_preload_reservation_abort(&reservation);
}

int
evoke_am_preload_clear(void)
{
    int cleared = 0;
    uint32 index;

    if (!evoke_am_preload_available())
    {
        return 0;
    }
    LWLockAcquire(evoke_preload_lock, LW_EXCLUSIVE);
    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        if (evoke_preload->entries[index].in_use)
        {
            evoke_am_preload_retire_entry_locked(
                &evoke_preload->entries[index]
            );
            cleared++;
        }
    }
    evoke_am_preload_try_rewind_locked();
    LWLockRelease(evoke_preload_lock);
    return cleared;
}

void
evoke_am_preload_status_snapshot(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_preload_kind requested_kind,
    Size candidate_bytes,
    evoke_am_preload_status *status_out
)
{
    uint32 index;

    if (status_out == NULL)
    {
        return;
    }
    memset(status_out, 0, sizeof(*status_out));
    status_out->candidate_bytes = candidate_bytes;
    status_out->available = evoke_am_preload_available();
    status_out->admission_state = EVOKE_AM_PRELOAD_ADMISSION_UNAVAILABLE;
    if (!status_out->available)
    {
        return;
    }
    LWLockAcquire(evoke_preload_lock, LW_SHARED);
    evoke_am_preload_exact_state_locked(
        index_relation,
        meta,
        requested_kind,
        &status_out->resident,
        &status_out->loading
    );
    evoke_am_preload_exact_state_locked(
        index_relation,
        meta,
        EVOKE_AM_PRELOAD_HOT_FOLD,
        &status_out->hot_fold_current,
        &status_out->hot_fold_loading
    );
    evoke_am_preload_exact_state_locked(
        index_relation,
        meta,
        EVOKE_AM_PRELOAD_RESIDENT_FOLD,
        &status_out->resident_fold_current,
        &status_out->resident_fold_loading
    );
    if (status_out->resident_fold_current)
    {
        int slot = evoke_am_preload_hash_lookup_locked(
            index_relation,
            meta,
            EVOKE_AM_PRELOAD_RESIDENT_FOLD
        );

        if (slot >= 0 && evoke_preload->entries[slot].ready)
        {
            status_out->resident_fold_bytes =
                evoke_preload->entries[slot].mapped_size;
        }
    }
    status_out->arena_size = evoke_preload->arena_size;
    status_out->entry_capacity = evoke_preload->entry_capacity;
    status_out->hash_capacity = evoke_preload->hash_capacity;
    status_out->used = evoke_preload->used;
    if (status_out->arena_size > status_out->used)
    {
        status_out->free_bytes =
            status_out->arena_size - status_out->used;
    }
    status_out->candidate_fits_empty =
        candidate_bytes <= status_out->arena_size;
    status_out->relation_entry_evictions =
        evoke_preload->relation_entry_evictions;
    status_out->relation_entry_eviction_bytes =
        evoke_preload->relation_entry_eviction_bytes;
    status_out->access_clock = pg_atomic_read_u64(
        &evoke_preload->access_clock
    );
    for (index = 0; index < evoke_am_preload_entry_capacity(); index++)
    {
        evoke_am_preload_entry *entry = &evoke_preload->entries[index];

        if (entry->in_use)
        {
            status_out->entries++;
            if (entry->generation_kind == EVOKE_AM_PRELOAD_UNIFIED_WARM)
            {
                status_out->unified_warm_entries++;
            }
            else if (entry->generation_kind == EVOKE_AM_PRELOAD_HOT_FOLD)
            {
                status_out->hot_fold_entries++;
            }
            else if (entry->generation_kind ==
                         EVOKE_AM_PRELOAD_RESIDENT_FOLD)
            {
                status_out->resident_fold_entries++;
            }
            else if (entry->generation_kind ==
                         EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS)
            {
                status_out->document_length_entries++;
            }
            else if (entry->generation_kind ==
                         EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP)
            {
                status_out->document_tid_lookup_entries++;
            }
            else if (entry->generation_kind ==
                         EVOKE_AM_PRELOAD_VALIDATED_PAGES)
            {
                status_out->validated_page_entries++;
            }
            if (entry->obsolete)
            {
                status_out->obsolete_entries++;
            }
            if (pg_atomic_read_u32(&entry->refcount) > 0)
            {
                status_out->refcounted_entries++;
            }
            if (entry->ready)
            {
                status_out->ready_entries++;
            }
        }
        else if (evoke_am_preload_entry_has_block(entry))
        {
            status_out->reusable_entries++;
            status_out->reusable_bytes += entry->allocation_size;
        }
        if (!entry->in_use)
        {
            if (evoke_am_preload_entry_has_block(entry) &&
                entry->allocation_size >= candidate_bytes)
            {
                status_out->has_reusable_slot = true;
            }
            else if (!evoke_am_preload_entry_has_block(entry))
            {
                status_out->has_free_slot = true;
            }
        }
    }
    status_out->has_evictable_relation =
        evoke_am_preload_has_evictable_relation_locked(
            index_relation,
            meta,
            candidate_bytes,
            true,
            requested_kind
        );
    LWLockRelease(evoke_preload_lock);

    if (status_out->resident)
    {
        status_out->admission_state = EVOKE_AM_PRELOAD_ADMISSION_RESIDENT;
    }
    else if (status_out->loading)
    {
        status_out->admission_state = EVOKE_AM_PRELOAD_ADMISSION_LOADING;
    }
    else if (candidate_bytes == 0)
    {
        status_out->admission_state =
            EVOKE_AM_PRELOAD_ADMISSION_ZERO_PAYLOAD;
    }
    else if (!status_out->candidate_fits_empty)
    {
        status_out->admission_state = EVOKE_AM_PRELOAD_ADMISSION_OVERSIZED;
    }
    else if (status_out->has_reusable_slot ||
             status_out->has_evictable_relation ||
             (status_out->has_free_slot &&
              candidate_bytes <= status_out->free_bytes))
    {
        status_out->admission_state = EVOKE_AM_PRELOAD_ADMISSION_ADMISSIBLE;
    }
    else if (!status_out->has_free_slot)
    {
        status_out->admission_state =
            EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SLOT;
    }
    else
    {
        status_out->admission_state =
            EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SPACE;
    }
}

const char *
evoke_am_preload_admission_state_name(evoke_am_preload_admission_state state)
{
    switch (state)
    {
        case EVOKE_AM_PRELOAD_ADMISSION_UNAVAILABLE:
            return "unavailable";
        case EVOKE_AM_PRELOAD_ADMISSION_RESIDENT:
            return "resident";
        case EVOKE_AM_PRELOAD_ADMISSION_LOADING:
            return "loading";
        case EVOKE_AM_PRELOAD_ADMISSION_ZERO_PAYLOAD:
            return "zero_payload";
        case EVOKE_AM_PRELOAD_ADMISSION_OVERSIZED:
            return "oversized";
        case EVOKE_AM_PRELOAD_ADMISSION_ADMISSIBLE:
            return "admissible";
        case EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SLOT:
            return "blocked_no_slot";
        case EVOKE_AM_PRELOAD_ADMISSION_BLOCKED_NO_SPACE:
            return "blocked_no_space";
    }
    return "unknown";
}

static void
evoke_am_preload_advise_hugepage(void)
{
#ifdef MADV_HUGEPAGE
    char *arena_base;
    long page_size;
    uintptr_t address;
    uintptr_t aligned_address;
    Size adjust;
    Size advised_size;

    if (!evoke_am_preload_cache_available())
    {
        return;
    }
    arena_base = evoke_am_preload_arena_base();
    if (arena_base == NULL)
    {
        return;
    }
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        return;
    }
    address = (uintptr_t) arena_base;
    aligned_address = address - (address % (uintptr_t) page_size);
    adjust = (Size) (address - aligned_address);
    if (!evoke_am_preload_checked_add(
            evoke_preload->arena_size,
            adjust,
            &advised_size))
    {
        return;
    }
    if (madvise(
            (void *) aligned_address,
            advised_size,
            MADV_HUGEPAGE) != 0)
    {
        ereport(
            DEBUG1,
            (errmsg("evoke shared preload hugepage advice failed: %m"))
        );
    }
#endif
}

void
evoke_am_preload_shmem_startup(LWLock *lock)
{
    Size shmem_size = evoke_am_preload_shmem_size();
    Size arena_size;
    uint32 index;
    bool found;

    if (lock == NULL)
    {
        ereport(FATAL, (errmsg("evoke preload lock is unavailable")));
    }
    evoke_preload_lock = lock;
    evoke_preload = ShmemInitStruct(
        "evoke shared runtime arena",
        shmem_size,
        &found
    );
    if (!found)
    {
        arena_size = evoke_am_preload_arena_size();
        memset(evoke_preload, 0, shmem_size);
        evoke_preload->magic = EVOKE_AM_PRELOAD_MAGIC;
        evoke_preload->version = EVOKE_AM_PRELOAD_VERSION;
        evoke_preload->arena_size = arena_size;
        evoke_preload->entry_capacity =
            evoke_am_preload_entry_capacity_for_arena(arena_size);
        evoke_preload->hash_capacity =
            evoke_am_preload_hash_capacity_for_entries(
                evoke_preload->entry_capacity
            );
        pg_atomic_init_u64(&evoke_preload->access_clock, 0);
        for (index = 0; index < evoke_preload->entry_capacity; index++)
        {
            pg_atomic_init_u32(&evoke_preload->entries[index].refcount, 0);
            pg_atomic_init_u64(
                &evoke_preload->entries[index].last_access_counter,
                0
            );
        }
        if (evoke_test_shared_preload_registry_fill > 0)
        {
            uint32 fill =
                (uint32) evoke_test_shared_preload_registry_fill;

            if (fill >= evoke_preload->entry_capacity)
            {
                ereport(
                    FATAL,
                    (
                        errmsg("evoke test registry fill exceeds its capacity"),
                        errdetail(
                            "fill=%u capacity=%u",
                            fill,
                            evoke_preload->entry_capacity
                        )
                    )
                );
            }
            for (index = 0; index < fill; index++)
            {
                evoke_am_preload_entry *entry =
                    &evoke_preload->entries[index];
                uint64 key_hash = UINT64_C(0xcbf29ce484222325);

                key_hash = evoke_am_preload_hash_mix(
                    key_hash,
                    (uint64) index + 1
                );
                key_hash = evoke_am_preload_hash_mix(key_hash, fill);
                entry->in_use = true;
                entry->ready = true;
                entry->generation_kind = EVOKE_AM_PRELOAD_UNIFIED_WARM;
                entry->key_hash = key_hash == 0 ? UINT64_C(1) : key_hash;
                evoke_am_preload_hash_insert_locked((int) index);
            }
            pg_atomic_write_u64(&evoke_preload->access_clock, fill);
        }
    }
    else if (!evoke_am_preload_available())
    {
        ereport(
            FATAL,
            (
                errmsg("evoke shared preload ABI does not match this binary"),
                errdetail(
                    "found magic=%08x version=%u, expected magic=%08x "
                    "version=%u",
                    evoke_preload->magic,
                    evoke_preload->version,
                    EVOKE_AM_PRELOAD_MAGIC,
                    EVOKE_AM_PRELOAD_VERSION
                ),
                errhint("Restart PostgreSQL with one consistent evoke binary.")
            )
        );
    }
    evoke_am_preload_advise_hugepage();
}

static bool
evoke_am_preload_check_test_registry_capacity(
    int *new_value,
    void **extra,
    GucSource source
)
{
    (void) extra;
    (void) source;
    if (*new_value != 0 && *new_value < EVOKE_AM_PRELOAD_MIN_ENTRIES)
    {
        GUC_check_errdetail(
            "The value must be zero or at least %u.",
            EVOKE_AM_PRELOAD_MIN_ENTRIES
        );
        return false;
    }
    return true;
}

void
evoke_am_preload_define_gucs(void)
{
    if (!process_shared_preload_libraries_in_progress ||
        evoke_am_preload_gucs_initialized)
    {
        return;
    }
    DefineCustomIntVariable(
        "evoke.shared_runtime_size",
        "Sets the shared runtime and derived-residency arena size.",
        "When evoke is loaded through shared_preload_libraries, this "
        "reserves shared memory for runtime queues, exact-root markers, and "
        "optional HOT_FOLD and exact-root resident projections. Durable "
        "posting authority remains in relation pages. A positive value is "
        "required for SAE indexes; BM25-only indexes remain page-native "
        "when it is 0.",
        &evoke_shared_runtime_size_mb,
        0,
        0,
        EVOKE_AM_PRELOAD_MAX_MB,
        PGC_POSTMASTER,
        GUC_UNIT_MB,
        NULL,
        NULL,
        NULL
    );
    DefineCustomIntVariable(
        "evoke.test_shared_preload_registry_capacity",
        "Overrides shared preload registry metadata capacity for tests.",
        "This restart-only test hook changes metadata capacity without "
        "inflating the payload arena. Zero keeps production auto-sizing.",
        &evoke_test_shared_preload_registry_capacity,
        0,
        0,
        EVOKE_AM_PRELOAD_MAX_ENTRIES,
        PGC_POSTMASTER,
        GUC_NOT_IN_SAMPLE,
        evoke_am_preload_check_test_registry_capacity,
        NULL,
        NULL
    );
    DefineCustomIntVariable(
        "evoke.test_shared_preload_registry_fill",
        "Pre-fills shared preload hash entries for capacity tests.",
        "This restart-only test hook creates metadata-only occupancy before "
        "backends start. Zero disables synthetic entries.",
        &evoke_test_shared_preload_registry_fill,
        0,
        0,
        EVOKE_AM_PRELOAD_MAX_ENTRIES - 1,
        PGC_POSTMASTER,
        GUC_NOT_IN_SAMPLE,
        NULL,
        NULL,
        NULL
    );
    evoke_am_preload_gucs_initialized = true;
}

int
evoke_am_preload_configured_mb(void)
{
    return evoke_shared_runtime_size_mb;
}

int
evoke_am_preload_test_registry_fill(void)
{
    return evoke_test_shared_preload_registry_fill;
}
