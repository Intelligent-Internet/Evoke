#include "postgres.h"

#include <stdlib.h>

#include "access/xact.h"
#include "access/xlog.h"
#include "miscadmin.h"
#include "port/atomics.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "utils/backend_status.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"

#include "evoke_am_scheduler.h"
#include "evoke_core.h"

#define EVOKE_AM_SCHEDULER_MAGIC UINT32_C(0x53433249)
#define EVOKE_AM_SCHEDULER_VERSION 3
#define EVOKE_AM_WORK_HINT_CAPACITY 4096
#define EVOKE_AM_RECONCILE_DB_CAPACITY 256
#define EVOKE_AM_SEMANTIC_TELEMETRY_CAPACITY 1024
#define EVOKE_AM_POSTING_HEAT_CAPACITY 4096
#define EVOKE_AM_CATALOG_RECONCILE_INTERVAL_MS 300000
#define EVOKE_AM_STANDBY_PRELOAD_RECONCILE_INTERVAL_MS 5000
#define EVOKE_AM_POSTING_HEAT_HALF_LIFE_MS UINT64_C(300000)
#define EVOKE_AM_POSTING_HEAT_MIN_QUERY_HEAT UINT64_C(64)
#define EVOKE_AM_POSTING_HEAT_MIN_ROOT_HEAT UINT64_C(32)
#define EVOKE_AM_POSTING_HEAT_RETRY_COOLDOWN_MS UINT64_C(300000)
#define EVOKE_AM_WORKER_LAUNCH_RESERVATION_TIMEOUT_MS 30000

typedef struct evoke_am_scheduler_work_hint
{
    Oid database_oid;
    Oid index_oid;
    uint8 flags;
    uint64 sequence;
    TimestampTz maintenance_first_marked_at;
    TimestampTz accelerator_retry_after;
} evoke_am_scheduler_work_hint;

typedef struct evoke_am_scheduler_reconcile_db
{
    Oid database_oid;
    Oid maintenance_last_served_index_oid;
    uint8 requested_flags;
    uint8 running_flags;
    TimestampTz maintenance_completed_at;
    TimestampTz preload_completed_at;
    uint64 maintenance_count;
    uint64 maintenance_last_rows;
    uint64 maintenance_last_duration_us;
    uint64 maintenance_max_duration_us;
    uint64 maintenance_service_count;
    uint64 preload_count;
    uint64 preload_last_rows;
    uint64 preload_last_duration_us;
    uint64 preload_max_duration_us;
    uint64 last_access_counter;
} evoke_am_scheduler_reconcile_db;

typedef struct evoke_am_scheduler_telemetry_entry
{
    Oid database_oid;
    Oid index_oid;
    uint64 access_sequence;
    uint64 attempts;
    uint64 completed;
    uint64 retired;
    uint64 retries;
    uint64 failures;
    uint32 last_batch_completed;
    uint32 last_batch_retired;
    uint64 last_duration_us;
    uint64 frontier_scanned_records;
    uint64 frontier_scanned_bytes;
    uint64 frontier_xid_lock_windows;
    uint64 frontier_reconstructions;
    uint32 last_frontier_scanned_records;
    uint64 last_frontier_scanned_bytes;
    uint32 last_frontier_xid_lock_windows;
    bool last_frontier_reconstructed;
    TimestampTz last_attempt_at;
    TimestampTz last_completed_at;
    TimestampTz last_error_at;
    char last_error[EVOKE_AM_SEMANTIC_ERROR_LEN];
} evoke_am_scheduler_telemetry_entry;

typedef struct evoke_am_scheduler_control
{
    uint32 magic;
    uint32 version;
    uint32 active_background_workers;
    uint32 pending_maintenance_worker_launches;
    uint32 active_preload_workers;
    uint32 active_index_maintenance_workers;
    pg_atomic_uint32 maintenance_worker_limit;
    Oid last_maintenance_db_oid;
    TimestampTz last_maintenance_cycle;
    TimestampTz last_maintenance_worker_launch;
    uint64 work_hint_sequence;
    uint64 work_hint_overflows;
    uint64 work_hints_consumed;
    uint64 reconcile_access_clock;
    uint64 semantic_telemetry_access_clock;
    uint64 posting_heat_flushes;
    uint64 posting_heat_query_observations;
    uint64 posting_heat_evictions;
    evoke_am_scheduler_work_hint work_hints[EVOKE_AM_WORK_HINT_CAPACITY];
    evoke_am_scheduler_reconcile_db
        reconcile_dbs[EVOKE_AM_RECONCILE_DB_CAPACITY];
    evoke_am_scheduler_telemetry_entry
        semantic_telemetry[EVOKE_AM_SEMANTIC_TELEMETRY_CAPACITY];
    evoke_posting_heat_entry posting_heat[EVOKE_AM_POSTING_HEAT_CAPACITY];
} evoke_am_scheduler_control;

static evoke_am_scheduler_control *evoke_scheduler = NULL;
static LWLock *evoke_scheduler_lock = NULL;
static bool evoke_scheduler_process_counted_active = false;
static bool evoke_scheduler_process_launch_reserved = false;
static bool evoke_scheduler_process_worker_exit_registered = false;
static bool evoke_scheduler_process_launch_exit_registered = false;
static evoke_am_scheduler_worker_phase evoke_scheduler_process_phase =
    EVOKE_AM_SCHEDULER_WORKER_IDLE;

static bool
evoke_am_scheduler_valid_work_class(uint8 flag)
{
    return flag == EVOKE_AM_WORK_HINT_MAINTENANCE ||
        flag == EVOKE_AM_WORK_HINT_PRELOAD;
}

static int
evoke_am_scheduler_compare_oid(const void *left, const void *right)
{
    Oid a = *(const Oid *) left;
    Oid b = *(const Oid *) right;

    if (a == b)
    {
        return 0;
    }
    return a < b ? -1 : 1;
}

static int
evoke_am_scheduler_compare_work_hint_token(
    const void *left,
    const void *right
)
{
    const evoke_am_scheduler_work_hint_token *a = left;
    const evoke_am_scheduler_work_hint_token *b = right;

    if (a->index_oid == b->index_oid)
    {
        return 0;
    }
    return a->index_oid < b->index_oid ? -1 : 1;
}

static int
evoke_am_scheduler_compare_database_hint_token(
    const void *left,
    const void *right
)
{
    const evoke_am_scheduler_database_hint_token *a = left;
    const evoke_am_scheduler_database_hint_token *b = right;

    if (a->sequence != b->sequence)
    {
        return a->sequence < b->sequence ? -1 : 1;
    }
    if (a->database_oid == b->database_oid)
    {
        return 0;
    }
    return a->database_oid < b->database_oid ? -1 : 1;
}

Size
evoke_am_scheduler_shmem_size(void)
{
    return MAXALIGN(sizeof(evoke_am_scheduler_control));
}

bool
evoke_am_scheduler_available(void)
{
    return evoke_scheduler != NULL &&
        evoke_scheduler->magic == EVOKE_AM_SCHEDULER_MAGIC &&
        evoke_scheduler->version == EVOKE_AM_SCHEDULER_VERSION;
}

void
evoke_am_scheduler_shmem_startup(
    LWLock *shared_lock,
    uint32 maintenance_worker_limit
)
{
    Size scheduler_size = evoke_am_scheduler_shmem_size();
    bool found;

    if (shared_lock == NULL)
    {
        ereport(FATAL, (errmsg("evoke scheduler shared lock is required")));
    }
    evoke_scheduler_lock = shared_lock;
    evoke_scheduler = ShmemInitStruct(
        "evoke scheduler state",
        scheduler_size,
        &found
    );
    if (!found)
    {
        memset(evoke_scheduler, 0, scheduler_size);
        evoke_scheduler->magic = EVOKE_AM_SCHEDULER_MAGIC;
        evoke_scheduler->version = EVOKE_AM_SCHEDULER_VERSION;
        pg_atomic_init_u32(
            &evoke_scheduler->maintenance_worker_limit,
            maintenance_worker_limit
        );
    }
    else if (!evoke_am_scheduler_available())
    {
        ereport(
            FATAL,
            (
                errmsg("evoke scheduler ABI does not match this binary"),
                errdetail(
                    "found magic=%08x version=%u, expected magic=%08x "
                    "version=%u",
                    evoke_scheduler->magic,
                    evoke_scheduler->version,
                    EVOKE_AM_SCHEDULER_MAGIC,
                    EVOKE_AM_SCHEDULER_VERSION
                ),
                errhint("Restart PostgreSQL with one consistent evoke binary.")
            )
        );
    }
}

uint32
evoke_am_scheduler_worker_limit(void)
{
    if (!evoke_am_scheduler_available())
    {
        return 0;
    }
    return pg_atomic_read_u32(&evoke_scheduler->maintenance_worker_limit);
}

void
evoke_am_scheduler_set_worker_limit(uint32 limit)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    pg_atomic_write_u32(
        &evoke_scheduler->maintenance_worker_limit,
        limit
    );
}

void
evoke_am_scheduler_note_maintenance_cycle(void)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    evoke_scheduler->last_maintenance_cycle = GetCurrentTimestamp();
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_work_hint_mark(Oid index_oid, uint8 flags)
{
    evoke_am_scheduler_work_hint *hint = NULL;
    bool maintenance_was_pending = false;
    uint64 sequence;
    uint64 key;
    uint32 start_slot;
    uint32 tombstone_slot = EVOKE_AM_WORK_HINT_CAPACITY;
    bool request_reconcile = false;

    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || flags == 0)
    {
        return;
    }

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    key = ((uint64) MyDatabaseId << 32) ^ (uint64) index_oid;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    start_slot = (uint32) (key % EVOKE_AM_WORK_HINT_CAPACITY);
    for (uint32 probe = 0;
         probe < EVOKE_AM_WORK_HINT_CAPACITY;
         probe++)
    {
        uint32 slot =
            (start_slot + probe) % EVOKE_AM_WORK_HINT_CAPACITY;
        evoke_am_scheduler_work_hint *candidate =
            &evoke_scheduler->work_hints[slot];

        if (candidate->flags != 0)
        {
            if (candidate->database_oid == MyDatabaseId &&
                candidate->index_oid == index_oid)
            {
                hint = candidate;
                break;
            }
            continue;
        }
        if (candidate->sequence == UINT64_MAX)
        {
            if (tombstone_slot == EVOKE_AM_WORK_HINT_CAPACITY)
            {
                tombstone_slot = slot;
            }
            continue;
        }
        hint = tombstone_slot == EVOKE_AM_WORK_HINT_CAPACITY
            ? candidate
            : &evoke_scheduler->work_hints[tombstone_slot];
        break;
    }
    if (hint == NULL && tombstone_slot != EVOKE_AM_WORK_HINT_CAPACITY)
    {
        hint = &evoke_scheduler->work_hints[tombstone_slot];
    }
    if (hint == NULL)
    {
        evoke_scheduler->work_hint_overflows++;
        request_reconcile = true;
    }
    else
    {
        maintenance_was_pending =
            (hint->flags & EVOKE_AM_WORK_HINT_MAINTENANCE) != 0;
        sequence = evoke_scheduler->work_hint_sequence;
        sequence = sequence == UINT64_MAX ? 1 : sequence + 1;
        evoke_scheduler->work_hint_sequence = sequence;
        if (hint->flags != 0)
        {
            hint->flags |= flags;
        }
        else
        {
            hint->database_oid = MyDatabaseId;
            hint->index_oid = index_oid;
            hint->flags = flags;
            hint->maintenance_first_marked_at = 0;
            hint->accelerator_retry_after = 0;
        }
        hint->sequence = sequence;
        if ((flags & EVOKE_AM_WORK_HINT_MAINTENANCE) != 0 &&
            !maintenance_was_pending)
        {
            hint->maintenance_first_marked_at = GetCurrentTimestamp();
        }
    }
    LWLockRelease(evoke_scheduler_lock);
    if (request_reconcile)
    {
        evoke_am_scheduler_request_catalog_reconcile(flags);
    }
}

Oid *
evoke_am_scheduler_work_hint_consume(uint8 flag, size_t *count_out)
{
    Oid *oids;
    size_t count = 0;
    size_t unique_count = 0;
    size_t active_count = 0;
    uint32 i;

    if (count_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke work hint count output is required")));
    }
    *count_out = 0;
    if (!evoke_am_scheduler_available())
    {
        return NULL;
    }
    if (!evoke_am_scheduler_valid_work_class(flag))
    {
        ereport(ERROR, (errmsg("invalid evoke work hint class")));
    }

    oids = palloc(sizeof(*oids) * EVOKE_AM_WORK_HINT_CAPACITY);
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[i];

        if (hint->database_oid != MyDatabaseId ||
            !OidIsValid(hint->index_oid) ||
            (hint->flags & flag) == 0)
        {
            continue;
        }
        oids[count++] = hint->index_oid;
        hint->flags &= (uint8) ~flag;
        if (flag == EVOKE_AM_WORK_HINT_MAINTENANCE)
        {
            hint->maintenance_first_marked_at = 0;
            hint->accelerator_retry_after = 0;
        }
        if (hint->flags == 0)
        {
            hint->database_oid = InvalidOid;
            hint->index_oid = InvalidOid;
            hint->sequence = UINT64_MAX;
        }
        evoke_scheduler->work_hints_consumed++;
    }
    for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        if (evoke_scheduler->work_hints[i].flags != 0)
        {
            active_count++;
        }
    }
    if (active_count == 0)
    {
        memset(
            evoke_scheduler->work_hints,
            0,
            sizeof(evoke_scheduler->work_hints)
        );
    }
    LWLockRelease(evoke_scheduler_lock);

    if (count == 0)
    {
        pfree(oids);
        return NULL;
    }
    qsort(oids, count, sizeof(*oids), evoke_am_scheduler_compare_oid);
    for (i = 0; i < count; i++)
    {
        if (unique_count == 0 || oids[i] != oids[unique_count - 1])
        {
            oids[unique_count++] = oids[i];
        }
    }
    *count_out = unique_count;
    return oids;
}

evoke_am_scheduler_work_hint_token *
evoke_am_scheduler_work_hint_snapshot(uint8 flag, size_t *count_out)
{
    evoke_am_scheduler_work_hint_token *tokens;
    size_t count = 0;
    uint32 i;

    if (count_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke work hint count output is required")));
    }
    *count_out = 0;
    if (!evoke_am_scheduler_available())
    {
        return NULL;
    }
    if (!evoke_am_scheduler_valid_work_class(flag))
    {
        ereport(ERROR, (errmsg("invalid evoke work hint class")));
    }

    tokens = palloc(sizeof(*tokens) * EVOKE_AM_WORK_HINT_CAPACITY);
    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        const evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[i];

        if (hint->database_oid != MyDatabaseId ||
            !OidIsValid(hint->index_oid) ||
            (hint->flags & flag) == 0)
        {
            continue;
        }
        tokens[count].index_oid = hint->index_oid;
        tokens[count].sequence = hint->sequence;
        tokens[count].maintenance_first_marked_at =
            hint->maintenance_first_marked_at;
        tokens[count].accelerator_retry_after =
            hint->accelerator_retry_after;
        count++;
    }
    LWLockRelease(evoke_scheduler_lock);

    if (count == 0)
    {
        pfree(tokens);
        return NULL;
    }
    if (count > 1)
    {
        qsort(
            tokens,
            count,
            sizeof(*tokens),
            evoke_am_scheduler_compare_work_hint_token
        );
    }
    *count_out = count;
    return tokens;
}

evoke_am_scheduler_database_hint_token *
evoke_am_scheduler_database_hint_snapshot(uint8 flags, size_t *count_out)
{
    evoke_am_scheduler_database_hint_token *tokens;
    const uint8 known_flags =
        EVOKE_AM_WORK_HINT_MAINTENANCE | EVOKE_AM_WORK_HINT_PRELOAD;
    size_t count = 0;
    uint32 i;

    if (count_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke database hint count output is required")));
    }
    *count_out = 0;
    if (!evoke_am_scheduler_available())
    {
        return NULL;
    }
    if ((flags & known_flags) == 0 || (flags & (uint8) ~known_flags) != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke database hint classes")));
    }

    tokens = palloc(sizeof(*tokens) * EVOKE_AM_WORK_HINT_CAPACITY);
    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        const evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[i];

        if (!OidIsValid(hint->database_oid) ||
            !OidIsValid(hint->index_oid) ||
            (hint->flags & flags) == 0)
        {
            continue;
        }
        tokens[count].database_oid = hint->database_oid;
        tokens[count].sequence = hint->sequence;
        count++;
    }
    LWLockRelease(evoke_scheduler_lock);

    if (count == 0)
    {
        pfree(tokens);
        return NULL;
    }
    if (count > 1)
    {
        qsort(
            tokens,
            count,
            sizeof(*tokens),
            evoke_am_scheduler_compare_database_hint_token
        );
    }
    *count_out = count;
    return tokens;
}

const evoke_am_scheduler_work_hint_token *
evoke_am_scheduler_work_hint_token_find(
    const evoke_am_scheduler_work_hint_token *tokens,
    size_t token_count,
    Oid index_oid
)
{
    evoke_am_scheduler_work_hint_token key;

    if (tokens == NULL || token_count == 0 || !OidIsValid(index_oid))
    {
        return NULL;
    }
    memset(&key, 0, sizeof(key));
    key.index_oid = index_oid;
    return bsearch(
        &key,
        tokens,
        token_count,
        sizeof(*tokens),
        evoke_am_scheduler_compare_work_hint_token
    );
}

void
evoke_am_scheduler_work_hint_clear_if_unchanged(
    Oid index_oid,
    uint8 flag,
    uint64 sequence
)
{
    uint64 key;
    uint32 start_slot;
    uint32 i;
    bool cleared_entry = false;

    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || sequence == 0 || flag == 0)
    {
        return;
    }

    key = ((uint64) MyDatabaseId << 32) ^ (uint64) index_oid;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    start_slot = (uint32) (key % EVOKE_AM_WORK_HINT_CAPACITY);

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        uint32 slot = (start_slot + i) % EVOKE_AM_WORK_HINT_CAPACITY;
        evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[slot];

        if (hint->flags == 0 && hint->sequence == 0)
        {
            break;
        }
        if (hint->database_oid != MyDatabaseId ||
            hint->index_oid != index_oid)
        {
            continue;
        }
        if (hint->sequence != sequence || (hint->flags & flag) == 0)
        {
            break;
        }
        hint->flags &= (uint8) ~flag;
        if (flag == EVOKE_AM_WORK_HINT_MAINTENANCE)
        {
            hint->maintenance_first_marked_at = 0;
            hint->accelerator_retry_after = 0;
        }
        if (hint->flags == 0)
        {
            hint->database_oid = InvalidOid;
            hint->index_oid = InvalidOid;
            hint->sequence = UINT64_MAX;
            cleared_entry = true;
        }
        evoke_scheduler->work_hints_consumed++;
        break;
    }
    if (cleared_entry)
    {
        bool active = false;

        for (i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
        {
            if (evoke_scheduler->work_hints[i].flags != 0)
            {
                active = true;
                break;
            }
        }
        if (!active)
        {
            memset(
                evoke_scheduler->work_hints,
                0,
                sizeof(evoke_scheduler->work_hints)
            );
        }
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_work_hint_reset_age(Oid index_oid, uint8 flag)
{
    uint64 key;
    uint32 start_slot;

    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || flag != EVOKE_AM_WORK_HINT_MAINTENANCE)
    {
        return;
    }

    key = ((uint64) MyDatabaseId << 32) ^ (uint64) index_oid;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    start_slot = (uint32) (key % EVOKE_AM_WORK_HINT_CAPACITY);

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    for (uint32 probe = 0;
         probe < EVOKE_AM_WORK_HINT_CAPACITY;
         probe++)
    {
        uint32 slot =
            (start_slot + probe) % EVOKE_AM_WORK_HINT_CAPACITY;
        evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[slot];

        if (hint->flags == 0 && hint->sequence == 0)
        {
            break;
        }
        if (hint->database_oid != MyDatabaseId ||
            hint->index_oid != index_oid)
        {
            continue;
        }
        if ((hint->flags & flag) != 0)
        {
            hint->maintenance_first_marked_at = GetCurrentTimestamp();
        }
        break;
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_work_hint_allow_accelerator_now(Oid index_oid)
{
    uint64 key;
    uint32 start_slot;

    if (!evoke_am_scheduler_available() || !OidIsValid(index_oid))
    {
        return;
    }

    key = ((uint64) MyDatabaseId << 32) ^ (uint64) index_oid;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    start_slot = (uint32) (key % EVOKE_AM_WORK_HINT_CAPACITY);

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    for (uint32 probe = 0;
         probe < EVOKE_AM_WORK_HINT_CAPACITY;
         probe++)
    {
        uint32 slot =
            (start_slot + probe) % EVOKE_AM_WORK_HINT_CAPACITY;
        evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[slot];

        if (hint->flags == 0 && hint->sequence == 0)
        {
            break;
        }
        if (hint->database_oid != MyDatabaseId ||
            hint->index_oid != index_oid)
        {
            continue;
        }
        hint->accelerator_retry_after = 0;
        break;
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_work_hint_defer_accelerator(
    Oid index_oid,
    int retry_interval_ms
)
{
    evoke_am_scheduler_work_hint *hint = NULL;
    TimestampTz now;
    TimestampTz retry_after;
    uint64 sequence;
    uint64 key;
    uint32 start_slot;
    uint32 tombstone_slot = EVOKE_AM_WORK_HINT_CAPACITY;
    bool request_reconcile = false;

    if (!evoke_am_scheduler_available() || !OidIsValid(index_oid))
    {
        return;
    }
    retry_interval_ms = Max(retry_interval_ms, 1);
    now = GetCurrentTimestamp();
    retry_after = PG_INT64_MAX - now <
        (int64) retry_interval_ms * INT64_C(1000)
        ? PG_INT64_MAX
        : now + (int64) retry_interval_ms * INT64_C(1000);

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    key = ((uint64) MyDatabaseId << 32) ^ (uint64) index_oid;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    start_slot = (uint32) (key % EVOKE_AM_WORK_HINT_CAPACITY);
    for (uint32 probe = 0;
         probe < EVOKE_AM_WORK_HINT_CAPACITY;
         probe++)
    {
        uint32 slot =
            (start_slot + probe) % EVOKE_AM_WORK_HINT_CAPACITY;
        evoke_am_scheduler_work_hint *candidate =
            &evoke_scheduler->work_hints[slot];

        if (candidate->flags != 0)
        {
            if (candidate->database_oid == MyDatabaseId &&
                candidate->index_oid == index_oid)
            {
                hint = candidate;
                break;
            }
            continue;
        }
        if (candidate->sequence == UINT64_MAX)
        {
            if (tombstone_slot == EVOKE_AM_WORK_HINT_CAPACITY)
            {
                tombstone_slot = slot;
            }
            continue;
        }
        hint = tombstone_slot == EVOKE_AM_WORK_HINT_CAPACITY
            ? candidate
            : &evoke_scheduler->work_hints[tombstone_slot];
        break;
    }
    if (hint == NULL && tombstone_slot != EVOKE_AM_WORK_HINT_CAPACITY)
    {
        hint = &evoke_scheduler->work_hints[tombstone_slot];
    }
    if (hint == NULL)
    {
        evoke_scheduler->work_hint_overflows++;
        request_reconcile = true;
    }
    else
    {
        if (hint->flags == 0)
        {
            sequence = evoke_scheduler->work_hint_sequence;
            sequence = sequence == UINT64_MAX ? 1 : sequence + 1;
            evoke_scheduler->work_hint_sequence = sequence;
            hint->database_oid = MyDatabaseId;
            hint->index_oid = index_oid;
            hint->flags = EVOKE_AM_WORK_HINT_MAINTENANCE;
            hint->sequence = sequence;
            hint->maintenance_first_marked_at = now;
        }
        else
        {
            hint->flags |= EVOKE_AM_WORK_HINT_MAINTENANCE;
            if (hint->maintenance_first_marked_at == 0)
            {
                hint->maintenance_first_marked_at = now;
            }
        }
        hint->accelerator_retry_after = retry_after;
    }
    LWLockRelease(evoke_scheduler_lock);
    if (request_reconcile)
    {
        evoke_am_scheduler_request_catalog_reconcile(
            EVOKE_AM_WORK_HINT_MAINTENANCE
        );
    }
}

static evoke_am_scheduler_reconcile_db *
evoke_am_scheduler_reconcile_db_locked(Oid database_oid, bool create)
{
    evoke_am_scheduler_reconcile_db *entry = NULL;
    evoke_am_scheduler_reconcile_db *free_entry = NULL;
    evoke_am_scheduler_reconcile_db *oldest = NULL;
    uint32 i;

    for (i = 0; i < EVOKE_AM_RECONCILE_DB_CAPACITY; i++)
    {
        evoke_am_scheduler_reconcile_db *candidate =
            &evoke_scheduler->reconcile_dbs[i];

        if (candidate->database_oid == database_oid)
        {
            entry = candidate;
            break;
        }
        if (!OidIsValid(candidate->database_oid))
        {
            if (free_entry == NULL)
            {
                free_entry = candidate;
            }
            continue;
        }
        if (candidate->running_flags == 0 &&
            (oldest == NULL ||
             candidate->last_access_counter < oldest->last_access_counter))
        {
            oldest = candidate;
        }
    }
    if (entry == NULL)
    {
        entry = free_entry != NULL ? free_entry : oldest;
    }
    if (entry != NULL && create && entry->database_oid != database_oid)
    {
        memset(entry, 0, sizeof(*entry));
        entry->database_oid = database_oid;
    }
    return entry;
}

static void
evoke_am_scheduler_reconcile_touch_locked(
    evoke_am_scheduler_reconcile_db *entry
)
{
    if (entry == NULL)
    {
        return;
    }
    if (evoke_scheduler->reconcile_access_clock == UINT64_MAX)
    {
        for (uint32 i = 0; i < EVOKE_AM_RECONCILE_DB_CAPACITY; i++)
        {
            evoke_scheduler->reconcile_dbs[i].last_access_counter = 0;
        }
        evoke_scheduler->reconcile_access_clock = 1;
    }
    else
    {
        evoke_scheduler->reconcile_access_clock++;
    }
    entry->last_access_counter = evoke_scheduler->reconcile_access_clock;
}

bool
evoke_am_scheduler_claim_catalog_reconcile(
    uint8 flag,
    int preload_timer_interval_ms
)
{
    evoke_am_scheduler_reconcile_db *entry;
    TimestampTz now;
    TimestampTz completed_at;
    int reconcile_interval_ms;

    if (!evoke_am_scheduler_available())
    {
        return true;
    }
    if (!evoke_am_scheduler_valid_work_class(flag))
    {
        ereport(ERROR, (errmsg("invalid evoke reconciliation class")));
    }

    now = GetCurrentTimestamp();
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, true);
    if (entry == NULL)
    {
        LWLockRelease(evoke_scheduler_lock);
        return true;
    }
    if ((entry->running_flags & flag) != 0)
    {
        LWLockRelease(evoke_scheduler_lock);
        return false;
    }
    completed_at = flag == EVOKE_AM_WORK_HINT_MAINTENANCE
        ? entry->maintenance_completed_at
        : entry->preload_completed_at;
    reconcile_interval_ms = EVOKE_AM_CATALOG_RECONCILE_INTERVAL_MS;
    if (flag == EVOKE_AM_WORK_HINT_PRELOAD && RecoveryInProgress())
    {
        reconcile_interval_ms = Max(
            EVOKE_AM_STANDBY_PRELOAD_RECONCILE_INTERVAL_MS,
            preload_timer_interval_ms
        );
    }
    if ((entry->requested_flags & flag) == 0 &&
        completed_at != 0 &&
        !TimestampDifferenceExceeds(
            completed_at,
            now,
            reconcile_interval_ms))
    {
        LWLockRelease(evoke_scheduler_lock);
        return false;
    }
    entry->requested_flags &= (uint8) ~flag;
    entry->running_flags |= flag;
    evoke_am_scheduler_reconcile_touch_locked(entry);
    LWLockRelease(evoke_scheduler_lock);
    return true;
}

void
evoke_am_scheduler_request_catalog_reconcile(uint8 flags)
{
    evoke_am_scheduler_reconcile_db *entry;

    if (!evoke_am_scheduler_available())
    {
        return;
    }
    if (flags == 0 ||
        (flags & (uint8) ~(
            EVOKE_AM_WORK_HINT_MAINTENANCE |
            EVOKE_AM_WORK_HINT_PRELOAD)) != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke reconciliation class")));
    }

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, true);
    if (entry != NULL)
    {
        entry->requested_flags |= flags;
        evoke_am_scheduler_reconcile_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_complete_catalog_reconcile(
    uint8 flag,
    uint64 rows,
    uint64 duration_us
)
{
    evoke_am_scheduler_reconcile_db *entry;

    if (!evoke_am_scheduler_available())
    {
        return;
    }
    if (!evoke_am_scheduler_valid_work_class(flag))
    {
        ereport(ERROR, (errmsg("invalid evoke reconciliation class")));
    }

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, false);
    if (entry != NULL)
    {
        entry->running_flags &= (uint8) ~flag;
        if (flag == EVOKE_AM_WORK_HINT_MAINTENANCE)
        {
            entry->maintenance_completed_at = GetCurrentTimestamp();
            entry->maintenance_count++;
            entry->maintenance_last_rows = rows;
            entry->maintenance_last_duration_us = duration_us;
            entry->maintenance_max_duration_us = Max(
                entry->maintenance_max_duration_us,
                duration_us
            );
        }
        else
        {
            entry->preload_completed_at = GetCurrentTimestamp();
            entry->preload_count++;
            entry->preload_last_rows = rows;
            entry->preload_last_duration_us = duration_us;
            entry->preload_max_duration_us = Max(
                entry->preload_max_duration_us,
                duration_us
            );
        }
        evoke_am_scheduler_reconcile_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_release_catalog_reconcile(uint8 flag)
{
    evoke_am_scheduler_reconcile_db *entry;

    if (!evoke_am_scheduler_available())
    {
        return;
    }
    if (!evoke_am_scheduler_valid_work_class(flag))
    {
        ereport(ERROR, (errmsg("invalid evoke reconciliation class")));
    }

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, false);
    if (entry != NULL)
    {
        entry->running_flags &= (uint8) ~flag;
        entry->requested_flags |= flag;
        evoke_am_scheduler_reconcile_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

Oid
evoke_am_scheduler_maintenance_fairness_cursor(void)
{
    evoke_am_scheduler_reconcile_db *entry;
    Oid cursor = InvalidOid;

    if (!evoke_am_scheduler_available())
    {
        return InvalidOid;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, false);
    if (entry != NULL)
    {
        cursor = entry->maintenance_last_served_index_oid;
    }
    LWLockRelease(evoke_scheduler_lock);
    return cursor;
}

void
evoke_am_scheduler_note_maintenance_service(Oid index_oid)
{
    evoke_am_scheduler_reconcile_db *entry;

    if (!evoke_am_scheduler_available() || !OidIsValid(index_oid))
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_reconcile_db_locked(MyDatabaseId, true);
    if (entry != NULL)
    {
        entry->maintenance_last_served_index_oid = index_oid;
        entry->maintenance_service_count =
            entry->maintenance_service_count == UINT64_MAX
                ? UINT64_MAX
                : entry->maintenance_service_count + 1u;
        evoke_am_scheduler_reconcile_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

static evoke_am_scheduler_telemetry_entry *
evoke_am_scheduler_semantic_entry_locked(Oid index_oid, bool create)
{
    evoke_am_scheduler_telemetry_entry *entry = NULL;
    evoke_am_scheduler_telemetry_entry *free_entry = NULL;
    evoke_am_scheduler_telemetry_entry *oldest = NULL;

    for (uint32 i = 0; i < EVOKE_AM_SEMANTIC_TELEMETRY_CAPACITY; i++)
    {
        evoke_am_scheduler_telemetry_entry *candidate =
            &evoke_scheduler->semantic_telemetry[i];

        if (candidate->database_oid == MyDatabaseId &&
            candidate->index_oid == index_oid)
        {
            entry = candidate;
            break;
        }
        if (!OidIsValid(candidate->database_oid) ||
            !OidIsValid(candidate->index_oid))
        {
            if (free_entry == NULL)
            {
                free_entry = candidate;
            }
            continue;
        }
        if (oldest == NULL ||
            candidate->access_sequence < oldest->access_sequence)
        {
            oldest = candidate;
        }
    }
    if (entry == NULL && create)
    {
        entry = free_entry != NULL ? free_entry : oldest;
        if (entry != NULL)
        {
            memset(entry, 0, sizeof(*entry));
            entry->database_oid = MyDatabaseId;
            entry->index_oid = index_oid;
        }
    }
    return entry;
}

static void
evoke_am_scheduler_semantic_touch_locked(
    evoke_am_scheduler_telemetry_entry *entry
)
{
    if (entry == NULL)
    {
        return;
    }
    if (evoke_scheduler->semantic_telemetry_access_clock == UINT64_MAX)
    {
        for (uint32 i = 0;
             i < EVOKE_AM_SEMANTIC_TELEMETRY_CAPACITY;
             i++)
        {
            evoke_scheduler->semantic_telemetry[i].access_sequence = 0;
        }
        evoke_scheduler->semantic_telemetry_access_clock = 1;
    }
    else
    {
        evoke_scheduler->semantic_telemetry_access_clock++;
    }
    entry->access_sequence =
        evoke_scheduler->semantic_telemetry_access_clock;
}

void
evoke_am_scheduler_semantic_note_result(
    Oid index_oid,
    const evoke_am_scheduler_semantic_result *result,
    uint64 duration_us
)
{
    evoke_am_scheduler_telemetry_entry *entry;
    TimestampTz now;

    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || result == NULL)
    {
        return;
    }
    now = GetCurrentTimestamp();
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_semantic_entry_locked(index_oid, true);
    if (entry != NULL)
    {
        entry->attempts++;
        entry->completed += result->completed;
        entry->retired += result->retired;
        if (result->pending > 0 &&
            result->completed == 0 && result->retired == 0)
        {
            entry->retries++;
        }
        entry->last_batch_completed = result->completed;
        entry->last_batch_retired = result->retired;
        entry->last_duration_us = duration_us;
        entry->frontier_scanned_records =
            UINT64_MAX - entry->frontier_scanned_records <
                result->frontier_scanned_records
                ? UINT64_MAX
                : entry->frontier_scanned_records +
                    result->frontier_scanned_records;
        entry->frontier_scanned_bytes =
            UINT64_MAX - entry->frontier_scanned_bytes <
                result->frontier_scanned_bytes
                ? UINT64_MAX
                : entry->frontier_scanned_bytes +
                    result->frontier_scanned_bytes;
        entry->frontier_xid_lock_windows =
            UINT64_MAX - entry->frontier_xid_lock_windows <
                result->frontier_xid_lock_windows
                ? UINT64_MAX
                : entry->frontier_xid_lock_windows +
                    result->frontier_xid_lock_windows;
        if (result->frontier_reconstructed)
        {
            entry->frontier_reconstructions =
                entry->frontier_reconstructions == UINT64_MAX
                    ? UINT64_MAX
                    : entry->frontier_reconstructions + 1u;
        }
        entry->last_frontier_scanned_records =
            result->frontier_scanned_records;
        entry->last_frontier_scanned_bytes =
            result->frontier_scanned_bytes;
        entry->last_frontier_xid_lock_windows =
            result->frontier_xid_lock_windows;
        entry->last_frontier_reconstructed =
            result->frontier_reconstructed;
        entry->last_attempt_at = now;
        if (result->completed > 0 || result->retired > 0)
        {
            entry->last_completed_at = now;
        }
        evoke_am_scheduler_semantic_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_semantic_note_error(
    Oid index_oid,
    const char *message,
    uint64 duration_us
)
{
    evoke_am_scheduler_telemetry_entry *entry;
    TimestampTz now;

    if (!evoke_am_scheduler_available() || !OidIsValid(index_oid))
    {
        return;
    }
    now = GetCurrentTimestamp();
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    entry = evoke_am_scheduler_semantic_entry_locked(index_oid, true);
    if (entry != NULL)
    {
        entry->attempts++;
        entry->retries++;
        entry->failures++;
        entry->last_batch_completed = 0;
        entry->last_batch_retired = 0;
        entry->last_duration_us = duration_us;
        entry->last_frontier_scanned_records = 0;
        entry->last_frontier_scanned_bytes = 0;
        entry->last_frontier_xid_lock_windows = 0;
        entry->last_frontier_reconstructed = false;
        entry->last_attempt_at = now;
        entry->last_error_at = now;
        strlcpy(
            entry->last_error,
            message != NULL ? message : "unknown semantic encoding error",
            sizeof(entry->last_error)
        );
        evoke_am_scheduler_semantic_touch_locked(entry);
    }
    LWLockRelease(evoke_scheduler_lock);
}

bool
evoke_am_scheduler_semantic_snapshot(
    Oid index_oid,
    evoke_am_scheduler_semantic_telemetry *snapshot_out
)
{
    evoke_am_scheduler_telemetry_entry *entry;
    bool found = false;

    if (snapshot_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke semantic telemetry output is required")));
    }
    memset(snapshot_out, 0, sizeof(*snapshot_out));
    if (!evoke_am_scheduler_available() || !OidIsValid(index_oid))
    {
        return false;
    }

    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    entry = evoke_am_scheduler_semantic_entry_locked(index_oid, false);
    if (entry != NULL)
    {
        snapshot_out->database_oid = entry->database_oid;
        snapshot_out->index_oid = entry->index_oid;
        snapshot_out->attempts = entry->attempts;
        snapshot_out->completed = entry->completed;
        snapshot_out->retired = entry->retired;
        snapshot_out->retries = entry->retries;
        snapshot_out->failures = entry->failures;
        snapshot_out->last_batch_completed = entry->last_batch_completed;
        snapshot_out->last_batch_retired = entry->last_batch_retired;
        snapshot_out->last_duration_us = entry->last_duration_us;
        snapshot_out->frontier_scanned_records =
            entry->frontier_scanned_records;
        snapshot_out->frontier_scanned_bytes =
            entry->frontier_scanned_bytes;
        snapshot_out->frontier_xid_lock_windows =
            entry->frontier_xid_lock_windows;
        snapshot_out->frontier_reconstructions =
            entry->frontier_reconstructions;
        snapshot_out->last_frontier_scanned_records =
            entry->last_frontier_scanned_records;
        snapshot_out->last_frontier_scanned_bytes =
            entry->last_frontier_scanned_bytes;
        snapshot_out->last_frontier_xid_lock_windows =
            entry->last_frontier_xid_lock_windows;
        snapshot_out->last_frontier_reconstructed =
            entry->last_frontier_reconstructed;
        snapshot_out->last_attempt_at = entry->last_attempt_at;
        snapshot_out->last_completed_at = entry->last_completed_at;
        snapshot_out->last_error_at = entry->last_error_at;
        strlcpy(
            snapshot_out->last_error,
            entry->last_error,
            sizeof(snapshot_out->last_error)
        );
        found = true;
    }
    LWLockRelease(evoke_scheduler_lock);
    return found;
}

static uint64
evoke_am_scheduler_posting_heat_now_ms(void)
{
    TimestampTz now = GetCurrentTimestamp();

    return now > 0 ? (uint64) now / UINT64_C(1000) : UINT64_C(1);
}

size_t
evoke_am_scheduler_posting_heat_merge(
    const evoke_posting_heat_observation *observations,
    size_t observation_count,
    Oid *scheduled_indexes,
    size_t scheduled_capacity
)
{
    size_t scheduled_count = 0;
    uint64 query_observations = 0;
    uint64 evictions = 0;

    if (observation_count > 0 && observations == NULL)
    {
        ereport(ERROR, (errmsg("evoke posting heat observations are required")));
    }
    if (scheduled_capacity > 0 && scheduled_indexes == NULL)
    {
        ereport(ERROR, (errmsg("evoke posting heat schedule output is required")));
    }
    if (!evoke_am_scheduler_available())
    {
        return 0;
    }

    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    for (size_t index = 0; index < observation_count; index++)
    {
        const evoke_posting_heat_observation *observation =
            &observations[index];
        evoke_posting_heat_merge_result merged;

        merged = evoke_posting_heat_merge(
            evoke_scheduler->posting_heat,
            EVOKE_AM_POSTING_HEAT_CAPACITY,
            observation,
            EVOKE_AM_POSTING_HEAT_HALF_LIFE_MS,
            EVOKE_AM_POSTING_HEAT_MIN_QUERY_HEAT,
            EVOKE_AM_POSTING_HEAT_MIN_ROOT_HEAT
        );
        if (!merged.valid)
        {
            continue;
        }
        query_observations = evoke_u64_saturating_add(
            query_observations,
            observation->query_count
        );
        if (merged.evicted)
        {
            evictions++;
        }
        if (merged.became_candidate &&
            scheduled_count < scheduled_capacity)
        {
            Oid candidate_index = (Oid) observation->index_id;
            bool already_scheduled = false;

            for (size_t scheduled_index = 0;
                 scheduled_index < scheduled_count;
                 scheduled_index++)
            {
                if (scheduled_indexes[scheduled_index] == candidate_index)
                {
                    already_scheduled = true;
                    break;
                }
            }
            if (!already_scheduled)
            {
                scheduled_indexes[scheduled_count++] = candidate_index;
            }
        }
    }
    evoke_scheduler->posting_heat_flushes = evoke_u64_saturating_add(
        evoke_scheduler->posting_heat_flushes,
        1
    );
    evoke_scheduler->posting_heat_query_observations =
        evoke_u64_saturating_add(
            evoke_scheduler->posting_heat_query_observations,
            query_observations
        );
    evoke_scheduler->posting_heat_evictions = evoke_u64_saturating_add(
        evoke_scheduler->posting_heat_evictions,
        evictions
    );
    LWLockRelease(evoke_scheduler_lock);
    return scheduled_count;
}

bool
evoke_am_scheduler_posting_heat_candidate(
    Oid index_oid,
    uint64 root_id,
    evoke_posting_heat_entry *candidate_out
)
{
    bool found;

    if (candidate_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke posting heat candidate is required")));
    }
    memset(candidate_out, 0, sizeof(*candidate_out));
    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || root_id == 0)
    {
        return false;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    found = evoke_posting_heat_select(
        evoke_scheduler->posting_heat,
        EVOKE_AM_POSTING_HEAT_CAPACITY,
        (uint32) MyDatabaseId,
        (uint32) index_oid,
        root_id,
        evoke_am_scheduler_posting_heat_now_ms(),
        EVOKE_AM_POSTING_HEAT_HALF_LIFE_MS,
        EVOKE_AM_POSTING_HEAT_MIN_QUERY_HEAT,
        EVOKE_AM_POSTING_HEAT_MIN_ROOT_HEAT,
        EVOKE_AM_POSTING_HEAT_RETRY_COOLDOWN_MS,
        candidate_out
    );
    LWLockRelease(evoke_scheduler_lock);
    return found;
}

void
evoke_am_scheduler_posting_heat_note_attempt(
    Oid index_oid,
    uint32 term_id,
    uint64 root_id
)
{
    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || root_id == 0)
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    (void) evoke_posting_heat_note_attempt(
        evoke_scheduler->posting_heat,
        EVOKE_AM_POSTING_HEAT_CAPACITY,
        (uint32) MyDatabaseId,
        (uint32) index_oid,
        term_id,
        root_id,
        evoke_am_scheduler_posting_heat_now_ms()
    );
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_posting_heat_note_hot_cache_resident(
    Oid index_oid,
    uint32 term_id,
    uint64 root_id
)
{
    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || root_id == 0)
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    (void) evoke_posting_heat_note_hot_cache_resident(
        evoke_scheduler->posting_heat,
        EVOKE_AM_POSTING_HEAT_CAPACITY,
        (uint32) MyDatabaseId,
        (uint32) index_oid,
        term_id,
        root_id
    );
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_posting_heat_summary(
    Oid index_oid,
    uint64 root_id,
    evoke_posting_heat_summary *summary_out,
    uint64 *flushes_out,
    uint64 *query_observations_out,
    uint64 *evictions_out
)
{
    if (summary_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke posting heat summary is required")));
    }
    memset(summary_out, 0, sizeof(*summary_out));
    if (flushes_out != NULL)
    {
        *flushes_out = 0;
    }
    if (query_observations_out != NULL)
    {
        *query_observations_out = 0;
    }
    if (evictions_out != NULL)
    {
        *evictions_out = 0;
    }
    if (!evoke_am_scheduler_available() ||
        !OidIsValid(index_oid) || root_id == 0)
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    evoke_posting_heat_summarize(
        evoke_scheduler->posting_heat,
        EVOKE_AM_POSTING_HEAT_CAPACITY,
        (uint32) MyDatabaseId,
        (uint32) index_oid,
        root_id,
        evoke_am_scheduler_posting_heat_now_ms(),
        EVOKE_AM_POSTING_HEAT_HALF_LIFE_MS,
        EVOKE_AM_POSTING_HEAT_MIN_QUERY_HEAT,
        EVOKE_AM_POSTING_HEAT_MIN_ROOT_HEAT,
        EVOKE_AM_POSTING_HEAT_RETRY_COOLDOWN_MS,
        summary_out
    );
    if (flushes_out != NULL)
    {
        *flushes_out = evoke_scheduler->posting_heat_flushes;
    }
    if (query_observations_out != NULL)
    {
        *query_observations_out =
            evoke_scheduler->posting_heat_query_observations;
    }
    if (evictions_out != NULL)
    {
        *evictions_out = evoke_scheduler->posting_heat_evictions;
    }
    LWLockRelease(evoke_scheduler_lock);
}

static uint32 *
evoke_am_scheduler_phase_counter(evoke_am_scheduler_worker_phase phase)
{
    switch (phase)
    {
        case EVOKE_AM_SCHEDULER_WORKER_PRELOAD:
            return &evoke_scheduler->active_preload_workers;
        case EVOKE_AM_SCHEDULER_WORKER_MAINTENANCE:
            return &evoke_scheduler->active_index_maintenance_workers;
        case EVOKE_AM_SCHEDULER_WORKER_IDLE:
        default:
            return NULL;
    }
}

void
evoke_am_scheduler_change_worker_phase(
    evoke_am_scheduler_worker_phase old_phase,
    evoke_am_scheduler_worker_phase new_phase
)
{
    uint32 *old_counter;
    uint32 *new_counter;

    if (!evoke_am_scheduler_available() || old_phase == new_phase)
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    old_counter = evoke_am_scheduler_phase_counter(old_phase);
    new_counter = evoke_am_scheduler_phase_counter(new_phase);
    if (old_counter != NULL && *old_counter > 0)
    {
        (*old_counter)--;
    }
    if (new_counter != NULL && *new_counter < UINT32_MAX)
    {
        (*new_counter)++;
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_note_worker_started(void)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    if (evoke_scheduler->active_background_workers < UINT32_MAX)
    {
        evoke_scheduler->active_background_workers++;
    }
    LWLockRelease(evoke_scheduler_lock);
}

void
evoke_am_scheduler_note_worker_finished(void)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    if (evoke_scheduler->active_background_workers > 0)
    {
        evoke_scheduler->active_background_workers--;
    }
    LWLockRelease(evoke_scheduler_lock);
}

uint32
evoke_am_scheduler_active_workers(void)
{
    uint32 active = 0;

    if (!evoke_am_scheduler_available())
    {
        return 0;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    active = evoke_scheduler->active_background_workers;
    LWLockRelease(evoke_scheduler_lock);
    return active;
}

static void
evoke_am_scheduler_prune_worker_launches_locked(TimestampTz now)
{
    if (evoke_scheduler->pending_maintenance_worker_launches == 0)
    {
        evoke_scheduler->last_maintenance_worker_launch = 0;
        return;
    }
    if (evoke_scheduler->last_maintenance_worker_launch == 0 ||
        TimestampDifferenceExceeds(
            evoke_scheduler->last_maintenance_worker_launch,
            now,
            EVOKE_AM_WORKER_LAUNCH_RESERVATION_TIMEOUT_MS))
    {
        evoke_scheduler->pending_maintenance_worker_launches = 0;
        evoke_scheduler->last_maintenance_worker_launch = 0;
    }
}

bool
evoke_am_scheduler_try_reserve_worker_launch(int limit)
{
    TimestampTz now;
    uint32 active;
    uint32 pending;

    if (!evoke_am_scheduler_available())
    {
        return true;
    }
    now = GetCurrentTimestamp();
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    evoke_am_scheduler_prune_worker_launches_locked(now);
    active = evoke_scheduler->active_background_workers;
    pending = evoke_scheduler->pending_maintenance_worker_launches;
    if ((uint64) active + (uint64) pending >= (uint64) Max(limit, 1))
    {
        LWLockRelease(evoke_scheduler_lock);
        return false;
    }
    if (evoke_scheduler->pending_maintenance_worker_launches >= UINT32_MAX)
    {
        LWLockRelease(evoke_scheduler_lock);
        return false;
    }
    evoke_scheduler->pending_maintenance_worker_launches++;
    evoke_scheduler->last_maintenance_worker_launch = now;
    LWLockRelease(evoke_scheduler_lock);
    return true;
}

void
evoke_am_scheduler_release_worker_launch(void)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    if (evoke_scheduler->pending_maintenance_worker_launches > 0)
    {
        evoke_scheduler->pending_maintenance_worker_launches--;
    }
    if (evoke_scheduler->pending_maintenance_worker_launches == 0)
    {
        evoke_scheduler->last_maintenance_worker_launch = 0;
    }
    LWLockRelease(evoke_scheduler_lock);
}

Oid
evoke_am_scheduler_last_maintenance_db_oid(void)
{
    Oid database_oid = InvalidOid;

    if (!evoke_am_scheduler_available())
    {
        return InvalidOid;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    database_oid = evoke_scheduler->last_maintenance_db_oid;
    LWLockRelease(evoke_scheduler_lock);
    return database_oid;
}

void
evoke_am_scheduler_set_last_maintenance_db_oid(Oid database_oid)
{
    if (!evoke_am_scheduler_available())
    {
        return;
    }
    LWLockAcquire(evoke_scheduler_lock, LW_EXCLUSIVE);
    evoke_scheduler->last_maintenance_db_oid = database_oid;
    LWLockRelease(evoke_scheduler_lock);
}

static void
evoke_am_scheduler_process_report_phase(
    evoke_am_scheduler_worker_phase phase
)
{
    const char *appname;
    const char *activity;
    BackendState state;

    switch (phase)
    {
        case EVOKE_AM_SCHEDULER_WORKER_PRELOAD:
            appname = "evoke preload";
            activity = "auto-preload";
            state = STATE_RUNNING;
            break;
        case EVOKE_AM_SCHEDULER_WORKER_MAINTENANCE:
            appname = "evoke maintenance";
            activity = "index maintenance";
            state = STATE_RUNNING;
            break;
        case EVOKE_AM_SCHEDULER_WORKER_IDLE:
        default:
            appname = "evoke background";
            activity = "idle";
            state = STATE_IDLE;
            break;
    }

    pgstat_report_appname(appname);
    pgstat_report_activity(state, activity);
}

void
evoke_am_scheduler_process_set_phase(
    evoke_am_scheduler_worker_phase phase
)
{
    if (evoke_scheduler_process_phase != phase)
    {
        evoke_am_scheduler_change_worker_phase(
            evoke_scheduler_process_phase,
            phase
        );
        evoke_scheduler_process_phase = phase;
    }
    evoke_am_scheduler_process_report_phase(phase);
}

void
evoke_am_scheduler_process_note_started(void)
{
    if (!evoke_am_scheduler_available() ||
        evoke_scheduler_process_counted_active)
    {
        return;
    }
    evoke_am_scheduler_note_worker_started();
    evoke_scheduler_process_counted_active = true;
}

void
evoke_am_scheduler_process_note_finished(void)
{
    evoke_am_scheduler_process_set_phase(
        EVOKE_AM_SCHEDULER_WORKER_IDLE
    );
    if (!evoke_scheduler_process_counted_active)
    {
        return;
    }
    evoke_am_scheduler_note_worker_finished();
    evoke_scheduler_process_counted_active = false;
}

static void
evoke_am_scheduler_process_worker_exit(int code, Datum arg)
{
    (void) code;
    (void) arg;
    evoke_am_scheduler_process_note_finished();
}

void
evoke_am_scheduler_process_initialize(void)
{
    evoke_am_scheduler_process_set_phase(
        EVOKE_AM_SCHEDULER_WORKER_IDLE
    );
    if (evoke_scheduler_process_worker_exit_registered)
    {
        return;
    }
    before_shmem_exit(evoke_am_scheduler_process_worker_exit, 0);
    evoke_scheduler_process_worker_exit_registered = true;
}

void
evoke_am_scheduler_process_release_worker_launch(void)
{
    if (!evoke_scheduler_process_launch_reserved)
    {
        return;
    }
    evoke_am_scheduler_release_worker_launch();
    evoke_scheduler_process_launch_reserved = false;
}

static void
evoke_am_scheduler_process_launch_exit(int code, Datum arg)
{
    (void) code;
    (void) arg;
    evoke_am_scheduler_process_release_worker_launch();
}

void
evoke_am_scheduler_process_adopt_worker_launch(void)
{
    evoke_scheduler_process_launch_reserved = true;
    /*
     * Count the process before it opens the target database.  Database
     * startup can block in the kernel when storage is unhealthy; retaining
     * only a timeout-based pending reservation would then admit another
     * worker every timeout interval and eventually exhaust postmaster slots.
     */
    evoke_am_scheduler_process_note_started();
    if (!evoke_scheduler_process_worker_exit_registered)
    {
        before_shmem_exit(evoke_am_scheduler_process_worker_exit, 0);
        evoke_scheduler_process_worker_exit_registered = true;
    }
    if (evoke_scheduler_process_launch_exit_registered)
    {
        return;
    }
    before_shmem_exit(evoke_am_scheduler_process_launch_exit, 0);
    evoke_scheduler_process_launch_exit_registered = true;
}

void
evoke_am_scheduler_status_snapshot(
    Oid database_oid,
    evoke_am_scheduler_status *status_out
)
{
    evoke_am_scheduler_reconcile_db *entry;

    if (status_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke scheduler status output is required")));
    }
    memset(status_out, 0, sizeof(*status_out));
    if (!evoke_am_scheduler_available())
    {
        return;
    }

    LWLockAcquire(evoke_scheduler_lock, LW_SHARED);
    status_out->available = true;
    status_out->active_background_workers =
        evoke_scheduler->active_background_workers;
    status_out->active_preload_workers =
        evoke_scheduler->active_preload_workers;
    status_out->active_index_maintenance_workers =
        evoke_scheduler->active_index_maintenance_workers;
    status_out->pending_maintenance_worker_launches =
        evoke_scheduler->pending_maintenance_worker_launches;
    status_out->work_hint_sequence = evoke_scheduler->work_hint_sequence;
    status_out->work_hint_overflows = evoke_scheduler->work_hint_overflows;
    status_out->work_hints_consumed = evoke_scheduler->work_hints_consumed;
    status_out->last_maintenance_cycle =
        evoke_scheduler->last_maintenance_cycle;
    for (uint32 i = 0; i < EVOKE_AM_WORK_HINT_CAPACITY; i++)
    {
        const evoke_am_scheduler_work_hint *hint =
            &evoke_scheduler->work_hints[i];

        if (hint->database_oid != database_oid)
        {
            continue;
        }
        if ((hint->flags & EVOKE_AM_WORK_HINT_MAINTENANCE) != 0)
        {
            status_out->pending_maintenance_hint_entries++;
        }
        if ((hint->flags & EVOKE_AM_WORK_HINT_PRELOAD) != 0)
        {
            status_out->pending_preload_hint_entries++;
        }
    }
    entry = evoke_am_scheduler_reconcile_db_locked(database_oid, false);
    if (entry != NULL && entry->database_oid != database_oid)
    {
        entry = NULL;
    }
    if (entry != NULL)
    {
        status_out->maintenance_reconcile_requested =
            (entry->requested_flags & EVOKE_AM_WORK_HINT_MAINTENANCE) != 0;
        status_out->preload_reconcile_requested =
            (entry->requested_flags & EVOKE_AM_WORK_HINT_PRELOAD) != 0;
        status_out->maintenance_reconcile_running =
            (entry->running_flags & EVOKE_AM_WORK_HINT_MAINTENANCE) != 0;
        status_out->preload_reconcile_running =
            (entry->running_flags & EVOKE_AM_WORK_HINT_PRELOAD) != 0;
        status_out->maintenance_reconcile_completed_at =
            entry->maintenance_completed_at;
        status_out->preload_reconcile_completed_at =
            entry->preload_completed_at;
        status_out->maintenance_reconcile_count = entry->maintenance_count;
        status_out->maintenance_reconcile_last_rows =
            entry->maintenance_last_rows;
        status_out->maintenance_reconcile_last_duration_us =
            entry->maintenance_last_duration_us;
        status_out->maintenance_reconcile_max_duration_us =
            entry->maintenance_max_duration_us;
        status_out->maintenance_last_served_index_oid =
            entry->maintenance_last_served_index_oid;
        status_out->maintenance_service_count =
            entry->maintenance_service_count;
        status_out->preload_reconcile_count = entry->preload_count;
        status_out->preload_reconcile_last_rows = entry->preload_last_rows;
        status_out->preload_reconcile_last_duration_us =
            entry->preload_last_duration_us;
        status_out->preload_reconcile_max_duration_us =
            entry->preload_max_duration_us;
    }
    LWLockRelease(evoke_scheduler_lock);
}
