#include "postgres.h"

#include "access/xact.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "storage/lock.h"
#include "utils/memutils.h"

#include "evoke_am_maintenance.h"
#include "evoke_am_options.h"

#define EVOKE_AM_APPEND_LOCK_TAG UINT32_C(0x32534241)
#define EVOKE_AM_ACCELERATOR_BUILD_LOCK_TAG UINT32_C(0x32534247)
#define EVOKE_AM_MAINTENANCE_LOCK_TAG UINT32_C(0x3253424D)
#define EVOKE_AM_RESIDENT_BUILD_LOCK_TAG UINT32_C(0x32534252)
#define EVOKE_AM_WRITER_BARRIER_LOCK_TAG UINT32_C(0x32534242)

typedef struct evoke_am_scoped_index
{
    Oid index_oid;
    SubTransactionId subxid;
    LOCKMODE lock_mode;
} evoke_am_scoped_index;

static List *evoke_am_pinned_maintenances = NIL;

static bool
evoke_am_has_pinned_maintenance(Oid index_oid)
{
    ListCell *cell;

    foreach (cell, evoke_am_pinned_maintenances)
    {
        const evoke_am_scoped_index *entry = lfirst(cell);

        if (entry != NULL && entry->index_oid == index_oid)
        {
            return true;
        }
    }
    return false;
}

bool
evoke_am_try_maintenance_lock(Oid index_oid)
{
    LOCKTAG tag;
    LockAcquireResult result;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    /*
     * A maintenance root can become buffer-visible before its WAL reaches the
     * durable commit boundary. Keep ownership until transaction end so another
     * maintenance action cannot use that root as an ancestor prematurely.
     */
    result = LockAcquire(&tag, ExclusiveLock, false, true);
    return result != LOCKACQUIRE_NOT_AVAIL;
}

void
evoke_am_lock_maintenance_xact(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    (void) LockAcquire(&tag, ExclusiveLock, false, false);
}

bool
evoke_am_try_accelerator_build_lock(Oid index_oid)
{
    LOCKTAG tag;
    LockAcquireResult result;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_ACCELERATOR_BUILD_LOCK_TAG,
        index_oid,
        0
    );
    result = LockAcquire(&tag, ExclusiveLock, false, true);
    return result != LOCKACQUIRE_NOT_AVAIL;
}

bool
evoke_am_accelerator_build_in_progress(Oid index_oid)
{
    LOCKTAG tag;
    LockAcquireResult result;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_ACCELERATOR_BUILD_LOCK_TAG,
        index_oid,
        0
    );
    result = LockAcquire(&tag, ShareLock, false, true);
    if (result == LOCKACQUIRE_NOT_AVAIL)
    {
        return true;
    }
    if (!LockRelease(&tag, ShareLock, false))
    {
        ereport(
            ERROR,
            (errmsg("evoke accelerator build probe lock is not held"))
        );
    }
    return false;
}

void
evoke_am_accelerator_build_unlock(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_ACCELERATOR_BUILD_LOCK_TAG,
        index_oid,
        0
    );
    if (!LockRelease(&tag, ExclusiveLock, false))
    {
        ereport(ERROR, (errmsg("evoke accelerator build lock is not held")));
    }
}

bool
evoke_am_try_resident_build_lock(void)
{
    LOCKTAG tag;
    LockAcquireResult result;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_RESIDENT_BUILD_LOCK_TAG,
        0,
        0
    );
    result = LockAcquire(&tag, ExclusiveLock, false, true);
    return result != LOCKACQUIRE_NOT_AVAIL;
}

void
evoke_am_resident_build_unlock(void)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_RESIDENT_BUILD_LOCK_TAG,
        0,
        0
    );
    if (!LockRelease(&tag, ExclusiveLock, false))
    {
        ereport(ERROR, (errmsg("evoke resident build lock is not held")));
    }
}

bool
evoke_am_try_session_maintenance_lock(Oid index_oid)
{
    LOCKTAG tag;
    LockAcquireResult result;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    result = LockAcquire(&tag, ExclusiveLock, true, true);
    return result != LOCKACQUIRE_NOT_AVAIL;
}

bool
evoke_am_maintenance_lock_held(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    return LockHeldByMe(&tag, ExclusiveLock, false);
}

bool
evoke_am_session_maintenance_lock_held(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    return LockHeldByMe(&tag, ExclusiveLock, true);
}

void
evoke_am_maintenance_unlock(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    LockRelease(&tag, ExclusiveLock, false);
}

void
evoke_am_session_maintenance_unlock(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_MAINTENANCE_LOCK_TAG,
        index_oid,
        0
    );
    LockRelease(&tag, ExclusiveLock, true);
}

void
evoke_am_lock_append(Relation index_relation)
{
    LOCKTAG tag;

    /*
     * Linked-L0 pages are append-only until a generation publication retires
     * them. Serialize physical appends and short metapage counter updates
     * without blocking readers of an immutable, already-snapshotted prefix.
     */
    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_APPEND_LOCK_TAG,
        RelationGetRelid(index_relation),
        0
    );
    (void) LockAcquire(&tag, ExclusiveLock, false, false);
}

void
evoke_am_unlock_append(Relation index_relation)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_APPEND_LOCK_TAG,
        RelationGetRelid(index_relation),
        0
    );
    if (!LockRelease(&tag, ExclusiveLock, false))
    {
        ereport(ERROR, (errmsg("evoke append lock is not held")));
    }
}

LockAcquireResult
evoke_am_lock_writer_barrier_oid(
    Oid index_oid,
    LOCKMODE lock_mode,
    bool dont_wait
)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_WRITER_BARRIER_LOCK_TAG,
        index_oid,
        0
    );
    return LockAcquire(&tag, lock_mode, false, dont_wait);
}

void
evoke_am_unlock_writer_barrier_oid(
    Oid index_oid,
    LOCKMODE lock_mode
)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_WRITER_BARRIER_LOCK_TAG,
        index_oid,
        0
    );
    if (!LockRelease(&tag, lock_mode, false))
    {
        ereport(ERROR, (errmsg("evoke writer barrier is not held")));
    }
}

LOCKMODE
evoke_am_pinned_maintenance_mode(Oid index_oid)
{
    ListCell *cell;

    foreach (cell, evoke_am_pinned_maintenances)
    {
        const evoke_am_scoped_index *entry = lfirst(cell);

        if (entry != NULL && entry->index_oid == index_oid)
        {
            return entry->lock_mode;
        }
    }
    return NoLock;
}

void
evoke_am_pin_maintenance_xact(Relation index_relation)
{
    MemoryContext oldcontext;
    evoke_am_scoped_index *entry;
    LOCKMODE lock_mode;
    Oid index_oid;

    if (!IsTransactionState())
    {
        return;
    }

    index_oid = RelationGetRelid(index_relation);
    if (evoke_am_has_pinned_maintenance(index_oid))
    {
        return;
    }

    /*
     * BM25-only realtime queries rebuild exact corpus statistics from a stable
     * committed sealed-root-plus-linked-L0 snapshot, so one writer transaction
     * owns the barrier until commit, abort, or prepared-transaction resolution.
     * SAE writes are exact through heap-MVCC-filtered linked L0 and therefore
     * share the transaction barrier, serializing only their physical append.
     */
    lock_mode =
        evoke_am_foreground_maintenance_enabled(index_relation) &&
        !evoke_am_sae_enabled(index_relation)
            ? ExclusiveLock
            : ShareLock;
    (void) evoke_am_lock_writer_barrier_oid(
        index_oid,
        lock_mode,
        false
    );

    oldcontext = MemoryContextSwitchTo(TopTransactionContext);
    entry = palloc0(sizeof(*entry));
    entry->index_oid = index_oid;
    entry->subxid = GetCurrentSubTransactionId();
    entry->lock_mode = lock_mode;
    evoke_am_pinned_maintenances = lappend(
        evoke_am_pinned_maintenances,
        entry
    );
    MemoryContextSwitchTo(oldcontext);
}

bool
evoke_am_pinned_maintenances_present(void)
{
    return evoke_am_pinned_maintenances != NIL;
}

void
evoke_am_reparent_pinned_maintenances(
    SubTransactionId my_subid,
    SubTransactionId parent_subid,
    bool aborting
)
{
    ListCell *cell;

    foreach (cell, evoke_am_pinned_maintenances)
    {
        evoke_am_scoped_index *entry = lfirst(cell);

        if (entry == NULL || entry->subxid != my_subid)
        {
            continue;
        }
        if (aborting)
        {
            entry->index_oid = InvalidOid;
        }
        else
        {
            entry->subxid = parent_subid;
        }
    }
}

void
evoke_am_clear_pinned_maintenances(void)
{
    evoke_am_pinned_maintenances = NIL;
}

bool
evoke_am_maintenance_tracking_enabled(Relation index_relation)
{
    return evoke_am_get_consistency(index_relation) !=
        EVOKE_AM_CONSISTENCY_MANUAL;
}

static bool
evoke_am_eventual_consistency_enabled(Relation index_relation)
{
    return evoke_am_get_consistency(index_relation) ==
        EVOKE_AM_CONSISTENCY_EVENTUAL;
}

bool
evoke_am_eventual_policy_enabled(Relation index_relation)
{
    evoke_am_validate_relation_policy(index_relation);
    return evoke_am_eventual_consistency_enabled(index_relation);
}

bool
evoke_am_automatic_policy_enabled(Relation index_relation)
{
    return evoke_am_get_consistency(index_relation) !=
        EVOKE_AM_CONSISTENCY_MANUAL;
}

bool
evoke_am_foreground_maintenance_enabled(Relation index_relation)
{
    return evoke_am_get_consistency(index_relation) ==
        EVOKE_AM_CONSISTENCY_REALTIME;
}

void
evoke_am_maintenance_codec_error(
    const char *operation,
    evoke_status status
)
{
    ereport(
        ERROR,
        (
            errmsg(
                "failed to %s during evoke segment maintenance",
                operation
            ),
            errdetail(
                "Validation failed: %s.",
                evoke_strerror(status)
            )
        )
    );
}
