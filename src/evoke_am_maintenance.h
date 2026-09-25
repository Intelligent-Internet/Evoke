#ifndef EVOKE_AM_MAINTENANCE_H
#define EVOKE_AM_MAINTENANCE_H

#include "postgres.h"

#include "access/transam.h"
#include "storage/lock.h"
#include "utils/rel.h"

#include "evoke_core.h"

bool evoke_am_try_maintenance_lock(Oid index_oid);
void evoke_am_lock_maintenance_xact(Oid index_oid);
bool evoke_am_try_accelerator_build_lock(Oid index_oid);
bool evoke_am_accelerator_build_in_progress(Oid index_oid);
void evoke_am_accelerator_build_unlock(Oid index_oid);
bool evoke_am_try_resident_build_lock(void);
void evoke_am_resident_build_unlock(void);
bool evoke_am_try_session_maintenance_lock(Oid index_oid);
bool evoke_am_maintenance_lock_held(Oid index_oid);
bool evoke_am_session_maintenance_lock_held(Oid index_oid);
void evoke_am_maintenance_unlock(Oid index_oid);
void evoke_am_session_maintenance_unlock(Oid index_oid);

void evoke_am_lock_append(Relation index_relation);
void evoke_am_unlock_append(Relation index_relation);

LockAcquireResult evoke_am_lock_writer_barrier_oid(
    Oid index_oid,
    LOCKMODE lock_mode,
    bool dont_wait
);
void evoke_am_unlock_writer_barrier_oid(
    Oid index_oid,
    LOCKMODE lock_mode
);

void evoke_am_pin_maintenance_xact(Relation index_relation);
LOCKMODE evoke_am_pinned_maintenance_mode(Oid index_oid);
bool evoke_am_pinned_maintenances_present(void);
void evoke_am_reparent_pinned_maintenances(
    SubTransactionId my_subid,
    SubTransactionId parent_subid,
    bool aborting
);
void evoke_am_clear_pinned_maintenances(void);

bool evoke_am_maintenance_tracking_enabled(Relation index_relation);
bool evoke_am_eventual_policy_enabled(Relation index_relation);
bool evoke_am_automatic_policy_enabled(Relation index_relation);
bool evoke_am_foreground_maintenance_enabled(Relation index_relation);
void evoke_am_maintenance_codec_error(
    const char *operation,
    evoke_status status
);

#endif
