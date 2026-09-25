#ifndef EVOKE_AM_SCHEDULER_H
#define EVOKE_AM_SCHEDULER_H

#include "postgres.h"

#include "storage/lwlock.h"

#include "evoke_posting_heat.h"

#define EVOKE_AM_WORK_HINT_MAINTENANCE UINT8_C(0x01)
#define EVOKE_AM_WORK_HINT_PRELOAD UINT8_C(0x02)
#define EVOKE_AM_SEMANTIC_ERROR_LEN 160

typedef enum evoke_am_scheduler_worker_phase
{
    EVOKE_AM_SCHEDULER_WORKER_IDLE = 0,
    EVOKE_AM_SCHEDULER_WORKER_PRELOAD,
    EVOKE_AM_SCHEDULER_WORKER_MAINTENANCE
} evoke_am_scheduler_worker_phase;

typedef struct evoke_am_scheduler_work_hint_token
{
    Oid index_oid;
    uint64 sequence;
    TimestampTz maintenance_first_marked_at;
    TimestampTz accelerator_retry_after;
} evoke_am_scheduler_work_hint_token;

typedef struct evoke_am_scheduler_database_hint_token
{
    Oid database_oid;
    uint64 sequence;
} evoke_am_scheduler_database_hint_token;

typedef struct evoke_am_scheduler_semantic_result
{
    uint32 pending;
    uint32 completed;
    uint32 retired;
    uint32 frontier_scanned_records;
    uint64 frontier_scanned_bytes;
    uint32 frontier_xid_lock_windows;
    bool frontier_reconstructed;
} evoke_am_scheduler_semantic_result;

typedef struct evoke_am_scheduler_semantic_telemetry
{
    Oid database_oid;
    Oid index_oid;
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
} evoke_am_scheduler_semantic_telemetry;

typedef struct evoke_am_scheduler_status
{
    bool available;
    uint32 active_background_workers;
    uint32 active_preload_workers;
    uint32 active_index_maintenance_workers;
    uint32 pending_maintenance_worker_launches;
    uint64 work_hint_sequence;
    uint64 work_hint_overflows;
    uint64 work_hints_consumed;
    uint32 pending_maintenance_hint_entries;
    uint32 pending_preload_hint_entries;
    TimestampTz last_maintenance_cycle;
    bool maintenance_reconcile_requested;
    bool preload_reconcile_requested;
    bool maintenance_reconcile_running;
    bool preload_reconcile_running;
    TimestampTz maintenance_reconcile_completed_at;
    TimestampTz preload_reconcile_completed_at;
    uint64 maintenance_reconcile_count;
    uint64 maintenance_reconcile_last_rows;
    uint64 maintenance_reconcile_last_duration_us;
    uint64 maintenance_reconcile_max_duration_us;
    Oid maintenance_last_served_index_oid;
    uint64 maintenance_service_count;
    uint64 preload_reconcile_count;
    uint64 preload_reconcile_last_rows;
    uint64 preload_reconcile_last_duration_us;
    uint64 preload_reconcile_max_duration_us;
} evoke_am_scheduler_status;

Size evoke_am_scheduler_shmem_size(void);
void evoke_am_scheduler_shmem_startup(
    LWLock *shared_lock,
    uint32 maintenance_worker_limit
);
bool evoke_am_scheduler_available(void);

uint32 evoke_am_scheduler_worker_limit(void);
void evoke_am_scheduler_set_worker_limit(uint32 limit);
void evoke_am_scheduler_note_maintenance_cycle(void);

void evoke_am_scheduler_work_hint_mark(Oid index_oid, uint8 flags);
Oid *evoke_am_scheduler_work_hint_consume(uint8 flag, size_t *count_out);
evoke_am_scheduler_work_hint_token *evoke_am_scheduler_work_hint_snapshot(
    uint8 flag,
    size_t *count_out
);
evoke_am_scheduler_database_hint_token *
evoke_am_scheduler_database_hint_snapshot(
    uint8 flags,
    size_t *count_out
);
const evoke_am_scheduler_work_hint_token *
evoke_am_scheduler_work_hint_token_find(
    const evoke_am_scheduler_work_hint_token *tokens,
    size_t token_count,
    Oid index_oid
);
void evoke_am_scheduler_work_hint_clear_if_unchanged(
    Oid index_oid,
    uint8 flag,
    uint64 sequence
);
void evoke_am_scheduler_work_hint_reset_age(Oid index_oid, uint8 flag);
void evoke_am_scheduler_work_hint_allow_accelerator_now(Oid index_oid);
void evoke_am_scheduler_work_hint_defer_accelerator(
    Oid index_oid,
    int retry_interval_ms
);

bool evoke_am_scheduler_claim_catalog_reconcile(
    uint8 flag,
    int preload_timer_interval_ms
);
void evoke_am_scheduler_request_catalog_reconcile(uint8 flags);
void evoke_am_scheduler_complete_catalog_reconcile(
    uint8 flag,
    uint64 rows,
    uint64 duration_us
);
void evoke_am_scheduler_release_catalog_reconcile(uint8 flag);
Oid evoke_am_scheduler_maintenance_fairness_cursor(void);
void evoke_am_scheduler_note_maintenance_service(Oid index_oid);

void evoke_am_scheduler_semantic_note_result(
    Oid index_oid,
    const evoke_am_scheduler_semantic_result *result,
    uint64 duration_us
);
void evoke_am_scheduler_semantic_note_error(
    Oid index_oid,
    const char *message,
    uint64 duration_us
);
bool evoke_am_scheduler_semantic_snapshot(
    Oid index_oid,
    evoke_am_scheduler_semantic_telemetry *snapshot_out
);

size_t evoke_am_scheduler_posting_heat_merge(
    const evoke_posting_heat_observation *observations,
    size_t observation_count,
    Oid *scheduled_indexes,
    size_t scheduled_capacity
);
bool evoke_am_scheduler_posting_heat_candidate(
    Oid index_oid,
    uint64 root_id,
    evoke_posting_heat_entry *candidate_out
);
void evoke_am_scheduler_posting_heat_note_attempt(
    Oid index_oid,
    uint32 term_id,
    uint64 root_id
);
void evoke_am_scheduler_posting_heat_note_hot_cache_resident(
    Oid index_oid,
    uint32 term_id,
    uint64 root_id
);
void evoke_am_scheduler_posting_heat_summary(
    Oid index_oid,
    uint64 root_id,
    evoke_posting_heat_summary *summary_out,
    uint64 *flushes_out,
    uint64 *query_observations_out,
    uint64 *evictions_out
);

void evoke_am_scheduler_change_worker_phase(
    evoke_am_scheduler_worker_phase old_phase,
    evoke_am_scheduler_worker_phase new_phase
);
void evoke_am_scheduler_note_worker_started(void);
void evoke_am_scheduler_note_worker_finished(void);
uint32 evoke_am_scheduler_active_workers(void);
bool evoke_am_scheduler_try_reserve_worker_launch(int limit);
void evoke_am_scheduler_release_worker_launch(void);
Oid evoke_am_scheduler_last_maintenance_db_oid(void);
void evoke_am_scheduler_set_last_maintenance_db_oid(Oid db_oid);

void evoke_am_scheduler_process_adopt_worker_launch(void);
void evoke_am_scheduler_process_release_worker_launch(void);
void evoke_am_scheduler_process_initialize(void);
void evoke_am_scheduler_process_note_started(void);
void evoke_am_scheduler_process_set_phase(
    evoke_am_scheduler_worker_phase phase
);
void evoke_am_scheduler_process_note_finished(void);

void evoke_am_scheduler_status_snapshot(
    Oid database_oid,
    evoke_am_scheduler_status *status_out
);

#endif
