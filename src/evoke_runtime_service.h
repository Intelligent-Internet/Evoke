#ifndef EVOKE_RUNTIME_SERVICE_H
#define EVOKE_RUNTIME_SERVICE_H

#include "postgres.h"

#include "port/atomics.h"
#include "storage/lwlock.h"
#include "storage/procnumber.h"
#include "utils/timestamp.h"

#define EVOKE_LWLOCK_TRANCHE_NAME "evoke"
#define EVOKE_LWLOCK_TRANCHE_COUNT 2
#define EVOKE_RUNTIME_SERVICE_LOCK_INDEX 0
#define EVOKE_SHARED_PRELOAD_LOCK_INDEX 1
#define EVOKE_RUNTIME_SERVICE_MAGIC UINT32_C(0x52543249)
#define EVOKE_RUNTIME_SERVICE_VERSION 26
#define EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES (1024 * 1024)
#define EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES (4 * 1024 * 1024)
#define EVOKE_RUNTIME_SERVICE_ERROR_MAX_BYTES 2048
#define EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES 32
#define EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES 16
#define EVOKE_RUNTIME_PRECISION_FP16 "fp16"
#define EVOKE_RUNTIME_PRECISION_FP32 "fp32"
#define EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY 32
#define EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY \
    EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY
#define EVOKE_RUNTIME_SERVICE_MAX_WORKERS 16
#define EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES 64U
#define EVOKE_RUNTIME_ACCELERATOR_MAX_WEIGHT 4096U
#define EVOKE_RUNTIME_ACCELERATOR_URL_MAX_BYTES 512
#define EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT UINT32_MAX
#define EVOKE_RUNTIME_SERVICE_INVALID_WORKER_SLOT UINT32_MAX
#define EVOKE_RUNTIME_SERVICE_DEFAULT_MAX_BATCH_SIZE 128
#define EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE 512
#define EVOKE_RUNTIME_SERVICE_DEFAULT_DOCUMENT_PIPELINE_DEPTH 16
#define EVOKE_RUNTIME_BUILDER_MAX_INFLIGHT 4096
#define EVOKE_RUNTIME_BUILDER_MAX_BUFFERED_BATCHES 4096
#define EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS \
    EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY
#define EVOKE_RUNTIME_SERVICE_RESERVED_QUERY_RESPONSES 1
#define EVOKE_RUNTIME_SERVICE_AFFINITY_BYPASS_LIMIT 4
#define EVOKE_RUNTIME_SERVICE_RESTART_SECONDS 5
#define EVOKE_RUNTIME_TERMINATION_NONE 0
#define EVOKE_RUNTIME_TERMINATION_CANCEL 1
#define EVOKE_RUNTIME_TERMINATION_LIVENESS 2
#define EVOKE_RUNTIME_TERMINATION_PROC_DIE 3

typedef enum evoke_runtime_request_kind
{
    EVOKE_RUNTIME_REQUEST_QUERY = 1,
    EVOKE_RUNTIME_REQUEST_DOCUMENT = 2
} evoke_runtime_request_kind;

typedef struct evoke_runtime_service_request
{
    bool occupied;
    bool canceled;
    uint64 request_id;
    pid_t caller_pid;
    ProcNumber caller_proc_number;
    uint32 response_slot;
    double enqueued_ms;
    uint32 request_kind;
    bool batch_response;
    uint32 batch_count;
    Size query_len;
    char model_path[MAXPGPATH];
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    uint32 text_offsets[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE + 1];
    char query_text[EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES];
} evoke_runtime_service_request;

typedef struct evoke_runtime_service_response
{
    bool occupied;
    bool ready;
    bool writing;
    bool error;
    uint64 request_id;
    pid_t owner_pid;
    ProcNumber owner_proc_number;
    uint32 request_kind;
    Size result_len;
    char result_json[EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES];
    char error_message[EVOKE_RUNTIME_SERVICE_ERROR_MAX_BYTES];
} evoke_runtime_service_response;

typedef struct EvokeRuntimeRequestHandle
{
    uint64 request_id;
    uint32 response_slot;
    int accelerator_fd;
    char *accelerator_url;
    char *accelerator_checkout_signature;
    char *accelerator_model_path;
    char *accelerator_runtime_precision;
    char **accelerator_texts;
    uint32 accelerator_batch_count;
    TimestampTz accelerator_started_at;
    TimestampTz local_runtime_started_at;
    uint32 local_runtime_batch_count;
    char *accelerator_response;
    Size accelerator_response_len;
    Size accelerator_response_cap;
    Size accelerator_response_header_len;
    Size accelerator_response_content_len;
    int accelerator_response_status;
    bool accelerator_response_header_parsed;
    bool accelerator_response_keep_alive;
    bool accelerator_health_active;
    bool local_runtime_health_active;
    bool accelerator_active;
    bool active;
    uint32 accelerator_retry_count;
    TimestampTz accelerator_retry_after;
    TimestampTz accelerator_deadline_at;
    TimestampTz accelerator_local_fallback_deadline_at;
} EvokeRuntimeRequestHandle;

typedef struct evoke_runtime_service_worker
{
    bool started;
    bool ready;
    bool processing;
    bool processing_document;
    pg_atomic_uint32 cancel_requested;
    pg_atomic_uint32 termination_reason;
    bool terminate_requested;
    pid_t pid;
    ProcNumber proc_number;
    pid_t processing_owner_pid;
    ProcNumber processing_owner_proc_number;
    uint32 processing_response_slot;
    uint32 processing_batch_count;
    uint64 processing_request_id;
    uint64 starts;
    uint64 recoveries;
    uint64 runtime_runs;
    uint64 successes;
    uint64 failures;
    uint64 runtime_total_us;
    uint64 runtime_max_us;
    uint64 encoded_texts;
    uint64 batch_successes;
    uint64 runtime_terminations;
    uint64 runtime_liveness_timeouts;
    uint64 session_cache_hits;
    uint64 session_cache_misses;
    uint64 session_cache_loads;
    uint64 session_cache_evictions;
    TimestampTz started_at;
    TimestampTz processing_started_at;
    TimestampTz last_progress_at;
    TimestampTz last_request_at;
    TimestampTz last_response_at;
    uint32 affinity_request_kind;
    char affinity_model_path[MAXPGPATH];
    char affinity_runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    char provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char active_provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
} evoke_runtime_service_worker;

typedef struct evoke_runtime_accelerator_state
{
    bool occupied;
    char url[EVOKE_RUNTIME_ACCELERATOR_URL_MAX_BYTES];
    uint32 in_flight;
    uint32 consecutive_failures;
    uint64 successes;
    uint64 request_failures;
    uint64 connect_failures;
    uint64 read_failures;
    uint64 status_failures;
    uint64 backpressure_failures;
    uint64 body_failures;
    uint64 status_400_failures;
    uint64 status_409_failures;
    uint64 status_500_failures;
    uint64 status_503_failures;
    uint64 status_504_failures;
    uint64 status_other_failures;
    uint32 last_status_failure;
    char last_status_body[256];
    double ewma_ms_per_text;
    double ewma_batch_ms;
    TimestampTz last_started_at;
    TimestampTz next_probe_at;
} evoke_runtime_accelerator_state;

typedef struct evoke_runtime_service_control
{
    uint32 magic;
    uint32 version;
    bool request_pending;
    uint32 configured_worker_count;
    uint64 request_id;
    uint64 last_completed_request_id;
    uint64 requests;
    uint64 successes;
    uint64 failures;
    uint64 busy_rejections;
    uint64 canceled_requests;
    uint64 orphan_responses;
    uint64 worker_recoveries;
    uint64 queue_waits;
    uint64 queue_total_wait_ms;
    uint64 queue_max_wait_ms;
    uint64 runtime_runs;
    uint64 runtime_total_us;
    uint64 runtime_max_us;
    uint64 encoded_texts;
    uint64 batch_successes;
    uint64 runtime_terminations;
    uint64 runtime_liveness_timeouts;
    uint64 affinity_dispatches;
    uint64 affinity_bypasses;
    uint64 query_dispatches;
    uint64 document_dispatches;
    uint64 accelerator_schedule_cursor;
    uint64 accelerator_local_successes;
    uint64 accelerator_local_abandoned;
    double accelerator_local_ewma_ms_per_text;
    double accelerator_local_ewma_batch_ms;
    uint32 last_batch_size;
    uint32 max_observed_batch_size;
    uint32 queue_count;
    uint32 queue_max_depth;
    TimestampTz last_request_at;
    TimestampTz last_response_at;
    char provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char active_provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    evoke_runtime_accelerator_state
        accelerators[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES];
    evoke_runtime_service_worker
        workers[EVOKE_RUNTIME_SERVICE_MAX_WORKERS];
    evoke_runtime_service_request queue[EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY];
    evoke_runtime_service_response
        responses[EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY];
} evoke_runtime_service_control;

extern evoke_runtime_service_control *evoke_runtime_service;
extern LWLock *evoke_runtime_service_lock;
extern int evoke_runtime_worker_count;
extern int evoke_runtime_max_batch_size;
extern int evoke_runtime_document_pipeline_depth;
extern int evoke_runtime_liveness_timeout_ms;
extern bool evoke_runtime_reserve_query_lane;
extern char *evoke_runtime_accelerators;
extern char *evoke_control_database;

extern Size evoke_runtime_service_shmem_size(void);
extern void evoke_runtime_service_shmem_startup(LWLock *lock);
extern void evoke_runtime_service_initialize_control(
    evoke_runtime_service_control *control
);
extern uint32 evoke_runtime_effective_document_pipeline_depth(void);
extern uint32 evoke_runtime_effective_local_document_pipeline_depth(void);
extern void evoke_runtime_service_submit_document_async(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    EvokeRuntimeRequestHandle *handle
);
extern void evoke_runtime_service_submit_document_prefix_async(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    uint32 *submitted_count_out,
    EvokeRuntimeRequestHandle *handle
);
extern void evoke_runtime_service_submit_document_prefix_checkout_async(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    uint32 *submitted_count_out,
    EvokeRuntimeRequestHandle *handle
);
extern uint32 evoke_runtime_service_recommended_document_batch_size(void);
extern char *evoke_runtime_service_wait_async(
    EvokeRuntimeRequestHandle *handle
);
extern bool evoke_runtime_service_try_complete_async(
    EvokeRuntimeRequestHandle *handle,
    char **result_json_out
);
extern bool evoke_runtime_service_request_ready(
    const EvokeRuntimeRequestHandle *handle
);
extern void evoke_runtime_service_cancel_async(
    EvokeRuntimeRequestHandle *handle
);
extern PGDLLEXPORT void evoke_runtime_worker_main(Datum main_arg);

#endif
