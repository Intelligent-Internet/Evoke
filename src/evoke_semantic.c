#include "postgres.h"

#include "access/xact.h"
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <netdb.h>
#include <poll.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "access/htup_details.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/transam.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "catalog/pg_authid_d.h"
#include "catalog/pg_database_d.h"
#include "catalog/pg_type_d.h"
#include "common/cryptohash.h"
#include "common/sha2.h"
#include "executor/spi.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "port/atomics.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/itemptr.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/procarray.h"
#include "storage/shmem.h"
#include "tcop/tcopprot.h"
#include "utils/array.h"
#include "utils/acl.h"
#include "utils/backend_status.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/hsearch.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"
#include "utils/varlena.h"

#include "evoke_semantic.h"
#include "evoke_am_preload.h"
#include "evoke_core.h"
#include "evoke_runtime_service.h"
#include "evoke_p2_runtime.h"

#ifdef EVOKE_ENABLE_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

#define SBMX_MAGIC_LEN 8
#define SBMX_CANDIDATE_RESULT_COLS 22
#define EVOKE_CHECKOUT_MAX_MANIFEST_BYTES (16 * 1024 * 1024)
#define EVOKE_CHECKOUT_MAX_ARTIFACT_TEXT_BYTES (64 * 1024 * 1024)
#define EVOKE_RUNTIME_ACCELERATOR_CONNECT_TIMEOUT_MS 1000
#define EVOKE_RUNTIME_ACCELERATOR_SOCKET_IO_TIMEOUT_MS 30000
#define EVOKE_RUNTIME_ACCELERATOR_BACKPRESSURE_MS 100
#define EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MIN_MS 1000
#define EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MAX_MS 30000
#define EVOKE_RUNTIME_ACCELERATOR_RETRY_LIMIT 4
#define EVOKE_RUNTIME_ACCELERATOR_ASYNC_RETRY_LIMIT 4
#define EVOKE_RUNTIME_ACCELERATOR_ASYNC_RETRY_WAIT_MS 100
#define EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_LIMIT 60
#define EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT 600
#define EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS 100
#define EVOKE_RUNTIME_ACCELERATOR_LOCAL_FAILOVER_WAIT_MS \
    (EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT * \
        EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS)
#define EVOKE_RUNTIME_ACCELERATOR_HEALTHY_FAILURE_SOFT_LIMIT 2
#define EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS 32
#define EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTION_MAX_AGE_MS 30000
#define EVOKE_RUNTIME_ACCELERATOR_IDLE_REUSE_ATTEMPTS 16
#define EVOKE_RUNTIME_ACCELERATOR_STALE_INFLIGHT_MULTIPLIER 4
#define EVOKE_RUNTIME_ACCELERATOR_UNKNOWN_MS_PER_TEXT 1.0
extern int evoke_onnxruntime_session_cache_size;
int evoke_onnxruntime_session_cache_size = 1;
extern int evoke_onnxruntime_intra_op_threads;
int evoke_onnxruntime_intra_op_threads = 0;
extern bool evoke_onnxruntime_document_cpu_mem_arena;
bool evoke_onnxruntime_document_cpu_mem_arena = false;
int evoke_runtime_worker_count = 2;
int evoke_runtime_max_batch_size =
    EVOKE_RUNTIME_SERVICE_DEFAULT_MAX_BATCH_SIZE;
int evoke_runtime_document_pipeline_depth =
    EVOKE_RUNTIME_SERVICE_DEFAULT_DOCUMENT_PIPELINE_DEPTH;
int evoke_runtime_liveness_timeout_ms = 300000;
bool evoke_runtime_reserve_query_lane = true;
char *evoke_runtime_accelerators = NULL;
char *evoke_control_database = NULL;
evoke_runtime_service_control *evoke_runtime_service = NULL;
LWLock *evoke_runtime_service_lock = NULL;
static volatile sig_atomic_t evoke_runtime_worker_got_sighup = false;

static uint32 evoke_runtime_service_configured_workers(void);
static uint32 evoke_runtime_service_pending_document_requests_locked(void);
static uint32 evoke_runtime_service_document_response_slots_locked(void);
static uint32 evoke_runtime_service_active_document_workers_locked(void);
static uint32 evoke_runtime_service_max_document_workers_for_ready(
    uint32 ready_workers
);
static void evoke_runtime_service_submit_document_checkout_async(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    EvokeRuntimeRequestHandle *handle
);
#ifdef EVOKE_ENABLE_ONNXRUNTIME
static uint32 evoke_runtime_active_worker_id =
    EVOKE_RUNTIME_SERVICE_INVALID_WORKER_SLOT;
static uint64 evoke_runtime_active_request_id = 0;
#endif

typedef struct EvokeCheckoutArtifactIdentity
{
    char *path;
    uint64 device;
    uint64 inode;
    int64 size;
    int64 mtime_sec;
    int64 mtime_nsec;
    int64 ctime_sec;
    int64 ctime_nsec;
} EvokeCheckoutArtifactIdentity;

typedef struct EvokeRuntimeAcceleratorTarget
{
    char *url;
    uint32 weight;
    uint32 max_batch_size;
} EvokeRuntimeAcceleratorTarget;

typedef struct EvokeRuntimeAcceleratorIdleConnection
{
    bool occupied;
    char url[EVOKE_RUNTIME_ACCELERATOR_URL_MAX_BYTES];
    int fd;
    TimestampTz last_used_at;
} EvokeRuntimeAcceleratorIdleConnection;

static EvokeRuntimeAcceleratorIdleConnection
    evoke_runtime_accelerator_idle_connections[
        EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS
    ];

#define EVOKE_CHECKOUT_VALIDATION_CACHE_LIMIT 16

typedef struct EvokeCheckoutValidationCacheEntry
{
    MemoryContext context;
    char *model_path;
    char *canonical_manifest;
    EvokeCheckoutArtifactIdentity *artifacts;
    uint64 artifact_count;
    uint64 access_seq;
} EvokeCheckoutValidationCacheEntry;

static EvokeCheckoutValidationCacheEntry
    evoke_checkout_validation_cache[EVOKE_CHECKOUT_VALIDATION_CACHE_LIMIT];
static uint64 evoke_checkout_validation_cache_access_seq = 0;

typedef const char *(*EvokeOrtGetVersionStringFn)(void);

typedef struct EvokeOrtApiBase
{
    const void *(*GetApi)(uint32 api_version);
    EvokeOrtGetVersionStringFn GetVersionString;
} EvokeOrtApiBase;

typedef const EvokeOrtApiBase *(*EvokeOrtGetApiBaseFn)(void);

static void
evoke_checkout_require_superuser(void)
{
    if (!superuser())
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                errmsg("evoke model checkout access requires superuser"),
                errhint(
                    "The model path is a server-local filesystem path. "
                    "Register models as a superuser, then expose SQL search "
                    "interfaces to ordinary users.")
            )
        );
    }
}

static void
evoke_checkout_validate_path(const char *model_path)
{
    if (model_path == NULL || model_path[0] == '\0')
    {
        ereport(ERROR, (errmsg("evoke model path must not be empty")));
    }
    if (model_path[0] != '/')
    {
        ereport(ERROR, (errmsg("evoke model path must be absolute")));
    }
    if (strchr(model_path, '\n') != NULL ||
        strchr(model_path, '\r') != NULL)
    {
        ereport(ERROR, (errmsg("evoke model path must not contain newlines")));
    }
}

static char *
evoke_checkout_manifest_path(const char *model_path)
{
    int written;
    char path[MAXPGPATH];

    evoke_checkout_validate_path(model_path);
    written = snprintf(path, sizeof(path), "%s/manifest.json", model_path);
    if (written <= 0 || (size_t) written >= sizeof(path))
    {
        ereport(ERROR, (errmsg("evoke model manifest path is too long")));
    }
    return pstrdup(path);
}

static char *evoke_checkout_artifact_path(
    const char *model_path,
    const char *artifact_key,
    const char *relpath);

static char *
evoke_checkout_read_manifest_text(const char *model_path, Size *len_out)
{
    char *path = evoke_checkout_manifest_path(model_path);
    struct stat st;
    int fd;
    char *buffer;
    Size len;
    Size pos = 0;

    if (stat(path, &st) != 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not stat evoke model manifest \"%s\": %m", path)
            )
        );
    }
    if (!S_ISREG(st.st_mode))
    {
        ereport(ERROR, (errmsg("evoke model manifest must be a regular file")));
    }
    if (st.st_size <= 0 || st.st_size > EVOKE_CHECKOUT_MAX_MANIFEST_BYTES)
    {
        ereport(
            ERROR,
            (errmsg("evoke model manifest size must be between 1 byte and 16 MiB"))
        );
    }

    fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not open evoke model manifest \"%s\": %m", path)
            )
        );
    }

    len = (Size) st.st_size;
    buffer = palloc(len + 1);
    while (pos < len)
    {
        ssize_t nread = read(fd, buffer + pos, len - pos);

        if (nread < 0)
        {
            int saved_errno = errno;

            close(fd);
            errno = saved_errno;
            ereport(
                ERROR,
                (
                    errcode_for_file_access(),
                    errmsg("could not read evoke model manifest \"%s\": %m", path)
                )
            );
        }
        if (nread == 0)
        {
            close(fd);
            ereport(ERROR, (errmsg("short read from evoke model manifest")));
        }
        pos += (Size) nread;
    }
    close(fd);
    buffer[len] = '\0';
    if (len_out != NULL)
    {
        *len_out = len;
    }
    return buffer;
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static char *
evoke_checkout_read_artifact_text(
    const char *model_path,
    const char *artifact_key,
    const char *relpath,
    Size *len_out)
{
    char *path = evoke_checkout_artifact_path(model_path, artifact_key, relpath);
    struct stat st;
    int fd;
    char *buffer;
    Size len;
    Size pos = 0;

    if (stat(path, &st) != 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not stat Evoke runtime model artifact \"%s\": %m", path)
            )
        );
    }
    if (!S_ISREG(st.st_mode))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" must be a regular file.",
                          artifact_key)
            )
        );
    }
    if (st.st_size <= 0 ||
        st.st_size > EVOKE_CHECKOUT_MAX_ARTIFACT_TEXT_BYTES)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail(
                    "artifact \"%s\" size must be between 1 byte and 64 MiB.",
                    artifact_key)
            )
        );
    }

    fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not open Evoke runtime model artifact \"%s\": %m", path)
            )
        );
    }

    len = (Size) st.st_size;
    buffer = palloc(len + 1);
    while (pos < len)
    {
        ssize_t nread = read(fd, buffer + pos, len - pos);

        if (nread < 0)
        {
            int saved_errno = errno;

            close(fd);
            errno = saved_errno;
            ereport(
                ERROR,
                (
                    errcode_for_file_access(),
                    errmsg("could not read Evoke runtime model artifact \"%s\": %m",
                           path)
                )
            );
        }
        if (nread == 0)
        {
            close(fd);
            ereport(
                ERROR,
                (
                    errmsg("short read from Evoke runtime model artifact \"%s\"", path)
                )
            );
        }
        pos += (Size) nread;
    }
    close(fd);
    buffer[len] = '\0';
    if (len_out != NULL)
    {
        *len_out = len;
    }
    return buffer;
}
#endif

static Datum
evoke_checkout_manifest_jsonb_from_text(const char *manifest_text)
{
    return DirectFunctionCall1(jsonb_in, CStringGetDatum(manifest_text));
}

static bool
evoke_checkout_is_hex_sha256(const char *value)
{
    if (value == NULL || strlen(value) != PG_SHA256_DIGEST_STRING_LENGTH - 1)
    {
        return false;
    }
    for (const char *ptr = value; *ptr != '\0'; ptr++)
    {
        if (!isxdigit((unsigned char) *ptr))
        {
            return false;
        }
    }
    return true;
}

static void
evoke_checkout_validate_artifact_relpath(
    const char *artifact_key,
    const char *relpath)
{
    const char *segment = relpath;

    if (relpath == NULL || relpath[0] == '\0')
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" path must not be empty.",
                          artifact_key)
            )
        );
    }
    if (relpath[0] == '/')
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" path must be relative.",
                          artifact_key)
            )
        );
    }
    if (strchr(relpath, '\n') != NULL || strchr(relpath, '\r') != NULL)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" path must not contain newlines.",
                          artifact_key)
            )
        );
    }

    while (*segment != '\0')
    {
        const char *slash = strchr(segment, '/');
        Size len = slash == NULL
            ? strlen(segment)
            : (Size) (slash - segment);

        if (len == 0 ||
            (len == 1 && segment[0] == '.') ||
            (len == 2 && segment[0] == '.' && segment[1] == '.'))
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid Evoke runtime model artifact"),
                    errdetail(
                        "artifact \"%s\" path must stay inside the checkout.",
                        artifact_key)
                )
            );
        }
        if (slash == NULL)
        {
            break;
        }
        segment = slash + 1;
    }
}

static char *
evoke_checkout_artifact_path(
    const char *model_path,
    const char *artifact_key,
    const char *relpath)
{
    int written;
    char path[MAXPGPATH];

    evoke_checkout_validate_artifact_relpath(artifact_key, relpath);
    written = snprintf(path, sizeof(path), "%s/%s", model_path, relpath);
    if (written <= 0 || (size_t) written >= sizeof(path))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" path is too long.", artifact_key)
            )
        );
    }
    return pstrdup(path);
}

static char *
evoke_checkout_sha256_file(const char *path)
{
    pg_cryptohash_ctx *ctx;
    uint8 digest[PG_SHA256_DIGEST_LENGTH];
    char *hex;
    int fd;
    uint8 buffer[8192];
    static const char hex_chars[] = "0123456789abcdef";

    fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not open Evoke runtime model artifact \"%s\": %m", path)
            )
        );
    }

    ctx = pg_cryptohash_create(PG_SHA256);
    if (ctx == NULL ||
        pg_cryptohash_init(ctx) < 0)
    {
        if (ctx != NULL)
        {
            pg_cryptohash_free(ctx);
        }
        close(fd);
        ereport(ERROR, (errmsg("could not initialize SHA-256 context")));
    }

    for (;;)
    {
        ssize_t nread = read(fd, buffer, sizeof(buffer));

        if (nread < 0)
        {
            int saved_errno = errno;

            pg_cryptohash_free(ctx);
            close(fd);
            errno = saved_errno;
            ereport(
                ERROR,
                (
                    errcode_for_file_access(),
                    errmsg("could not read Evoke runtime model artifact \"%s\": %m",
                           path)
                )
            );
        }
        if (nread == 0)
        {
            break;
        }
        if (pg_cryptohash_update(ctx, buffer, (size_t) nread) < 0)
        {
            const char *detail = pg_cryptohash_error(ctx);

            pg_cryptohash_free(ctx);
            close(fd);
            ereport(
                ERROR,
                (
                    errmsg("could not update SHA-256 context"),
                    errdetail("%s", detail == NULL ? "unknown error" : detail)
                )
            );
        }
    }
    close(fd);

    if (pg_cryptohash_final(ctx, digest, sizeof(digest)) < 0)
    {
        const char *detail = pg_cryptohash_error(ctx);

        pg_cryptohash_free(ctx);
        ereport(
            ERROR,
            (
                errmsg("could not finalize SHA-256 context"),
                errdetail("%s", detail == NULL ? "unknown error" : detail)
            )
        );
    }
    pg_cryptohash_free(ctx);

    hex = palloc(PG_SHA256_DIGEST_STRING_LENGTH);
    for (int i = 0; i < PG_SHA256_DIGEST_LENGTH; i++)
    {
        hex[i * 2] = hex_chars[digest[i] >> 4];
        hex[i * 2 + 1] = hex_chars[digest[i] & 0x0f];
    }
    hex[PG_SHA256_DIGEST_STRING_LENGTH - 1] = '\0';
    return hex;
}

static int64
evoke_checkout_stat_mtime_nsec(const struct stat *st)
{
#if defined(__APPLE__)
    return (int64) st->st_mtimespec.tv_nsec;
#elif defined(_WIN32)
    return 0;
#else
    return (int64) st->st_mtim.tv_nsec;
#endif
}

static int64
evoke_checkout_stat_ctime_nsec(const struct stat *st)
{
#if defined(__APPLE__)
    return (int64) st->st_ctimespec.tv_nsec;
#elif defined(_WIN32)
    return 0;
#else
    return (int64) st->st_ctim.tv_nsec;
#endif
}

static void
evoke_checkout_artifact_identity_set(
    EvokeCheckoutArtifactIdentity *identity,
    const char *path,
    const struct stat *st
)
{
    memset(identity, 0, sizeof(*identity));
    identity->path = (char *) path;
    identity->device = (uint64) st->st_dev;
    identity->inode = (uint64) st->st_ino;
    identity->size = (int64) st->st_size;
    identity->mtime_sec = (int64) st->st_mtime;
    identity->mtime_nsec = evoke_checkout_stat_mtime_nsec(st);
    identity->ctime_sec = (int64) st->st_ctime;
    identity->ctime_nsec = evoke_checkout_stat_ctime_nsec(st);
}

static bool
evoke_checkout_artifact_identity_equal(
    const EvokeCheckoutArtifactIdentity *left,
    const EvokeCheckoutArtifactIdentity *right
)
{
    return left->device == right->device &&
        left->inode == right->inode &&
        left->size == right->size &&
        left->mtime_sec == right->mtime_sec &&
        left->mtime_nsec == right->mtime_nsec &&
        left->ctime_sec == right->ctime_sec &&
        left->ctime_nsec == right->ctime_nsec;
}

static bool
evoke_checkout_validated_artifacts_unchanged(
    const EvokeCheckoutValidationCacheEntry *entry
)
{
    for (uint64 i = 0; i < entry->artifact_count; i++)
    {
        EvokeCheckoutArtifactIdentity current;
        struct stat st;

        if (stat(entry->artifacts[i].path, &st) != 0 ||
            !S_ISREG(st.st_mode))
        {
            return false;
        }
        evoke_checkout_artifact_identity_set(
            &current,
            entry->artifacts[i].path,
            &st
        );
        if (!evoke_checkout_artifact_identity_equal(
                &current,
                &entry->artifacts[i]
            ))
        {
            return false;
        }
    }
    return entry->artifact_count > 0;
}

static void
evoke_checkout_validate_artifact_file(
    const char *model_path,
    const char *artifact_key,
    const char *relpath,
    const char *expected_sha256,
    MemoryContext identity_context,
    EvokeCheckoutArtifactIdentity *identity_out)
{
    char *path;
    struct stat before_stat;
    struct stat after_stat;
    EvokeCheckoutArtifactIdentity before_identity;
    EvokeCheckoutArtifactIdentity after_identity;
    char *actual_sha256;

    if (!evoke_checkout_is_hex_sha256(expected_sha256))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" sha256 must be 64 hex chars.",
                          artifact_key)
            )
        );
    }

    path = evoke_checkout_artifact_path(model_path, artifact_key, relpath);
    if (stat(path, &before_stat) != 0)
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg("could not stat Evoke runtime model artifact \"%s\": %m", path)
            )
        );
    }
    if (!S_ISREG(before_stat.st_mode))
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" must be a regular file.",
                          artifact_key)
            )
        );
    }

    evoke_checkout_artifact_identity_set(&before_identity, path, &before_stat);
    actual_sha256 = evoke_checkout_sha256_file(path);
    if (stat(path, &after_stat) != 0 || !S_ISREG(after_stat.st_mode))
    {
        ereport(
            ERROR,
            (
                errcode_for_file_access(),
                errmsg(
                    "could not restat Evoke runtime model artifact \"%s\": %m",
                    path)
            )
        );
    }
    evoke_checkout_artifact_identity_set(&after_identity, path, &after_stat);
    if (!evoke_checkout_artifact_identity_equal(
            &before_identity,
            &after_identity
        ))
    {
        ereport(
            ERROR,
            (
                errmsg("Evoke runtime model artifact changed during validation"),
                errdetail("artifact \"%s\" was modified while hashing.",
                          artifact_key),
                errhint("Publish model checkouts atomically, then REINDEX.")
            )
        );
    }
    if (pg_strcasecmp(actual_sha256, expected_sha256) != 0)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail(
                    "artifact \"%s\" sha256 mismatch: expected %s got %s.",
                    artifact_key,
                    expected_sha256,
                    actual_sha256)
            )
        );
    }
    *identity_out = after_identity;
    identity_out->path = MemoryContextStrdup(identity_context, path);
}

static void
evoke_checkout_validate_manifest_artifacts(
    Datum manifest,
    const char *model_path)
{
    Oid argtypes[1] = {JSONBOID};
    Datum values[1] = {manifest};
    char nulls[1] = {' '};
    int spi_result;
    MemoryContext caller_context = CurrentMemoryContext;
    const char *query =
        "SELECT key, value->>'path', value->>'sha256' "
        "FROM jsonb_each(COALESCE($1->'artifacts', '{}'::jsonb)) "
        "ORDER BY key";
    char *canonical_manifest;
    EvokeCheckoutArtifactIdentity *validated_artifacts;
    uint64 validated_artifact_count;
    int exact_cache_slot = -1;
    int empty_cache_slot = -1;
    int lru_cache_slot = -1;
    int cache_slot;

    if (model_path == NULL)
    {
        return;
    }
    canonical_manifest = DatumGetCString(DirectFunctionCall1(
        jsonb_out,
        manifest
    ));
    for (int i = 0; i < EVOKE_CHECKOUT_VALIDATION_CACHE_LIMIT; i++)
    {
        EvokeCheckoutValidationCacheEntry *entry =
            &evoke_checkout_validation_cache[i];

        if (entry->context == NULL)
        {
            if (empty_cache_slot < 0)
            {
                empty_cache_slot = i;
            }
            continue;
        }
        if (lru_cache_slot < 0 ||
            entry->access_seq <
                evoke_checkout_validation_cache[lru_cache_slot].access_seq)
        {
            lru_cache_slot = i;
        }
        if (strcmp(entry->model_path, model_path) == 0 &&
            strcmp(entry->canonical_manifest, canonical_manifest) == 0)
        {
            if (evoke_checkout_validated_artifacts_unchanged(entry))
            {
                evoke_checkout_validation_cache_access_seq++;
                entry->access_seq =
                    evoke_checkout_validation_cache_access_seq;
                pfree(canonical_manifest);
                return;
            }
            exact_cache_slot = i;
            break;
        }
    }

    spi_result = SPI_connect();
    if (spi_result != SPI_OK_CONNECT)
    {
        ereport(ERROR, (errmsg("could not connect to SPI")));
    }
    spi_result = SPI_execute_with_args(
        query,
        1,
        argtypes,
        values,
        nulls,
        true,
        0);
    if (spi_result < 0)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("could not validate Evoke runtime model artifacts")));
    }

    validated_artifact_count = SPI_processed;
    if (validated_artifact_count == 0 ||
        validated_artifact_count >
            (uint64) MaxAllocSize / sizeof(*validated_artifacts))
    {
        SPI_finish();
        ereport(
            ERROR,
            (errmsg("Evoke runtime model manifest has invalid artifacts"))
        );
    }
    validated_artifacts = MemoryContextAllocZero(
        caller_context,
        sizeof(*validated_artifacts) * validated_artifact_count
    );

    for (uint64 row = 0; row < SPI_processed; row++)
    {
        bool isnull = false;
        Datum key_datum;
        Datum path_datum;
        Datum sha_datum;
        char *artifact_key;
        char *relpath;
        char *expected_sha256;

        key_datum = SPI_getbinval(
            SPI_tuptable->vals[row],
            SPI_tuptable->tupdesc,
            1,
            &isnull);
        if (isnull)
        {
            SPI_finish();
            ereport(ERROR, (errmsg("Evoke runtime artifact key must not be null")));
        }
        artifact_key = TextDatumGetCString(key_datum);

        path_datum = SPI_getbinval(
            SPI_tuptable->vals[row],
            SPI_tuptable->tupdesc,
            2,
            &isnull);
        if (isnull)
        {
            SPI_finish();
            ereport(
                ERROR,
                (
                    errmsg("invalid Evoke runtime model artifact"),
                    errdetail("artifact \"%s\" path is required.",
                              artifact_key)
                )
            );
        }
        relpath = TextDatumGetCString(path_datum);

        sha_datum = SPI_getbinval(
            SPI_tuptable->vals[row],
            SPI_tuptable->tupdesc,
            3,
            &isnull);
        if (isnull)
        {
            SPI_finish();
            ereport(
                ERROR,
                (
                    errmsg("invalid Evoke runtime model artifact"),
                    errdetail("artifact \"%s\" sha256 is required.",
                              artifact_key)
                )
            );
        }
        expected_sha256 = TextDatumGetCString(sha_datum);
        evoke_checkout_validate_artifact_file(
            model_path,
            artifact_key,
            relpath,
            expected_sha256,
            caller_context,
            &validated_artifacts[row]);
    }
    SPI_finish();

    if (exact_cache_slot >= 0)
    {
        cache_slot = exact_cache_slot;
    }
    else if (empty_cache_slot >= 0)
    {
        cache_slot = empty_cache_slot;
    }
    else
    {
        Assert(lru_cache_slot >= 0);
        cache_slot = lru_cache_slot;
    }
    {
        EvokeCheckoutValidationCacheEntry *entry =
            &evoke_checkout_validation_cache[cache_slot];

        if (entry->context != NULL)
        {
            MemoryContextDelete(entry->context);
        }
        memset(entry, 0, sizeof(*entry));
        entry->context = AllocSetContextCreate(
            TopMemoryContext,
            "evoke model validation cache entry",
            ALLOCSET_SMALL_SIZES
        );
        entry->model_path = MemoryContextStrdup(
            entry->context,
            model_path
        );
        entry->canonical_manifest = MemoryContextStrdup(
            entry->context,
            canonical_manifest
        );
        entry->artifact_count = validated_artifact_count;
        entry->artifacts = MemoryContextAllocZero(
            entry->context,
            sizeof(*entry->artifacts) * entry->artifact_count
        );
        for (uint64 i = 0; i < entry->artifact_count; i++)
        {
            entry->artifacts[i] = validated_artifacts[i];
            entry->artifacts[i].path = MemoryContextStrdup(
                entry->context,
                validated_artifacts[i].path
            );
            pfree(validated_artifacts[i].path);
        }
        evoke_checkout_validation_cache_access_seq++;
        entry->access_seq = evoke_checkout_validation_cache_access_seq;
    }
    pfree(validated_artifacts);
    pfree(canonical_manifest);
}

static void
evoke_checkout_validation_cache_clear(void)
{
    for (int i = 0; i < EVOKE_CHECKOUT_VALIDATION_CACHE_LIMIT; i++)
    {
        EvokeCheckoutValidationCacheEntry *entry =
            &evoke_checkout_validation_cache[i];

        if (entry->context != NULL)
        {
            MemoryContextDelete(entry->context);
        }
        memset(entry, 0, sizeof(*entry));
    }
    evoke_checkout_validation_cache_access_seq = 0;
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static char *
evoke_checkout_manifest_artifact_relpath(
    Datum manifest,
    const char *artifact_key)
{
    Oid argtypes[2] = {JSONBOID, TEXTOID};
    Datum values[2] = {
        manifest,
        CStringGetTextDatum(artifact_key)
    };
    char nulls[2] = {' ', ' '};
    MemoryContext caller_context = CurrentMemoryContext;
    int spi_result;
    bool isnull = false;
    Datum path_datum;
    char *relpath;
    const char *query =
        "SELECT $1#>>ARRAY['artifacts', $2, 'path']";

    spi_result = SPI_connect();
    if (spi_result != SPI_OK_CONNECT)
    {
        ereport(ERROR, (errmsg("could not connect to SPI")));
    }
    spi_result = SPI_execute_with_args(
        query,
        2,
        argtypes,
        values,
        nulls,
        true,
        1);
    if (spi_result < 0 || SPI_processed != 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("could not inspect Evoke runtime model artifact")));
    }
    path_datum = SPI_getbinval(
        SPI_tuptable->vals[0],
        SPI_tuptable->tupdesc,
        1,
        &isnull);
    if (isnull)
    {
        SPI_finish();
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model artifact"),
                errdetail("artifact \"%s\" is not declared.", artifact_key)
            )
        );
    }

    relpath = MemoryContextStrdup(
        caller_context,
        TextDatumGetCString(path_datum));
    SPI_finish();
    return relpath;
}
#endif

static char *
evoke_checkout_manifest_text_path(
    Datum manifest,
    const char *path0,
    const char *path1,
    const char *fallback)
{
    Oid argtypes[3] = {JSONBOID, TEXTOID, TEXTOID};
    Datum values[3] = {
        manifest,
        CStringGetTextDatum(path0),
        CStringGetTextDatum(path1)
    };
    char nulls[3] = {' ', ' ', ' '};
    MemoryContext caller_context = CurrentMemoryContext;
    int spi_result;
    bool isnull = false;
    Datum value_datum;
    char *value;
    const char *query =
        "SELECT $1#>>ARRAY[$2, $3]";

    spi_result = SPI_connect();
    if (spi_result != SPI_OK_CONNECT)
    {
        ereport(ERROR, (errmsg("could not connect to SPI")));
    }
    spi_result = SPI_execute_with_args(
        query,
        3,
        argtypes,
        values,
        nulls,
        true,
        1);
    if (spi_result < 0 || SPI_processed != 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("could not inspect Evoke runtime model manifest")));
    }

    value_datum = SPI_getbinval(
        SPI_tuptable->vals[0],
        SPI_tuptable->tupdesc,
        1,
        &isnull);
    if (isnull)
    {
        SPI_finish();
        if (fallback != NULL)
        {
            return MemoryContextStrdup(caller_context, fallback);
        }
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model manifest"),
                errdetail("manifest path \"%s.%s\" is required.", path0, path1)
            )
        );
    }

    value = TextDatumGetCString(value_datum);
    if (value[0] == '\0')
    {
        SPI_finish();
        if (fallback != NULL)
        {
            return MemoryContextStrdup(caller_context, fallback);
        }
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime model manifest"),
                errdetail(
                    "manifest path \"%s.%s\" must not be empty.",
                    path0,
                    path1)
            )
        );
    }

    value = MemoryContextStrdup(caller_context, value);
    SPI_finish();
    return value;
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static char *
evoke_checkout_jsonb_text_key(
    Datum jsonb_value,
    const char *key,
    const char *fallback)
{
    Oid argtypes[2] = {JSONBOID, TEXTOID};
    Datum values[2] = {
        jsonb_value,
        CStringGetTextDatum(key)
    };
    char nulls[2] = {' ', ' '};
    MemoryContext caller_context = CurrentMemoryContext;
    int spi_result;
    bool isnull = false;
    Datum value_datum;
    char *value;
    const char *query = "SELECT $1->>$2";

    spi_result = SPI_connect();
    if (spi_result != SPI_OK_CONNECT)
    {
        ereport(ERROR, (errmsg("could not connect to SPI")));
    }
    spi_result = SPI_execute_with_args(
        query,
        2,
        argtypes,
        values,
        nulls,
        true,
        1);
    if (spi_result < 0 || SPI_processed != 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("could not inspect Evoke runtime JSON artifact")));
    }

    value_datum = SPI_getbinval(
        SPI_tuptable->vals[0],
        SPI_tuptable->tupdesc,
        1,
        &isnull);
    if (isnull)
    {
        SPI_finish();
        return fallback == NULL
            ? NULL
            : MemoryContextStrdup(caller_context, fallback);
    }

    value = TextDatumGetCString(value_datum);
    if (value[0] == '\0')
    {
        SPI_finish();
        return fallback == NULL
            ? NULL
            : MemoryContextStrdup(caller_context, fallback);
    }

    value = MemoryContextStrdup(caller_context, value);
    SPI_finish();
    return value;
}
#endif

static char *
evoke_checkout_jsonb_text_path(
    Datum jsonb_value,
    const char *path0,
    const char *path1,
    const char *fallback)
{
    return evoke_checkout_manifest_text_path(
        jsonb_value,
        path0,
        path1,
        fallback);
}

static int32
evoke_checkout_parse_int32_setting(
    const char *value,
    const char *setting,
    int32 fallback,
    int32 min_value,
    int32 max_value)
{
    char *endptr;
    long parsed;

    if (value == NULL)
    {
        return fallback;
    }
    errno = 0;
    parsed = strtol(value, &endptr, 10);
    if (errno != 0 || endptr == value || *endptr != '\0'
        || parsed < min_value || parsed > max_value)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime setting"),
                errdetail(
                    "%s must be an integer between %d and %d.",
                    setting,
                    min_value,
                    max_value)
            )
        );
    }
    return (int32) parsed;
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static int32
evoke_checkout_jsonb_int_key(
    Datum jsonb_value,
    const char *key,
    int32 fallback,
    int32 min_value,
    int32 max_value)
{
    char fallback_text[32];
    char *value;

    snprintf(fallback_text, sizeof(fallback_text), "%d", fallback);
    value = evoke_checkout_jsonb_text_key(jsonb_value, key, fallback_text);

    return evoke_checkout_parse_int32_setting(
        value,
        key,
        fallback,
        min_value,
        max_value);
}
#endif

static int32
evoke_checkout_jsonb_int_path(
    Datum jsonb_value,
    const char *path0,
    const char *path1,
    int32 fallback,
    int32 min_value,
    int32 max_value)
{
    char fallback_text[32];
    char *value;
    char setting[128];

    snprintf(fallback_text, sizeof(fallback_text), "%d", fallback);
    value = evoke_checkout_jsonb_text_path(
        jsonb_value,
        path0,
        path1,
        fallback_text
    );
    snprintf(setting, sizeof(setting), "%s.%s", path0, path1);
    return evoke_checkout_parse_int32_setting(
        value,
        setting,
        fallback,
        min_value,
        max_value);
}

static double
evoke_checkout_parse_double_setting(
    const char *value,
    const char *setting,
    double min_value,
    double max_value)
{
    char *endptr;
    double parsed;

    if (value == NULL)
    {
        ereport(
            ERROR,
            (
                errmsg("missing Evoke runtime setting"),
                errdetail("%s is required.", setting)
            )
        );
    }
    errno = 0;
    parsed = strtod(value, &endptr);
    if (errno != 0 || endptr == value || *endptr != '\0' ||
        !isfinite(parsed) || parsed < min_value || parsed > max_value)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid Evoke runtime setting"),
                errdetail(
                    "%s must be a finite number between %.9g and %.9g.",
                    setting,
                    min_value,
                    max_value
                )
            )
        );
    }
    return parsed;
}

static double
evoke_checkout_jsonb_double_path(
    Datum jsonb_value,
    const char *path0,
    const char *path1,
    double min_value,
    double max_value)
{
    char *value = evoke_checkout_jsonb_text_path(
        jsonb_value,
        path0,
        path1,
        NULL
    );
    char setting[128];

    snprintf(setting, sizeof(setting), "%s.%s", path0, path1);
    return evoke_checkout_parse_double_setting(
        value,
        setting,
        min_value,
        max_value
    );
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static Datum
evoke_checkout_artifact_jsonb(
    Datum manifest,
    const char *model_path,
    const char *artifact_key)
{
    char *relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        artifact_key);
    char *artifact_text = evoke_checkout_read_artifact_text(
        model_path,
        artifact_key,
        relpath,
        NULL);

    return evoke_checkout_manifest_jsonb_from_text(artifact_text);
}
#endif

static void
evoke_checkout_validate_manifest_shape(
    Datum manifest,
    text *model_id,
    const char *model_path
)
{
    Oid argtypes[2] = {JSONBOID, TEXTOID};
    Datum values[2] = {manifest, (Datum) 0};
    char nulls[2] = {' ', 'n'};
    int spi_result;
    bool isnull = false;
    Datum valid_datum;
    bool valid;
    const char *query =
        "SELECT COALESCE("
        "jsonb_typeof($1) = 'object' "
        "AND $1->'schema_version' = '1'::jsonb "
        "AND NULLIF($1->>'model_id', '') IS NOT NULL "
        "AND ($2 IS NULL OR $1->>'model_id' = $2) "
        "AND $1->>'api_version' = 'evoke_model_v1' "
        "AND NOT ($1 ? 'runtime_parameters') "
        "AND NULLIF($1->>'encoder_type', '') IS NOT NULL "
        "AND $1->>'runtime' = 'onnxruntime' "
        "AND $1->>'runtime_abi' = 'evoke_p2_unified_text_atoms_v2' "
        "AND $1->>'model_format' = 'onnx' "
        "AND jsonb_typeof($1->'latent_dims') = 'number' "
        "AND jsonb_typeof($1->'index_compatibility') = 'object' "
        "AND (NOT ($1 ? 'artifacts') "
        "OR (jsonb_typeof($1->'artifacts') = 'object' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM jsonb_each($1->'artifacts') AS artifact(key, value) "
        "WHERE jsonb_typeof(value) <> 'object' "
        "OR NULLIF(value->>'path', '') IS NULL "
        "OR NULLIF(value->>'sha256', '') IS NULL))) "
        "AND jsonb_typeof($1->'runtime_io') = 'object' "
        "AND NULLIF($1#>>'{runtime_io,input_ids}', '') IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_io,attention_mask}', '') IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_io,semantic_ids}', '') IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_io,semantic_weights}', '') IS NOT NULL "
        "AND jsonb_typeof($1->'runtime_output') = 'object' "
        "AND $1#>>'{runtime_output,atom_id_dtype}' = 'int64' "
        "AND $1#>>'{runtime_output,atom_weight_dtype}' = 'float32' "
        "AND NULLIF($1#>>'{runtime_output,max_atoms}', '') IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_output,query_semantic_max_atoms}', '') "
        "IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_output,document_semantic_max_atoms}', '') "
        "IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_output,lexical_dims}', '') IS NOT NULL "
        "AND NULLIF($1#>>'{runtime_output,semantic_dims}', '') IS NOT NULL "
        "AND NULLIF("
        "$1#>>'{runtime_output,document_semantic_budget_ratio}', '') "
        "IS NOT NULL "
        "AND jsonb_typeof($1->'artifacts') = 'object' "
        "AND jsonb_typeof($1#>'{artifacts,scoring_profile}') = 'object' "
        "AND jsonb_typeof($1#>'{artifacts,query_encoder}') = 'object' "
        "AND jsonb_typeof($1#>'{artifacts,document_encoder}') = 'object' "
        "AND jsonb_typeof($1#>'{artifacts,tokenizer_vocabulary}') "
        "= 'object' "
        "AND jsonb_typeof($1#>'{artifacts,tokenizer_merges}') "
        "= 'object' "
        "AND jsonb_typeof($1#>'{artifacts,lexical_vocabulary}') "
        "= 'object' "
        "AND jsonb_typeof($1#>'{artifacts,query_calibration_runtime}') "
        "= 'object' "
        "AND jsonb_typeof($1#>'{artifacts,atom_space}') = 'object' "
        "AND NULLIF("
        "$1#>>'{index_compatibility,atom_space}', '') IS NOT NULL "
        "AND NULLIF("
        "$1#>>'{index_compatibility,scoring}', '') IS NOT NULL, "
        "false)";

    if (model_id != NULL)
    {
        values[1] = PointerGetDatum(model_id);
        nulls[1] = ' ';
    }

    spi_result = SPI_connect();
    if (spi_result != SPI_OK_CONNECT)
    {
        ereport(ERROR, (errmsg("could not connect to SPI")));
    }
    spi_result = SPI_execute_with_args(
        query,
        2,
        argtypes,
        values,
        nulls,
        true,
        1);
    if (spi_result < 0 || SPI_processed != 1)
    {
        SPI_finish();
        ereport(ERROR, (errmsg("could not validate evoke model manifest")));
    }
    valid_datum = SPI_getbinval(
        SPI_tuptable->vals[0],
        SPI_tuptable->tupdesc,
        1,
        &isnull);
    valid = !isnull && DatumGetBool(valid_datum);
    SPI_finish();

    if (!valid)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke model manifest"),
                errdetail(
                    "manifest.json must use schema_version=1, "
                    "api_version=evoke_model_v1, and contain model_id, "
                    "encoder_type, runtime, latent_dims, and no "
                    "runtime_parameters field. It must contain "
                    "index_compatibility.atom_space/scoring. Manifests must "
                    "use runtime=onnxruntime, "
                    "runtime_abi=evoke_p2_unified_text_atoms_v2, "
                    "model_format=onnx, provide the current runtime IO, "
                    "tokenizer/compiler artifacts, query and document "
                    "encoder artifacts, and "
                    "scoring_profile artifacts. Registered model_id "
                    "must match manifest model_id.")
            )
        );
    }
}

static void
evoke_checkout_validate_manifest(
    Datum manifest,
    text *model_id,
    const char *model_path
)
{
    evoke_checkout_validate_manifest_shape(manifest, model_id, model_path);
    evoke_checkout_validate_manifest_artifacts(manifest, model_path);
}

static void
evoke_checkout_manifest_signature(
    Datum manifest,
    char signature_out[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1]
)
{
    char *canonical_manifest;
    pg_cryptohash_ctx *ctx;
    uint8 digest[PG_SHA256_DIGEST_LENGTH];
    const char *detail;
    static const char hex_chars[] = "0123456789abcdef";

    if (signature_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke model signature output is required")));
    }

    canonical_manifest = DatumGetCString(DirectFunctionCall1(
        jsonb_out,
        manifest
    ));
    ctx = pg_cryptohash_create(PG_SHA256);
    if (ctx == NULL ||
        pg_cryptohash_init(ctx) < 0 ||
        pg_cryptohash_update(
            ctx,
            (const uint8 *) canonical_manifest,
            strlen(canonical_manifest)
        ) < 0 ||
        pg_cryptohash_final(ctx, digest, sizeof(digest)) < 0)
    {
        detail = ctx == NULL ? NULL : pg_cryptohash_error(ctx);
        if (ctx != NULL)
        {
            pg_cryptohash_free(ctx);
        }
        pfree(canonical_manifest);
        ereport(
            ERROR,
            (
                errmsg("could not identify evoke model checkout"),
                errdetail(
                    "%s",
                    detail == NULL ? "unknown error" : detail
                )
            )
        );
    }
    pg_cryptohash_free(ctx);

    for (int i = 0; i < PG_SHA256_DIGEST_LENGTH; i++)
    {
        signature_out[i * 2] = hex_chars[digest[i] >> 4];
        signature_out[i * 2 + 1] = hex_chars[digest[i] & 0x0f];
    }
    signature_out[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN] = '\0';
    pfree(canonical_manifest);
}

void
evoke_checkout_signature(
    const char *model_path,
    char signature_out[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1]
)
{
    char *manifest_text;
    Datum manifest;

    if (model_path == NULL || model_path[0] == '\0')
    {
        ereport(ERROR, (errmsg("evoke model checkout path is required")));
    }
    if (signature_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke model signature output is required")));
    }

    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    /*
     * The manifest already binds every runtime artifact by SHA-256.  Runtime
     * workers verify those files before inference; generation readers only
     * need the manifest identity and must not re-hash a large model once per
     * PostgreSQL backend.
     */
    evoke_checkout_validate_manifest_shape(manifest, NULL, model_path);
    evoke_checkout_manifest_signature(manifest, signature_out);
    pfree(DatumGetPointer(manifest));
    pfree(manifest_text);
}

void
evoke_checkout_unified_contract(
    const char *model_path,
    uint32 *lexical_dims_out,
    uint32 *total_dims_out,
    float4 *semantic_budget_ratio_out
)
{
    char *manifest_text;
    Datum manifest;
    int32 lexical_dims;
    int32 semantic_dims;
    double semantic_budget_ratio;

    if (model_path == NULL || model_path[0] == '\0' ||
        lexical_dims_out == NULL || total_dims_out == NULL ||
        semantic_budget_ratio_out == NULL)
    {
        ereport(
            ERROR,
            (errmsg("evoke unified model contract output is required"))
        );
    }

    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    evoke_checkout_validate_manifest_shape(manifest, NULL, model_path);
    lexical_dims = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        "lexical_dims",
        1,
        1,
        PG_INT32_MAX
    );
    semantic_dims = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        "semantic_dims",
        1,
        1,
        PG_INT32_MAX
    );
    if (lexical_dims > PG_INT32_MAX - semantic_dims)
    {
        pfree(manifest_text);
        ereport(
            ERROR,
            (errmsg("evoke unified model dimensions overflow"))
        );
    }
    semantic_budget_ratio = evoke_checkout_jsonb_double_path(
        manifest,
        "runtime_output",
        "document_semantic_budget_ratio",
        (double) FLT_MIN,
        (double) FLT_MAX
    );

    *lexical_dims_out = (uint32) lexical_dims;
    *total_dims_out = (uint32) (lexical_dims + semantic_dims);
    *semantic_budget_ratio_out = (float4) semantic_budget_ratio;
    pfree(DatumGetPointer(manifest));
    pfree(manifest_text);
}

PG_FUNCTION_INFO_V1(evoke_onnxruntime_probe);
Datum
evoke_onnxruntime_probe(PG_FUNCTION_ARGS)
{
    const char *env_path = getenv("EVOKE_ONNXRUNTIME_LIBRARY");
    const char *candidates[] = {
        "libonnxruntime.so.1",
        "libonnxruntime.so",
        "libonnxruntime.1.dylib",
        "libonnxruntime.dylib",
        NULL
    };
    const char *last_error = NULL;

#ifdef EVOKE_ENABLE_ONNXRUNTIME
    const OrtApiBase *linked_api_base = OrtGetApiBase();

    if (linked_api_base != NULL && linked_api_base->GetVersionString != NULL)
    {
        const char *version = linked_api_base->GetVersionString();

        PG_RETURN_TEXT_P(cstring_to_text(psprintf(
            "available:linked:version=%s",
            version == NULL ? "unknown" : version
        )));
    }
#endif

    if (env_path != NULL && env_path[0] != '\0')
    {
        void *handle;
        EvokeOrtGetApiBaseFn get_api_base;
        const EvokeOrtApiBase *api_base;
        const char *version;

        dlerror();
        handle = dlopen(env_path, RTLD_LAZY | RTLD_LOCAL);
        if (handle != NULL)
        {
            get_api_base = (EvokeOrtGetApiBaseFn) dlsym(
                handle,
                "OrtGetApiBase");
            if (get_api_base == NULL)
            {
                last_error = dlerror();
                dlclose(handle);
                PG_RETURN_TEXT_P(cstring_to_text(psprintf(
                    "missing:%s:OrtGetApiBase:%s",
                    env_path,
                    last_error == NULL ? "symbol not found" : last_error)));
            }
            api_base = get_api_base();
            if (api_base == NULL || api_base->GetVersionString == NULL)
            {
                dlclose(handle);
                PG_RETURN_TEXT_P(cstring_to_text(psprintf(
                    "missing:%s:OrtApiBase incomplete",
                    env_path)));
            }
            version = api_base->GetVersionString();
            dlclose(handle);
            PG_RETURN_TEXT_P(cstring_to_text(psprintf(
                "available:%s:version=%s",
                env_path,
                version == NULL ? "unknown" : version)));
        }
        last_error = dlerror();
    }

    for (int i = 0; candidates[i] != NULL; i++)
    {
        const char *candidate = candidates[i];
        void *handle;
        EvokeOrtGetApiBaseFn get_api_base;
        const EvokeOrtApiBase *api_base;
        const char *version;

        dlerror();
        handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
        if (handle != NULL)
        {
            get_api_base = (EvokeOrtGetApiBaseFn) dlsym(
                handle,
                "OrtGetApiBase");
            if (get_api_base == NULL)
            {
                last_error = dlerror();
                dlclose(handle);
                continue;
            }
            api_base = get_api_base();
            if (api_base == NULL || api_base->GetVersionString == NULL)
            {
                last_error = "OrtApiBase incomplete";
                dlclose(handle);
                continue;
            }
            version = api_base->GetVersionString();
            dlclose(handle);
            PG_RETURN_TEXT_P(cstring_to_text(psprintf(
                "available:%s:version=%s",
                candidate,
                version == NULL ? "unknown" : version)));
        }

        last_error = dlerror();
    }

    PG_RETURN_TEXT_P(cstring_to_text(psprintf(
        "missing:%s",
        last_error == NULL ? "no candidate library loaded" : last_error)));
}

PG_FUNCTION_INFO_V1(evoke_onnxruntime_build_info);
Datum
evoke_onnxruntime_build_info(PG_FUNCTION_ARGS)
{
#ifdef EVOKE_ENABLE_ONNXRUNTIME
    PG_RETURN_TEXT_P(cstring_to_text(psprintf(
        "enabled:api=%d",
        ORT_API_VERSION)));
#else
    PG_RETURN_TEXT_P(cstring_to_text("disabled"));
#endif
}

static double
evoke_monotonic_ms(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        return 0.0;
    }
    return ((double) ts.tv_sec * 1000.0)
        + ((double) ts.tv_nsec / 1000000.0);
}

static void
evoke_append_json_string(StringInfo out, const char *value)
{
    const unsigned char *ptr = (const unsigned char *) value;

    appendStringInfoChar(out, '"');
    while (*ptr != '\0')
    {
        if (*ptr == '"' || *ptr == '\\')
        {
            appendStringInfoChar(out, '\\');
            appendStringInfoChar(out, (char) *ptr);
        }
        else if (*ptr == '\n')
        {
            appendStringInfoString(out, "\\n");
        }
        else if (*ptr == '\r')
        {
            appendStringInfoString(out, "\\r");
        }
        else if (*ptr == '\t')
        {
            appendStringInfoString(out, "\\t");
        }
        else if (*ptr < 0x20)
        {
            appendStringInfo(out, "\\u%04x", (unsigned int) *ptr);
        }
        else
        {
            appendStringInfoChar(out, (char) *ptr);
        }
        ptr++;
    }
    appendStringInfoChar(out, '"');
}

static int
evoke_runtime_effective_max_batch_size(void)
{
    return (int) Min(
        Max(evoke_runtime_max_batch_size, 1),
        EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
    );
}

static int
evoke_runtime_effective_intra_op_threads(void)
{
    long processor_count;
    long thread_count;

    if (evoke_onnxruntime_intra_op_threads > 0)
    {
        return evoke_onnxruntime_intra_op_threads;
    }
    if (evoke_runtime_worker_count <= 1)
    {
        return 0;
    }

    processor_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (processor_count <= 0)
    {
        return 1;
    }

    /*
     * Multiple unrestricted ORT pools oversubscribe the host. Reserve half
     * of the online CPUs for PostgreSQL and divide the other half between
     * inference workers. Explicit GUC values always override this policy.
     */
    thread_count = processor_count / (2L * evoke_runtime_worker_count);
    return (int) Min(Max(thread_count, 1L), 16L);
}

static void
evoke_runtime_record_worker_termination_locked(
    evoke_runtime_service_worker *worker
)
{
    uint32 reason;

    if (worker == NULL || worker->terminate_requested)
    {
        return;
    }
    reason = pg_atomic_read_u32(&worker->termination_reason);
    if (reason == EVOKE_RUNTIME_TERMINATION_NONE)
    {
        return;
    }

    worker->terminate_requested = true;
    worker->runtime_terminations++;
    worker->last_progress_at = GetCurrentTimestamp();
    evoke_runtime_service->runtime_terminations++;
    if (reason == EVOKE_RUNTIME_TERMINATION_LIVENESS)
    {
        worker->runtime_liveness_timeouts++;
        evoke_runtime_service->runtime_liveness_timeouts++;
    }
}

static void
evoke_runtime_precision_normalize(
    const char *runtime_precision,
    char out[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES]
)
{
    size_t len;

    if (runtime_precision == NULL || runtime_precision[0] == '\0')
    {
        runtime_precision = EVOKE_RUNTIME_PRECISION_FP16;
    }
    len = strlen(runtime_precision);
    if (len == 0 ||
        len >= EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("unsupported evoke runtime precision"),
                errdetail("runtime_precision=%s", runtime_precision)
            )
        );
    }
    for (size_t i = 0; i <= len; i++)
    {
        out[i] = (char) tolower((unsigned char) runtime_precision[i]);
    }
    if (strcmp(out, EVOKE_RUNTIME_PRECISION_FP16) != 0 &&
        strcmp(out, EVOKE_RUNTIME_PRECISION_FP32) != 0)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                errmsg("unsupported evoke runtime precision"),
                errdetail(
                    "runtime_precision must be one of fp16 or fp32, got %s",
                    out
                )
            )
        );
    }
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static uint64 evoke_ort_error_count = 0;
static void evoke_ort_prepare_process(void);

static void
evoke_ort_check(const OrtApi *ort, OrtStatus *status, const char *step)
{
    const char *message;
    char *message_copy;

    if (status == NULL)
    {
        return;
    }
    evoke_ort_error_count++;
    message = ort->GetErrorMessage(status);
    message_copy = pstrdup(message == NULL ? "unknown ONNX Runtime error" : message);
    ort->ReleaseStatus(status);
    ereport(
        ERROR,
        (
            errmsg("ONNX Runtime execution failed during %s", step),
            errdetail("%s", message_copy)
        )
    );
}

#define EVOKE_ORT_TERMINATE_GRACE_MS 5000.0

typedef struct EvokeOrtRunMonitor
{
    pg_atomic_uint32 done;
    const OrtApi *ort;
    OrtRunOptions *run_options;
    pg_atomic_uint32 *cancel_flag;
    pg_atomic_uint32 *termination_reason;
    OrtStatus *terminate_status;
    int liveness_timeout_ms;
    bool cancel_requested;
    bool liveness_timeout;
    bool proc_die_requested;
    bool terminate_requested;
    bool termination_grace_expired;
} EvokeOrtRunMonitor;

static double
evoke_ort_monitor_monotonic_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    {
        return 0.0;
    }
    return (double) now.tv_sec * 1000.0 +
        (double) now.tv_nsec / 1000000.0;
}

static void *
evoke_ort_run_monitor_main(void *user_data)
{
    EvokeOrtRunMonitor *monitor = user_data;
    double started_ms = evoke_ort_monitor_monotonic_ms();
    double terminate_started_ms = 0.0;
    struct timespec pause = {
        .tv_sec = 0,
        .tv_nsec = 10000000L
    };

    while (pg_atomic_read_u32(&monitor->done) == 0)
    {
        double now_ms = evoke_ort_monitor_monotonic_ms();
        bool cancel_requested =
            monitor->cancel_flag != NULL &&
            pg_atomic_read_u32(monitor->cancel_flag) != 0;
        bool liveness_timeout =
            monitor->liveness_timeout_ms > 0 &&
            now_ms - started_ms >=
                (double) monitor->liveness_timeout_ms;
        bool proc_die_requested = ProcDiePending;
        uint32 termination_reason = EVOKE_RUNTIME_TERMINATION_NONE;

        if (!monitor->terminate_requested &&
            (
                cancel_requested ||
                liveness_timeout ||
                proc_die_requested
            ))
        {
            if (cancel_requested)
            {
                monitor->cancel_requested = true;
                termination_reason = EVOKE_RUNTIME_TERMINATION_CANCEL;
            }
            else if (liveness_timeout)
            {
                monitor->liveness_timeout = true;
                termination_reason = EVOKE_RUNTIME_TERMINATION_LIVENESS;
            }
            else
            {
                monitor->proc_die_requested = true;
                termination_reason = EVOKE_RUNTIME_TERMINATION_PROC_DIE;
            }
            if (monitor->termination_reason != NULL)
            {
                pg_atomic_write_u32(
                    monitor->termination_reason,
                    termination_reason
                );
            }
            monitor->terminate_status =
                monitor->ort->RunOptionsSetTerminate(
                    monitor->run_options
                );
            monitor->terminate_requested = true;
            terminate_started_ms = now_ms;
        }
        if (monitor->terminate_requested &&
            !monitor->termination_grace_expired &&
            now_ms - terminate_started_ms >=
                EVOKE_ORT_TERMINATE_GRACE_MS)
        {
            /*
             * Do not hard-exit from this fixed background worker. A non-zero
             * worker exit is treated as a crash by the postmaster and forces
             * cluster-wide recovery, which is worse than one stuck runtime
             * lane. RunOptionsSetTerminate remains the cooperative boundary;
             * if the provider does not return, other runtime workers continue
             * serving requests and status exposes the termination reason.
             */
            monitor->termination_grace_expired = true;
        }
        (void) nanosleep(&pause, NULL);
    }
    return NULL;
}

static void
evoke_runtime_note_termination(void)
{
    evoke_runtime_service_worker *worker;

    if (evoke_runtime_active_request_id == 0 ||
        evoke_runtime_active_worker_id >=
            EVOKE_RUNTIME_SERVICE_MAX_WORKERS ||
        evoke_runtime_service == NULL ||
        evoke_runtime_service_lock == NULL)
    {
        return;
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    if (
        evoke_runtime_active_worker_id <
            evoke_runtime_service_configured_workers()
    )
    {
        worker = &evoke_runtime_service->workers[
            evoke_runtime_active_worker_id];
        if (worker->processing &&
            worker->processing_request_id ==
                evoke_runtime_active_request_id)
        {
            evoke_runtime_record_worker_termination_locked(worker);
        }
    }
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_ort_run_interruptible(
    const OrtApi *ort,
    OrtSession *session,
    const char *const *input_names,
    const OrtValue *const *inputs,
    size_t input_count,
    const char *const *output_names,
    size_t output_count,
    OrtValue **outputs,
    const char *step
)
{
    EvokeOrtRunMonitor monitor = {0};
    OrtRunOptions *run_options = NULL;
    pthread_t monitor_thread;
    OrtStatus *status;
    int thread_status;

    pg_atomic_init_u32(&monitor.done, 0);
    status = ort->CreateRunOptions(&run_options);
    evoke_ort_check(ort, status, "CreateRunOptions");
    monitor.ort = ort;
    monitor.run_options = run_options;
    monitor.liveness_timeout_ms = evoke_runtime_liveness_timeout_ms;
    if (evoke_runtime_active_worker_id <
            EVOKE_RUNTIME_SERVICE_MAX_WORKERS &&
        evoke_runtime_service != NULL)
    {
        monitor.cancel_flag =
            &evoke_runtime_service->workers[
                evoke_runtime_active_worker_id].cancel_requested;
        monitor.termination_reason =
            &evoke_runtime_service->workers[
                evoke_runtime_active_worker_id].termination_reason;
    }
    thread_status = pthread_create(
        &monitor_thread,
        NULL,
        evoke_ort_run_monitor_main,
        &monitor
    );
    if (thread_status != 0)
    {
        ort->ReleaseRunOptions(run_options);
        ereport(
            ERROR,
            (
                errmsg("could not start isolated ONNX Runtime execution"),
                errdetail("pthread_create failed: %s", strerror(thread_status))
            )
        );
    }
    status = ort->Run(
        session,
        run_options,
        input_names,
        inputs,
        input_count,
        output_names,
        output_count,
        outputs
    );
    pg_atomic_write_u32(&monitor.done, 1);
    thread_status = pthread_join(monitor_thread, NULL);
    ort->ReleaseRunOptions(run_options);
    if (thread_status != 0)
    {
        ereport(
            ERROR,
            (
                errmsg("could not join isolated ONNX Runtime execution"),
                errdetail("pthread_join failed: %s", strerror(thread_status))
            )
        );
    }
    if (monitor.terminate_requested)
    {
        evoke_runtime_note_termination();
    }
    if (monitor.terminate_status != NULL)
    {
        if (status != NULL)
        {
            ort->ReleaseStatus(status);
        }
        evoke_ort_check(
            ort,
            monitor.terminate_status,
            "RunOptionsSetTerminate"
        );
    }
    if (monitor.cancel_requested)
    {
        if (status != NULL)
        {
            ort->ReleaseStatus(status);
        }
        ereport(ERROR, (errmsg("evoke runtime request was canceled")));
    }
    if (monitor.liveness_timeout)
    {
        if (status != NULL)
        {
            ort->ReleaseStatus(status);
        }
        ereport(
            ERROR,
            (
                errmsg("evoke ONNX Runtime execution exceeded liveness limit"),
                errdetail(
                    "limit_ms=%d",
                    evoke_runtime_liveness_timeout_ms
                )
            )
        );
    }
    if (monitor.proc_die_requested)
    {
        if (status != NULL)
        {
            ort->ReleaseStatus(status);
        }
        CHECK_FOR_INTERRUPTS();
    }
    evoke_ort_check(ort, status, step);
    CHECK_FOR_INTERRUPTS();
}

#define EVOKE_ORT_SESSION_CACHE_LIMIT 16

typedef struct EvokeOrtSessionCacheEntry
{
    char *model_path;
    char *encoder_path;
    char *requested_provider;
    char *active_provider;
    char *runtime_precision;
    bool document_mode;
    char checkout_signature[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1];
    OrtEnv *env;
    OrtSessionOptions *session_options;
    OrtSession *session;
    double last_load_ms;
    double last_run_ms;
    size_t last_atom_count;
    TimestampTz loaded_at;
    TimestampTz last_used_at;
    uint64 access_seq;
} EvokeOrtSessionCacheEntry;

typedef struct EvokeOrtSessionCache
{
    EvokeOrtSessionCacheEntry entries[EVOKE_ORT_SESSION_CACHE_LIMIT];
    int entry_count;
    uint64 access_seq;
    uint64 hits;
    uint64 misses;
    uint64 loads;
    uint64 evictions;
    double last_load_ms;
    double last_run_ms;
    size_t last_atom_count;
    TimestampTz loaded_at;
    TimestampTz last_used_at;
    char last_requested_provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char last_active_provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char last_runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
} EvokeOrtSessionCache;

typedef struct EvokeOrtSessionBuild
{
    OrtEnv *env;
    OrtSessionOptions *session_options;
    OrtSession *session;
    char *model_path;
    char *encoder_path;
    char *requested_provider;
    char *active_provider;
    char *runtime_precision;
} EvokeOrtSessionBuild;

typedef struct EvokeOrtRunResources
{
    const OrtApi *ort;
    OrtEnv *owned_env;
    OrtSessionOptions *owned_session_options;
    OrtSession *owned_session;
    OrtMemoryInfo *memory_info;
    OrtValue *inputs[2];
    OrtValue *outputs[2];
    OrtTensorTypeAndShapeInfo *atom_shape;
    OrtTensorTypeAndShapeInfo *weight_shape;
} EvokeOrtRunResources;

static EvokeOrtSessionCache evoke_ort_session_cache = {0};
static bool evoke_ort_cache_exit_registered = false;

static void
evoke_ort_release_session_build(
    const OrtApi *ort,
    EvokeOrtSessionBuild *build
)
{
    if (build->session != NULL)
    {
        ort->ReleaseSession(build->session);
    }
    if (build->session_options != NULL)
    {
        ort->ReleaseSessionOptions(build->session_options);
    }
    if (build->env != NULL)
    {
        ort->ReleaseEnv(build->env);
    }
    if (build->model_path != NULL)
    {
        pfree(build->model_path);
    }
    if (build->encoder_path != NULL)
    {
        pfree(build->encoder_path);
    }
    if (build->requested_provider != NULL)
    {
        pfree(build->requested_provider);
    }
    if (build->active_provider != NULL)
    {
        pfree(build->active_provider);
    }
    if (build->runtime_precision != NULL)
    {
        pfree(build->runtime_precision);
    }
}

static void
evoke_ort_release_run_resources(EvokeOrtRunResources *resources)
{
    const OrtApi *ort = resources->ort;

    if (resources->atom_shape != NULL)
    {
        ort->ReleaseTensorTypeAndShapeInfo(resources->atom_shape);
        resources->atom_shape = NULL;
    }
    if (resources->weight_shape != NULL)
    {
        ort->ReleaseTensorTypeAndShapeInfo(resources->weight_shape);
        resources->weight_shape = NULL;
    }
    for (int i = 0; i < 2; i++)
    {
        if (resources->outputs[i] != NULL)
        {
            ort->ReleaseValue(resources->outputs[i]);
            resources->outputs[i] = NULL;
        }
        if (resources->inputs[i] != NULL)
        {
            ort->ReleaseValue(resources->inputs[i]);
            resources->inputs[i] = NULL;
        }
    }
    if (resources->memory_info != NULL)
    {
        ort->ReleaseMemoryInfo(resources->memory_info);
        resources->memory_info = NULL;
    }
    if (resources->owned_session != NULL)
    {
        ort->ReleaseSession(resources->owned_session);
        resources->owned_session = NULL;
    }
    if (resources->owned_session_options != NULL)
    {
        ort->ReleaseSessionOptions(resources->owned_session_options);
        resources->owned_session_options = NULL;
    }
    if (resources->owned_env != NULL)
    {
        ort->ReleaseEnv(resources->owned_env);
        resources->owned_env = NULL;
    }
}

static const OrtApi *
evoke_ort_api(void)
{
    const OrtApiBase *api_base;
    const OrtApi *ort;

    evoke_ort_prepare_process();
    api_base = OrtGetApiBase();
    if (api_base == NULL)
    {
        ereport(ERROR, (errmsg("ONNX Runtime API base is unavailable")));
    }
    ort = api_base->GetApi(ORT_API_VERSION);
    if (ort == NULL)
    {
        ereport(ERROR, (errmsg("ONNX Runtime API is unavailable")));
    }
    return ort;
}

static void
evoke_ort_cache_release_entry(
    const OrtApi *ort,
    EvokeOrtSessionCacheEntry *entry)
{
    if (entry->session != NULL)
    {
        ort->ReleaseSession(entry->session);
    }
    if (entry->session_options != NULL)
    {
        ort->ReleaseSessionOptions(entry->session_options);
    }
    if (entry->env != NULL)
    {
        ort->ReleaseEnv(entry->env);
    }

    entry->session = NULL;
    entry->session_options = NULL;
    entry->env = NULL;
    if (entry->model_path != NULL)
    {
        pfree(entry->model_path);
    }
    if (entry->encoder_path != NULL)
    {
        pfree(entry->encoder_path);
    }
    if (entry->requested_provider != NULL)
    {
        pfree(entry->requested_provider);
    }
    if (entry->active_provider != NULL)
    {
        pfree(entry->active_provider);
    }
    if (entry->runtime_precision != NULL)
    {
        pfree(entry->runtime_precision);
    }
    memset(entry, 0, sizeof(EvokeOrtSessionCacheEntry));
}

static void
evoke_ort_cache_remove_entry(const OrtApi *ort, int entry_index)
{
    Assert(entry_index >= 0);
    Assert(entry_index < evoke_ort_session_cache.entry_count);

    evoke_ort_cache_release_entry(
        ort,
        &evoke_ort_session_cache.entries[entry_index]);
    if (entry_index + 1 < evoke_ort_session_cache.entry_count)
    {
        memmove(
            &evoke_ort_session_cache.entries[entry_index],
            &evoke_ort_session_cache.entries[entry_index + 1],
            sizeof(EvokeOrtSessionCacheEntry)
                * (size_t) (
                    evoke_ort_session_cache.entry_count - entry_index - 1));
    }
    evoke_ort_session_cache.entry_count--;
    memset(
        &evoke_ort_session_cache.entries[evoke_ort_session_cache.entry_count],
        0,
        sizeof(EvokeOrtSessionCacheEntry));
}

static int
evoke_ort_effective_cache_size(void)
{
    if (evoke_onnxruntime_session_cache_size <= 0)
    {
        return 0;
    }
    if (evoke_onnxruntime_session_cache_size > EVOKE_ORT_SESSION_CACHE_LIMIT)
    {
        return EVOKE_ORT_SESSION_CACHE_LIMIT;
    }
    return evoke_onnxruntime_session_cache_size;
}

static int
evoke_ort_cache_lru_index(void)
{
    int lru_index = 0;

    Assert(evoke_ort_session_cache.entry_count > 0);
    for (int i = 1; i < evoke_ort_session_cache.entry_count; i++)
    {
        if (evoke_ort_session_cache.entries[i].access_seq
            < evoke_ort_session_cache.entries[lru_index].access_seq)
        {
            lru_index = i;
        }
    }
    return lru_index;
}

static void
evoke_ort_cache_apply_size_limit(const OrtApi *ort, int max_sessions)
{
    while (evoke_ort_session_cache.entry_count > max_sessions)
    {
        evoke_ort_session_cache.evictions++;
        evoke_ort_cache_remove_entry(ort, evoke_ort_cache_lru_index());
    }
}

static void
evoke_ort_cache_shmem_exit(int code, Datum arg)
{
    const OrtApi *ort;

    (void) code;
    (void) arg;
    if (ProcDiePending)
    {
        /*
         * ORT teardown is not signal-safe. A runtime worker exits as a
         * whole process, so the kernel reclaims its cache after SIGTERM.
         * Calling ReleaseSession/ReleaseEnv here has caused postmaster stop
         * to become crash recovery instead of a clean restart.
         */
        return;
    }
    if (evoke_ort_session_cache.entry_count == 0)
    {
        return;
    }
    ort = evoke_ort_api();
    evoke_ort_cache_apply_size_limit(ort, 0);
}

static void
evoke_ort_prepare_process(void)
{
    const char *disable_telemetry = getenv("ORT_DISABLE_TELEMETRY");
#ifdef __APPLE__
    const char *activity_mode;
#endif

    /*
     * ORT 1.29 can include POSIX telemetry. Runtime workers must remain
     * self-contained data-plane processes, so disable it by default before
     * the first ORT API or environment initialization. An explicit operator
     * value remains authoritative.
     */
    if (disable_telemetry == NULL || disable_telemetry[0] == '\0')
    {
        if (setenv("ORT_DISABLE_TELEMETRY", "1", 0) != 0)
        {
            ereport(
                WARNING,
                (
                    errmsg("could not disable ONNX Runtime telemetry"),
                    errdetail("%m")
                )
            );
        }
    }
#ifdef __APPLE__
    activity_mode = getenv("OS_ACTIVITY_MODE");

    /*
     * CoreML compilation reports metrics through CoreAnalytics/os_log. In a
     * PostgreSQL background worker on macOS that path can crash inside Apple's
     * fork-sensitive logging initialization before ORT can return an error.
     * Runtime workers do not need unified activity logging, so disable it
     * before the first CoreML provider/session initialization in the process.
     */
    if (activity_mode == NULL || activity_mode[0] == '\0')
    {
        if (setenv("OS_ACTIVITY_MODE", "disable", 0) != 0)
        {
            ereport(
                WARNING,
                (
                    errmsg("could not disable CoreML activity logging"),
                    errdetail("%m")
                )
            );
        }
    }
#endif
}

static void
evoke_ort_record_provider(
    const char *requested_provider,
    const char *active_provider,
    const char *runtime_precision
)
{
    strlcpy(
        evoke_ort_session_cache.last_requested_provider,
        requested_provider,
        sizeof(evoke_ort_session_cache.last_requested_provider));
    strlcpy(
        evoke_ort_session_cache.last_active_provider,
        active_provider,
        sizeof(evoke_ort_session_cache.last_active_provider));
    strlcpy(
        evoke_ort_session_cache.last_runtime_precision,
        runtime_precision,
        sizeof(evoke_ort_session_cache.last_runtime_precision));
}

static bool
evoke_ort_provider_available(const OrtApi *ort, const char *ort_provider_name)
{
    OrtStatus *status;
    char **providers = NULL;
    int provider_count = 0;
    bool available = false;

    status = ort->GetAvailableProviders(&providers, &provider_count);
    if (status != NULL)
    {
        const char *message = ort->GetErrorMessage(status);

        ereport(
            DEBUG1,
            (
                errmsg("could not inspect ONNX Runtime execution providers"),
                errdetail(
                    "%s",
                    message == NULL ? "unknown ONNX Runtime error" : message)
            )
        );
        ort->ReleaseStatus(status);
        return false;
    }

    for (int i = 0; i < provider_count; i++)
    {
        if (strcmp(providers[i], ort_provider_name) == 0)
        {
            available = true;
            break;
        }
    }
    status = ort->ReleaseAvailableProviders(providers, provider_count);
    if (status != NULL)
    {
        ort->ReleaseStatus(status);
    }
    return available;
}

static void
evoke_ort_append_available_providers_json(const OrtApi *ort, StringInfo out)
{
    OrtStatus *status;
    char **providers = NULL;
    int provider_count = 0;

    status = ort->GetAvailableProviders(&providers, &provider_count);
    if (status != NULL)
    {
        const char *message = ort->GetErrorMessage(status);

        appendStringInfoString(out, "[]");
        ereport(
            DEBUG1,
            (
                errmsg("could not inspect ONNX Runtime execution providers"),
                errdetail(
                    "%s",
                    message == NULL ? "unknown ONNX Runtime error" : message)
            )
        );
        ort->ReleaseStatus(status);
        return;
    }

    appendStringInfoChar(out, '[');
    for (int i = 0; i < provider_count; i++)
    {
        if (i > 0)
        {
            appendStringInfoChar(out, ',');
        }
        evoke_append_json_string(out, providers[i]);
    }
    appendStringInfoChar(out, ']');
    status = ort->ReleaseAvailableProviders(providers, provider_count);
    if (status != NULL)
    {
        ort->ReleaseStatus(status);
    }
}

static char *
evoke_ort_auto_provider(
    const OrtApi *ort,
    const char *runtime_precision)
{
    if (strcmp(runtime_precision, EVOKE_RUNTIME_PRECISION_FP16) == 0)
    {
        if (evoke_ort_provider_available(ort, "TensorrtExecutionProvider"))
        {
            return "tensorrt";
        }
        if (evoke_ort_provider_available(ort, "CoreMLExecutionProvider"))
        {
            return "coreml";
        }
        if (evoke_ort_provider_available(ort, "CUDAExecutionProvider"))
        {
            return "cuda";
        }
        return "cpu";
    }

    if (evoke_ort_provider_available(ort, "CUDAExecutionProvider"))
    {
        return "cuda";
    }
    if (evoke_ort_provider_available(ort, "TensorrtExecutionProvider"))
    {
        return "tensorrt";
    }
    if (evoke_ort_provider_available(ort, "CoreMLExecutionProvider"))
    {
        return "coreml";
    }
    return "cpu";
}

static char *
evoke_p2_ort_active_provider(
    const OrtApi *ort,
    Datum semantic_runtime,
    const char *runtime_precision)
{
    char *validated_provider = evoke_checkout_jsonb_text_path(
        semantic_runtime,
        "validation",
        "provider",
        "CPUExecutionProvider"
    );

    if (strcmp(validated_provider, "CPUExecutionProvider") != 0 &&
        strcmp(validated_provider, "CUDAExecutionProvider") != 0 &&
        strcmp(validated_provider, "CoreMLExecutionProvider") != 0)
    {
        ereport(
            ERROR,
            (
                errmsg("unsupported P2 validation provider"),
                errdetail(
                    "semantic_runtime.validation.provider=%s",
                    validated_provider)
            )
        );
    }

    /*
     * P2 validation records the reference provider used by the exporter, not
     * the only provider a deployment may use. Prefer available GPU execution
     * providers by default, then fall back to CPU. Serious provider
     * incompatibilities still fail closed during provider attachment, session
     * creation, runtime execution, or native index qualification.
     */
    return evoke_ort_auto_provider(ort, runtime_precision);
}

static void
evoke_ort_append_cuda_provider(
    const OrtApi *ort,
    OrtSessionOptions *session_options)
{
    OrtCUDAProviderOptionsV2 *cuda_options = NULL;
    char device_id[16] = "0";
    const char *keys[1] = {"device_id"};
    const char *values[1] = {device_id};

    evoke_ort_check(
        ort,
        ort->CreateCUDAProviderOptions(&cuda_options),
        "CreateCUDAProviderOptions");
    PG_TRY();
    {
        evoke_ort_check(
            ort,
            ort->UpdateCUDAProviderOptions(cuda_options, keys, values, 1),
            "UpdateCUDAProviderOptions");
        evoke_ort_check(
            ort,
            ort->SessionOptionsAppendExecutionProvider_CUDA_V2(
                session_options,
                cuda_options),
            "SessionOptionsAppendExecutionProvider_CUDA_V2");
    }
    PG_FINALLY();
    {
        if (cuda_options != NULL)
        {
            ort->ReleaseCUDAProviderOptions(cuda_options);
        }
    }
    PG_END_TRY();
}

static void
evoke_ort_append_tensorrt_provider(
    const OrtApi *ort,
    OrtSessionOptions *session_options,
    const char *runtime_precision)
{
    OrtTensorRTProviderOptionsV2 *trt_options = NULL;
    char device_id[16] = "0";
    const char *keys[2] = {"device_id", "trt_fp16_enable"};
    const char *values[2] = {
        device_id,
        strcmp(runtime_precision, EVOKE_RUNTIME_PRECISION_FP16) == 0
            ? "1"
            : "0"
    };

    evoke_ort_check(
        ort,
        ort->CreateTensorRTProviderOptions(&trt_options),
        "CreateTensorRTProviderOptions");
    PG_TRY();
    {
        evoke_ort_check(
            ort,
            ort->UpdateTensorRTProviderOptions(
                trt_options,
                keys,
                values,
                lengthof(keys)),
            "UpdateTensorRTProviderOptions");
        evoke_ort_check(
            ort,
            ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                session_options,
                trt_options),
            "SessionOptionsAppendExecutionProvider_TensorRT_V2");
    }
    PG_FINALLY();
    {
        if (trt_options != NULL)
        {
            ort->ReleaseTensorRTProviderOptions(trt_options);
        }
    }
    PG_END_TRY();
}

static void
evoke_ort_append_coreml_provider(
    const OrtApi *ort,
    OrtSessionOptions *session_options,
    const char *runtime_precision)
{
    const char *keys[3];
    const char *values[3];
    size_t count = 0;

    evoke_ort_prepare_process();

    keys[count] = "MLComputeUnits";
    values[count] = "CPUAndGPU";
    count++;

    keys[count] = "RequireStaticInputShapes";
    values[count] = "1";
    count++;

    keys[count] = "AllowLowPrecisionAccumulationOnGPU";
    values[count] =
        strcmp(runtime_precision, EVOKE_RUNTIME_PRECISION_FP16) == 0
            ? "1"
            : "0";
    count++;

    evoke_ort_check(
        ort,
        ort->SessionOptionsAppendExecutionProvider(
            session_options,
            "CoreML",
            keys,
            values,
            count),
        "SessionOptionsAppendExecutionProvider(CoreML)");
}

static void
evoke_ort_require_runtime_precision(
    const char *active_provider,
    const char *runtime_precision)
{
    (void) active_provider;

    if (strcmp(runtime_precision, EVOKE_RUNTIME_PRECISION_FP16) == 0 ||
        strcmp(runtime_precision, EVOKE_RUNTIME_PRECISION_FP32) == 0)
    {
        return;
    }
    ereport(
        ERROR,
        (
            errmsg("unsupported evoke runtime precision"),
            errdetail(
                "runtime_precision must be one of fp16 or fp32, got %s",
                runtime_precision
            )
        )
    );
}

static void
evoke_ort_configure_provider(
    const OrtApi *ort,
    OrtSessionOptions *session_options,
    Datum manifest,
    const char *active_provider,
    const char *runtime_precision)
{
    (void) manifest;
    evoke_ort_require_runtime_precision(active_provider, runtime_precision);
    if (strcmp(active_provider, "cpu") == 0)
    {
        return;
    }
    if (strcmp(active_provider, "tensorrt") == 0)
    {
        evoke_ort_append_tensorrt_provider(
            ort,
            session_options,
            runtime_precision);
        return;
    }
    if (strcmp(active_provider, "cuda") == 0)
    {
        evoke_ort_append_cuda_provider(ort, session_options);
        return;
    }
    if (strcmp(active_provider, "coreml") == 0)
    {
        evoke_ort_append_coreml_provider(
            ort,
            session_options,
            runtime_precision);
        return;
    }
    ereport(
        ERROR,
        (
            errmsg("unsupported ONNX Runtime active provider"),
            errdetail("active_provider=%s", active_provider)
        )
    );
}

static OrtSession *
evoke_ort_cache_session(
    const OrtApi *ort,
    Datum manifest,
    const char *model_path,
    const char *encoder_path,
    bool document_mode,
    const char *requested_provider,
    const char *active_provider,
    const char *runtime_precision,
    const char *checkout_signature,
    OrtEnv **transient_env,
    OrtSessionOptions **transient_session_options,
    bool *cache_hit,
    double *load_ms)
{
    EvokeOrtSessionBuild *build;
    EvokeOrtSessionCacheEntry *entry;
    bool runtime_worker_owner = false;
    double started_ms;
    int max_sessions;

    if (
        evoke_runtime_service != NULL &&
        evoke_runtime_service->magic == EVOKE_RUNTIME_SERVICE_MAGIC &&
        evoke_runtime_service->version == EVOKE_RUNTIME_SERVICE_VERSION
    )
    {
        LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
        for (uint32 i = 0;
            i < evoke_runtime_service_configured_workers();
            i++)
        {
            if (evoke_runtime_service->workers[i].ready &&
                evoke_runtime_service->workers[i].pid == MyProcPid &&
                evoke_runtime_service->workers[i].proc_number ==
                    MyProcNumber)
            {
                runtime_worker_owner = true;
                break;
            }
        }
        LWLockRelease(evoke_runtime_service_lock);
    }
    if (!runtime_worker_owner)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke model session requested outside runtime worker"),
                errdetail(
                    "ONNX Runtime sessions are owned only by the shared "
                    "evoke runtime worker pool."
                )
            )
        );
    }

    *transient_env = NULL;
    *transient_session_options = NULL;
    *cache_hit = false;
    *load_ms = 0.0;

    max_sessions = evoke_ort_effective_cache_size();
    evoke_ort_cache_apply_size_limit(ort, max_sessions);

    if (max_sessions > 0)
    {
        for (int i = 0; i < evoke_ort_session_cache.entry_count; i++)
        {
            entry = &evoke_ort_session_cache.entries[i];
            if (strcmp(entry->model_path, model_path) == 0
                && strcmp(entry->encoder_path, encoder_path) == 0
                && entry->document_mode == document_mode
                && strcmp(entry->requested_provider, requested_provider) == 0
                && strcmp(entry->active_provider, active_provider) == 0
                && strcmp(entry->runtime_precision, runtime_precision) == 0
                && strcmp(
                    entry->checkout_signature,
                    checkout_signature
                ) == 0)
            {
                evoke_ort_session_cache.hits++;
                evoke_ort_session_cache.access_seq++;
                evoke_ort_session_cache.last_used_at = GetCurrentTimestamp();
                evoke_ort_record_provider(
                    requested_provider,
                    active_provider,
                    runtime_precision
                );
                entry->last_used_at = evoke_ort_session_cache.last_used_at;
                entry->access_seq = evoke_ort_session_cache.access_seq;
                *cache_hit = true;
                return entry->session;
            }
        }
    }

    evoke_ort_session_cache.misses++;
    build = palloc0(sizeof(*build));
    started_ms = evoke_monotonic_ms();
    PG_TRY();
    {
        evoke_ort_check(
            ort,
            ort->CreateEnv(
                ORT_LOGGING_LEVEL_WARNING,
                "evoke",
                &build->env
            ),
            "CreateEnv"
        );
        evoke_ort_check(
            ort,
            ort->CreateSessionOptions(&build->session_options),
            "CreateSessionOptions"
        );
        if (evoke_runtime_effective_intra_op_threads() > 0)
        {
            evoke_ort_check(
                ort,
                ort->SetIntraOpNumThreads(
                    build->session_options,
                    evoke_runtime_effective_intra_op_threads()
                ),
                "SetIntraOpNumThreads"
            );
        }
        if (document_mode && !evoke_onnxruntime_document_cpu_mem_arena)
        {
            /*
             * Document completion sees dynamic batch and sequence shapes.
             * Every execution provider can leave shape and fallback nodes on
             * CPU. Retaining those high-water allocations in the CPU arena
             * can pin several GiB in a long-lived runtime worker. Query
             * sessions retain the arena because foreground latency is more
             * important.
             */
            evoke_ort_check(
                ort,
                ort->DisableMemPattern(build->session_options),
                "DisableMemPattern(document)"
            );
            evoke_ort_check(
                ort,
                ort->DisableCpuMemArena(build->session_options),
                "DisableCpuMemArena(document)"
            );
        }
        evoke_ort_configure_provider(
            ort,
            build->session_options,
            manifest,
            active_provider,
            runtime_precision
        );
        evoke_ort_check(
            ort,
            ort->CreateSession(
                build->env,
                encoder_path,
                build->session_options,
                &build->session
            ),
            "CreateSession"
        );
        if (max_sessions > 0)
        {
            build->model_path = MemoryContextStrdup(
                TopMemoryContext,
                model_path
            );
            build->encoder_path = MemoryContextStrdup(
                TopMemoryContext,
                encoder_path
            );
            build->requested_provider = MemoryContextStrdup(
                TopMemoryContext,
                requested_provider
            );
            build->active_provider = MemoryContextStrdup(
                TopMemoryContext,
                active_provider
            );
            build->runtime_precision = MemoryContextStrdup(
                TopMemoryContext,
                runtime_precision
            );
        }
    }
    PG_CATCH();
    {
        evoke_ort_release_session_build(ort, build);
        pfree(build);
        PG_RE_THROW();
    }
    PG_END_TRY();

    evoke_ort_session_cache.loads++;
    evoke_ort_session_cache.loaded_at = GetCurrentTimestamp();
    evoke_ort_session_cache.last_used_at = evoke_ort_session_cache.loaded_at;
    evoke_ort_session_cache.last_load_ms = evoke_monotonic_ms() - started_ms;
    evoke_ort_record_provider(
        requested_provider,
        active_provider,
        runtime_precision
    );

    if (max_sessions <= 0)
    {
        OrtSession *session = build->session;

        *transient_env = build->env;
        *transient_session_options = build->session_options;
        *load_ms = evoke_ort_session_cache.last_load_ms;
        pfree(build);
        return session;
    }

    if (evoke_ort_session_cache.entry_count >= max_sessions)
    {
        evoke_ort_session_cache.evictions++;
        evoke_ort_cache_remove_entry(ort, evoke_ort_cache_lru_index());
    }
    Assert(evoke_ort_session_cache.entry_count < max_sessions);
    if (!evoke_ort_cache_exit_registered)
    {
        before_shmem_exit(evoke_ort_cache_shmem_exit, 0);
        evoke_ort_cache_exit_registered = true;
    }
    entry = &evoke_ort_session_cache.entries[
        evoke_ort_session_cache.entry_count
    ];
    memset(entry, 0, sizeof(*entry));
    entry->model_path = build->model_path;
    entry->encoder_path = build->encoder_path;
    entry->requested_provider = build->requested_provider;
    entry->active_provider = build->active_provider;
    entry->runtime_precision = build->runtime_precision;
    entry->document_mode = document_mode;
    strlcpy(
        entry->checkout_signature,
        checkout_signature,
        sizeof(entry->checkout_signature)
    );
    entry->env = build->env;
    entry->session_options = build->session_options;
    entry->session = build->session;
    evoke_ort_session_cache.entry_count++;
    evoke_ort_session_cache.access_seq++;
    entry->loaded_at = evoke_ort_session_cache.loaded_at;
    entry->last_used_at = entry->loaded_at;
    entry->last_load_ms = evoke_ort_session_cache.last_load_ms;
    entry->access_seq = evoke_ort_session_cache.access_seq;

    *load_ms = evoke_ort_session_cache.last_load_ms;
    pfree(build);
    return entry->session;
}
#endif

#ifdef EVOKE_ENABLE_ONNXRUNTIME
#define EVOKE_P2_DEFAULT_WINDOW_OVERLAP 64
#define EVOKE_P2_DEFAULT_MAX_WINDOWS 64
#define EVOKE_P2_MAX_RUNTIME_WINDOWS 256

typedef struct EvokeP2SemanticAtoms
{
    int64 *ids;
    float *weights;
    int32 count;
} EvokeP2SemanticAtoms;

typedef struct EvokeP2WindowReference
{
    uint32 row_index;
    const EvokeP2TokenizedInput *input;
} EvokeP2WindowReference;

typedef struct EvokeP2SemanticCandidate
{
    int64 id;
    float weight;
} EvokeP2SemanticCandidate;

static int
evoke_p2_compare_semantic_candidate(const void *left, const void *right)
{
    const EvokeP2SemanticCandidate *left_candidate = left;
    const EvokeP2SemanticCandidate *right_candidate = right;

    if (left_candidate->weight > right_candidate->weight)
    {
        return -1;
    }
    if (left_candidate->weight < right_candidate->weight)
    {
        return 1;
    }
    if (left_candidate->id < right_candidate->id)
    {
        return -1;
    }
    if (left_candidate->id > right_candidate->id)
    {
        return 1;
    }
    return 0;
}

static void
evoke_p2_semantic_atoms_free(
    EvokeP2SemanticAtoms *semantic,
    uint32 row_count
)
{
    if (semantic == NULL)
    {
        return;
    }
    for (uint32 i = 0; i < row_count; i++)
    {
        if (semantic[i].ids != NULL)
        {
            pfree(semantic[i].ids);
        }
        if (semantic[i].weights != NULL)
        {
            pfree(semantic[i].weights);
        }
    }
    pfree(semantic);
}

static EvokeP2SemanticAtoms *
evoke_p2_run_semantic_windows(
    const OrtApi *ort,
    OrtSession *session,
    const char *const input_names[2],
    const char *const output_names[2],
    const EvokeP2TokenizedWindows *tokenized,
    uint32 row_count,
    int32 semantic_dims,
    int32 semantic_max_atoms,
    int32 max_batch_size
)
{
    EvokeP2WindowReference *references;
    EvokeP2SemanticAtoms *result;
    EvokeP2SemanticCandidate *candidates;
    float *max_weights;
    uint32 total_windows = 0;
    uint32 reference_index = 0;
    uint64 dense_value_count;

    if (row_count == 0 || semantic_dims <= 0 || semantic_max_atoms <= 0 ||
        max_batch_size <= 0)
    {
        ereport(ERROR, (errmsg("invalid P2 semantic window configuration")));
    }
    for (uint32 row = 0; row < row_count; row++)
    {
        if (tokenized[row].window_count <= 0)
        {
            ereport(ERROR, (errmsg("P2 tokenizer produced no windows")));
        }
        if ((uint64) total_windows +
            (uint64) tokenized[row].window_count > PG_UINT32_MAX)
        {
            ereport(ERROR, (errmsg("P2 semantic window count is too large")));
        }
        total_windows += (uint32) tokenized[row].window_count;
    }
    dense_value_count = (uint64) row_count * (uint64) semantic_dims;
    if (dense_value_count > (uint64) MaxAllocSize / sizeof(*max_weights))
    {
        ereport(ERROR, (errmsg("P2 semantic aggregation is too large")));
    }

    references = palloc(
        sizeof(*references) * (Size) total_windows
    );
    result = palloc0(sizeof(*result) * (Size) row_count);
    max_weights = palloc0(
        sizeof(*max_weights) * (Size) dense_value_count
    );
    candidates = palloc(
        sizeof(*candidates) * (Size) semantic_dims
    );
    for (uint32 row = 0; row < row_count; row++)
    {
        for (int32 window = 0;
             window < tokenized[row].window_count;
             window++)
        {
            references[reference_index].row_index = row;
            references[reference_index].input =
                &tokenized[row].windows[window];
            reference_index++;
        }
    }

    for (uint32 offset = 0; offset < total_windows;)
    {
        uint32 chunk_count = Min(
            (uint32) max_batch_size,
            total_windows - offset
        );
        EvokeOrtRunResources *run_resources;
        int64 *input_ids = NULL;
        int64 *attention_mask = NULL;
        int64 input_shape[2];
        int32 sequence_length = 0;

        run_resources = palloc0(sizeof(*run_resources));
        run_resources->ort = ort;
        for (uint32 i = 0; i < chunk_count; i++)
        {
            sequence_length = Max(
                sequence_length,
                references[offset + i].input->token_count
            );
        }
        input_ids = palloc0(
            sizeof(*input_ids) * (Size) chunk_count *
            (Size) sequence_length
        );
        attention_mask = palloc0(
            sizeof(*attention_mask) * (Size) chunk_count *
            (Size) sequence_length
        );
        for (uint32 i = 0; i < chunk_count; i++)
        {
            const EvokeP2TokenizedInput *window =
                references[offset + i].input;
            Size row_offset = (Size) i * (Size) sequence_length;
            Size row_bytes = sizeof(*input_ids) *
                (Size) window->token_count;

            memcpy(input_ids + row_offset, window->input_ids, row_bytes);
            memcpy(
                attention_mask + row_offset,
                window->attention_mask,
                row_bytes
            );
        }
        input_shape[0] = (int64) chunk_count;
        input_shape[1] = (int64) sequence_length;

        PG_TRY();
        {
            size_t atom_dimension_count = 0;
            size_t weight_dimension_count = 0;
            int64 atom_dimensions[2] = {0, 0};
            int64 weight_dimensions[2] = {0, 0};
            size_t atom_count = 0;
            size_t weight_count = 0;
            enum ONNXTensorElementDataType atom_element_type;
            enum ONNXTensorElementDataType weight_element_type;
            int64 *atom_data = NULL;
            float *weight_data = NULL;
            int32 semantic_width;

            evoke_ort_check(
                ort,
                ort->CreateCpuMemoryInfo(
                    OrtArenaAllocator,
                    OrtMemTypeDefault,
                    &run_resources->memory_info
                ),
                "CreateCpuMemoryInfo(P2 windows)"
            );
            evoke_ort_check(
                ort,
                ort->CreateTensorWithDataAsOrtValue(
                    run_resources->memory_info,
                    input_ids,
                    sizeof(*input_ids) * (Size) chunk_count *
                        (Size) sequence_length,
                    input_shape,
                    2,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                    &run_resources->inputs[0]
                ),
                "CreateTensorWithDataAsOrtValue(P2 window input_ids)"
            );
            evoke_ort_check(
                ort,
                ort->CreateTensorWithDataAsOrtValue(
                    run_resources->memory_info,
                    attention_mask,
                    sizeof(*attention_mask) * (Size) chunk_count *
                        (Size) sequence_length,
                    input_shape,
                    2,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                    &run_resources->inputs[1]
                ),
                "CreateTensorWithDataAsOrtValue(P2 window attention_mask)"
            );
            evoke_ort_run_interruptible(
                ort,
                session,
                input_names,
                (const OrtValue *const *) run_resources->inputs,
                2,
                output_names,
                2,
                run_resources->outputs,
                "Run(P2 windows)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorTypeAndShape(
                    run_resources->outputs[0],
                    &run_resources->atom_shape
                ),
                "GetTensorTypeAndShape(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorTypeAndShape(
                    run_resources->outputs[1],
                    &run_resources->weight_shape
                ),
                "GetTensorTypeAndShape(P2 window semantic_weights)"
            );
            evoke_ort_check(
                ort,
                ort->GetDimensionsCount(
                    run_resources->atom_shape,
                    &atom_dimension_count
                ),
                "GetDimensionsCount(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetDimensionsCount(
                    run_resources->weight_shape,
                    &weight_dimension_count
                ),
                "GetDimensionsCount(P2 window semantic_weights)"
            );
            if (atom_dimension_count != 2 || weight_dimension_count != 2)
            {
                ereport(ERROR, (errmsg("invalid P2 ONNX window output rank")));
            }
            evoke_ort_check(
                ort,
                ort->GetDimensions(
                    run_resources->atom_shape,
                    atom_dimensions,
                    2
                ),
                "GetDimensions(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetDimensions(
                    run_resources->weight_shape,
                    weight_dimensions,
                    2
                ),
                "GetDimensions(P2 window semantic_weights)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorShapeElementCount(
                    run_resources->atom_shape,
                    &atom_count
                ),
                "GetTensorShapeElementCount(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorShapeElementCount(
                    run_resources->weight_shape,
                    &weight_count
                ),
                "GetTensorShapeElementCount(P2 window semantic_weights)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorElementType(
                    run_resources->atom_shape,
                    &atom_element_type
                ),
                "GetTensorElementType(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorElementType(
                    run_resources->weight_shape,
                    &weight_element_type
                ),
                "GetTensorElementType(P2 window semantic_weights)"
            );
            if (atom_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                weight_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                atom_dimensions[0] != (int64) chunk_count ||
                weight_dimensions[0] != (int64) chunk_count ||
                atom_dimensions[1] <= 0 ||
                atom_dimensions[1] > semantic_max_atoms ||
                weight_dimensions[1] != atom_dimensions[1] ||
                atom_count != weight_count)
            {
                ereport(
                    ERROR,
                    (
                        errmsg("invalid P2 ONNX window output shape"),
                        errdetail(
                            "ids=[%lld,%lld] weights=[%lld,%lld] "
                            "batch=%u max=%d",
                            (long long) atom_dimensions[0],
                            (long long) atom_dimensions[1],
                            (long long) weight_dimensions[0],
                            (long long) weight_dimensions[1],
                            chunk_count,
                            semantic_max_atoms
                        )
                    )
                );
            }
            semantic_width = (int32) atom_dimensions[1];
            evoke_ort_check(
                ort,
                ort->GetTensorMutableData(
                    run_resources->outputs[0],
                    (void **) &atom_data
                ),
                "GetTensorMutableData(P2 window semantic_ids)"
            );
            evoke_ort_check(
                ort,
                ort->GetTensorMutableData(
                    run_resources->outputs[1],
                    (void **) &weight_data
                ),
                "GetTensorMutableData(P2 window semantic_weights)"
            );
            for (uint32 i = 0; i < chunk_count; i++)
            {
                uint32 row = references[offset + i].row_index;
                Size semantic_offset = (Size) i * (Size) semantic_width;
                Size dense_offset = (Size) row * (Size) semantic_dims;

                for (int32 j = 0; j < semantic_width; j++)
                {
                    int64 atom_id = atom_data[semantic_offset + (Size) j];
                    float weight = weight_data[semantic_offset + (Size) j];
                    float *current;

                    if (atom_id < 0 || atom_id >= semantic_dims)
                    {
                        ereport(
                            ERROR,
                            (errmsg("P2 semantic atom id is out of range"))
                        );
                    }
                    if (!isfinite((double) weight) || weight < 0.0f)
                    {
                        ereport(
                            ERROR,
                            (errmsg("invalid P2 semantic atom weight"))
                        );
                    }
                    if (weight == 0.0f)
                    {
                        continue;
                    }
                    current = &max_weights[
                        dense_offset + (Size) atom_id
                    ];
                    if (weight > *current)
                    {
                        *current = weight;
                    }
                }
            }
        }
        PG_FINALLY();
        {
            evoke_ort_release_run_resources(run_resources);
            pfree(run_resources);
        }
        PG_END_TRY();

        pfree(input_ids);
        pfree(attention_mask);
        offset += chunk_count;
    }

    for (uint32 row = 0; row < row_count; row++)
    {
        Size dense_offset = (Size) row * (Size) semantic_dims;
        int32 candidate_count = 0;
        int32 output_count;

        for (int32 atom_id = 0; atom_id < semantic_dims; atom_id++)
        {
            float weight = max_weights[dense_offset + (Size) atom_id];

            if (weight > 0.0f)
            {
                candidates[candidate_count].id = atom_id;
                candidates[candidate_count].weight = weight;
                candidate_count++;
            }
        }
        if (candidate_count == 0)
        {
            ereport(ERROR, (errmsg("P2 semantic encoder produced no atoms")));
        }
        qsort(
            candidates,
            candidate_count,
            sizeof(*candidates),
            evoke_p2_compare_semantic_candidate
        );
        output_count = Min(candidate_count, semantic_max_atoms);
        result[row].ids = palloc(
            sizeof(*result[row].ids) * (Size) output_count
        );
        result[row].weights = palloc(
            sizeof(*result[row].weights) * (Size) output_count
        );
        result[row].count = output_count;
        for (int32 i = 0; i < output_count; i++)
        {
            result[row].ids[i] = candidates[i].id;
            result[row].weights[i] = candidates[i].weight;
        }
    }

    pfree(candidates);
    pfree(max_weights);
    pfree(references);
    return result;
}

static char *
evoke_p2_onnxruntime_query_atoms_json(
    const char *model_path,
    const char *query_text,
    const char *requested_runtime_precision,
    Datum manifest,
    const char *encoder_path,
    const char *runtime_abi
)
{
    struct
    {
        EvokeP2TokenizedWindows tokenized;
        EvokeP2SemanticAtoms *semantic;
        EvokeP2CompiledAtoms compiled;
    } *cleanup;
    char *tokenizer_vocabulary_relpath;
    char *tokenizer_merges_relpath;
    char *lexical_vocabulary_relpath;
    char *calibration_relpath;
    char *tokenizer_vocabulary_path;
    char *tokenizer_merges_path;
    char *lexical_vocabulary_path;
    char *calibration_path;
    Datum scoring;
    Datum atom_space;
    Datum semantic_runtime;
    const OrtApi *ort;
    EvokeOrtRunResources *resources;
    OrtSession *session = NULL;
    char *input_ids_name;
    char *attention_mask_name;
    char *semantic_ids_name;
    char *semantic_weights_name;
    char *requested_provider;
    char *active_provider;
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    const char *input_names[2];
    const char *output_names[2];
    int32 max_length;
    int32 window_overlap;
    int32 window_stride;
    int32 max_windows;
    int32 max_batch_size;
    int32 max_output_atoms;
    int32 semantic_max_atoms;
    int32 lexical_dims;
    int32 total_dims;
    int32 manifest_dims;
    double global_scale;
    double multiplier;
    double clip_min;
    double clip_max;
    StringInfoData json;
    double started_ms;
    double elapsed_ms;
    double load_ms;
    bool cache_hit;
    char checkout_signature[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1];

    tokenizer_vocabulary_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "tokenizer_vocabulary"
    );
    tokenizer_merges_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "tokenizer_merges"
    );
    tokenizer_vocabulary_path = evoke_checkout_artifact_path(
        model_path,
        "tokenizer_vocabulary",
        tokenizer_vocabulary_relpath
    );
    tokenizer_merges_path = evoke_checkout_artifact_path(
        model_path,
        "tokenizer_merges",
        tokenizer_merges_relpath
    );
    lexical_vocabulary_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "lexical_vocabulary"
    );
    calibration_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "query_calibration_runtime"
    );
    lexical_vocabulary_path = evoke_checkout_artifact_path(
        model_path,
        "lexical_vocabulary",
        lexical_vocabulary_relpath
    );
    calibration_path = evoke_checkout_artifact_path(
        model_path,
        "query_calibration_runtime",
        calibration_relpath
    );
    scoring = evoke_checkout_artifact_jsonb(
        manifest,
        model_path,
        "scoring_profile"
    );
    atom_space = evoke_checkout_artifact_jsonb(
        manifest,
        model_path,
        "atom_space"
    );
    semantic_runtime = evoke_checkout_artifact_jsonb(
        manifest,
        model_path,
        "semantic_runtime"
    );
    max_length = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "max_length",
        512,
        3,
        4096
    );
    window_overlap = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "window_overlap",
        Min(EVOKE_P2_DEFAULT_WINDOW_OVERLAP, max_length - 3),
        0,
        max_length - 3
    );
    window_stride = max_length - 2 - window_overlap;
    max_windows = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "max_windows",
        EVOKE_P2_DEFAULT_MAX_WINDOWS,
        1,
        EVOKE_P2_MAX_RUNTIME_WINDOWS
    );
    max_batch_size = evoke_runtime_effective_max_batch_size();
    max_output_atoms = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        "max_atoms",
        512,
        1,
        65536
    );
    semantic_max_atoms = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        "query_semantic_max_atoms",
        50,
        1,
        65536
    );
    lexical_dims = evoke_checkout_jsonb_int_path(
        atom_space,
        "lexical",
        "end_exclusive",
        0,
        1,
        PG_INT32_MAX
    );
    total_dims = evoke_checkout_jsonb_int_key(
        atom_space,
        "latent_dims",
        0,
        lexical_dims + 1,
        PG_INT32_MAX
    );
    manifest_dims = evoke_checkout_jsonb_int_key(
        manifest,
        "latent_dims",
        0,
        lexical_dims + 1,
        PG_INT32_MAX
    );
    if (manifest_dims != total_dims)
    {
        ereport(ERROR, (errmsg("P2 atom-space dimensions do not match")));
    }
    global_scale = evoke_checkout_jsonb_double_path(
        scoring,
        "query_compiler",
        "global_scale",
        0.0,
        1.0e12
    );
    multiplier = evoke_checkout_jsonb_double_path(
        scoring,
        "query_compiler",
        "multiplier",
        0.0,
        1.0e12
    );
    clip_min = evoke_checkout_jsonb_double_path(
        scoring,
        "query_compiler",
        "clip_min",
        0.0,
        1.0e12
    );
    clip_max = evoke_checkout_jsonb_double_path(
        scoring,
        "query_compiler",
        "clip_max",
        clip_min,
        1.0e12
    );

    input_ids_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "input_ids",
        NULL
    );
    attention_mask_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "attention_mask",
        NULL
    );
    semantic_ids_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "semantic_ids",
        NULL
    );
    semantic_weights_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "semantic_weights",
        NULL
    );
    input_names[0] = input_ids_name;
    input_names[1] = attention_mask_name;
    output_names[0] = semantic_ids_name;
    output_names[1] = semantic_weights_name;

    ort = evoke_ort_api();
    evoke_runtime_precision_normalize(
        requested_runtime_precision,
        runtime_precision
    );
    requested_provider = pstrdup("auto");
    active_provider = evoke_p2_ort_active_provider(
        ort,
        semantic_runtime,
        runtime_precision
    );
    evoke_checkout_manifest_signature(manifest, checkout_signature);
    evoke_ort_record_provider(
        requested_provider,
        active_provider,
        runtime_precision
    );
    resources = palloc0(sizeof(*resources));
    resources->ort = ort;
    cleanup = palloc0(sizeof(*cleanup));
    session = evoke_ort_cache_session(
        ort,
        manifest,
        model_path,
        encoder_path,
        false,
        requested_provider,
        active_provider,
        runtime_precision,
        checkout_signature,
        &resources->owned_env,
        &resources->owned_session_options,
        &cache_hit,
        &load_ms
    );
    if (resources->owned_env != NULL)
    {
        resources->owned_session = session;
    }

    PG_TRY();
    {
        started_ms = evoke_monotonic_ms();
        evoke_p2_tokenize_roberta_windows(
            tokenizer_vocabulary_path,
            tokenizer_merges_path,
            checkout_signature,
            query_text,
            max_length,
            window_stride,
            max_windows,
            &cleanup->tokenized
        );
        cleanup->semantic = evoke_p2_run_semantic_windows(
            ort,
            session,
            input_names,
            output_names,
            &cleanup->tokenized,
            1,
            total_dims - lexical_dims,
            semantic_max_atoms,
            max_batch_size
        );
        evoke_p2_compile_unified_atoms(
            lexical_vocabulary_path,
            calibration_path,
            checkout_signature,
            query_text,
            cleanup->semantic[0].ids,
            cleanup->semantic[0].weights,
            cleanup->semantic[0].count,
            lexical_dims,
            total_dims,
            global_scale,
            multiplier,
            clip_min,
            clip_max,
            &cleanup->compiled
        );
        if (cleanup->compiled.atom_count > max_output_atoms)
        {
            ereport(
                ERROR,
                (
                    errmsg("P2 compiler output exceeds runtime atom limit"),
                    errdetail(
                        "atoms=%d max_atoms=%d",
                        cleanup->compiled.atom_count,
                        max_output_atoms
                    )
                )
            );
        }
        /*
         * A checkout publisher may replace artifacts while inference is
         * running. Revalidate the exact manifest used for this response before
         * exposing its output so an in-place update fails closed.
         */
        evoke_checkout_validate_manifest(manifest, NULL, model_path);
        elapsed_ms = evoke_monotonic_ms() - started_ms;
        evoke_ort_session_cache.last_run_ms =
            elapsed_ms < 0.0 ? 0.0 : elapsed_ms;
        evoke_ort_session_cache.last_atom_count =
            (size_t) cleanup->compiled.atom_count;
        evoke_ort_session_cache.last_used_at = GetCurrentTimestamp();
        for (int i = 0; i < evoke_ort_session_cache.entry_count; i++)
        {
            if (evoke_ort_session_cache.entries[i].session == session)
            {
                evoke_ort_session_cache.entries[i].last_run_ms =
                    evoke_ort_session_cache.last_run_ms;
                evoke_ort_session_cache.entries[i].last_atom_count =
                    (size_t) cleanup->compiled.atom_count;
                evoke_ort_session_cache.entries[i].last_used_at =
                    evoke_ort_session_cache.last_used_at;
                break;
            }
        }

        initStringInfo(&json);
        appendStringInfoString(&json, "{\"atoms\":[");
        for (int32 i = 0; i < cleanup->compiled.atom_count; i++)
        {
            if (i > 0)
            {
                appendStringInfoChar(&json, ',');
            }
            appendStringInfo(
                &json,
                "%d",
                cleanup->compiled.atom_ids[i]
            );
        }
        appendStringInfoString(&json, "],\"weights\":[");
        for (int32 i = 0; i < cleanup->compiled.atom_count; i++)
        {
            if (i > 0)
            {
                appendStringInfoChar(&json, ',');
            }
            appendStringInfo(
                &json,
                "%.9g",
                (double) cleanup->compiled.atom_weights[i]
            );
        }
        appendStringInfoString(
            &json,
            "],\"runtime\":\"onnxruntime\",\"runtime_abi\":"
        );
        evoke_append_json_string(&json, runtime_abi);
        appendStringInfoString(&json, ",\"checkout_signature\":");
        evoke_append_json_string(&json, checkout_signature);
        appendStringInfoString(&json, ",\"runtime_io\":{\"input_ids\":");
        evoke_append_json_string(&json, input_ids_name);
        appendStringInfoString(&json, ",\"attention_mask\":");
        evoke_append_json_string(&json, attention_mask_name);
        appendStringInfoString(&json, ",\"semantic_ids\":");
        evoke_append_json_string(&json, semantic_ids_name);
        appendStringInfoString(&json, ",\"semantic_weights\":");
        evoke_append_json_string(&json, semantic_weights_name);
        appendStringInfoString(&json, "},\"provider\":{\"requested\":");
        evoke_append_json_string(&json, requested_provider);
        appendStringInfoString(&json, ",\"active\":");
        evoke_append_json_string(&json, active_provider);
        appendStringInfoString(&json, ",\"available\":");
        evoke_ort_append_available_providers_json(ort, &json);
        appendStringInfoString(&json, "},\"runtime_precision\":");
        evoke_append_json_string(&json, runtime_precision);
        appendStringInfo(
            &json,
            ",\"tokenizer\":{\"type\":\"roberta_byte_bpe\","
            "\"max_length\":%d,\"token_count\":%d,"
            "\"window_count\":%d,\"window_stride\":%d,"
            "\"window_overlap\":%d,\"aggregation\":"
            "\"dimension_max_top_k\",\"truncated\":false},"
            "\"compiler\":{\"lexical_atoms\":%d,"
            "\"semantic_atoms\":%d,\"lexical_proxy\":%.9g,"
            "\"semantic_proxy\":%.9g,\"query_scale\":%.9g},"
            "\"atom_count\":%d,\"latency_ms\":%.3f,"
            "\"session_cache_hit\":%s,\"session_load_ms\":%.3f}",
            max_length,
            cleanup->tokenized.full_token_count,
            cleanup->tokenized.window_count,
            cleanup->tokenized.window_stride,
            window_overlap,
            cleanup->compiled.lexical_atom_count,
            cleanup->compiled.semantic_atom_count,
            cleanup->compiled.lexical_proxy,
            cleanup->compiled.semantic_proxy,
            cleanup->compiled.query_scale,
            cleanup->compiled.atom_count,
            elapsed_ms < 0.0 ? 0.0 : elapsed_ms,
            cache_hit ? "true" : "false",
            load_ms < 0.0 ? 0.0 : load_ms
        );
    }
    PG_FINALLY();
    {
        if (cleanup->compiled.atom_ids != NULL)
        {
            pfree(cleanup->compiled.atom_ids);
        }
        if (cleanup->compiled.atom_weights != NULL)
        {
            pfree(cleanup->compiled.atom_weights);
        }
        evoke_p2_semantic_atoms_free(cleanup->semantic, 1);
        evoke_p2_tokenized_windows_free(&cleanup->tokenized);
        evoke_ort_release_run_resources(resources);
        pfree(resources);
        pfree(cleanup);
    }
    PG_END_TRY();
    return json.data;
}

static void
evoke_p2_append_compiled_atoms_json(
    StringInfo json,
    const EvokeP2CompiledAtoms *compiled,
    const EvokeP2TokenizedWindows *tokenized,
    const char *compiler_role
)
{
    appendStringInfoString(json, "{\"compiler_role\":");
    evoke_append_json_string(json, compiler_role);
    appendStringInfoString(json, ",\"atoms\":[");
    for (int32 i = 0; i < compiled->atom_count; i++)
    {
        if (i > 0)
        {
            appendStringInfoChar(json, ',');
        }
        appendStringInfo(json, "%d", compiled->atom_ids[i]);
    }
    appendStringInfoString(json, "],\"weights\":[");
    for (int32 i = 0; i < compiled->atom_count; i++)
    {
        if (i > 0)
        {
            appendStringInfoChar(json, ',');
        }
        appendStringInfo(
            json,
            "%.9g",
            (double) compiled->atom_weights[i]
        );
    }
    appendStringInfo(
        json,
        "],\"token_count\":%d,\"window_count\":%d,"
        "\"window_stride\":%d,\"aggregation\":"
        "\"dimension_max_top_k\",\"truncated\":false,\"compiler\":{"
        "\"lexical_atoms\":%d,\"semantic_atoms\":%d,"
        "\"lexical_proxy\":%.9g,\"semantic_proxy\":%.9g,"
        "\"query_scale\":%.9g},\"atom_count\":%d}",
        tokenized->full_token_count,
        tokenized->window_count,
        tokenized->window_stride,
        compiled->lexical_atom_count,
        compiled->semantic_atom_count,
        compiled->lexical_proxy,
        compiled->semantic_proxy,
        compiled->query_scale,
        compiled->atom_count
    );
}

static char *
evoke_p2_onnxruntime_atoms_batch_json(
    const char *model_path,
    const char *const *texts,
    uint32 batch_count,
    const char *requested_runtime_precision,
    Datum manifest,
    const char *encoder_path,
    const char *runtime_abi,
    bool document_mode
)
{
    char *tokenizer_vocabulary_relpath;
    char *tokenizer_merges_relpath;
    char *lexical_vocabulary_relpath = NULL;
    char *calibration_relpath = NULL;
    char *tokenizer_vocabulary_path;
    char *tokenizer_merges_path;
    char *lexical_vocabulary_path = NULL;
    char *calibration_path = NULL;
    Datum scoring = (Datum) 0;
    Datum atom_space;
    Datum semantic_runtime;
    const OrtApi *ort;
    EvokeOrtRunResources *resources;
    OrtSession *session = NULL;
    EvokeP2TokenizedWindows *tokenized;
    EvokeP2SemanticAtoms *volatile semantic = NULL;
    EvokeP2CompiledAtoms *compiled;
    char *input_ids_name;
    char *attention_mask_name;
    char *semantic_ids_name;
    char *semantic_weights_name;
    char *requested_provider;
    char *active_provider;
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    const char *input_names[2];
    const char *output_names[2];
    int32 max_length;
    int32 window_overlap;
    int32 window_stride;
    int32 max_windows;
    int32 max_output_atoms;
    int32 semantic_max_atoms;
    int32 lexical_dims;
    int32 total_dims;
    int32 manifest_dims;
    int32 max_batch_size;
    int32 sequence_length = 0;
    double global_scale = 0.0;
    double multiplier = 0.0;
    double clip_min = 0.0;
    double clip_max = 0.0;
    StringInfoData json;
    double started_ms;
    double elapsed_ms;
    double load_ms;
    size_t total_compiled_atoms = 0;
    uint32 total_window_count = 0;
    bool cache_hit;
    char checkout_signature[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1];

    if (batch_count == 0 ||
        batch_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    {
        ereport(ERROR, (errmsg("invalid P2 ONNX batch size")));
    }
    max_batch_size = evoke_runtime_effective_max_batch_size();
    if (batch_count > (uint32) max_batch_size)
    {
        ereport(
            ERROR,
            (
                errmsg("P2 ONNX batch exceeds the model runtime limit"),
                errdetail(
                    "batch_count=%u max_batch_size=%d",
                    batch_count,
                    max_batch_size
                )
            )
        );
    }

    tokenizer_vocabulary_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "tokenizer_vocabulary"
    );
    tokenizer_merges_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "tokenizer_merges"
    );
    tokenizer_vocabulary_path = evoke_checkout_artifact_path(
        model_path,
        "tokenizer_vocabulary",
        tokenizer_vocabulary_relpath
    );
    tokenizer_merges_path = evoke_checkout_artifact_path(
        model_path,
        "tokenizer_merges",
        tokenizer_merges_relpath
    );
    if (!document_mode)
    {
        lexical_vocabulary_relpath = evoke_checkout_manifest_artifact_relpath(
            manifest,
            "lexical_vocabulary"
        );
        calibration_relpath = evoke_checkout_manifest_artifact_relpath(
            manifest,
            "query_calibration_runtime"
        );
        lexical_vocabulary_path = evoke_checkout_artifact_path(
            model_path,
            "lexical_vocabulary",
            lexical_vocabulary_relpath
        );
        calibration_path = evoke_checkout_artifact_path(
            model_path,
            "query_calibration_runtime",
            calibration_relpath
        );
        scoring = evoke_checkout_artifact_jsonb(
            manifest,
            model_path,
            "scoring_profile"
        );
    }
    atom_space = evoke_checkout_artifact_jsonb(
        manifest,
        model_path,
        "atom_space"
    );
    semantic_runtime = evoke_checkout_artifact_jsonb(
        manifest,
        model_path,
        "semantic_runtime"
    );
    max_length = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "max_length",
        512,
        3,
        4096
    );
    window_overlap = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "window_overlap",
        Min(EVOKE_P2_DEFAULT_WINDOW_OVERLAP, max_length - 3),
        0,
        max_length - 3
    );
    window_stride = max_length - 2 - window_overlap;
    max_windows = evoke_checkout_jsonb_int_key(
        semantic_runtime,
        "max_windows",
        EVOKE_P2_DEFAULT_MAX_WINDOWS,
        1,
        EVOKE_P2_MAX_RUNTIME_WINDOWS
    );
    max_output_atoms = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        "max_atoms",
        512,
        1,
        65536
    );
    semantic_max_atoms = evoke_checkout_jsonb_int_path(
        manifest,
        "runtime_output",
        document_mode
            ? "document_semantic_max_atoms"
            : "query_semantic_max_atoms",
        document_mode ? 192 : 50,
        1,
        65536
    );
    lexical_dims = evoke_checkout_jsonb_int_path(
        atom_space,
        "lexical",
        "end_exclusive",
        0,
        1,
        PG_INT32_MAX
    );
    total_dims = evoke_checkout_jsonb_int_key(
        atom_space,
        "latent_dims",
        0,
        lexical_dims + 1,
        PG_INT32_MAX
    );
    manifest_dims = evoke_checkout_jsonb_int_key(
        manifest,
        "latent_dims",
        0,
        lexical_dims + 1,
        PG_INT32_MAX
    );
    if (manifest_dims != total_dims)
    {
        ereport(ERROR, (errmsg("P2 atom-space dimensions do not match")));
    }
    if (!document_mode)
    {
        global_scale = evoke_checkout_jsonb_double_path(
            scoring,
            "query_compiler",
            "global_scale",
            0.0,
            1.0e12
        );
        multiplier = evoke_checkout_jsonb_double_path(
            scoring,
            "query_compiler",
            "multiplier",
            0.0,
            1.0e12
        );
        clip_min = evoke_checkout_jsonb_double_path(
            scoring,
            "query_compiler",
            "clip_min",
            0.0,
            1.0e12
        );
        clip_max = evoke_checkout_jsonb_double_path(
            scoring,
            "query_compiler",
            "clip_max",
            clip_min,
            1.0e12
        );
    }

    input_ids_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "input_ids",
        NULL
    );
    attention_mask_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "attention_mask",
        NULL
    );
    semantic_ids_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "semantic_ids",
        NULL
    );
    semantic_weights_name = evoke_checkout_manifest_text_path(
        manifest,
        "runtime_io",
        "semantic_weights",
        NULL
    );
    input_names[0] = input_ids_name;
    input_names[1] = attention_mask_name;
    output_names[0] = semantic_ids_name;
    output_names[1] = semantic_weights_name;

    ort = evoke_ort_api();
    evoke_runtime_precision_normalize(
        requested_runtime_precision,
        runtime_precision
    );
    requested_provider = pstrdup("auto");
    active_provider = evoke_p2_ort_active_provider(
        ort,
        semantic_runtime,
        runtime_precision
    );
    evoke_checkout_manifest_signature(manifest, checkout_signature);
    evoke_ort_record_provider(
        requested_provider,
        active_provider,
        runtime_precision
    );
    tokenized = palloc0(sizeof(*tokenized) * (Size) batch_count);
    compiled = palloc0(sizeof(*compiled) * (Size) batch_count);
    resources = palloc0(sizeof(*resources));
    resources->ort = ort;
    session = evoke_ort_cache_session(
        ort,
        manifest,
        model_path,
        encoder_path,
        document_mode,
        requested_provider,
        active_provider,
        runtime_precision,
        checkout_signature,
        &resources->owned_env,
        &resources->owned_session_options,
        &cache_hit,
        &load_ms
    );
    if (resources->owned_env != NULL)
    {
        resources->owned_session = session;
    }

    PG_TRY();
    {
        started_ms = evoke_monotonic_ms();
        for (uint32 i = 0; i < batch_count; i++)
        {
            evoke_p2_tokenize_roberta_windows(
                tokenizer_vocabulary_path,
                tokenizer_merges_path,
                checkout_signature,
                texts[i],
                max_length,
                window_stride,
                max_windows,
                &tokenized[i]
            );
            total_window_count += (uint32) tokenized[i].window_count;
            for (int32 window = 0;
                 window < tokenized[i].window_count;
                 window++)
            {
                sequence_length = Max(
                    sequence_length,
                    tokenized[i].windows[window].token_count
                );
            }
        }
        semantic = evoke_p2_run_semantic_windows(
            ort,
            session,
            input_names,
            output_names,
            tokenized,
            batch_count,
            total_dims - lexical_dims,
            semantic_max_atoms,
            max_batch_size
        );
        for (uint32 i = 0; i < batch_count; i++)
        {
            if (document_mode)
            {
                evoke_p2_compile_document_semantic_atoms(
                    semantic[i].ids,
                    semantic[i].weights,
                    semantic[i].count,
                    lexical_dims,
                    total_dims,
                    &compiled[i]
                );
            }
            else
            {
                evoke_p2_compile_unified_atoms(
                    lexical_vocabulary_path,
                    calibration_path,
                    checkout_signature,
                    texts[i],
                    semantic[i].ids,
                    semantic[i].weights,
                    semantic[i].count,
                    lexical_dims,
                    total_dims,
                    global_scale,
                    multiplier,
                    clip_min,
                    clip_max,
                    &compiled[i]
                );
            }
            if (compiled[i].atom_count > max_output_atoms)
            {
                ereport(
                    ERROR,
                    (
                        errmsg("P2 compiler output exceeds runtime atom limit"),
                        errdetail(
                            "row=%u atoms=%d max_atoms=%d",
                            i + 1,
                            compiled[i].atom_count,
                            max_output_atoms
                        )
                    )
                );
            }
            total_compiled_atoms += (size_t) compiled[i].atom_count;
        }
        /*
         * Keep a completed batch tied to the exact manifest and artifact
         * identities that were validated before inference.
         */
        evoke_checkout_validate_manifest(manifest, NULL, model_path);
        elapsed_ms = evoke_monotonic_ms() - started_ms;
        evoke_ort_session_cache.last_run_ms =
            elapsed_ms < 0.0 ? 0.0 : elapsed_ms;
        evoke_ort_session_cache.last_atom_count = total_compiled_atoms;
        evoke_ort_session_cache.last_used_at = GetCurrentTimestamp();
        for (int i = 0; i < evoke_ort_session_cache.entry_count; i++)
        {
            if (evoke_ort_session_cache.entries[i].session == session)
            {
                evoke_ort_session_cache.entries[i].last_run_ms =
                    evoke_ort_session_cache.last_run_ms;
                evoke_ort_session_cache.entries[i].last_atom_count =
                    total_compiled_atoms;
                evoke_ort_session_cache.entries[i].last_used_at =
                    evoke_ort_session_cache.last_used_at;
                break;
            }
        }

        initStringInfo(&json);
        appendStringInfoString(&json, "{\"results\":[");
        for (uint32 i = 0; i < batch_count; i++)
        {
            if (i > 0)
            {
                appendStringInfoChar(&json, ',');
            }
            evoke_p2_append_compiled_atoms_json(
                &json,
                &compiled[i],
                &tokenized[i],
                document_mode ? "document" : "query"
            );
        }
        appendStringInfoString(
            &json,
            "],\"runtime\":\"onnxruntime\","
            "\"runtime_abi\":"
        );
        evoke_append_json_string(&json, runtime_abi);
        appendStringInfoString(&json, ",\"checkout_signature\":");
        evoke_append_json_string(&json, checkout_signature);
        appendStringInfoString(&json, ",\"compiler_role\":");
        evoke_append_json_string(
            &json,
            document_mode ? "document" : "query"
        );
        appendStringInfoString(&json, ",\"provider\":{\"requested\":");
        evoke_append_json_string(&json, requested_provider);
        appendStringInfoString(&json, ",\"active\":");
        evoke_append_json_string(&json, active_provider);
        appendStringInfoString(&json, "},\"runtime_precision\":");
        evoke_append_json_string(&json, runtime_precision);
        appendStringInfo(
            &json,
            ",\"batch_size\":%u,\"sequence_length\":%d,"
            "\"window_count\":%u,"
            "\"latency_ms\":%.3f,\"session_cache_hit\":%s,"
            "\"session_load_ms\":%.3f}",
            batch_count,
            sequence_length,
            total_window_count,
            elapsed_ms < 0.0 ? 0.0 : elapsed_ms,
            cache_hit ? "true" : "false",
            load_ms < 0.0 ? 0.0 : load_ms
        );
    }
    PG_FINALLY();
    {
        for (uint32 i = 0; i < batch_count; i++)
        {
            evoke_p2_tokenized_windows_free(&tokenized[i]);
            if (compiled[i].atom_ids != NULL)
            {
                pfree(compiled[i].atom_ids);
            }
            if (compiled[i].atom_weights != NULL)
            {
                pfree(compiled[i].atom_weights);
            }
        }
        evoke_p2_semantic_atoms_free(semantic, batch_count);
        pfree(tokenized);
        pfree(compiled);
        evoke_ort_release_run_resources(resources);
        pfree(resources);
    }
    PG_END_TRY();
    return json.data;
}

static char *
evoke_p2_query_atoms_json(
    const char *model_path,
    const char *runtime_precision,
    const char *query_text
)
{
    char *manifest_text;
    Datum manifest;
    char *runtime_abi;
    char *encoder_relpath;
    char *encoder_path;

    if (model_path == NULL || model_path[0] == '\0')
    {
        ereport(ERROR, (errmsg("model path is required")));
    }
    if (query_text == NULL || query_text[0] == '\0')
    {
        ereport(ERROR, (errmsg("query text must not be empty")));
    }

    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    evoke_checkout_validate_manifest(manifest, NULL, model_path);
    runtime_abi = evoke_checkout_jsonb_text_key(
        manifest,
        "runtime_abi",
        ""
    );
    if (strcmp(runtime_abi, "evoke_p2_unified_text_atoms_v2") != 0)
    {
        ereport(
            ERROR,
            (
                errmsg("unsupported Evoke ONNX runtime ABI"),
                errdetail(
                    "runtime_abi=%s; expected "
                    "evoke_p2_unified_text_atoms_v2",
                    runtime_abi
                )
            )
        );
    }
    encoder_relpath = evoke_checkout_manifest_artifact_relpath(
        manifest,
        "query_encoder"
    );
    encoder_path = evoke_checkout_artifact_path(
        model_path,
        "query_encoder",
        encoder_relpath
    );
    return evoke_p2_onnxruntime_query_atoms_json(
        model_path,
        query_text,
        runtime_precision,
        manifest,
        encoder_path,
        "evoke_p2_unified_text_atoms_v2"
    );
}

static char *
evoke_onnxruntime_atoms_batch_json(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    bool document_mode
)
{
    char *manifest_text;
    Datum manifest;
    char *runtime_abi;
    char *encoder_relpath;
    char *encoder_path;
    int32 max_batch_size;

    if (model_path == NULL || model_path[0] == '\0')
    {
        ereport(ERROR, (errmsg("model path is required")));
    }
    if (texts == NULL ||
        batch_count == 0 ||
        batch_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    {
        ereport(ERROR, (errmsg("invalid ONNX text-to-atoms batch size")));
    }
    for (uint32 i = 0; i < batch_count; i++)
    {
        if (texts[i] == NULL || texts[i][0] == '\0')
        {
            ereport(
                ERROR,
                (errmsg("ONNX text-to-atoms batch must not contain empty text"))
            );
        }
    }

    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    evoke_checkout_validate_manifest(manifest, NULL, model_path);
    max_batch_size = evoke_runtime_effective_max_batch_size();
    if (batch_count > (uint32) max_batch_size)
    {
        ereport(
            ERROR,
            (
                errmsg("ONNX batch exceeds the model runtime limit"),
                errdetail(
                    "batch_count=%u max_batch_size=%d",
                    batch_count,
                    max_batch_size
                )
            )
        );
    }
    runtime_abi = evoke_checkout_jsonb_text_key(
        manifest,
        "runtime_abi",
        ""
    );
    if (strcmp(runtime_abi, "evoke_p2_unified_text_atoms_v2") != 0)
    {
        ereport(
            ERROR,
            (
                errmsg("unsupported Evoke ONNX runtime ABI"),
                errdetail(
                    "runtime_abi=%s; expected "
                    "evoke_p2_unified_text_atoms_v2",
                    runtime_abi
                )
            )
        );
    }
    {
        const char *artifact_name =
            document_mode ? "document_encoder" : "query_encoder";

        encoder_relpath = evoke_checkout_manifest_artifact_relpath(
            manifest,
            artifact_name
        );
        encoder_path = evoke_checkout_artifact_path(
            model_path,
            artifact_name,
            encoder_relpath
        );
    }
    return evoke_p2_onnxruntime_atoms_batch_json(
        model_path,
        texts,
        batch_count,
        runtime_precision,
        manifest,
        encoder_path,
        "evoke_p2_unified_text_atoms_v2",
        document_mode
    );
}
#endif

static bool
evoke_runtime_service_signal_backend(
    pid_t pid,
    ProcNumber proc_number
)
{
    PGPROC *proc;

    if (pid <= 0 || proc_number == INVALID_PROC_NUMBER)
    {
        return false;
    }
    proc = ProcNumberGetProc(proc_number);
    if (proc == NULL || proc->pid != pid)
    {
        return false;
    }
    SetLatch(&proc->procLatch);
    return true;
}

static void
evoke_runtime_worker_sighup(SIGNAL_ARGS)
{
    int save_errno = errno;

    evoke_runtime_worker_got_sighup = true;
    if (MyProc != NULL)
    {
        SetLatch(&MyProc->procLatch);
    }
    errno = save_errno;
}

static bool
evoke_runtime_service_signal_worker(pid_t pid, ProcNumber proc_number)
{
    PGPROC *proc;

    if (pid <= 0 || proc_number == INVALID_PROC_NUMBER)
    {
        return false;
    }
    proc = ProcNumberGetProc(proc_number);
    if (proc == NULL || proc->pid != pid)
    {
        return false;
    }
    SetLatch(&proc->procLatch);
    return true;
}

static bool
evoke_runtime_service_preload_configured(void)
{
    const char *preload;
    char *raw_preload;
    List *libraries = NIL;
    ListCell *cell;
    bool configured = false;

    preload = GetConfigOptionByName(
        "shared_preload_libraries",
        NULL,
        true
    );
    if (preload == NULL || preload[0] == '\0')
    {
        return false;
    }

    raw_preload = pstrdup(preload);
    if (!SplitDirectoriesString(raw_preload, ',', &libraries))
    {
        list_free(libraries);
        pfree(raw_preload);
        return false;
    }

    foreach (cell, libraries)
    {
        const char *library = lfirst(cell);
        const char *basename = last_dir_separator(library);

        basename = basename == NULL ? library : basename + 1;
        if (strcmp(basename, "evoke") == 0)
        {
            configured = true;
            break;
        }
    }

    list_free(libraries);
    pfree(raw_preload);
    return configured;
}

void
evoke_runtime_service_initialize_control(evoke_runtime_service_control *control)
{
    uint32 configured_workers = (uint32) Max(evoke_runtime_worker_count, 1);

    memset(control, 0, sizeof(*control));
    control->magic = EVOKE_RUNTIME_SERVICE_MAGIC;
    control->version = EVOKE_RUNTIME_SERVICE_VERSION;
    control->configured_worker_count = Min(
        configured_workers,
        (uint32) EVOKE_RUNTIME_SERVICE_MAX_WORKERS
    );
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_MAX_WORKERS; i++)
    {
        pg_atomic_init_u32(&control->workers[i].cancel_requested, 0);
        pg_atomic_init_u32(&control->workers[i].termination_reason, 0);
        control->workers[i].processing_response_slot =
            EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
        control->workers[i].proc_number = INVALID_PROC_NUMBER;
        control->workers[i].processing_owner_proc_number =
            INVALID_PROC_NUMBER;
        snprintf(
            control->workers[i].provider,
            sizeof(control->workers[i].provider),
            "auto"
        );
        snprintf(
            control->workers[i].active_provider,
            sizeof(control->workers[i].active_provider),
            "cpu"
        );
        snprintf(
            control->workers[i].runtime_precision,
            sizeof(control->workers[i].runtime_precision),
            "%s",
            EVOKE_RUNTIME_PRECISION_FP16
        );
    }
    snprintf(control->provider, sizeof(control->provider), "auto");
    snprintf(control->active_provider, sizeof(control->active_provider), "cpu");
    snprintf(
        control->runtime_precision,
        sizeof(control->runtime_precision),
        "%s",
        EVOKE_RUNTIME_PRECISION_FP16
    );
}

Size
evoke_runtime_service_shmem_size(void)
{
    return MAXALIGN(sizeof(evoke_runtime_service_control));
}

void
evoke_runtime_service_shmem_startup(LWLock *lock)
{
    bool found;

    evoke_runtime_service_lock = lock;
    evoke_runtime_service = ShmemInitStruct(
        "evoke runtime service",
        evoke_runtime_service_shmem_size(),
        &found
    );
    if (!found)
    {
        evoke_runtime_service_initialize_control(evoke_runtime_service);
    }
    else if (
        evoke_runtime_service->magic != EVOKE_RUNTIME_SERVICE_MAGIC ||
        evoke_runtime_service->version != EVOKE_RUNTIME_SERVICE_VERSION
    )
    {
        ereport(
            FATAL,
            (
                errmsg("evoke shared runtime ABI does not match this binary"),
                errdetail(
                    "found magic=%08x version=%u, expected magic=%08x "
                    "version=%u",
                    evoke_runtime_service->magic,
                    evoke_runtime_service->version,
                    EVOKE_RUNTIME_SERVICE_MAGIC,
                    EVOKE_RUNTIME_SERVICE_VERSION
                ),
                errhint(
                    "Restart PostgreSQL with one consistent evoke binary."
                )
            )
        );
    }
}

static uint32
evoke_runtime_service_configured_workers(void)
{
    if (evoke_runtime_service == NULL)
    {
        return 0;
    }
    return Min(
        evoke_runtime_service->configured_worker_count,
        (uint32) EVOKE_RUNTIME_SERVICE_MAX_WORKERS
    );
}

static bool
evoke_runtime_service_attach_if_possible(void)
{
    bool found;

    if (evoke_runtime_service != NULL &&
        evoke_runtime_service->magic == EVOKE_RUNTIME_SERVICE_MAGIC &&
        evoke_runtime_service->version == EVOKE_RUNTIME_SERVICE_VERSION)
    {
        return true;
    }
    if (!evoke_runtime_service_preload_configured())
    {
        return false;
    }

    if (evoke_runtime_service_lock == NULL)
    {
        LWLockPadded *locks =
            GetNamedLWLockTranche(EVOKE_LWLOCK_TRANCHE_NAME);

        evoke_runtime_service_lock =
            &locks[EVOKE_RUNTIME_SERVICE_LOCK_INDEX].lock;
    }
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    evoke_runtime_service = ShmemInitStruct(
        "evoke runtime service",
        evoke_runtime_service_shmem_size(),
        &found
    );
    if (!found)
    {
        evoke_runtime_service_initialize_control(evoke_runtime_service);
    }
    else if (
        evoke_runtime_service->magic != EVOKE_RUNTIME_SERVICE_MAGIC ||
        evoke_runtime_service->version != EVOKE_RUNTIME_SERVICE_VERSION
    )
    {
        uint32 found_magic = evoke_runtime_service->magic;
        uint32 found_version = evoke_runtime_service->version;

        LWLockRelease(AddinShmemInitLock);
        ereport(
            ERROR,
            (
                errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                errmsg("evoke shared runtime ABI does not match this binary"),
                errdetail(
                    "found magic=%08x version=%u, expected magic=%08x "
                    "version=%u",
                    found_magic,
                    found_version,
                    EVOKE_RUNTIME_SERVICE_MAGIC,
                    EVOKE_RUNTIME_SERVICE_VERSION
                ),
                errhint(
                    "Restart the PostgreSQL postmaster after installing or "
                    "upgrading evoke."
                )
            )
        );
    }
    LWLockRelease(AddinShmemInitLock);

    return evoke_runtime_service->magic == EVOKE_RUNTIME_SERVICE_MAGIC &&
        evoke_runtime_service->version == EVOKE_RUNTIME_SERVICE_VERSION;
}

static uint32
evoke_runtime_accelerator_service_count(void)
{
    const char *config = evoke_runtime_accelerators;
    Jsonb *parsed = NULL;
    MemoryContext old_context = CurrentMemoryContext;
    uint32 count = 0;

    if (config == NULL || config[0] == '\0')
    {
        return 0;
    }

    PG_TRY();
    {
        parsed = DatumGetJsonbP(
            DirectFunctionCall1(jsonb_in, CStringGetDatum(config))
        );
        if (JB_ROOT_IS_ARRAY(parsed))
        {
            count = JB_ROOT_COUNT(parsed);
        }
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(old_context);
        FlushErrorState();
        count = 0;
    }
    PG_END_TRY();

    return count;
}

static bool
evoke_runtime_jsonb_string_equals(
    const JsonbValue *value,
    const char *expected
)
{
    size_t expected_len;

    if (value == NULL || expected == NULL || value->type != jbvString)
    {
        return false;
    }
    expected_len = strlen(expected);
    return value->val.string.len == (int) expected_len &&
        strncmp(value->val.string.val, expected, expected_len) == 0;
}

static JsonbValue *
evoke_runtime_jsonb_object_get(JsonbContainer *container, const char *key)
{
    JsonbValue key_value;

    if (container == NULL || key == NULL || !JsonContainerIsObject(container))
    {
        return NULL;
    }
    memset(&key_value, 0, sizeof(key_value));
    key_value.type = jbvString;
    key_value.val.string.val = unconstify(char *, key);
    key_value.val.string.len = (int) strlen(key);
    return findJsonbValueFromContainer(container, JB_FOBJECT, &key_value);
}

static bool
evoke_runtime_jsonb_positive_uint(
    const JsonbValue *value,
    uint32 max_value,
    uint32 *result_out
)
{
    char *number_text;
    char *endptr = NULL;
    unsigned long parsed;

    if (result_out != NULL)
    {
        *result_out = 0;
    }
    if (value == NULL || value->type != jbvNumeric ||
        result_out == NULL || max_value == 0)
    {
        return false;
    }
    number_text = DatumGetCString(
        DirectFunctionCall1(numeric_out, NumericGetDatum(value->val.numeric))
    );
    errno = 0;
    parsed = strtoul(number_text, &endptr, 10);
    if (errno != 0 || endptr == NULL || *endptr != '\0' ||
        parsed == 0)
    {
        pfree(number_text);
        return false;
    }
    *result_out = (uint32) Min(parsed, (unsigned long) max_value);
    pfree(number_text);
    return true;
}

static uint32
evoke_runtime_accelerator_targets_parse(
    EvokeRuntimeAcceleratorTarget *targets,
    uint32 capacity,
    MemoryContext output_context
)
{
    const char *config = evoke_runtime_accelerators;
    Jsonb *parsed = NULL;
    JsonbIterator *iterator;
    JsonbIteratorToken token;
    JsonbValue item;
    uint32 count = 0;
    bool expect_url_value = false;
    bool expect_weight_value = false;
    bool expect_max_batch_value = false;
    char *pending_url = NULL;
    uint32 pending_weight = 0;
    uint32 pending_max_batch_size = 0;

    #define EVOKE_ACCELERATOR_REMEMBER_URL(value, raw_weight, raw_batch) \
        do { \
            if (count < capacity) \
            { \
                targets[count].url = MemoryContextStrdup( \
                    output_context, \
                    (value) \
                ); \
                targets[count].weight = (raw_weight); \
                targets[count].max_batch_size = (raw_batch); \
                count++; \
            } \
        } while (0)

    if (targets == NULL || capacity == 0 ||
        config == NULL || config[0] == '\0')
    {
        return 0;
    }
    parsed = DatumGetJsonbP(
        DirectFunctionCall1(jsonb_in, CStringGetDatum(config))
    );
    if (!JB_ROOT_IS_ARRAY(parsed))
    {
        return 0;
    }
    iterator = JsonbIteratorInit(&parsed->root);
    while ((token = JsonbIteratorNext(
                &iterator,
                &item,
                false)) != WJB_DONE)
    {
        if (token == WJB_KEY)
        {
            expect_url_value =
                evoke_runtime_jsonb_string_equals(&item, "url");
            expect_weight_value =
                evoke_runtime_jsonb_string_equals(&item, "weight");
            expect_max_batch_value =
                evoke_runtime_jsonb_string_equals(&item, "max_batch_size");
            continue;
        }
        if (token == WJB_BEGIN_OBJECT)
        {
            if (pending_url != NULL)
            {
                pfree(pending_url);
            }
            pending_url = NULL;
            pending_weight = 0;
            pending_max_batch_size = 0;
            expect_url_value = false;
            expect_weight_value = false;
            expect_max_batch_value = false;
            continue;
        }
        if (token == WJB_END_OBJECT)
        {
            if (pending_url != NULL && pending_url[0] != '\0')
            {
                EVOKE_ACCELERATOR_REMEMBER_URL(
                    pending_url,
                    pending_weight,
                    pending_max_batch_size
                );
                pfree(pending_url);
                pending_url = NULL;
            }
            pending_weight = 0;
            pending_max_batch_size = 0;
            expect_url_value = false;
            expect_weight_value = false;
            expect_max_batch_value = false;
            continue;
        }
        if (token == WJB_VALUE)
        {
            if (expect_url_value && item.type == jbvString &&
                item.val.string.len > 0)
            {
                if (pending_url != NULL)
                {
                    pfree(pending_url);
                }
                pending_url = pnstrdup(
                    item.val.string.val,
                    item.val.string.len
                );
            }
            else if (expect_weight_value && item.type == jbvNumeric)
            {
                (void) evoke_runtime_jsonb_positive_uint(
                    &item,
                    EVOKE_RUNTIME_ACCELERATOR_MAX_WEIGHT,
                    &pending_weight
                );
            }
            else if (expect_max_batch_value && item.type == jbvNumeric)
            {
                (void) evoke_runtime_jsonb_positive_uint(
                    &item,
                    EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE,
                    &pending_max_batch_size
                );
            }
            expect_url_value = false;
            expect_weight_value = false;
            expect_max_batch_value = false;
            continue;
        }
        if (token == WJB_ELEM && item.type == jbvString &&
            item.val.string.len > 0 && count < capacity)
        {
            char *url = pnstrdup(
                item.val.string.val,
                item.val.string.len
            );
            EVOKE_ACCELERATOR_REMEMBER_URL(url, 1, 0);
            pfree(url);
        }
    }
    if (pending_url != NULL)
    {
        pfree(pending_url);
    }
    #undef EVOKE_ACCELERATOR_REMEMBER_URL
    return count;
}

static uint32
evoke_runtime_accelerator_targets(
    EvokeRuntimeAcceleratorTarget *targets,
    uint32 capacity
)
{
    static MemoryContext cache_context = NULL;
    static char *cached_config = NULL;
    static EvokeRuntimeAcceleratorTarget
        cached_targets[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    static uint32 cached_count = 0;
    MemoryContext caller_context = CurrentMemoryContext;
    MemoryContext old_context;
    uint32 count;

    if (targets == NULL || capacity == 0)
    {
        return 0;
    }
    if (evoke_runtime_accelerators == NULL ||
        evoke_runtime_accelerators[0] == '\0')
    {
        if (cache_context != NULL && cached_config != NULL)
        {
            MemoryContextReset(cache_context);
            cached_config = NULL;
            cached_count = 0;
            memset(cached_targets, 0, sizeof(cached_targets));
        }
        return 0;
    }
    if (cache_context == NULL)
    {
        cache_context = AllocSetContextCreate(
            TopMemoryContext,
            "evoke accelerator target cache",
            ALLOCSET_SMALL_SIZES
        );
    }
    if (cached_config == NULL ||
        strcmp(cached_config, evoke_runtime_accelerators) != 0)
    {
        MemoryContextReset(cache_context);
        cached_config = NULL;
        cached_count = 0;
        memset(cached_targets, 0, sizeof(cached_targets));
        old_context = MemoryContextSwitchTo(cache_context);
        PG_TRY();
        {
            cached_count = evoke_runtime_accelerator_targets_parse(
                cached_targets,
                EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES,
                cache_context
            );
            cached_config = pstrdup(evoke_runtime_accelerators);
        }
        PG_CATCH();
        {
            MemoryContextSwitchTo(old_context);
            MemoryContextReset(cache_context);
            cached_config = NULL;
            cached_count = 0;
            memset(cached_targets, 0, sizeof(cached_targets));
            PG_RE_THROW();
        }
        PG_END_TRY();
        MemoryContextSwitchTo(old_context);
    }
    count = Min(cached_count, capacity);
    for (uint32 i = 0; i < count; i++)
    {
        targets[i] = cached_targets[i];
        targets[i].url = MemoryContextStrdup(
            caller_context,
            cached_targets[i].url
        );
    }
    return count;
}

static bool
evoke_runtime_accelerator_parse_url(
    const char *url,
    char **host_out,
    int *port_out,
    char **path_out
)
{
    const char *prefix = "http://";
    const char *host_start;
    const char *cursor;
    const char *host_end;
    const char *path_start;
    char *port_text = NULL;
    long port = 8042;

    if (url == NULL || host_out == NULL || port_out == NULL ||
        path_out == NULL || strncmp(url, prefix, strlen(prefix)) != 0)
    {
        return false;
    }
    host_start = url + strlen(prefix);
    cursor = host_start;
    while (*cursor != '\0' && *cursor != ':' && *cursor != '/')
    {
        cursor++;
    }
    host_end = cursor;
    if (host_end == host_start)
    {
        return false;
    }
    if (*cursor == ':')
    {
        char *endptr;
        const char *port_start = ++cursor;
        bool valid_port;

        while (*cursor != '\0' && *cursor != '/')
        {
            cursor++;
        }
        if (cursor == port_start)
        {
            return false;
        }
        port_text = pnstrdup(port_start, cursor - port_start);
        errno = 0;
        port = strtol(port_text, &endptr, 10);
        valid_port = errno == 0 && endptr != NULL && *endptr == '\0' &&
            port > 0 && port <= 65535;
        pfree(port_text);
        if (!valid_port)
        {
            return false;
        }
    }
    path_start = *cursor == '/' ? cursor : "/";
    *host_out = pnstrdup(host_start, host_end - host_start);
    *port_out = (int) port;
    if (strcmp(path_start, "/") == 0)
    {
        *path_out = pstrdup("/v1/encode");
    }
    else if (strlen(path_start) >= strlen("/v1/encode") &&
             strcmp(path_start + strlen(path_start) -
                    strlen("/v1/encode"), "/v1/encode") == 0)
    {
        *path_out = pstrdup(path_start);
    }
    else if (path_start[strlen(path_start) - 1] == '/')
    {
        *path_out = psprintf("%sv1/encode", path_start);
    }
    else
    {
        *path_out = psprintf("%s/v1/encode", path_start);
    }
    return true;
}

static bool
evoke_runtime_accelerator_idle_connection_reusable(int fd)
{
    char probe;
    int flags;

    if (fd < 0)
    {
        return false;
    }
    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        return false;
    }
    if ((flags & O_NONBLOCK) == 0 &&
        fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
    {
        return false;
    }
    for (;;)
    {
        ssize_t got = recv(fd, &probe, sizeof(probe), MSG_PEEK);

        if (got > 0)
        {
            return false;
        }
        if (got == 0)
        {
            return false;
        }
        if (errno == EINTR)
        {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return true;
        }
        return false;
    }
}

static int
evoke_runtime_accelerator_take_idle_connection(const char *url)
{
    TimestampTz now;

    if (url == NULL || url[0] == '\0')
    {
        return -1;
    }
    now = GetCurrentTimestamp();
    for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS; i++)
    {
        EvokeRuntimeAcceleratorIdleConnection *entry =
            &evoke_runtime_accelerator_idle_connections[i];

        if (entry->occupied && strcmp(entry->url, url) == 0)
        {
            int fd = entry->fd;
            bool expired = entry->last_used_at > 0 &&
                now - entry->last_used_at >
                    (TimestampTz)
                        EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTION_MAX_AGE_MS *
                        1000;

            memset(entry, 0, sizeof(*entry));
            entry->fd = -1;
            if (!expired &&
                evoke_runtime_accelerator_idle_connection_reusable(fd))
            {
                return fd;
            }
            close(fd);
        }
    }
    return -1;
}

static uint32
evoke_runtime_accelerator_idle_connection_count(const char *url)
{
    uint32 count = 0;
    TimestampTz now;

    if (url == NULL || url[0] == '\0')
    {
        return 0;
    }
    now = GetCurrentTimestamp();
    for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS; i++)
    {
        EvokeRuntimeAcceleratorIdleConnection *entry =
            &evoke_runtime_accelerator_idle_connections[i];

        if (entry->occupied && entry->last_used_at > 0 &&
            now - entry->last_used_at >
                (TimestampTz)
                    EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTION_MAX_AGE_MS *
                    1000)
        {
            if (entry->fd >= 0)
            {
                close(entry->fd);
            }
            memset(entry, 0, sizeof(*entry));
            entry->fd = -1;
            continue;
        }
        if (entry->occupied && strcmp(entry->url, url) == 0)
        {
            count++;
        }
    }
    return count;
}

static void
evoke_runtime_accelerator_release_idle_connection(
    const char *url,
    int fd
)
{
    EvokeRuntimeAcceleratorIdleConnection *slot = NULL;
    EvokeRuntimeAcceleratorIdleConnection *oldest = NULL;
    TimestampTz now;

    if (url == NULL || url[0] == '\0' || fd < 0)
    {
        if (fd >= 0)
        {
            close(fd);
        }
        return;
    }
    now = GetCurrentTimestamp();
    for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS; i++)
    {
        EvokeRuntimeAcceleratorIdleConnection *entry =
            &evoke_runtime_accelerator_idle_connections[i];

        if (entry->occupied && entry->last_used_at > 0 &&
            now - entry->last_used_at >
                (TimestampTz)
                    EVOKE_RUNTIME_ACCELERATOR_IDLE_CONNECTION_MAX_AGE_MS *
                    1000)
        {
            if (entry->fd >= 0)
            {
                close(entry->fd);
            }
            memset(entry, 0, sizeof(*entry));
            entry->fd = -1;
        }
        if (!entry->occupied)
        {
            slot = entry;
            break;
        }
        if (oldest == NULL ||
            entry->last_used_at < oldest->last_used_at)
        {
            oldest = entry;
        }
    }
    if (slot == NULL)
    {
        slot = oldest;
        if (slot == NULL)
        {
            close(fd);
            return;
        }
        if (slot->fd >= 0)
        {
            close(slot->fd);
        }
    }
    memset(slot, 0, sizeof(*slot));
    slot->occupied = true;
    slot->fd = fd;
    slot->last_used_at = now;
    strlcpy(slot->url, url, sizeof(slot->url));
}

static bool
evoke_runtime_accelerator_send_all(int fd, const char *data, Size len)
{
    Size sent = 0;

    while (sent < len)
    {
        ssize_t written = send(fd, data + sent, len - sent, 0);

        if (written < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                struct pollfd poll_fd;
                int rc;

                memset(&poll_fd, 0, sizeof(poll_fd));
                poll_fd.fd = fd;
                poll_fd.events = POLLOUT | POLLERR | POLLHUP;
                rc = poll(
                    &poll_fd,
                    1,
                    EVOKE_RUNTIME_ACCELERATOR_SOCKET_IO_TIMEOUT_MS
                );
                if (rc > 0 &&
                    (poll_fd.revents & POLLOUT) != 0 &&
                    (poll_fd.revents & (POLLERR | POLLHUP)) == 0)
                {
                    continue;
                }
            }
            return false;
        }
        if (written == 0)
        {
            return false;
        }
        sent += (Size) written;
    }
    return true;
}

static bool
evoke_runtime_accelerator_set_nonblocking(int fd, int *flags_out)
{
    int flags;

    if (flags_out == NULL)
    {
        return false;
    }
    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        return false;
    }
    *flags_out = flags;
    if ((flags & O_NONBLOCK) != 0)
    {
        return true;
    }
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void
evoke_runtime_accelerator_restore_flags(int fd, int flags)
{
    if (fd >= 0 && flags >= 0)
    {
        (void) fcntl(fd, F_SETFL, flags);
    }
}

static bool
evoke_runtime_accelerator_connect_with_timeout(
    int fd,
    const struct sockaddr *address,
    socklen_t address_len
)
{
    int flags = -1;
    int rc;
    struct pollfd poll_fd;
    int socket_error = 0;
    socklen_t socket_error_len = sizeof(socket_error);

    if (!evoke_runtime_accelerator_set_nonblocking(fd, &flags))
    {
        return false;
    }
    rc = connect(fd, address, address_len);
    if (rc == 0)
    {
        evoke_runtime_accelerator_restore_flags(fd, flags);
        return true;
    }
    if (errno != EINPROGRESS && errno != EWOULDBLOCK && errno != EINTR)
    {
        evoke_runtime_accelerator_restore_flags(fd, flags);
        return false;
    }

    for (;;)
    {
        memset(&poll_fd, 0, sizeof(poll_fd));
        poll_fd.fd = fd;
        poll_fd.events = POLLOUT | POLLERR | POLLHUP;
        rc = poll(&poll_fd, 1, EVOKE_RUNTIME_ACCELERATOR_CONNECT_TIMEOUT_MS);
        if (rc < 0 && errno == EINTR)
        {
            CHECK_FOR_INTERRUPTS();
            continue;
        }
        break;
    }
    if (rc <= 0 ||
        (poll_fd.revents & (POLLOUT | POLLERR | POLLHUP)) == 0)
    {
        evoke_runtime_accelerator_restore_flags(fd, flags);
        return false;
    }
    if (getsockopt(
            fd,
            SOL_SOCKET,
            SO_ERROR,
            &socket_error,
            &socket_error_len) != 0 ||
        socket_error != 0)
    {
        if (socket_error != 0)
        {
            errno = socket_error;
        }
        evoke_runtime_accelerator_restore_flags(fd, flags);
        return false;
    }
    evoke_runtime_accelerator_restore_flags(fd, flags);
    return true;
}

static uint32
evoke_runtime_accelerator_backoff_ms(uint32 failures)
{
    uint32 backoff = EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MIN_MS;

    if (failures == 0)
    {
        return 0;
    }
    for (uint32 step = 1; step < failures; step++)
    {
        if (backoff >= EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MAX_MS / 2)
        {
            return EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MAX_MS;
        }
        backoff *= 2;
    }
    if (backoff > EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MAX_MS)
    {
        return EVOKE_RUNTIME_ACCELERATOR_BACKOFF_MAX_MS;
    }
    return backoff;
}

static uint32
evoke_runtime_accelerator_failure_backoff_ms(
    const evoke_runtime_accelerator_state *entry
)
{
    if (entry == NULL || entry->consecutive_failures == 0)
    {
        return 0;
    }
    if (entry->successes > 0 &&
        entry->consecutive_failures <
            EVOKE_RUNTIME_ACCELERATOR_HEALTHY_FAILURE_SOFT_LIMIT)
    {
        return 0;
    }
    return evoke_runtime_accelerator_backoff_ms(entry->consecutive_failures);
}

static double
evoke_runtime_accelerator_elapsed_ms(TimestampTz start, TimestampTz end)
{
    if (start == 0 || end <= start)
    {
        return 0.0;
    }
    return (double) (end - start) / 1000.0;
}

static int
evoke_runtime_accelerator_request_timeout_ms(void)
{
    return evoke_runtime_liveness_timeout_ms;
}

static int64
evoke_runtime_accelerator_stale_inflight_ms(void)
{
    int timeout_ms = evoke_runtime_accelerator_request_timeout_ms();

    if (timeout_ms <= 0)
    {
        return 0;
    }
    return (int64) timeout_ms *
        EVOKE_RUNTIME_ACCELERATOR_STALE_INFLIGHT_MULTIPLIER;
}

static bool
evoke_runtime_accelerator_request_timed_out(
    const EvokeRuntimeRequestHandle *handle
)
{
    double elapsed_ms;
    int timeout_ms;

    if (handle == NULL || handle->accelerator_started_at == 0)
    {
        return false;
    }
    timeout_ms = evoke_runtime_accelerator_request_timeout_ms();
    if (timeout_ms <= 0)
    {
        return false;
    }
    elapsed_ms = evoke_runtime_accelerator_elapsed_ms(
        handle->accelerator_started_at,
        GetCurrentTimestamp()
    );
    return elapsed_ms >= (double) timeout_ms;
}

static double
evoke_runtime_accelerator_ewma(double previous, double observed)
{
    if (observed <= 0.0)
    {
        return previous;
    }
    if (previous <= 0.0)
    {
        return observed;
    }
    return previous * 0.8 + observed * 0.2;
}

static bool
evoke_runtime_accelerator_stale_in_flight(
    const evoke_runtime_accelerator_state *entry,
    TimestampTz now
)
{
    TimestampTz stale_after;
    int64 stale_ms;

    if (entry == NULL || entry->in_flight == 0 ||
        entry->last_started_at == 0)
    {
        return false;
    }
    stale_ms = evoke_runtime_accelerator_stale_inflight_ms();
    if (stale_ms <= 0)
    {
        return false;
    }
    stale_after =
        entry->last_started_at +
        (TimestampTz) stale_ms * 1000;
    return now > stale_after;
}

static uint32
evoke_runtime_accelerator_effective_in_flight(
    const evoke_runtime_accelerator_state *entry,
    TimestampTz now
)
{
    if (evoke_runtime_accelerator_stale_in_flight(entry, now))
    {
        return 0;
    }
    return entry == NULL ? 0 : entry->in_flight;
}

static void
evoke_runtime_accelerator_expire_stale_locked(
    evoke_runtime_accelerator_state *entry,
    TimestampTz now
)
{
    if (evoke_runtime_accelerator_stale_in_flight(entry, now))
    {
        entry->in_flight = 0;
        entry->last_started_at = 0;
    }
}

static evoke_runtime_accelerator_state *
evoke_runtime_accelerator_find_locked(const char *url, bool create)
{
    evoke_runtime_accelerator_state *empty = NULL;

    if (url == NULL || url[0] == '\0' ||
        strlen(url) >= EVOKE_RUNTIME_ACCELERATOR_URL_MAX_BYTES)
    {
        return NULL;
    }
    for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES; i++)
    {
        evoke_runtime_accelerator_state *entry =
            &evoke_runtime_service->accelerators[i];

        if (entry->occupied && strcmp(entry->url, url) == 0)
        {
            return entry;
        }
        if (!entry->occupied && empty == NULL)
        {
            empty = entry;
        }
    }
    if (!create)
    {
        return NULL;
    }
    if (empty == NULL)
    {
        for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES; i++)
        {
            evoke_runtime_accelerator_state *entry =
                &evoke_runtime_service->accelerators[i];

            if (entry->in_flight == 0)
            {
                empty = entry;
                break;
            }
        }
    }
    if (empty == NULL)
    {
        return NULL;
    }
    memset(empty, 0, sizeof(*empty));
    empty->occupied = true;
    strlcpy(empty->url, url, sizeof(empty->url));
    return empty;
}

static bool
evoke_runtime_accelerator_copy_state(
    const char *url,
    evoke_runtime_accelerator_state *out
)
{
    evoke_runtime_accelerator_state *entry;
    bool found = false;

    if (out == NULL || !evoke_runtime_service_attach_if_possible())
    {
        return false;
    }
    memset(out, 0, sizeof(*out));
    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    entry = evoke_runtime_accelerator_find_locked(url, false);
    if (entry != NULL)
    {
        *out = *entry;
        found = true;
    }
    LWLockRelease(evoke_runtime_service_lock);
    return found;
}

static bool
evoke_runtime_accelerator_backoff_active(const char *url, TimestampTz now)
{
    evoke_runtime_accelerator_state entry;

    return evoke_runtime_accelerator_copy_state(url, &entry) &&
        entry.next_probe_at != 0 &&
        entry.next_probe_at > now;
}

static bool
evoke_runtime_accelerator_probe_needed(const char *url, TimestampTz now)
{
    evoke_runtime_accelerator_state entry;

    if (!evoke_runtime_accelerator_copy_state(url, &entry))
    {
        return true;
    }
    if (evoke_runtime_accelerator_effective_in_flight(&entry, now) > 0)
    {
        return false;
    }
    if (entry.successes == 0 &&
        entry.consecutive_failures == 0 &&
        entry.request_failures == 0)
    {
        return true;
    }
    if (entry.next_probe_at != 0 &&
        entry.next_probe_at <= now &&
        (entry.consecutive_failures > 0 || entry.request_failures > 0))
    {
        return true;
    }
    return false;
}

static uint64
evoke_runtime_accelerator_next_schedule_cursor(void)
{
    uint64 cursor = 0;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return 0;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    cursor = evoke_runtime_service->accelerator_schedule_cursor++;
    LWLockRelease(evoke_runtime_service_lock);
    return cursor;
}

static uint64
evoke_runtime_accelerator_schedule_cursor_snapshot(void)
{
    uint64 cursor = 0;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return 0;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    cursor = evoke_runtime_service->accelerator_schedule_cursor;
    LWLockRelease(evoke_runtime_service_lock);
    return cursor;
}

#ifndef EVOKE_ENABLE_ONNXRUNTIME
static bool
evoke_runtime_accelerator_wait_required_retry(uint32 *waited_ms)
{
    if (waited_ms == NULL ||
        *waited_ms >=
            EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT *
            EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS)
    {
        return false;
    }
    CHECK_FOR_INTERRUPTS();
    ResetLatch(&MyProc->procLatch);
    (void) WaitLatch(
        &MyProc->procLatch,
        WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
        EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS,
        0
    );
    if (UINT32_MAX - *waited_ms <
        EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS)
    {
        *waited_ms = UINT32_MAX;
    }
    else
    {
        *waited_ms += EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS;
    }
    return true;
}
#endif

static uint32
evoke_runtime_accelerator_local_capacity(void)
{
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    return 0;
#else
    uint32 ready_workers = 0;
    uint32 pending_window;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return 1;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    for (uint32 i = 0; i < evoke_runtime_service_configured_workers(); i++)
    {
        if (evoke_runtime_service->workers[i].ready)
        {
            ready_workers++;
        }
    }
    ready_workers = evoke_runtime_service_max_document_workers_for_ready(
        ready_workers
    );
    LWLockRelease(evoke_runtime_service_lock);
    if (ready_workers == 0)
    {
        return 1;
    }
    pending_window = evoke_runtime_effective_local_document_pipeline_depth();
    if (UINT32_MAX - ready_workers < pending_window)
    {
        return UINT32_MAX;
    }
    return Max(ready_workers + pending_window, 1);
#endif
}

static uint32
evoke_runtime_configured_document_pipeline_depth(void)
{
    uint32 configured = (uint32) Max(
        evoke_runtime_document_pipeline_depth,
        1
    );

    configured = Min(configured, (uint32) EVOKE_RUNTIME_BUILDER_MAX_INFLIGHT);
    return Max(configured, 1);
}

static uint32
evoke_runtime_accelerator_local_max_batch_size(void)
{
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    return 0;
#else
    return (uint32) evoke_runtime_effective_max_batch_size();
#endif
}

static uint32
evoke_runtime_accelerator_target_max_batch_size(
    const EvokeRuntimeAcceleratorTarget *target
)
{
    if (target != NULL && target->max_batch_size > 0)
    {
        return Min(
            target->max_batch_size,
            (uint32) EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
        );
    }
    return (uint32) evoke_runtime_effective_max_batch_size();
}

static uint32
evoke_runtime_accelerator_candidate_batch_size(
    uint32 available_count,
    uint32 max_batch_size,
    bool exact_batch_count
)
{
    if (available_count == 0 || max_batch_size == 0)
    {
        return 0;
    }
    if (exact_batch_count && available_count > max_batch_size)
    {
        return 0;
    }
    if (exact_batch_count)
    {
        return Min(
            available_count,
            (uint32) EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
        );
    }
    return Min(
        available_count,
        Min(max_batch_size, (uint32) EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    );
}

static uint32
evoke_runtime_accelerator_default_remote_capacity(uint32 target_count)
{
    (void) target_count;
    return 1;
}

static uint32
evoke_runtime_accelerator_target_capacity(
    const EvokeRuntimeAcceleratorTarget *target,
    const EvokeRuntimeAcceleratorTarget *targets,
    uint32 target_count
)
{
    uint32 capacity;
    uint32 pipeline_depth;

    (void) targets;
    if (target == NULL || target_count == 0)
    {
        return evoke_runtime_accelerator_default_remote_capacity(target_count);
    }
    pipeline_depth = evoke_runtime_configured_document_pipeline_depth();
    capacity = Max(target->weight, 1);
    return Max(Min(capacity, pipeline_depth), 1);
}

static uint32
evoke_runtime_accelerator_total_capacity(void)
{
    EvokeRuntimeAcceleratorTarget
        targets[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    uint32 target_count;
    uint32 configured = evoke_runtime_configured_document_pipeline_depth();
    uint32 total = 0;

#ifdef EVOKE_ENABLE_ONNXRUNTIME
    total = evoke_runtime_accelerator_local_capacity();
#endif
    target_count = evoke_runtime_accelerator_targets(
        targets,
        EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES
    );
    for (uint32 i = 0; i < target_count; i++)
    {
        uint32 capacity = evoke_runtime_accelerator_target_capacity(
            &targets[i],
            targets,
            target_count
        );

        if (UINT32_MAX - total < capacity)
        {
            total = UINT32_MAX;
            break;
        }
        total += capacity;
    }
    for (uint32 i = 0; i < target_count; i++)
    {
        if (targets[i].url != NULL)
        {
            pfree(targets[i].url);
        }
    }
    if (total == 0)
    {
        total = 1;
    }
    return Max(Min(total, configured), 1);
}

static uint32
evoke_runtime_accelerator_capacity_for_url(
    const char *url,
    const EvokeRuntimeAcceleratorTarget *targets,
    uint32 target_count,
    uint32 *configured_weight_out,
    uint32 *configured_max_batch_size_out
)
{
    if (configured_weight_out != NULL)
    {
        *configured_weight_out = 0;
    }
    if (configured_max_batch_size_out != NULL)
    {
        *configured_max_batch_size_out = 0;
    }
    if (url == NULL || targets == NULL)
    {
        return evoke_runtime_accelerator_default_remote_capacity(target_count);
    }
    for (uint32 i = 0; i < target_count; i++)
    {
        if (targets[i].url != NULL && strcmp(targets[i].url, url) == 0)
        {
            if (configured_weight_out != NULL)
            {
                *configured_weight_out = targets[i].weight;
            }
            if (configured_max_batch_size_out != NULL)
            {
                *configured_max_batch_size_out =
                    targets[i].max_batch_size;
            }
            return evoke_runtime_accelerator_target_capacity(
                &targets[i],
                targets,
                target_count
            );
        }
    }
    return evoke_runtime_accelerator_default_remote_capacity(target_count);
}

static uint32
evoke_runtime_accelerator_local_in_flight_locked(void)
{
    uint32 pending = evoke_runtime_service_pending_document_requests_locked();
    uint32 active = evoke_runtime_service_active_document_workers_locked();
    uint32 responses = evoke_runtime_service_document_response_slots_locked();
    uint32 running;

    if (UINT32_MAX - pending < active)
    {
        return UINT32_MAX;
    }
    running = pending + active;
    return Max(running, responses);
}

static double
evoke_runtime_accelerator_score(
    const EvokeRuntimeAcceleratorTarget *target,
    const EvokeRuntimeAcceleratorTarget *targets,
    uint32 target_count,
    uint32 batch_count
)
{
    evoke_runtime_accelerator_state entry;
    double batch_ms;
    uint32 capacity;
    double queue_factor;
    TimestampTz now = GetCurrentTimestamp();
    uint32 in_flight = 0;

    if (target == NULL || target->url == NULL)
    {
        return DBL_MAX;
    }
    if (batch_count > evoke_runtime_accelerator_target_max_batch_size(target))
    {
        return DBL_MAX;
    }
    capacity = evoke_runtime_accelerator_target_capacity(
        target,
        targets,
        target_count
    );
    batch_ms =
        EVOKE_RUNTIME_ACCELERATOR_UNKNOWN_MS_PER_TEXT *
        (double) Max(batch_count, 1);
    memset(&entry, 0, sizeof(entry));
    if (evoke_runtime_accelerator_copy_state(target->url, &entry) &&
        entry.ewma_ms_per_text > 0.0)
    {
        batch_ms =
            entry.ewma_ms_per_text * (double) Max(batch_count, 1);
        in_flight = evoke_runtime_accelerator_effective_in_flight(
            &entry,
            now
        );
    }
    else
    {
        in_flight = evoke_runtime_accelerator_effective_in_flight(
            &entry,
            now
        );
    }
    if (capacity == 0 || in_flight >= capacity)
    {
        return DBL_MAX;
    }
    queue_factor =
        ((double) (in_flight + 1)) / (double) capacity;
    return batch_ms * queue_factor;
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static double
evoke_runtime_accelerator_local_score(uint32 batch_count)
{
    double batch_ms =
        EVOKE_RUNTIME_ACCELERATOR_UNKNOWN_MS_PER_TEXT *
        (double) Max(batch_count, 1);
    uint32 capacity = evoke_runtime_accelerator_local_capacity();
    uint32 in_flight = 0;

    if (batch_count > evoke_runtime_accelerator_local_max_batch_size())
    {
        return DBL_MAX;
    }
    if (evoke_runtime_service_attach_if_possible())
    {
        LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
        in_flight = evoke_runtime_accelerator_local_in_flight_locked();
        if (evoke_runtime_service->accelerator_local_ewma_ms_per_text > 0.0)
        {
            batch_ms =
                evoke_runtime_service->accelerator_local_ewma_ms_per_text *
                (double) Max(batch_count, 1);
        }
        LWLockRelease(evoke_runtime_service_lock);
    }
    if (capacity == 0)
    {
        capacity = 1;
    }
    if (in_flight >= capacity)
    {
        return DBL_MAX;
    }
    return batch_ms * ((double) (in_flight + 1)) / (double) capacity;
}
#endif

uint32
evoke_runtime_service_recommended_document_batch_size(void)
{
    EvokeRuntimeAcceleratorTarget
        targets[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    uint32 target_count;
    uint32 best_batch_size =
        (uint32) evoke_runtime_effective_max_batch_size();
    double best_score_per_text = DBL_MAX;
    uint64 schedule_cursor;
    uint32 start;
    TimestampTz now = GetCurrentTimestamp();

    target_count = evoke_runtime_accelerator_targets(
        targets,
        EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES
    );
    schedule_cursor = evoke_runtime_accelerator_schedule_cursor_snapshot();
    start = target_count == 0
        ? 0
        : (uint32) (schedule_cursor % (uint64) (target_count + 1));

    for (uint32 offset = 0; offset <= target_count; offset++)
    {
        uint32 index = target_count == 0
            ? target_count
            : (start + offset) % (target_count + 1);
        uint32 candidate_batch_size;
        double score;
        double score_per_text;

        if (index == target_count)
        {
#ifdef EVOKE_ENABLE_ONNXRUNTIME
            candidate_batch_size =
                evoke_runtime_accelerator_local_max_batch_size();
            if (candidate_batch_size == 0)
            {
                continue;
            }
            score = evoke_runtime_accelerator_local_score(
                candidate_batch_size
            );
#else
            continue;
#endif
        }
        else
        {
            if (evoke_runtime_accelerator_backoff_active(
                    targets[index].url,
                    now) &&
                !evoke_runtime_accelerator_probe_needed(
                    targets[index].url,
                    now))
            {
                continue;
            }
            candidate_batch_size =
                evoke_runtime_accelerator_target_max_batch_size(
                    &targets[index]
                );
            score = evoke_runtime_accelerator_score(
                &targets[index],
                targets,
                target_count,
                candidate_batch_size
            );
        }
        if (candidate_batch_size == 0 || score == DBL_MAX)
        {
            continue;
        }
        score_per_text = score / (double) candidate_batch_size;
        if (score_per_text < best_score_per_text ||
            (
                score_per_text == best_score_per_text &&
                candidate_batch_size > best_batch_size
            ))
        {
            best_score_per_text = score_per_text;
            best_batch_size = candidate_batch_size;
        }
    }
    for (uint32 i = 0; i < target_count; i++)
    {
        if (targets[i].url != NULL)
        {
            pfree(targets[i].url);
        }
    }
    return Min(
        Max(best_batch_size, 1U),
        (uint32) EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
    );
}

static TimestampTz
evoke_runtime_accelerator_note_start(const char *url)
{
    evoke_runtime_accelerator_state *entry;
    TimestampTz now = GetCurrentTimestamp();

    if (!evoke_runtime_service_attach_if_possible())
    {
        return now;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, true);
    if (entry != NULL)
    {
        evoke_runtime_accelerator_expire_stale_locked(entry, now);
        if (entry->in_flight < UINT32_MAX)
        {
            entry->in_flight++;
        }
        entry->last_started_at = now;
    }
    LWLockRelease(evoke_runtime_service_lock);
    return now;
}

static void
evoke_runtime_accelerator_note_abandoned(const char *url)
{
    evoke_runtime_accelerator_state *entry;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, false);
    if (entry != NULL && entry->in_flight > 0)
    {
        entry->in_flight--;
        if (entry->in_flight == 0)
        {
            entry->last_started_at = 0;
        }
    }
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_runtime_accelerator_note_success(
    const char *url,
    uint32 batch_count,
    TimestampTz started_at
)
{
    evoke_runtime_accelerator_state *entry;
    double elapsed_ms;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, true);
    if (entry == NULL)
    {
        LWLockRelease(evoke_runtime_service_lock);
        return;
    }
    if (entry->in_flight > 0)
    {
        entry->in_flight--;
        if (entry->in_flight == 0)
        {
            entry->last_started_at = 0;
        }
    }
    entry->consecutive_failures = 0;
    entry->next_probe_at = 0;
    entry->successes++;
    elapsed_ms = evoke_runtime_accelerator_elapsed_ms(
        started_at,
        GetCurrentTimestamp()
    );
    if (elapsed_ms > 0.0)
    {
        entry->ewma_batch_ms = evoke_runtime_accelerator_ewma(
            entry->ewma_batch_ms,
            elapsed_ms
        );
        entry->ewma_ms_per_text = evoke_runtime_accelerator_ewma(
            entry->ewma_ms_per_text,
            elapsed_ms / (double) Max(batch_count, 1)
        );
    }
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_runtime_accelerator_note_connect_failure(const char *url)
{
    evoke_runtime_accelerator_state *entry;
    uint32 backoff;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, true);
    if (entry == NULL)
    {
        LWLockRelease(evoke_runtime_service_lock);
        return;
    }
    if (entry->consecutive_failures < UINT32_MAX)
    {
        entry->consecutive_failures++;
    }
    entry->connect_failures++;
    backoff = evoke_runtime_accelerator_failure_backoff_ms(entry);
    entry->next_probe_at = GetCurrentTimestamp() + (TimestampTz) backoff * 1000;
    LWLockRelease(evoke_runtime_service_lock);
}

typedef enum evoke_runtime_accelerator_failure_kind
{
    EVOKE_RUNTIME_ACCELERATOR_FAILURE_READ,
    EVOKE_RUNTIME_ACCELERATOR_FAILURE_STATUS,
    EVOKE_RUNTIME_ACCELERATOR_FAILURE_BODY
} evoke_runtime_accelerator_failure_kind;

static void
evoke_runtime_accelerator_note_request_failure(
    const char *url,
    evoke_runtime_accelerator_failure_kind failure_kind,
    int status,
    const char *status_body
)
{
    evoke_runtime_accelerator_state *entry;
    uint32 backoff;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, true);
    if (entry == NULL)
    {
        LWLockRelease(evoke_runtime_service_lock);
        return;
    }
    if (entry->in_flight > 0)
    {
        entry->in_flight--;
        if (entry->in_flight == 0)
        {
            entry->last_started_at = 0;
        }
    }
    entry->request_failures++;
    switch (failure_kind)
    {
        case EVOKE_RUNTIME_ACCELERATOR_FAILURE_READ:
            entry->read_failures++;
            break;
        case EVOKE_RUNTIME_ACCELERATOR_FAILURE_STATUS:
            entry->status_failures++;
            entry->last_status_failure = (uint32) Max(status, 0);
            if (status_body != NULL)
            {
                strlcpy(
                    entry->last_status_body,
                    status_body,
                    sizeof(entry->last_status_body)
                );
            }
            switch (status)
            {
                case 400:
                    entry->status_400_failures++;
                    break;
                case 409:
                    entry->status_409_failures++;
                    break;
                case 500:
                    entry->status_500_failures++;
                    break;
                case 504:
                    entry->status_504_failures++;
                    break;
                default:
                    entry->status_other_failures++;
                    break;
            }
            break;
        case EVOKE_RUNTIME_ACCELERATOR_FAILURE_BODY:
            entry->body_failures++;
            break;
    }
    if (entry->consecutive_failures < UINT32_MAX)
    {
        entry->consecutive_failures++;
    }
    backoff = evoke_runtime_accelerator_failure_backoff_ms(entry);
    entry->next_probe_at = GetCurrentTimestamp() + (TimestampTz) backoff * 1000;
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_runtime_accelerator_note_backpressure(const char *url)
{
    evoke_runtime_accelerator_state *entry;

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    entry = evoke_runtime_accelerator_find_locked(url, true);
    if (entry == NULL)
    {
        LWLockRelease(evoke_runtime_service_lock);
        return;
    }
    if (entry->in_flight > 0)
    {
        entry->in_flight--;
        if (entry->in_flight == 0)
        {
            entry->last_started_at = 0;
        }
    }
    entry->request_failures++;
    entry->backpressure_failures++;
    entry->status_503_failures++;
    entry->last_status_failure = 503;
    entry->next_probe_at = GetCurrentTimestamp() +
        (TimestampTz) EVOKE_RUNTIME_ACCELERATOR_BACKPRESSURE_MS * 1000;
    LWLockRelease(evoke_runtime_service_lock);
}

static TimestampTz
evoke_runtime_accelerator_note_local_start(uint32 batch_count)
{
    (void) batch_count;
    return GetCurrentTimestamp();
}

static void
evoke_runtime_accelerator_note_local_success(
    uint32 batch_count,
    TimestampTz started_at
)
{
    double elapsed_ms = evoke_runtime_accelerator_elapsed_ms(
        started_at,
        GetCurrentTimestamp()
    );

    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    evoke_runtime_service->accelerator_local_successes++;
    if (elapsed_ms > 0.0)
    {
        evoke_runtime_service->accelerator_local_ewma_batch_ms =
            evoke_runtime_accelerator_ewma(
                evoke_runtime_service->accelerator_local_ewma_batch_ms,
                elapsed_ms
            );
        evoke_runtime_service->accelerator_local_ewma_ms_per_text =
            evoke_runtime_accelerator_ewma(
                evoke_runtime_service->accelerator_local_ewma_ms_per_text,
                elapsed_ms / (double) Max(batch_count, 1)
            );
    }
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_runtime_accelerator_note_local_abandoned(void)
{
    if (!evoke_runtime_service_attach_if_possible())
    {
        return;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    evoke_runtime_service->accelerator_local_abandoned++;
    LWLockRelease(evoke_runtime_service_lock);
}

static void
evoke_runtime_accelerator_append_metrics_json(StringInfo json)
{
    EvokeRuntimeAcceleratorTarget
        targets[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    uint32 target_count;
    uint32 local_capacity;
    bool first = true;
    TimestampTz now = GetCurrentTimestamp();

    target_count = evoke_runtime_accelerator_targets(
        targets,
        EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES
    );
    local_capacity = evoke_runtime_accelerator_local_capacity();
    appendStringInfoChar(json, '[');
    if (!evoke_runtime_service_attach_if_possible())
    {
        appendStringInfoChar(json, ']');
        goto done;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    appendStringInfo(
        json,
        "{\"url\":\"local\","
        "\"capacity\":%u,"
        "\"configured_weight\":0,"
        "\"max_batch_size\":%u,"
        "\"configured_max_batch_size\":0,"
            "\"in_flight\":%u,"
            "\"successes\":%llu,"
            "\"request_failures\":%llu,"
            "\"connect_failures\":0,"
            "\"read_failures\":0,"
            "\"status_failures\":0,"
            "\"backpressure_failures\":0,"
            "\"body_failures\":0,"
            "\"status_400_failures\":0,"
            "\"status_409_failures\":0,"
            "\"status_500_failures\":0,"
            "\"status_503_failures\":0,"
            "\"status_504_failures\":0,"
            "\"status_other_failures\":0,"
            "\"last_status_failure\":0,"
            "\"last_status_body\":\"\","
            "\"consecutive_failures\":0,"
            "\"backoff_active\":false,"
            "\"ewma_ms_per_text\":%.6f,"
        "\"ewma_batch_ms\":%.6f}",
        local_capacity,
        evoke_runtime_accelerator_local_max_batch_size(),
        local_capacity == 0
            ? 0
            : evoke_runtime_accelerator_local_in_flight_locked(),
        (unsigned long long)
            evoke_runtime_service->accelerator_local_successes,
        (unsigned long long)
            evoke_runtime_service->accelerator_local_abandoned,
        evoke_runtime_service->accelerator_local_ewma_ms_per_text,
        evoke_runtime_service->accelerator_local_ewma_batch_ms
    );
    first = false;
    for (uint32 i = 0; i < EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES; i++)
    {
        evoke_runtime_accelerator_state *entry =
            &evoke_runtime_service->accelerators[i];
        uint32 configured_weight = 0;
        uint32 configured_max_batch_size = 0;
        uint32 capacity;

        if (!entry->occupied)
        {
            continue;
        }
        if (!first)
        {
            appendStringInfoChar(json, ',');
        }
        first = false;
        appendStringInfoString(json, "{\"url\":");
        evoke_append_json_string(json, entry->url);
        capacity = evoke_runtime_accelerator_capacity_for_url(
            entry->url,
            targets,
            target_count,
            &configured_weight,
            &configured_max_batch_size
        );
        appendStringInfo(
            json,
            ",\"capacity\":%u,"
            "\"configured_weight\":%u,"
            "\"max_batch_size\":%u,"
            "\"configured_max_batch_size\":%u,"
            "\"in_flight\":%u,"
            "\"successes\":%llu,"
            "\"request_failures\":%llu,"
            "\"connect_failures\":%llu,"
            "\"read_failures\":%llu,"
            "\"status_failures\":%llu,"
            "\"backpressure_failures\":%llu,"
            "\"body_failures\":%llu,"
            "\"status_400_failures\":%llu,"
            "\"status_409_failures\":%llu,"
            "\"status_500_failures\":%llu,"
            "\"status_503_failures\":%llu,"
            "\"status_504_failures\":%llu,"
            "\"status_other_failures\":%llu,"
            "\"last_status_failure\":%u,"
            "\"last_status_body\":",
            capacity,
            configured_weight,
            configured_max_batch_size == 0
                ? (uint32) evoke_runtime_effective_max_batch_size()
                : configured_max_batch_size,
            configured_max_batch_size,
            evoke_runtime_accelerator_effective_in_flight(entry, now),
            (unsigned long long) entry->successes,
            (unsigned long long) entry->request_failures,
            (unsigned long long) entry->connect_failures,
            (unsigned long long) entry->read_failures,
            (unsigned long long) entry->status_failures,
            (unsigned long long) entry->backpressure_failures,
            (unsigned long long) entry->body_failures,
            (unsigned long long) entry->status_400_failures,
            (unsigned long long) entry->status_409_failures,
            (unsigned long long) entry->status_500_failures,
            (unsigned long long) entry->status_503_failures,
            (unsigned long long) entry->status_504_failures,
            (unsigned long long) entry->status_other_failures,
            entry->last_status_failure
        );
        evoke_append_json_string(json, entry->last_status_body);
        appendStringInfo(
            json,
            ",\"consecutive_failures\":%u,"
            "\"backoff_active\":%s,"
            "\"ewma_ms_per_text\":%.6f,"
            "\"ewma_batch_ms\":%.6f}",
            entry->consecutive_failures,
            entry->next_probe_at != 0 &&
                entry->next_probe_at > now
                ? "true"
                : "false",
            entry->ewma_ms_per_text,
            entry->ewma_batch_ms
        );
    }
    LWLockRelease(evoke_runtime_service_lock);
    appendStringInfoChar(json, ']');

done:
    for (uint32 i = 0; i < target_count; i++)
    {
        if (targets[i].url != NULL)
        {
            pfree(targets[i].url);
        }
    }
}

static int
evoke_runtime_accelerator_http_open_post(
    const char *url,
    const char *body
)
{
    char *host = NULL;
    char *path = NULL;
    int port = 0;
    char port_text[16];
    struct addrinfo hints;
    struct addrinfo *addresses = NULL;
    struct addrinfo *address;
    struct timeval timeout;
    int fd = -1;
    int flags = -1;
    int result_fd = -1;
    StringInfoData request;
    bool request_initialized = false;

    if (body == NULL)
    {
        return -1;
    }
    if (!evoke_runtime_accelerator_parse_url(url, &host, &port, &path))
    {
        return -1;
    }
    snprintf(port_text, sizeof(port_text), "%d", port);
    initStringInfo(&request);
    request_initialized = true;
    appendStringInfo(
        &request,
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Accept: application/json\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: keep-alive\r\n\r\n",
        path,
        host,
        port,
        strlen(body)
    );
    appendStringInfoString(&request, body);

    for (uint32 attempt = 0;
         attempt < EVOKE_RUNTIME_ACCELERATOR_IDLE_REUSE_ATTEMPTS;
         attempt++)
    {
        fd = evoke_runtime_accelerator_take_idle_connection(url);
        if (fd < 0)
        {
            break;
        }
        if (evoke_runtime_accelerator_send_all(
                fd,
                request.data,
                (Size) request.len))
        {
            result_fd = fd;
            fd = -1;
            goto done;
        }
        close(fd);
        fd = -1;
    }

    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host, port_text, &hints, &addresses) != 0)
    {
        goto done;
    }
    timeout.tv_sec = EVOKE_RUNTIME_ACCELERATOR_SOCKET_IO_TIMEOUT_MS / 1000;
    timeout.tv_usec =
        (EVOKE_RUNTIME_ACCELERATOR_SOCKET_IO_TIMEOUT_MS % 1000) * 1000;
    for (address = addresses; address != NULL; address = address->ai_next)
    {
        fd = socket(
            address->ai_family,
            address->ai_socktype,
            address->ai_protocol
        );
        if (fd < 0)
        {
            continue;
        }
        (void) setsockopt(
            fd,
            SOL_SOCKET,
            SO_RCVTIMEO,
            &timeout,
            sizeof(timeout)
        );
        (void) setsockopt(
            fd,
            SOL_SOCKET,
            SO_SNDTIMEO,
            &timeout,
            sizeof(timeout)
        );
        if (evoke_runtime_accelerator_connect_with_timeout(
                fd,
                address->ai_addr,
                address->ai_addrlen))
        {
            break;
        }
        close(fd);
        fd = -1;
    }
    if (fd < 0)
    {
        goto done;
    }
    if (!evoke_runtime_accelerator_send_all(
            fd,
            request.data,
            (Size) request.len))
    {
        goto done;
    }
    if (!evoke_runtime_accelerator_set_nonblocking(fd, &flags))
    {
        goto done;
    }
    result_fd = fd;
    fd = -1;

done:
    if (fd >= 0)
    {
        close(fd);
    }
    if (request_initialized)
    {
        pfree(request.data);
    }
    if (addresses != NULL)
    {
        freeaddrinfo(addresses);
    }
    if (host != NULL)
    {
        pfree(host);
    }
    if (path != NULL)
    {
        pfree(path);
    }
    return result_fd;
}

static bool
evoke_runtime_http_response_metadata(
    const char *data,
    Size len,
    int *status_out,
    Size *header_len_out,
    Size *content_len_out,
    bool *keep_alive_out
)
{
    const char *header_end = NULL;
    const char *line;
    bool content_length_found = false;
    bool keep_alive = true;

    if (data == NULL || len < 12 || status_out == NULL ||
        header_len_out == NULL || content_len_out == NULL ||
        keep_alive_out == NULL)
    {
        return false;
    }
    if (strncmp(data, "HTTP/1.", 7) != 0)
    {
        return false;
    }
    if (strncmp(data, "HTTP/1.0", 8) == 0)
    {
        keep_alive = false;
    }
    for (Size i = 0; i + 3 < len; i++)
    {
        if (data[i] == '\r' && data[i + 1] == '\n' &&
            data[i + 2] == '\r' && data[i + 3] == '\n')
        {
            header_end = data + i;
            break;
        }
    }
    if (header_end == NULL)
    {
        return false;
    }
    *status_out = atoi(data + 9);
    *header_len_out = (Size) (header_end - data) + 4;
    *content_len_out = 0;

    line = data;
    while (line < header_end)
    {
        const char *line_end = line;
        Size line_len;

        while (line_end < header_end &&
               !(line_end[0] == '\r' && line_end + 1 < header_end &&
                 line_end[1] == '\n'))
        {
            line_end++;
        }
        line_len = (Size) (line_end - line);
        if (line_len > 15 &&
            pg_strncasecmp(line, "Content-Length:", 15) == 0)
        {
            const char *value = line + 15;
            unsigned long long parsed;
            char *end_ptr;

            while (value < line_end && isspace((unsigned char) *value))
            {
                value++;
            }
            errno = 0;
            parsed = strtoull(value, &end_ptr, 10);
            if (errno != 0 || end_ptr == value || end_ptr > line_end ||
                parsed > (unsigned long long) SIZE_MAX)
            {
                return false;
            }
            *content_len_out = (Size) parsed;
            content_length_found = true;
        }
        else if (line_len > 11 &&
                 pg_strncasecmp(line, "Connection:", 11) == 0)
        {
            const char *value = line + 11;
            Size value_len;
            char connection[64];

            while (value < line_end && isspace((unsigned char) *value))
            {
                value++;
            }
            value_len = (Size) (line_end - value);
            if (value_len >= sizeof(connection))
            {
                value_len = sizeof(connection) - 1;
            }
            for (Size i = 0; i < value_len; i++)
            {
                connection[i] = (char) tolower((unsigned char) value[i]);
            }
            connection[value_len] = '\0';
            if (strstr(connection, "close") != NULL)
            {
                keep_alive = false;
            }
            else if (strstr(connection, "keep-alive") != NULL)
            {
                keep_alive = true;
            }
        }
        line = line_end;
        if (line < header_end && line[0] == '\r' &&
            line + 1 < header_end && line[1] == '\n')
        {
            line += 2;
        }
    }
    if (!content_length_found)
    {
        return false;
    }
    *keep_alive_out = keep_alive;
    return true;
}

static void
evoke_runtime_accelerator_http_clear_response(
    EvokeRuntimeRequestHandle *handle
)
{
    if (handle == NULL)
    {
        return;
    }
    if (handle->accelerator_response != NULL)
    {
        pfree(handle->accelerator_response);
        handle->accelerator_response = NULL;
    }
    handle->accelerator_response_len = 0;
    handle->accelerator_response_cap = 0;
    handle->accelerator_response_header_len = 0;
    handle->accelerator_response_content_len = 0;
    handle->accelerator_response_status = 0;
    handle->accelerator_response_header_parsed = false;
    handle->accelerator_response_keep_alive = false;
}

static bool
evoke_runtime_accelerator_http_response_complete(
    const EvokeRuntimeRequestHandle *handle
)
{
    if (handle == NULL || !handle->accelerator_response_header_parsed)
    {
        return false;
    }
    return handle->accelerator_response_len >=
        handle->accelerator_response_header_len +
            handle->accelerator_response_content_len;
}

static bool
evoke_runtime_accelerator_http_reserve_response(
    EvokeRuntimeRequestHandle *handle,
    Size needed
)
{
    Size new_cap;

    if (needed > EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES + 65536U ||
        needed + 1 > MaxAllocSize)
    {
        return false;
    }
    if (handle->accelerator_response_cap >= needed + 1)
    {
        return true;
    }
    new_cap = handle->accelerator_response_cap;
    if (new_cap < 8192)
    {
        new_cap = 8192;
    }
    while (new_cap < needed + 1)
    {
        if (new_cap > MaxAllocSize / 2)
        {
            new_cap = needed + 1;
            break;
        }
        new_cap *= 2;
    }
    if (new_cap > MaxAllocSize)
    {
        return false;
    }
    if (handle->accelerator_response == NULL)
    {
        handle->accelerator_response = palloc(new_cap);
    }
    else
    {
        handle->accelerator_response = repalloc(
            handle->accelerator_response,
            new_cap
        );
    }
    handle->accelerator_response_cap = new_cap;
    return true;
}

static bool
evoke_runtime_accelerator_http_append_response(
    EvokeRuntimeRequestHandle *handle,
    const char *data,
    Size len
)
{
    Size needed;

    if (handle == NULL || data == NULL)
    {
        return false;
    }
    needed = handle->accelerator_response_len + len;
    if (needed < handle->accelerator_response_len ||
        !evoke_runtime_accelerator_http_reserve_response(handle, needed))
    {
        return false;
    }
    memcpy(handle->accelerator_response + handle->accelerator_response_len,
           data,
           len);
    handle->accelerator_response_len = needed;
    handle->accelerator_response[handle->accelerator_response_len] = '\0';

    if (!handle->accelerator_response_header_parsed)
    {
        int status = 0;
        Size header_len = 0;
        Size content_len = 0;
        bool keep_alive = false;

        if (evoke_runtime_http_response_metadata(
                handle->accelerator_response,
                handle->accelerator_response_len,
                &status,
                &header_len,
                &content_len,
                &keep_alive))
        {
            if (content_len > EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES ||
                header_len + content_len < header_len ||
                header_len + content_len >
                    EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES + 65536U)
            {
                return false;
            }
            handle->accelerator_response_status = status;
            handle->accelerator_response_header_len = header_len;
            handle->accelerator_response_content_len = content_len;
            handle->accelerator_response_header_parsed = true;
            handle->accelerator_response_keep_alive = keep_alive;
        }
    }
    return true;
}

static char *
evoke_runtime_accelerator_http_take_body(
    EvokeRuntimeRequestHandle *handle,
    int *status_out
)
{
    if (!evoke_runtime_accelerator_http_response_complete(handle))
    {
        return NULL;
    }
    if (status_out != NULL)
    {
        *status_out = handle->accelerator_response_status;
    }
    return pnstrdup(
        handle->accelerator_response +
            handle->accelerator_response_header_len,
        handle->accelerator_response_content_len
    );
}

static bool
evoke_runtime_accelerator_http_try_read_response(
    EvokeRuntimeRequestHandle *handle,
    bool *complete_out
)
{
    char buffer[8192];

    if (complete_out == NULL)
    {
        return false;
    }
    *complete_out = false;
    if (handle == NULL || handle->accelerator_fd < 0)
    {
        return false;
    }
    if (evoke_runtime_accelerator_http_response_complete(handle))
    {
        *complete_out = true;
        return true;
    }
    for (;;)
    {
        ssize_t got = recv(
            handle->accelerator_fd,
            buffer,
            sizeof(buffer),
            0
        );

        if (got < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                *complete_out =
                    evoke_runtime_accelerator_http_response_complete(handle);
                return true;
            }
            return false;
        }
        if (got == 0)
        {
            *complete_out =
                evoke_runtime_accelerator_http_response_complete(handle);
            return *complete_out;
        }
        if (!evoke_runtime_accelerator_http_append_response(
                handle,
                buffer,
                (Size) got))
        {
            return false;
        }
        if (evoke_runtime_accelerator_http_response_complete(handle))
        {
            *complete_out = true;
            return true;
        }
    }
}

static bool
evoke_runtime_accelerator_http_wait_response(
    EvokeRuntimeRequestHandle *handle,
    int *status_out,
    char **body_out
)
{
    if (status_out == NULL || body_out == NULL)
    {
        return false;
    }
    *status_out = 0;
    *body_out = NULL;
    for (;;)
    {
        bool complete = false;

        CHECK_FOR_INTERRUPTS();
        if (evoke_runtime_accelerator_request_timed_out(handle))
        {
            return false;
        }
        if (!evoke_runtime_accelerator_http_try_read_response(
                handle,
                &complete))
        {
            return false;
        }
        if (complete)
        {
            *body_out = evoke_runtime_accelerator_http_take_body(
                handle,
                status_out
            );
            return *body_out != NULL;
        }
        pg_usleep(1000L);
    }
}

static char *
evoke_runtime_accelerator_checkout_signature(const char *model_path)
{
    char *manifest_text;
    Datum manifest;
    char checkout_signature[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1];

    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    evoke_checkout_validate_manifest(manifest, NULL, model_path);
    evoke_checkout_manifest_signature(manifest, checkout_signature);
    return pstrdup(checkout_signature);
}

static char *
evoke_runtime_accelerator_build_request(
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count
)
{
    StringInfoData json;

    initStringInfo(&json);
    appendStringInfoString(&json, "{\"mode\":\"document\",");
    appendStringInfoString(&json, "\"checkout_signature\":");
    evoke_append_json_string(&json, checkout_signature);
    appendStringInfoString(&json, ",\"runtime_precision\":");
    evoke_append_json_string(&json, runtime_precision);
    appendStringInfoString(&json, ",\"texts\":[");
    for (uint32 i = 0; i < batch_count; i++)
    {
        if (i > 0)
        {
            appendStringInfoChar(&json, ',');
        }
        evoke_append_json_string(&json, texts[i]);
    }
    appendStringInfoString(&json, "]}");
    return json.data;
}

static int
evoke_runtime_accelerator_start_document_batch_internal(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    bool exact_batch_count,
    uint32 *selected_count_out,
    char **selected_url_out,
    TimestampTz *started_at_out
)
{
    EvokeRuntimeAcceleratorTarget
        targets[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    bool tried[EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES] = {0};
    uint32 count;
    uint32 start;
    uint64 schedule_cursor;
    const char *request_signature = checkout_signature;
    char *owned_checkout_signature = NULL;
    TimestampTz now;
    int fd = -1;
    uint32 attempt_limit;
    bool ignore_backoff = false;

    if (selected_count_out != NULL)
    {
        *selected_count_out = 0;
    }
    if (selected_url_out != NULL)
    {
        *selected_url_out = NULL;
    }
    if (started_at_out != NULL)
    {
        *started_at_out = 0;
    }
    if (model_path == NULL || runtime_precision == NULL ||
        texts == NULL || available_count == 0)
    {
        return -1;
    }
    count = evoke_runtime_accelerator_targets(
        targets,
        EVOKE_RUNTIME_ACCELERATOR_MAX_SERVICES
    );
    if (count == 0)
    {
        return -1;
    }
    attempt_limit = count;
    schedule_cursor = evoke_runtime_accelerator_next_schedule_cursor();
    if (request_signature == NULL || request_signature[0] == '\0')
    {
        owned_checkout_signature =
            evoke_runtime_accelerator_checkout_signature(model_path);
        request_signature = owned_checkout_signature;
    }
    start = (uint32) (schedule_cursor % (uint64) (count + 1));
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    attempt_limit = Max(
        count,
        count * (EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_LIMIT + 1)
    );
#endif
    for (uint32 attempt = 0; attempt < attempt_limit; attempt++)
    {
        bool best_is_local = true;
        int best_index = -1;
        uint32 best_batch_count = 0;
        double best_score = DBL_MAX;
        char *candidate_url = NULL;
        char *request = NULL;

        CHECK_FOR_INTERRUPTS();
        now = GetCurrentTimestamp();
        for (uint32 offset = 0; offset < count; offset++)
        {
            uint32 index =
                ((uint32) schedule_cursor + offset) % count;
            uint32 candidate_batch_count;

            if (tried[index])
            {
                continue;
            }
            candidate_batch_count =
                evoke_runtime_accelerator_candidate_batch_size(
                    available_count,
                    evoke_runtime_accelerator_target_max_batch_size(
                        &targets[index]
                    ),
                    exact_batch_count
                );
            if (candidate_batch_count == 0)
            {
                tried[index] = true;
                continue;
            }
            if (evoke_runtime_accelerator_probe_needed(
                    targets[index].url,
                    now))
            {
                best_is_local = false;
                best_index = (int) index;
                best_batch_count = candidate_batch_count;
                goto start_remote_attempt;
            }
        }
        for (uint32 offset = 0; offset <= count; offset++)
        {
            uint32 index = (start + offset) % (count + 1);
            uint32 candidate_batch_count;
            double score;
            double score_per_text;

            if (index == count)
            {
                continue;
            }
            candidate_batch_count =
                evoke_runtime_accelerator_candidate_batch_size(
                    available_count,
                    evoke_runtime_accelerator_target_max_batch_size(
                        &targets[index]
                    ),
                    exact_batch_count
                );
            if (tried[index] ||
                candidate_batch_count == 0 ||
                (
                    !ignore_backoff &&
                    evoke_runtime_accelerator_backoff_active(
                        targets[index].url,
                        now
                    )
                ))
            {
                continue;
            }
            score = evoke_runtime_accelerator_score(
                &targets[index],
                targets,
                count,
                candidate_batch_count
            );
            if (score == DBL_MAX)
            {
                continue;
            }
            score_per_text =
                score / (double) Max(candidate_batch_count, 1);
            if (score_per_text < best_score ||
                (
                    score_per_text == best_score &&
                    candidate_batch_count > best_batch_count
                ))
            {
                best_score = score_per_text;
                best_is_local = false;
                best_index = (int) index;
                best_batch_count = candidate_batch_count;
            }
        }
        if (best_is_local || best_index < 0)
        {
#ifndef EVOKE_ENABLE_ONNXRUNTIME
            if (!ignore_backoff)
            {
                ignore_backoff = true;
                memset(tried, 0, sizeof(tried));
                continue;
            }
            break;
#else
            if (selected_count_out != NULL)
            {
                *selected_count_out = best_batch_count;
            }
            break;
#endif
        }
start_remote_attempt:
        if (best_index < 0 ||
            (uint32) best_index >= count ||
            targets[best_index].url == NULL ||
            best_batch_count == 0)
        {
            break;
        }
        tried[best_index] = true;
        request = evoke_runtime_accelerator_build_request(
            request_signature,
            runtime_precision,
            texts,
            best_batch_count
        );
        if (selected_url_out != NULL)
        {
            candidate_url = pstrdup(targets[best_index].url);
        }
        fd = evoke_runtime_accelerator_http_open_post(
            targets[best_index].url,
            request
        );
        if (request != NULL)
        {
            pfree(request);
            request = NULL;
        }
        if (fd >= 0)
        {
            if (selected_count_out != NULL)
            {
                *selected_count_out = best_batch_count;
            }
            if (selected_url_out != NULL)
            {
                *selected_url_out = candidate_url;
                candidate_url = NULL;
            }
            if (started_at_out != NULL)
            {
                *started_at_out = evoke_runtime_accelerator_note_start(
                    targets[best_index].url
                );
            }
            break;
        }
        if (candidate_url != NULL)
        {
            pfree(candidate_url);
        }
        if (evoke_runtime_accelerator_idle_connection_count(
                targets[best_index].url) == 0)
        {
            evoke_runtime_accelerator_note_connect_failure(
                targets[best_index].url
            );
        }
    }
    for (uint32 index = 0; index < count; index++)
    {
        if (targets[index].url != NULL)
        {
            pfree(targets[index].url);
        }
    }
    if (owned_checkout_signature != NULL)
    {
        pfree(owned_checkout_signature);
    }
    return fd;
}

static int
evoke_runtime_accelerator_start_document_batch(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    char **selected_url_out,
    TimestampTz *started_at_out
)
{
    return evoke_runtime_accelerator_start_document_batch_internal(
        model_path,
        checkout_signature,
        runtime_precision,
        texts,
        batch_count,
        true,
        NULL,
        selected_url_out,
        started_at_out
    );
}

static int
evoke_runtime_accelerator_start_document_prefix_batch(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    uint32 *selected_count_out,
    char **selected_url_out,
    TimestampTz *started_at_out
)
{
    return evoke_runtime_accelerator_start_document_batch_internal(
        model_path,
        checkout_signature,
        runtime_precision,
        texts,
        available_count,
        false,
        selected_count_out,
        selected_url_out,
        started_at_out
    );
}

static void
evoke_runtime_request_clear_accelerator(
    EvokeRuntimeRequestHandle *handle,
    bool close_fd
)
{
    if (handle == NULL)
    {
        return;
    }
    if (close_fd && handle->accelerator_fd >= 0)
    {
        close(handle->accelerator_fd);
    }
    if (handle->accelerator_health_active &&
        handle->accelerator_url != NULL)
    {
        evoke_runtime_accelerator_note_abandoned(handle->accelerator_url);
    }
    if (handle->local_runtime_health_active)
    {
        evoke_runtime_accelerator_note_local_abandoned();
    }
    handle->accelerator_fd = -1;
    handle->accelerator_started_at = 0;
    handle->accelerator_retry_after = 0;
    handle->accelerator_deadline_at = 0;
    handle->accelerator_local_fallback_deadline_at = 0;
    handle->local_runtime_started_at = 0;
    handle->local_runtime_batch_count = 0;
    handle->accelerator_health_active = false;
    handle->local_runtime_health_active = false;
    handle->accelerator_active = false;
    if (handle->accelerator_url != NULL)
    {
        pfree(handle->accelerator_url);
        handle->accelerator_url = NULL;
    }
    if (handle->accelerator_checkout_signature != NULL)
    {
        pfree(handle->accelerator_checkout_signature);
        handle->accelerator_checkout_signature = NULL;
    }
    if (handle->accelerator_model_path != NULL)
    {
        pfree(handle->accelerator_model_path);
        handle->accelerator_model_path = NULL;
    }
    if (handle->accelerator_runtime_precision != NULL)
    {
        pfree(handle->accelerator_runtime_precision);
        handle->accelerator_runtime_precision = NULL;
    }
    if (handle->accelerator_texts != NULL)
    {
        for (uint32 i = 0; i < handle->accelerator_batch_count; i++)
        {
            if (handle->accelerator_texts[i] != NULL)
            {
                pfree(handle->accelerator_texts[i]);
            }
        }
        pfree(handle->accelerator_texts);
        handle->accelerator_texts = NULL;
    }
    handle->accelerator_batch_count = 0;
    evoke_runtime_accelerator_http_clear_response(handle);
}

static void
evoke_runtime_request_save_accelerator_fallback(
    EvokeRuntimeRequestHandle *handle,
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count
)
{
    if (handle == NULL || model_path == NULL || runtime_precision == NULL ||
        texts == NULL || batch_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke accelerator fallback request")));
    }
    if (checkout_signature != NULL && checkout_signature[0] != '\0')
    {
        handle->accelerator_checkout_signature = pstrdup(checkout_signature);
    }
    handle->accelerator_model_path = pstrdup(model_path);
    handle->accelerator_runtime_precision = pstrdup(runtime_precision);
    handle->accelerator_texts = palloc0(
        sizeof(*handle->accelerator_texts) * batch_count
    );
    handle->accelerator_batch_count = batch_count;
    handle->accelerator_retry_after = 0;
    handle->accelerator_local_fallback_deadline_at = 0;
    handle->accelerator_deadline_at = GetCurrentTimestamp() +
        (TimestampTz) (
            evoke_runtime_liveness_timeout_ms > 0
                ? evoke_runtime_liveness_timeout_ms
                : EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT *
                    EVOKE_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS
        ) * 1000;
    for (uint32 i = 0; i < batch_count; i++)
    {
        handle->accelerator_texts[i] = pstrdup(texts[i]);
    }
}

static bool
evoke_runtime_service_available(void)
{
    (void) evoke_runtime_service_attach_if_possible();
    return evoke_runtime_service != NULL &&
        evoke_runtime_service->magic == EVOKE_RUNTIME_SERVICE_MAGIC &&
        evoke_runtime_service->version == EVOKE_RUNTIME_SERVICE_VERSION;
}

void
evoke_unified_require_shared_runtime(void)
{
    if (!evoke_runtime_service_available())
    {
        ereport(
            ERROR,
            (
                errmsg("evoke SAE runtime service is not available"),
                errdetail(
                    "SAE indexes require shared_preload_libraries='evoke' "
                    "so the runtime worker pool owns model sessions."
                ),
                errhint(
                    "Add evoke to shared_preload_libraries and restart "
                    "PostgreSQL."
                )
            )
        );
    }
    if (!evoke_am_preload_cache_available())
    {
        ereport(
            ERROR,
            (
                errmsg("evoke SAE shared arena is not available"),
                errdetail(
                    "SAE indexes require a positive "
                    "evoke.shared_runtime_size so exact-root "
                    "residency and page-native preload state remain "
                    "postmaster-owned."
                ),
                errhint(
                    "Set evoke.shared_runtime_size for the expected "
                    "hot-root working set and restart PostgreSQL."
                )
            )
        );
    }
}

static void evoke_runtime_service_clear_response_locked(uint32 response_slot);

static pid_t
evoke_runtime_service_mark_worker_unavailable_locked(
    uint32 worker_id,
    pid_t expected_pid,
    ProcNumber expected_proc_number,
    const char *message,
    ProcNumber *owner_proc_number_out
)
{
    evoke_runtime_service_worker *worker;
    pid_t owner_pid = 0;

    if (owner_proc_number_out != NULL)
    {
        *owner_proc_number_out = INVALID_PROC_NUMBER;
    }
    if (worker_id >= evoke_runtime_service_configured_workers())
    {
        return 0;
    }
    worker = &evoke_runtime_service->workers[worker_id];
    if ((expected_pid > 0 && worker->pid != expected_pid) ||
        (expected_proc_number != INVALID_PROC_NUMBER &&
         worker->proc_number != expected_proc_number))
    {
        return 0;
    }

    if (worker->processing && worker->processing_request_id != 0)
    {
        uint32 response_slot = worker->processing_response_slot;

        evoke_runtime_record_worker_termination_locked(worker);
        owner_pid = worker->processing_owner_pid;
        if (owner_proc_number_out != NULL)
        {
            *owner_proc_number_out =
                worker->processing_owner_proc_number;
        }
        if (response_slot < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY)
        {
            evoke_runtime_service_response *response =
                &evoke_runtime_service->responses[response_slot];

            if (response->occupied &&
                response->request_id == worker->processing_request_id)
            {
                response->writing = false;
                response->error = true;
                response->result_len = 0;
                response->result_json[0] = '\0';
                strlcpy(
                    response->error_message,
                    message,
                    sizeof(response->error_message)
                );
                if (response->owner_pid > 0 &&
                    response->owner_proc_number != INVALID_PROC_NUMBER)
                {
                    response->ready = true;
                    evoke_runtime_service->last_completed_request_id =
                        response->request_id;
                    evoke_runtime_service->last_response_at =
                        GetCurrentTimestamp();
                }
                else
                {
                    evoke_runtime_service_clear_response_locked(response_slot);
                    owner_pid = 0;
                    if (owner_proc_number_out != NULL)
                    {
                        *owner_proc_number_out = INVALID_PROC_NUMBER;
                    }
                    evoke_runtime_service->orphan_responses++;
                }
            }
        }
        worker->failures++;
        evoke_runtime_service->failures++;
    }

    worker->ready = false;
    worker->processing = false;
    worker->processing_document = false;
    pg_atomic_write_u32(&worker->cancel_requested, 0);
    pg_atomic_write_u32(
        &worker->termination_reason,
        EVOKE_RUNTIME_TERMINATION_NONE
    );
    worker->terminate_requested = false;
    worker->pid = 0;
    worker->proc_number = INVALID_PROC_NUMBER;
    worker->processing_request_id = 0;
    worker->processing_owner_pid = 0;
    worker->processing_owner_proc_number = INVALID_PROC_NUMBER;
    worker->processing_response_slot =
        EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    worker->processing_batch_count = 0;
    worker->processing_started_at = 0;
    worker->last_progress_at = 0;
    worker->affinity_model_path[0] = '\0';
    worker->affinity_runtime_precision[0] = '\0';
    worker->active_provider[0] = '\0';
    worker->runtime_precision[0] = '\0';
    return owner_pid;
}

static uint32
evoke_runtime_service_ready_worker_count(void)
{
    pid_t worker_pids[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};
    ProcNumber worker_proc_numbers[EVOKE_RUNTIME_SERVICE_MAX_WORKERS];
    pid_t interrupted_owners[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};
    ProcNumber interrupted_owner_proc_numbers[
        EVOKE_RUNTIME_SERVICE_MAX_WORKERS
    ];
    uint32 configured_workers;
    uint32 ready_workers = 0;

    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_MAX_WORKERS; i++)
    {
        worker_proc_numbers[i] = INVALID_PROC_NUMBER;
        interrupted_owner_proc_numbers[i] = INVALID_PROC_NUMBER;
    }
    if (!evoke_runtime_service_available())
    {
        return 0;
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    configured_workers = evoke_runtime_service_configured_workers();
    for (uint32 i = 0; i < configured_workers; i++)
    {
        if (evoke_runtime_service->workers[i].ready)
        {
            worker_pids[i] = evoke_runtime_service->workers[i].pid;
            worker_proc_numbers[i] =
                evoke_runtime_service->workers[i].proc_number;
        }
    }
    LWLockRelease(evoke_runtime_service_lock);

    for (uint32 i = 0; i < configured_workers; i++)
    {
        PGPROC *proc = worker_proc_numbers[i] == INVALID_PROC_NUMBER
            ? NULL
            : ProcNumberGetProc(worker_proc_numbers[i]);

        if (worker_pids[i] > 0 &&
            (proc == NULL || proc->pid != worker_pids[i]))
        {
            LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
            interrupted_owners[i] =
                evoke_runtime_service_mark_worker_unavailable_locked(
                    i,
                    worker_pids[i],
                    worker_proc_numbers[i],
                    "evoke SAE runtime worker exited while processing request",
                    &interrupted_owner_proc_numbers[i]
                );
            LWLockRelease(evoke_runtime_service_lock);
        }
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    for (uint32 i = 0; i < configured_workers; i++)
    {
        if (evoke_runtime_service->workers[i].ready &&
            evoke_runtime_service->workers[i].pid > 0)
        {
            ready_workers++;
        }
    }
    LWLockRelease(evoke_runtime_service_lock);

    for (uint32 i = 0; i < configured_workers; i++)
    {
        (void) evoke_runtime_service_signal_backend(
            interrupted_owners[i],
            interrupted_owner_proc_numbers[i]
        );
    }
    return ready_workers;
}

static bool
evoke_runtime_service_worker_is_ready(void)
{
    return evoke_runtime_service_ready_worker_count() > 0;
}

static void
evoke_runtime_service_signal_workers(void)
{
    pid_t worker_pids[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};
    ProcNumber worker_proc_numbers[EVOKE_RUNTIME_SERVICE_MAX_WORKERS];
    uint32 configured_workers;

    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_MAX_WORKERS; i++)
    {
        worker_proc_numbers[i] = INVALID_PROC_NUMBER;
    }
    if (!evoke_runtime_service_available())
    {
        return;
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    configured_workers = evoke_runtime_service_configured_workers();
    for (uint32 i = 0; i < configured_workers; i++)
    {
        if (evoke_runtime_service->workers[i].ready)
        {
            worker_pids[i] = evoke_runtime_service->workers[i].pid;
            worker_proc_numbers[i] =
                evoke_runtime_service->workers[i].proc_number;
        }
    }
    LWLockRelease(evoke_runtime_service_lock);

    for (uint32 i = 0; i < configured_workers; i++)
    {
        (void) evoke_runtime_service_signal_worker(
            worker_pids[i],
            worker_proc_numbers[i]
        );
    }
}

static void
evoke_runtime_service_clear_response_locked(uint32 response_slot)
{
    evoke_runtime_service_response *response;

    if (response_slot >= EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY)
    {
        return;
    }
    response = &evoke_runtime_service->responses[response_slot];
    response->occupied = false;
    response->ready = false;
    response->writing = false;
    response->error = false;
    response->request_id = 0;
    response->owner_pid = 0;
    response->owner_proc_number = INVALID_PROC_NUMBER;
    response->request_kind = 0;
    response->result_len = 0;
    response->result_json[0] = '\0';
    response->error_message[0] = '\0';
}

static uint32
evoke_runtime_service_find_response_slot_locked(void)
{
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY; i++)
    {
        if (!evoke_runtime_service->responses[i].occupied)
        {
            return i;
        }
    }
    return EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
}

static void
evoke_runtime_service_cancel_request(uint64 request_id)
{
    bool canceled = false;
    pid_t worker_pids[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};
    ProcNumber worker_proc_numbers[EVOKE_RUNTIME_SERVICE_MAX_WORKERS];
    uint32 configured_workers;

    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_MAX_WORKERS; i++)
    {
        worker_proc_numbers[i] = INVALID_PROC_NUMBER;
    }
    if (request_id == 0 || !evoke_runtime_service_available())
    {
        return;
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY; i++)
    {
        evoke_runtime_service_request *request =
            &evoke_runtime_service->queue[i];

        if (request->occupied && request->request_id == request_id)
        {
            request->canceled = true;
            if (request->response_slot <
                    EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY &&
                evoke_runtime_service->responses[
                    request->response_slot].request_id == request_id)
            {
                evoke_runtime_service_clear_response_locked(
                    request->response_slot
                );
            }
            canceled = true;
            break;
        }
    }
    configured_workers = evoke_runtime_service_configured_workers();
    for (uint32 i = 0; i < configured_workers; i++)
    {
        evoke_runtime_service_worker *worker =
            &evoke_runtime_service->workers[i];

        if (!worker->processing ||
            worker->processing_request_id != request_id)
        {
            continue;
        }

        /*
         * A dedicated monitor thread polls this flag while the worker runs
         * ONNX Runtime and requests cooperative termination.
         */
        pg_atomic_write_u32(&worker->cancel_requested, 1);
        worker->processing_owner_pid = 0;
        worker->processing_owner_proc_number = INVALID_PROC_NUMBER;
        if (worker->processing_response_slot <
            EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY)
        {
            evoke_runtime_service->responses[
                worker->processing_response_slot].owner_pid = 0;
            evoke_runtime_service->responses[
                worker->processing_response_slot].owner_proc_number =
                INVALID_PROC_NUMBER;
        }
        worker_pids[i] = worker->pid;
        worker_proc_numbers[i] = worker->proc_number;
        canceled = true;
    }
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY; i++)
    {
        evoke_runtime_service_response *response =
            &evoke_runtime_service->responses[i];

        if (!response->occupied || response->request_id != request_id)
        {
            continue;
        }
        response->owner_pid = 0;
        response->owner_proc_number = INVALID_PROC_NUMBER;
        if (response->ready)
        {
            evoke_runtime_service_clear_response_locked(i);
        }
        canceled = true;
        break;
    }
    if (canceled)
    {
        evoke_runtime_service->canceled_requests++;
    }
    LWLockRelease(evoke_runtime_service_lock);
    for (uint32 i = 0; i < configured_workers; i++)
    {
        (void) evoke_runtime_service_signal_worker(
            worker_pids[i],
            worker_proc_numbers[i]
        );
    }
    if (canceled)
    {
        evoke_runtime_service_signal_workers();
    }
}

static void
evoke_runtime_service_drop_orphan_responses(void)
{
    if (!evoke_runtime_service_available())
    {
        return;
    }

    for (uint32 slot = 0;
         slot < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY;
         slot++)
    {
        bool occupied;
        pid_t owner_pid;
        ProcNumber owner_proc_number;
        uint64 request_id;
        PGPROC *owner_proc;

        LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
        occupied = evoke_runtime_service->responses[slot].occupied;
        owner_pid = evoke_runtime_service->responses[slot].owner_pid;
        owner_proc_number =
            evoke_runtime_service->responses[slot].owner_proc_number;
        request_id = evoke_runtime_service->responses[slot].request_id;
        LWLockRelease(evoke_runtime_service_lock);

        owner_proc = owner_proc_number == INVALID_PROC_NUMBER
            ? NULL
            : ProcNumberGetProc(owner_proc_number);
        if (!occupied || owner_pid <= 0 ||
            (owner_proc != NULL && owner_proc->pid == owner_pid))
        {
            continue;
        }

        LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
        if (evoke_runtime_service->responses[slot].occupied &&
            evoke_runtime_service->responses[slot].owner_pid == owner_pid &&
            evoke_runtime_service->responses[
                slot].owner_proc_number == owner_proc_number &&
            evoke_runtime_service->responses[slot].request_id == request_id)
        {
            bool processing = false;

            for (uint32 i = 0;
                 i < evoke_runtime_service_configured_workers();
                 i++)
            {
                evoke_runtime_service_worker *worker =
                    &evoke_runtime_service->workers[i];

                if (worker->processing &&
                    worker->processing_request_id == request_id)
                {
                    worker->processing_owner_pid = 0;
                    worker->processing_owner_proc_number =
                        INVALID_PROC_NUMBER;
                    processing = true;
                    break;
                }
            }
            if (processing ||
                evoke_runtime_service->responses[slot].writing)
            {
                evoke_runtime_service->responses[slot].owner_pid = 0;
                evoke_runtime_service->responses[
                    slot].owner_proc_number = INVALID_PROC_NUMBER;
            }
            else
            {
                for (uint32 i = 0;
                     i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
                     i++)
                {
                    evoke_runtime_service_request *request =
                        &evoke_runtime_service->queue[i];

                    if (request->occupied &&
                        request->request_id == request_id)
                    {
                        request->canceled = true;
                        break;
                    }
                }
                evoke_runtime_service_clear_response_locked(slot);
                evoke_runtime_service->orphan_responses++;
            }
        }
        LWLockRelease(evoke_runtime_service_lock);
    }
}

static void
evoke_runtime_service_drop_canceled_requests_locked(void)
{
    for (uint32 slot = 0;
         slot < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
         slot++)
    {
        evoke_runtime_service_request *request =
            &evoke_runtime_service->queue[slot];
        uint32 response_slot =
            request->response_slot;
        uint64 request_id = request->request_id;

        if (!request->occupied || !request->canceled)
        {
            continue;
        }
        if (response_slot < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY &&
            evoke_runtime_service->responses[
                response_slot].request_id == request_id)
        {
            evoke_runtime_service_clear_response_locked(response_slot);
        }
        memset(
            &evoke_runtime_service->queue[slot],
            0,
            sizeof(evoke_runtime_service->queue[slot])
        );
        if (evoke_runtime_service->queue_count > 0)
        {
            evoke_runtime_service->queue_count--;
        }
    }
    evoke_runtime_service->request_pending =
        evoke_runtime_service->queue_count > 0;
}

static uint32
evoke_runtime_service_find_request_slot_locked(void)
{
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY; i++)
    {
        if (!evoke_runtime_service->queue[i].occupied)
        {
            return i;
        }
    }
    return EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
}

static uint32
evoke_runtime_service_pending_document_requests_locked(void)
{
    uint32 pending = 0;

    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY; i++)
    {
        const evoke_runtime_service_request *request =
            &evoke_runtime_service->queue[i];

        if (
            request->occupied &&
            !request->canceled &&
            request->request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT
        )
        {
            pending++;
        }
    }
    return pending;
}

static uint32
evoke_runtime_service_document_response_slots_locked(void)
{
    uint32 occupied = 0;

    for (uint32 i = 0;
         i < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY;
         i++)
    {
        const evoke_runtime_service_response *response =
            &evoke_runtime_service->responses[i];

        if (response->occupied &&
            response->request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT)
        {
            occupied++;
        }
    }
    return occupied;
}

static uint32
evoke_runtime_service_active_document_workers_locked(void)
{
    uint32 active = 0;

    for (uint32 i = 0;
         i < evoke_runtime_service_configured_workers();
         i++)
    {
        const evoke_runtime_service_worker *worker =
            &evoke_runtime_service->workers[i];

        if (worker->ready &&
            worker->processing &&
            worker->processing_document)
        {
            active++;
        }
    }
    return active;
}

static uint32
evoke_runtime_service_reserved_query_workers(uint32 ready_workers)
{
    if (!evoke_runtime_reserve_query_lane || ready_workers <= 1)
    {
        return 0;
    }
    return 1;
}

static uint32
evoke_runtime_service_reserved_query_responses(void)
{
    if (!evoke_runtime_reserve_query_lane ||
        EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY <= 1)
    {
        return 0;
    }
    return Min(
        EVOKE_RUNTIME_SERVICE_RESERVED_QUERY_RESPONSES,
        EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY - 1
    );
}

uint32
evoke_runtime_effective_local_document_pipeline_depth(void)
{
    uint32 configured_workers = (uint32) Max(
        evoke_runtime_worker_count,
        1
    );
    uint32 reserved_workers =
        evoke_runtime_service_reserved_query_workers(configured_workers);
    uint32 worker_limit = configured_workers > reserved_workers
        ? configured_workers - reserved_workers
        : 1;
    uint32 configured = (uint32) Max(
        evoke_runtime_document_pipeline_depth,
        1
    );

    configured = Min(
        configured,
        (uint32) EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS
    );
    return Max(Min(configured, worker_limit), 1);
}

uint32
evoke_runtime_effective_document_pipeline_depth(void)
{
    return evoke_runtime_accelerator_total_capacity();
}

static uint32
evoke_runtime_service_max_document_workers_for_ready(uint32 ready_workers)
{
    uint32 reserved = evoke_runtime_service_reserved_query_workers(ready_workers);

    return ready_workers > reserved ? ready_workers - reserved : 0;
}

static uint32
evoke_runtime_service_max_document_workers_locked(void)
{
    uint32 ready_workers = 0;

    for (uint32 i = 0;
         i < evoke_runtime_service_configured_workers();
         i++)
    {
        if (evoke_runtime_service->workers[i].ready)
        {
            ready_workers++;
        }
    }
    return evoke_runtime_service_max_document_workers_for_ready(ready_workers);
}

static bool
evoke_runtime_service_other_idle_worker_has_affinity_locked(
    uint32 worker_id,
    const char *model_path,
    const char *runtime_precision,
    uint32 request_kind
)
{
    uint32 configured_workers =
        evoke_runtime_service_configured_workers();

    for (uint32 i = 0; i < configured_workers; i++)
    {
        const evoke_runtime_service_worker *worker =
            &evoke_runtime_service->workers[i];

        if (i == worker_id ||
            !worker->ready ||
            worker->processing ||
            worker->affinity_model_path[0] == '\0')
        {
            continue;
        }
        if (worker->affinity_request_kind == request_kind &&
            strcmp(worker->affinity_model_path, model_path) == 0 &&
            strcmp(
                worker->affinity_runtime_precision,
                runtime_precision
            ) == 0)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_runtime_service_model_has_additional_request_locked(
    uint64 request_id,
    const char *model_path,
    const char *runtime_precision,
    uint32 request_kind,
    bool document_capacity
)
{
    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY; i++)
    {
        const evoke_runtime_service_request *request =
            &evoke_runtime_service->queue[i];

        if (!request->occupied ||
            request->canceled ||
            request->request_id == request_id ||
            request->request_kind != request_kind ||
            strcmp(request->model_path, model_path) != 0 ||
            strcmp(request->runtime_precision, runtime_precision) != 0)
        {
            continue;
        }
        if (request->request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT &&
            !document_capacity)
        {
            continue;
        }
        return true;
    }
    return false;
}

static uint32
evoke_runtime_service_select_request_locked(uint32 worker_id)
{
    const evoke_runtime_service_worker *worker =
        &evoke_runtime_service->workers[worker_id];
    uint32 oldest_slot = EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
    uint32 affinity_slot = EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
    uint64 oldest_id = UINT64_MAX;
    uint64 affinity_id = UINT64_MAX;
    uint32 max_document_workers;
    bool document_capacity;

    max_document_workers =
        evoke_runtime_service_max_document_workers_locked();
    document_capacity =
        evoke_runtime_service_active_document_workers_locked() <
            max_document_workers;

    for (uint32 i = 0; i < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY; i++)
    {
        const evoke_runtime_service_request *request =
            &evoke_runtime_service->queue[i];

        if (!request->occupied || request->canceled)
        {
            continue;
        }
        if (request->request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT &&
            !document_capacity)
        {
            continue;
        }
        if (
            (worker->affinity_model_path[0] == '\0' ||
             worker->affinity_request_kind != request->request_kind ||
             strcmp(worker->affinity_model_path, request->model_path) != 0 ||
             strcmp(
                 worker->affinity_runtime_precision,
                 request->runtime_precision
             ) != 0) &&
            evoke_runtime_service_other_idle_worker_has_affinity_locked(
                worker_id,
                request->model_path,
                request->runtime_precision,
                request->request_kind
            ) &&
            !evoke_runtime_service_model_has_additional_request_locked(
                request->request_id,
                request->model_path,
                request->runtime_precision,
                request->request_kind,
                document_capacity
            )
        )
        {
            continue;
        }
        if (request->request_id < oldest_id)
        {
            oldest_id = request->request_id;
            oldest_slot = i;
        }
        if (worker->affinity_model_path[0] != '\0' &&
            worker->affinity_request_kind == request->request_kind &&
            strcmp(worker->affinity_model_path, request->model_path) == 0 &&
            strcmp(
                worker->affinity_runtime_precision,
                request->runtime_precision
            ) == 0 &&
            request->request_id < affinity_id)
        {
            affinity_id = request->request_id;
            affinity_slot = i;
        }
    }

    if (affinity_slot < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY &&
        affinity_id >= oldest_id &&
        affinity_id - oldest_id <=
            EVOKE_RUNTIME_SERVICE_AFFINITY_BYPASS_LIMIT)
    {
        evoke_runtime_service->affinity_dispatches++;
        if (affinity_slot != oldest_slot)
        {
            evoke_runtime_service->affinity_bypasses++;
        }
        return affinity_slot;
    }
    return oldest_slot;
}

static void
evoke_runtime_service_copy_status(StringInfoData *json)
{
    bool available;
    bool worker_started = false;
    bool worker_ready = false;
    bool request_pending = false;
    bool request_processing = false;
    bool response_ready = false;
    pid_t worker_pid = 0;
    pid_t processing_owner_pid = 0;
    uint32 configured_workers = 0;
    uint32 ready_workers = 0;
    uint32 busy_workers = 0;
    uint32 document_workers_busy = 0;
    uint32 document_worker_limit = 0;
    uint32 document_pipeline_depth = 0;
    uint32 document_queue_limit = 0;
    uint32 reserved_query_worker_slots = 0;
    uint32 reserved_query_response_slots = 0;
    uint64 request_id = 0;
    uint64 processing_request_id = 0;
    uint64 completed_request_id = 0;
    uint64 requests = 0;
    uint64 successes = 0;
    uint64 failures = 0;
    uint64 busy_rejections = 0;
    uint64 canceled_requests = 0;
    uint64 orphan_responses = 0;
    uint64 worker_recoveries = 0;
    uint64 queue_waits = 0;
    uint64 queue_total_wait_ms = 0;
    uint64 queue_max_wait_ms = 0;
    uint64 runtime_runs = 0;
    uint64 runtime_total_us = 0;
    uint64 runtime_max_us = 0;
    uint64 encoded_texts = 0;
    uint64 batch_successes = 0;
    uint64 runtime_terminations = 0;
    uint64 runtime_liveness_timeouts = 0;
    uint64 affinity_dispatches = 0;
    uint64 affinity_bypasses = 0;
    uint64 query_dispatches = 0;
    uint64 document_dispatches = 0;
    uint64 session_cache_hits = 0;
    uint64 session_cache_misses = 0;
    uint64 session_cache_loads = 0;
    uint64 session_cache_evictions = 0;
    uint32 queue_depth = 0;
    uint32 queue_capacity = EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY;
    uint32 queue_max_depth = 0;
    uint32 response_slots_in_use = 0;
    uint32 response_slots_ready = 0;
    uint32 response_slots_writing = 0;
    uint32 document_response_slots_in_use = 0;
    uint32 processing_response_slot =
        EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    uint32 processing_batch_count = 0;
    uint32 termination_reasons[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};
    uint32 last_batch_size = 0;
    uint32 max_observed_batch_size = 0;
    TimestampTz worker_started_at = 0;
    TimestampTz last_request_at = 0;
    TimestampTz last_response_at = 0;
    char provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char active_provider[EVOKE_RUNTIME_SERVICE_PROVIDER_MAX_BYTES];
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    evoke_runtime_service_worker
        workers[EVOKE_RUNTIME_SERVICE_MAX_WORKERS] = {0};

    provider[0] = '\0';
    active_provider[0] = '\0';
    runtime_precision[0] = '\0';
    available = evoke_runtime_service_available();
    if (available)
    {
        (void) evoke_runtime_service_ready_worker_count();
    }
    if (available)
    {
        LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
        request_pending = evoke_runtime_service->request_pending;
        configured_workers = evoke_runtime_service_configured_workers();
        memcpy(
            workers,
            evoke_runtime_service->workers,
            sizeof(workers)
        );
        for (uint32 i = 0; i < configured_workers; i++)
        {
            const evoke_runtime_service_worker *worker = &workers[i];

            termination_reasons[i] = pg_atomic_read_u32(
                &evoke_runtime_service->workers[i].termination_reason
            );
            worker_started = worker_started || worker->started;
            if (worker->ready)
            {
                ready_workers++;
                worker_ready = true;
                if (worker_pid == 0)
                {
                    worker_pid = worker->pid;
                }
            }
            if (worker->processing)
            {
                busy_workers++;
                request_processing = true;
                if (worker->processing_document)
                {
                    document_workers_busy++;
                }
                if (processing_request_id == 0)
                {
                    processing_owner_pid =
                        worker->processing_owner_pid;
                    processing_request_id =
                        worker->processing_request_id;
                    processing_response_slot =
                        worker->processing_response_slot;
                    processing_batch_count =
                        worker->processing_batch_count;
                }
            }
            if (worker->started_at > 0 &&
                (worker_started_at == 0 ||
                 worker->started_at < worker_started_at))
            {
                worker_started_at = worker->started_at;
            }
            session_cache_hits += worker->session_cache_hits;
            session_cache_misses += worker->session_cache_misses;
            session_cache_loads += worker->session_cache_loads;
            session_cache_evictions += worker->session_cache_evictions;
        }
        request_id = evoke_runtime_service->request_id;
        completed_request_id =
            evoke_runtime_service->last_completed_request_id;
        requests = evoke_runtime_service->requests;
        successes = evoke_runtime_service->successes;
        failures = evoke_runtime_service->failures;
        busy_rejections = evoke_runtime_service->busy_rejections;
        canceled_requests = evoke_runtime_service->canceled_requests;
        orphan_responses = evoke_runtime_service->orphan_responses;
        worker_recoveries = evoke_runtime_service->worker_recoveries;
        queue_waits = evoke_runtime_service->queue_waits;
        queue_total_wait_ms = evoke_runtime_service->queue_total_wait_ms;
        queue_max_wait_ms = evoke_runtime_service->queue_max_wait_ms;
        runtime_runs = evoke_runtime_service->runtime_runs;
        runtime_total_us = evoke_runtime_service->runtime_total_us;
        runtime_max_us = evoke_runtime_service->runtime_max_us;
        encoded_texts = evoke_runtime_service->encoded_texts;
        batch_successes = evoke_runtime_service->batch_successes;
        runtime_terminations =
            evoke_runtime_service->runtime_terminations;
        runtime_liveness_timeouts =
            evoke_runtime_service->runtime_liveness_timeouts;
        affinity_dispatches = evoke_runtime_service->affinity_dispatches;
        affinity_bypasses = evoke_runtime_service->affinity_bypasses;
        query_dispatches = evoke_runtime_service->query_dispatches;
        document_dispatches = evoke_runtime_service->document_dispatches;
        queue_depth = evoke_runtime_service->queue_count;
        queue_max_depth = evoke_runtime_service->queue_max_depth;
        for (uint32 i = 0;
             i < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY;
             i++)
        {
            if (evoke_runtime_service->responses[i].occupied)
            {
                response_slots_in_use++;
                if (evoke_runtime_service->responses[i].request_kind ==
                    EVOKE_RUNTIME_REQUEST_DOCUMENT)
                {
                    document_response_slots_in_use++;
                }
            }
            if (evoke_runtime_service->responses[i].ready)
            {
                response_slots_ready++;
            }
            if (evoke_runtime_service->responses[i].writing)
            {
                response_slots_writing++;
            }
        }
        response_ready = response_slots_ready > 0;
        reserved_query_worker_slots =
            evoke_runtime_service_reserved_query_workers(ready_workers);
        reserved_query_response_slots =
            evoke_runtime_service_reserved_query_responses();
        document_worker_limit =
            evoke_runtime_service_max_document_workers_for_ready(ready_workers);
        document_pipeline_depth =
            evoke_runtime_effective_document_pipeline_depth();
        document_queue_limit =
            evoke_runtime_effective_local_document_pipeline_depth();
        last_batch_size = evoke_runtime_service->last_batch_size;
        max_observed_batch_size =
            evoke_runtime_service->max_observed_batch_size;
        last_request_at = evoke_runtime_service->last_request_at;
        last_response_at = evoke_runtime_service->last_response_at;
        strlcpy(provider, evoke_runtime_service->provider, sizeof(provider));
        strlcpy(
            active_provider,
            evoke_runtime_service->active_provider,
            sizeof(active_provider)
        );
        strlcpy(
            runtime_precision,
            evoke_runtime_service->runtime_precision,
            sizeof(runtime_precision)
        );
        LWLockRelease(evoke_runtime_service_lock);
    }

    initStringInfo(json);
    appendStringInfoString(
        json,
        "{\"api_version\":\"evoke_index_v1\","
        "\"operation\":\"runtime_service_status\","
    );
    appendStringInfo(json, "\"shared_memory_available\":%s,",
        available ? "true" : "false");
    appendStringInfo(json, "\"worker_started\":%s,",
        worker_started ? "true" : "false");
    appendStringInfo(json, "\"worker_ready\":%s,",
        worker_ready ? "true" : "false");
    appendStringInfo(json, "\"request_pending\":%s,",
        request_pending ? "true" : "false");
    appendStringInfo(json, "\"request_processing\":%s,",
        request_processing ? "true" : "false");
    appendStringInfo(json, "\"response_ready\":%s,",
        response_ready ? "true" : "false");
    appendStringInfo(json, "\"worker_pid\":%d,", (int) worker_pid);
    appendStringInfo(
        json,
        "\"worker_count_configured\":%u,",
        configured_workers
    );
    appendStringInfo(json, "\"worker_count_ready\":%u,", ready_workers);
    appendStringInfo(json, "\"worker_count_busy\":%u,", busy_workers);
    appendStringInfo(
        json,
        "\"document_workers_busy\":%u,",
        document_workers_busy
    );
    appendStringInfo(
        json,
        "\"document_worker_limit\":%u,",
        document_worker_limit
    );
    appendStringInfo(
        json,
        "\"reserved_query_worker_slots\":%u,",
        reserved_query_worker_slots
    );
    appendStringInfo(
        json,
        "\"query_execution_lane_reserved\":%s,",
        (
            reserved_query_worker_slots > 0 ||
            reserved_query_response_slots > 0
        ) ? "true" : "false"
    );
    appendStringInfo(
        json,
        "\"processing_owner_pid\":%d,",
        (int) processing_owner_pid
    );
    appendStringInfo(json, "\"request_id\":%llu,",
        (unsigned long long) request_id);
    appendStringInfo(
        json,
        "\"processing_request_id\":%llu,",
        (unsigned long long) processing_request_id
    );
    appendStringInfo(json, "\"completed_request_id\":%llu,",
        (unsigned long long) completed_request_id);
    appendStringInfo(json, "\"requests\":%llu,",
        (unsigned long long) requests);
    appendStringInfo(json, "\"successes\":%llu,",
        (unsigned long long) successes);
    appendStringInfo(json, "\"failures\":%llu,",
        (unsigned long long) failures);
    appendStringInfo(json, "\"busy_rejections\":%llu,",
        (unsigned long long) busy_rejections);
    appendStringInfo(json, "\"canceled_requests\":%llu,",
        (unsigned long long) canceled_requests);
    appendStringInfo(json, "\"orphan_responses\":%llu,",
        (unsigned long long) orphan_responses);
    appendStringInfo(json, "\"worker_recoveries\":%llu,",
        (unsigned long long) worker_recoveries);
    appendStringInfo(json, "\"queue_waits\":%llu,",
        (unsigned long long) queue_waits);
    appendStringInfo(json, "\"queue_total_wait_ms\":%llu,",
        (unsigned long long) queue_total_wait_ms);
    appendStringInfo(json, "\"queue_max_wait_ms\":%llu,",
        (unsigned long long) queue_max_wait_ms);
    appendStringInfo(json, "\"runtime_runs\":%llu,",
        (unsigned long long) runtime_runs);
    appendStringInfo(json, "\"runtime_total_us\":%llu,",
        (unsigned long long) runtime_total_us);
    appendStringInfo(json, "\"runtime_max_us\":%llu,",
        (unsigned long long) runtime_max_us);
    appendStringInfo(json, "\"encoded_texts\":%llu,",
        (unsigned long long) encoded_texts);
    appendStringInfo(json, "\"batch_successes\":%llu,",
        (unsigned long long) batch_successes);
    appendStringInfo(json, "\"runtime_terminations\":%llu,",
        (unsigned long long) runtime_terminations);
    appendStringInfo(json, "\"runtime_liveness_timeouts\":%llu,",
        (unsigned long long) runtime_liveness_timeouts);
    appendStringInfo(json, "\"affinity_dispatches\":%llu,",
        (unsigned long long) affinity_dispatches);
    appendStringInfo(json, "\"affinity_bypasses\":%llu,",
        (unsigned long long) affinity_bypasses);
    appendStringInfo(json, "\"query_dispatches\":%llu,",
        (unsigned long long) query_dispatches);
    appendStringInfo(json, "\"document_dispatches\":%llu,",
        (unsigned long long) document_dispatches);
    appendStringInfo(json, "\"session_cache_hits\":%llu,",
        (unsigned long long) session_cache_hits);
    appendStringInfo(json, "\"session_cache_misses\":%llu,",
        (unsigned long long) session_cache_misses);
    appendStringInfo(json, "\"session_cache_loads\":%llu,",
        (unsigned long long) session_cache_loads);
    appendStringInfo(json, "\"session_cache_evictions\":%llu,",
        (unsigned long long) session_cache_evictions);
    appendStringInfo(json, "\"queue_depth\":%u,", queue_depth);
    appendStringInfo(json, "\"queue_capacity\":%u,", queue_capacity);
    appendStringInfo(
        json,
        "\"response_capacity\":%u,",
        EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY
    );
    appendStringInfo(
        json,
        "\"response_slots_in_use\":%u,",
        response_slots_in_use
    );
    appendStringInfo(
        json,
        "\"response_slots_ready\":%u,",
        response_slots_ready
    );
    appendStringInfo(
        json,
        "\"response_slots_writing\":%u,",
        response_slots_writing
    );
    appendStringInfo(
        json,
        "\"document_response_slots_in_use\":%u,",
        document_response_slots_in_use
    );
    appendStringInfo(
        json,
        "\"document_pipeline_depth\":%u,",
        document_pipeline_depth
    );
    appendStringInfo(
        json,
        "\"document_queue_limit\":%u,",
        document_queue_limit
    );
    appendStringInfo(
        json,
        "\"document_queue_capacity\":%u,",
        EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS
    );
    appendStringInfo(
        json,
        "\"reserved_query_response_slots\":%u,",
        reserved_query_response_slots
    );
    appendStringInfo(
        json,
        "\"document_response_limit\":%u,",
        EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY - reserved_query_response_slots
    );
    appendStringInfo(json, "\"queue_max_depth\":%u,", queue_max_depth);
    appendStringInfo(
        json,
        "\"processing_batch_count\":%u,",
        processing_batch_count
    );
    appendStringInfo(
        json,
        "\"processing_response_slot\":%u,",
        processing_response_slot
    );
    appendStringInfo(json, "\"last_batch_size\":%u,", last_batch_size);
    appendStringInfo(
        json,
        "\"max_observed_batch_size\":%u,",
        max_observed_batch_size
    );
    appendStringInfo(
        json,
        "\"max_supported_batch_size\":%u,",
        evoke_runtime_effective_max_batch_size()
    );
    appendStringInfo(
        json,
        "\"protocol_max_batch_size\":%u,",
        EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
    );
    appendStringInfo(
        json,
        "\"onnxruntime_intra_op_threads\":%d,",
        evoke_onnxruntime_intra_op_threads
    );
    appendStringInfo(
        json,
        "\"onnxruntime_effective_intra_op_threads\":%d,",
        evoke_runtime_effective_intra_op_threads()
    );
    appendStringInfo(
        json,
        "\"onnxruntime_document_cpu_mem_arena\":%s,",
        evoke_onnxruntime_document_cpu_mem_arena ? "true" : "false"
    );
    appendStringInfo(
        json,
        "\"accelerator_service_count\":%u,",
        evoke_runtime_accelerator_service_count()
    );
    appendStringInfoString(json, "\"accelerator_services\":");
    appendStringInfoString(
        json,
        (evoke_runtime_accelerators != NULL &&
         evoke_runtime_accelerators[0] != '\0')
            ? evoke_runtime_accelerators
            : "[]"
    );
    appendStringInfoChar(json, ',');
    appendStringInfoString(json, "\"accelerator_metrics\":");
    evoke_runtime_accelerator_append_metrics_json(json);
    appendStringInfoChar(json, ',');
    appendStringInfoString(
        json,
        "\"queue_policy\":\"bounded_affinity_worker_pool\","
    );
    appendStringInfoString(json, "\"queue_timeout_policy\":\"none\",");
    appendStringInfo(json, "\"worker_started_at\":%lld,",
        (long long) worker_started_at);
    appendStringInfo(json, "\"last_request_at\":%lld,",
        (long long) last_request_at);
    appendStringInfo(json, "\"last_response_at\":%lld,",
        (long long) last_response_at);
    appendStringInfoString(json, "\"provider\":");
    evoke_append_json_string(json, provider[0] == '\0' ? "auto" : provider);
    appendStringInfoString(json, ",\"active_provider\":");
    evoke_append_json_string(
        json,
        active_provider[0] == '\0' ? "cpu" : active_provider
    );
    appendStringInfoString(json, ",\"runtime_precision\":");
    evoke_append_json_string(
        json,
        runtime_precision[0] == '\0'
            ? EVOKE_RUNTIME_PRECISION_FP16
            : runtime_precision
    );
    appendStringInfoString(
        json,
        ",\"model_owner\":\"runtime_worker_pool\","
    );
    appendStringInfo(
        json,
        "\"runtime_liveness_timeout_ms\":%d,",
        evoke_runtime_liveness_timeout_ms
    );
    appendStringInfoString(json, "\"control_database\":");
    evoke_append_json_string(
        json,
        evoke_control_database == NULL ? "" : evoke_control_database
    );
    appendStringInfoString(json, ",\"workers\":[");
    for (uint32 i = 0; i < configured_workers; i++)
    {
        const evoke_runtime_service_worker *worker = &workers[i];

        if (i > 0)
        {
            appendStringInfoChar(json, ',');
        }
        appendStringInfo(
            json,
            "{\"slot\":%u,\"pid\":%d,\"started\":%s,\"ready\":%s,"
            "\"processing\":%s,\"processing_document\":%s,"
            "\"processing_request_id\":%llu,\"starts\":%llu,"
            "\"recoveries\":%llu,\"runtime_runs\":%llu,"
            "\"successes\":%llu,\"failures\":%llu,"
            "\"runtime_total_us\":%llu,\"runtime_max_us\":%llu,"
            "\"encoded_texts\":%llu,\"runtime_terminations\":%llu,"
            "\"runtime_liveness_timeouts\":%llu,"
            "\"terminate_requested\":%s,"
            "\"processing_started_at\":%lld,\"last_progress_at\":%lld,"
            "\"session_cache_hits\":%llu,"
            "\"session_cache_misses\":%llu,\"session_cache_loads\":%llu,"
            "\"session_cache_evictions\":%llu,"
            "\"affinity_model_loaded\":%s,\"affinity_role\":",
            i,
            (int) worker->pid,
            worker->started ? "true" : "false",
            worker->ready ? "true" : "false",
            worker->processing ? "true" : "false",
            worker->processing_document ? "true" : "false",
            (unsigned long long) worker->processing_request_id,
            (unsigned long long) worker->starts,
            (unsigned long long) worker->recoveries,
            (unsigned long long) worker->runtime_runs,
            (unsigned long long) worker->successes,
            (unsigned long long) worker->failures,
            (unsigned long long) worker->runtime_total_us,
            (unsigned long long) worker->runtime_max_us,
            (unsigned long long) worker->encoded_texts,
            (unsigned long long) worker->runtime_terminations,
            (unsigned long long) worker->runtime_liveness_timeouts,
            (
                worker->terminate_requested ||
                termination_reasons[i] != EVOKE_RUNTIME_TERMINATION_NONE
            ) ? "true" : "false",
            (long long) worker->processing_started_at,
            (long long) worker->last_progress_at,
            (unsigned long long) worker->session_cache_hits,
            (unsigned long long) worker->session_cache_misses,
            (unsigned long long) worker->session_cache_loads,
            (unsigned long long) worker->session_cache_evictions,
            worker->affinity_model_path[0] != '\0' ? "true" : "false"
        );
        evoke_append_json_string(
            json,
            worker->affinity_request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT
                ? "document"
                : worker->affinity_request_kind ==
                    EVOKE_RUNTIME_REQUEST_QUERY
                    ? "query"
                    : "none"
        );
        appendStringInfoString(json, ",\"provider\":");
        evoke_append_json_string(
            json,
            worker->provider[0] == '\0' ? "auto" : worker->provider
        );
        appendStringInfoString(json, ",\"active_provider\":");
        evoke_append_json_string(
            json,
            worker->active_provider[0] == '\0'
                ? "cpu"
                : worker->active_provider
        );
        appendStringInfoString(json, ",\"runtime_precision\":");
        evoke_append_json_string(
            json,
            worker->runtime_precision[0] == '\0'
                ? EVOKE_RUNTIME_PRECISION_FP16
                : worker->runtime_precision
        );
        appendStringInfoChar(json, '}');
    }
    appendStringInfoString(json, "],");
    appendStringInfoString(json, "\"backend_model_loading_allowed\":false,");
    appendStringInfoString(json, "\"cpu_builtin\":");
#ifdef EVOKE_ENABLE_ONNXRUNTIME
    appendStringInfoString(json, "true,");
#else
    appendStringInfoString(json, "false,");
#endif
    appendStringInfoString(
        json,
        "\"accelerator_packages\":[\"tensorrt\",\"cuda\",\"coreml\"],"
        "\"fp16_fallback_order\":[\"tensorrt\",\"coreml\",\"cuda\",\"cpu\"],"
        "\"fp32_fallback_order\":[\"cuda\",\"tensorrt\",\"coreml\",\"cpu\"],"
        "\"ready_for_text_encoding\":"
    );
#ifdef EVOKE_ENABLE_ONNXRUNTIME
    appendStringInfoString(json, worker_ready ? "true" : "false");
#else
    appendStringInfoString(json, "false");
#endif
    appendStringInfoChar(json, '}');
}

PG_FUNCTION_INFO_V1(evoke_runtime_service_status);
Datum
evoke_runtime_service_status(PG_FUNCTION_ARGS)
{
    StringInfoData json;

    evoke_runtime_service_copy_status(&json);
    PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

static void
evoke_runtime_service_finish_request_locked(
    uint32 worker_id,
    uint64 request_id,
    uint32 batch_count,
    uint64 runtime_us,
    bool is_error
)
{
    evoke_runtime_service_worker *worker;

    if (worker_id >= evoke_runtime_service_configured_workers())
    {
        return;
    }
    worker = &evoke_runtime_service->workers[worker_id];
    if (!worker->processing ||
        worker->processing_request_id != request_id)
    {
        return;
    }

    worker->processing = false;
    worker->processing_document = false;
    pg_atomic_write_u32(&worker->cancel_requested, 0);
    pg_atomic_write_u32(
        &worker->termination_reason,
        EVOKE_RUNTIME_TERMINATION_NONE
    );
    worker->terminate_requested = false;
    worker->processing_request_id = 0;
    worker->processing_owner_pid = 0;
    worker->processing_owner_proc_number = INVALID_PROC_NUMBER;
    worker->processing_response_slot =
        EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    worker->processing_batch_count = 0;
    worker->processing_started_at = 0;
    worker->last_progress_at = GetCurrentTimestamp();
    worker->runtime_runs++;
    worker->runtime_total_us += runtime_us;
    worker->runtime_max_us = Max(worker->runtime_max_us, runtime_us);
    worker->last_response_at = GetCurrentTimestamp();
    evoke_runtime_service->runtime_runs++;
    evoke_runtime_service->runtime_total_us += runtime_us;
    evoke_runtime_service->runtime_max_us = Max(
        evoke_runtime_service->runtime_max_us,
        runtime_us
    );
    evoke_runtime_service->last_batch_size = batch_count;
    evoke_runtime_service->max_observed_batch_size = Max(
        evoke_runtime_service->max_observed_batch_size,
        batch_count
    );
    if (is_error)
    {
        worker->failures++;
        evoke_runtime_service->failures++;
    }
    else
    {
        worker->successes++;
        worker->encoded_texts += batch_count;
        evoke_runtime_service->successes++;
        evoke_runtime_service->encoded_texts += batch_count;
        if (batch_count > 1)
        {
            worker->batch_successes++;
            evoke_runtime_service->batch_successes++;
        }
    }
}

static void
evoke_runtime_service_store_response(
    uint32 worker_id,
    uint64 request_id,
    uint32 response_slot,
    uint32 batch_count,
    uint64 runtime_us,
    const char *result_json,
    const char *error_message
)
{
    evoke_runtime_service_response *response;
    bool deliver = false;
    bool is_error;
    Size result_len = 0;
    pid_t response_owner_pid = 0;
    ProcNumber response_owner_proc_number = INVALID_PROC_NUMBER;

    if (error_message == NULL && result_json == NULL)
    {
        error_message = "runtime returned no result";
    }
    is_error = error_message != NULL;
    if (!is_error)
    {
        result_len = strlen(result_json);
        if (result_len >= EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES)
        {
            result_len = EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES - 1;
        }
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    if (response_slot < EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY)
    {
        response = &evoke_runtime_service->responses[response_slot];
        if (response->occupied &&
            response->request_id == request_id &&
            response->owner_pid > 0 &&
            response->owner_proc_number != INVALID_PROC_NUMBER)
        {
            response->ready = false;
            response->writing = true;
            response->error = is_error;
            response->result_len = result_len;
            response_owner_pid = response->owner_pid;
            response_owner_proc_number = response->owner_proc_number;
            deliver = true;
        }
        else if (response->occupied &&
                 response->request_id == request_id)
        {
            evoke_runtime_service_clear_response_locked(response_slot);
            evoke_runtime_service->orphan_responses++;
        }
    }
    if (!deliver)
    {
        evoke_runtime_service_finish_request_locked(
            worker_id,
            request_id,
            batch_count,
            runtime_us,
            is_error
        );
    }
    LWLockRelease(evoke_runtime_service_lock);

    if (!deliver)
    {
        return;
    }

    response = &evoke_runtime_service->responses[response_slot];
    if (is_error)
    {
        strlcpy(
            response->error_message,
            error_message,
            sizeof(response->error_message)
        );
        response->result_json[0] = '\0';
    }
    else
    {
        response->error_message[0] = '\0';
        memcpy(response->result_json, result_json, result_len);
        response->result_json[result_len] = '\0';
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    response = &evoke_runtime_service->responses[response_slot];
    if (response->occupied &&
        response->request_id == request_id &&
        response->owner_pid == response_owner_pid &&
        response->owner_proc_number == response_owner_proc_number)
    {
        response->writing = false;
        response->ready = true;
        evoke_runtime_service->last_completed_request_id = request_id;
        evoke_runtime_service->last_response_at = GetCurrentTimestamp();
    }
    else
    {
        evoke_runtime_service_clear_response_locked(response_slot);
        response_owner_pid = 0;
        response_owner_proc_number = INVALID_PROC_NUMBER;
        evoke_runtime_service->orphan_responses++;
    }
    evoke_runtime_service_finish_request_locked(
        worker_id,
        request_id,
        batch_count,
        runtime_us,
        is_error
    );
    LWLockRelease(evoke_runtime_service_lock);
    (void) evoke_runtime_service_signal_backend(
        response_owner_pid,
        response_owner_proc_number
    );
}

static void
evoke_runtime_service_init_handle(EvokeRuntimeRequestHandle *handle)
{
    memset(handle, 0, sizeof(*handle));
    handle->response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    handle->accelerator_fd = -1;
}

static bool
evoke_runtime_service_try_enqueue_prepared(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    const Size *text_lengths,
    Size query_len,
    uint32 batch_count,
    evoke_runtime_request_kind request_kind,
    bool batch_response,
    EvokeRuntimeRequestHandle *handle
)
{
    bool queue_full;
    uint32 pending_document_requests;
    uint32 document_response_slots;
    uint32 reserved_query_response_slots;
    uint32 response_slot;
    uint32 slot;
    Size write_len = 0;

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    pending_document_requests =
        evoke_runtime_service_pending_document_requests_locked();
    document_response_slots =
        evoke_runtime_service_document_response_slots_locked();
    reserved_query_response_slots =
        evoke_runtime_service_reserved_query_responses();
    response_slot = evoke_runtime_service_find_response_slot_locked();
    slot = evoke_runtime_service_find_request_slot_locked();
    queue_full = evoke_runtime_service->queue_count >=
            EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY ||
        slot >= EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY ||
        response_slot == EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT ||
        (
            request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT &&
            (
                pending_document_requests >=
                    evoke_runtime_effective_local_document_pipeline_depth() ||
                document_response_slots >=
                    EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY -
                        reserved_query_response_slots
            )
        );
    if (queue_full)
    {
        LWLockRelease(evoke_runtime_service_lock);
        handle->response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
        return false;
    }

    evoke_runtime_service->request_id++;
    handle->request_id = evoke_runtime_service->request_id;
    handle->response_slot = response_slot;
    evoke_runtime_service->requests++;
    evoke_runtime_service->last_request_at = GetCurrentTimestamp();
    evoke_runtime_service_clear_response_locked(handle->response_slot);
    evoke_runtime_service->responses[handle->response_slot].occupied = true;
    evoke_runtime_service->responses[handle->response_slot].request_id =
        handle->request_id;
    evoke_runtime_service->responses[handle->response_slot].owner_pid =
        MyProcPid;
    evoke_runtime_service->responses[handle->response_slot].owner_proc_number =
        MyProcNumber;
    evoke_runtime_service->responses[handle->response_slot].request_kind =
        request_kind;
    evoke_runtime_service->queue[slot].occupied = true;
    evoke_runtime_service->queue[slot].canceled = false;
    evoke_runtime_service->queue[slot].request_id = handle->request_id;
    evoke_runtime_service->queue[slot].caller_pid = MyProcPid;
    evoke_runtime_service->queue[slot].caller_proc_number = MyProcNumber;
    evoke_runtime_service->queue[slot].response_slot = handle->response_slot;
    evoke_runtime_service->queue[slot].enqueued_ms = evoke_monotonic_ms();
    evoke_runtime_service->queue[slot].request_kind = request_kind;
    evoke_runtime_service->queue[slot].batch_response = batch_response;
    evoke_runtime_service->queue[slot].batch_count = batch_count;
    strlcpy(
        evoke_runtime_service->queue[slot].model_path,
        model_path,
        sizeof(evoke_runtime_service->queue[slot].model_path)
    );
    strlcpy(
        evoke_runtime_service->queue[slot].runtime_precision,
        runtime_precision,
        sizeof(evoke_runtime_service->queue[slot].runtime_precision)
    );
    for (uint32 i = 0; i < batch_count; i++)
    {
        evoke_runtime_service->queue[slot].text_offsets[i] =
            (uint32) write_len;
        memcpy(
            evoke_runtime_service->queue[slot].query_text + write_len,
            texts[i],
            text_lengths[i]
        );
        write_len += text_lengths[i];
        evoke_runtime_service->queue[slot].query_text[write_len++] = '\0';
    }
    Assert(write_len == query_len);
    evoke_runtime_service->queue[slot].text_offsets[batch_count] =
        (uint32) write_len;
    evoke_runtime_service->queue[slot].query_len = write_len;
    evoke_runtime_service->queue_count++;
    evoke_runtime_service->request_pending =
        evoke_runtime_service->queue_count > 0;
    if (evoke_runtime_service->queue_count >
        evoke_runtime_service->queue_max_depth)
    {
        evoke_runtime_service->queue_max_depth =
            evoke_runtime_service->queue_count;
    }
    LWLockRelease(evoke_runtime_service_lock);
    handle->active = true;
    evoke_runtime_service_signal_workers();
    return true;
}

static void
evoke_runtime_service_enqueue(
    const char *model_path,
    const char *requested_runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    evoke_runtime_request_kind request_kind,
    bool batch_response,
    EvokeRuntimeRequestHandle *handle
)
{
    Size text_lengths[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE];
    Size query_len = 0;
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];

    if (handle == NULL)
    {
        ereport(ERROR, (errmsg("evoke runtime request handle is missing")));
    }
    evoke_runtime_service_init_handle(handle);
    if (model_path == NULL || model_path[0] == '\0')
    {
        ereport(ERROR, (errmsg("model path must not be empty")));
    }
    evoke_runtime_precision_normalize(
        requested_runtime_precision,
        runtime_precision
    );
    if (batch_count == 0 ||
        batch_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke runtime batch size"),
                errdetail(
                    "batch_count=%u max_batch_size=%u",
                    batch_count,
                    EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
                )
            )
        );
    }
    if (
        request_kind != EVOKE_RUNTIME_REQUEST_QUERY &&
        request_kind != EVOKE_RUNTIME_REQUEST_DOCUMENT
    )
    {
        ereport(ERROR, (errmsg("invalid evoke runtime request kind")));
    }
    if (request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT && !batch_response)
    {
        ereport(
            ERROR,
            (errmsg("evoke document runtime requests require batch output"))
        );
    }
    if (strlen(model_path) >= MAXPGPATH)
    {
        ereport(ERROR, (errmsg("model path is too long for evoke runtime service")));
    }
    for (uint32 i = 0; i < batch_count; i++)
    {
        Size text_len;

        if (texts[i] == NULL || texts[i][0] == '\0')
        {
            ereport(
                ERROR,
                (errmsg("runtime batch text %u must not be empty", i + 1))
            );
        }
        text_len = strlen(texts[i]);
        if (text_len >= EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES ||
            query_len > EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES - text_len - 1)
        {
            ereport(
                ERROR,
                (
                    errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                    errmsg(
                        "evoke runtime service text batch exceeds the "
                        "transport limit"
                    ),
                    errdetail(
                        "batch_count=%u total_limit=%u bytes",
                        batch_count,
                        EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES - 1
                    )
                )
            );
        }
        text_lengths[i] = text_len;
        query_len += text_len + 1;
    }
    if (!evoke_runtime_service_available())
    {
        ereport(
            ERROR,
            (
                errmsg("evoke SAE runtime service is not available"),
                errdetail(
                    "Load evoke through shared_preload_libraries so the "
                    "shared runtime worker pool owns model sessions.")
            )
        );
    }

    for (;;)
    {
        CHECK_FOR_INTERRUPTS();
        ResetLatch(&MyProc->procLatch);
        if (!evoke_runtime_service_worker_is_ready())
        {
            ereport(
                ERROR,
                (
                    errmsg("no evoke SAE runtime worker is ready"),
                    errhint(
                        "Retry after a shared runtime worker has started "
                        "or restarted."
                    )
                )
            );
        }
        evoke_runtime_service_drop_orphan_responses();
        if (evoke_runtime_service_try_enqueue_prepared(
                model_path,
                runtime_precision,
                texts,
                text_lengths,
                query_len,
                batch_count,
                request_kind,
                batch_response,
                handle))
        {
            return;
        }
        (void) WaitLatch(
            &MyProc->procLatch,
            WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
            10,
            0
        );
    }
}

static void
evoke_runtime_service_append_document_chunk_results(
    StringInfo merged,
    const char *result_json,
    const char *expected_checkout_signature,
    const char *expected_runtime_precision,
    uint32 expected_result_count,
    bool *first_result
)
{
    Jsonb *result;
    JsonbValue *checkout_signature;
    JsonbValue *runtime_precision;
    JsonbValue *results_value;
    JsonbContainer *results;
    uint32 result_count;

    if (merged == NULL || result_json == NULL ||
        expected_checkout_signature == NULL ||
        expected_runtime_precision == NULL || first_result == NULL)
    {
        ereport(ERROR, (errmsg("evoke document runtime merge is invalid")));
    }
    result = DatumGetJsonbP(
        DirectFunctionCall1(jsonb_in, CStringGetDatum(result_json))
    );
    if (!JB_ROOT_IS_OBJECT(result))
    {
        ereport(ERROR, (errmsg("evoke document runtime response is invalid")));
    }
    checkout_signature = evoke_runtime_jsonb_object_get(
        &result->root,
        "checkout_signature"
    );
    if (!evoke_runtime_jsonb_string_equals(
            checkout_signature,
            expected_checkout_signature))
    {
        ereport(
            ERROR,
            (errmsg("evoke document runtime checkout changed during encoding"))
        );
    }
    pfree(checkout_signature);
    runtime_precision = evoke_runtime_jsonb_object_get(
        &result->root,
        "runtime_precision"
    );
    if (!evoke_runtime_jsonb_string_equals(
            runtime_precision,
            expected_runtime_precision))
    {
        ereport(
            ERROR,
            (errmsg("evoke document runtime precision changed during encoding"))
        );
    }
    pfree(runtime_precision);
    results_value = evoke_runtime_jsonb_object_get(&result->root, "results");
    if (results_value == NULL || results_value->type != jbvBinary ||
        !JsonContainerIsArray(results_value->val.binary.data))
    {
        ereport(ERROR, (errmsg("evoke document runtime response is invalid")));
    }
    results = results_value->val.binary.data;
    result_count = JsonContainerSize(results);
    if (result_count != expected_result_count)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke document runtime result count is invalid"),
                errdetail(
                    "expected_count=%u result_count=%u",
                    expected_result_count,
                    result_count
                )
            )
        );
    }
    for (uint32 i = 0; i < result_count; i++)
    {
        JsonbValue *item = getIthJsonbValueFromContainer(results, i);
        Jsonb *item_jsonb;
        char *item_json;

        if (item == NULL)
        {
            ereport(
                ERROR,
                (errmsg("evoke document runtime response is invalid"))
            );
        }
        item_jsonb = JsonbValueToJsonb(item);
        item_json = DatumGetCString(
            DirectFunctionCall1(jsonb_out, JsonbPGetDatum(item_jsonb))
        );
        if (!*first_result)
        {
            appendStringInfoChar(merged, ',');
        }
        appendStringInfoString(merged, item_json);
        *first_result = false;
        pfree(item_json);
        pfree(item_jsonb);
        pfree(item);
    }
    pfree(results_value);
    pfree(result);
}

static char *
evoke_runtime_service_wait_document_pipeline(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count
)
{
    EvokeRuntimeRequestHandle
        handles[EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS] = {0};
    uint32 expected_counts[
        EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS
    ] = {0};
    char normalized_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    StringInfoData merged;
    char *checkout_signature;
    uint32 pipeline_depth;
    uint32 submitted_offset = 0;
    uint32 inflight_count = 0;
    uint32 head = 0;
    uint32 tail = 0;
    bool first_result = true;

    if (model_path == NULL || runtime_precision == NULL || texts == NULL ||
        batch_count == 0 ||
        batch_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    {
        ereport(ERROR, (errmsg("invalid evoke document runtime pipeline input")));
    }
    evoke_runtime_precision_normalize(
        runtime_precision,
        normalized_precision
    );
    checkout_signature =
        evoke_runtime_accelerator_checkout_signature(model_path);
    pipeline_depth = Min(
        evoke_runtime_effective_document_pipeline_depth(),
        (uint32) EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS
    );
    pipeline_depth = Max(pipeline_depth, 1U);

    initStringInfo(&merged);
    appendStringInfoString(&merged, "{\"checkout_signature\":");
    evoke_append_json_string(&merged, checkout_signature);
    appendStringInfoString(&merged, ",\"runtime_precision\":");
    evoke_append_json_string(&merged, normalized_precision);
    appendStringInfoString(&merged, ",\"results\":[");

    PG_TRY();
    {
        while (submitted_offset < batch_count || inflight_count > 0)
        {
            while (submitted_offset < batch_count &&
                   inflight_count < pipeline_depth)
            {
                EvokeRuntimeRequestHandle *handle = &handles[tail];
                uint32 submitted_count = 0;

                memset(handle, 0, sizeof(*handle));
                handle->response_slot =
                    EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
                handle->accelerator_fd = -1;
                evoke_runtime_service_submit_document_prefix_checkout_async(
                    model_path,
                    checkout_signature,
                    normalized_precision,
                    texts + submitted_offset,
                    batch_count - submitted_offset,
                    &submitted_count,
                    handle
                );
                if (submitted_count == 0)
                {
                    if (inflight_count == 0)
                    {
                        CHECK_FOR_INTERRUPTS();
                        pg_usleep(1000L);
                    }
                    break;
                }
                expected_counts[tail] = submitted_count;
                submitted_offset += submitted_count;
                inflight_count++;
                tail = (tail + 1U) %
                    EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS;
            }

            if (inflight_count > 0)
            {
                EvokeRuntimeRequestHandle *handle = &handles[head];
                char *chunk_json = NULL;

                chunk_json = evoke_runtime_service_wait_async(handle);
                evoke_runtime_service_append_document_chunk_results(
                    &merged,
                    chunk_json,
                    checkout_signature,
                    normalized_precision,
                    expected_counts[head],
                    &first_result
                );
                pfree(chunk_json);
                expected_counts[head] = 0;
                head = (head + 1U) %
                    EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS;
                inflight_count--;
            }
        }
    }
    PG_CATCH();
    {
        for (uint32 i = 0;
             i < EVOKE_RUNTIME_SERVICE_MAX_PENDING_DOCUMENT_REQUESTS;
             i++)
        {
            evoke_runtime_service_cancel_async(&handles[i]);
        }
        pfree(checkout_signature);
        PG_RE_THROW();
    }
    PG_END_TRY();

    appendStringInfoString(&merged, "]}");
    pfree(checkout_signature);
    return merged.data;
}

static char *
evoke_runtime_service_wait_local_document_chunks(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count
)
{
    uint32 local_max_batch_size =
        evoke_runtime_accelerator_local_max_batch_size();
    StringInfoData merged;
    char *checkout_signature;
    bool first_result = true;

    if (local_max_batch_size == 0)
    {
        ereport(ERROR, (errmsg("evoke local runtime fallback is unavailable")));
    }
    if (batch_count <= local_max_batch_size)
    {
        EvokeRuntimeRequestHandle local_handle = {0};
        char *result_json = NULL;

        local_handle.response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
        local_handle.accelerator_fd = -1;
        PG_TRY();
        {
            evoke_runtime_service_enqueue(
                model_path,
                runtime_precision,
                texts,
                batch_count,
                EVOKE_RUNTIME_REQUEST_DOCUMENT,
                true,
                &local_handle
            );
            local_handle.local_runtime_started_at =
                evoke_runtime_accelerator_note_local_start(batch_count);
            local_handle.local_runtime_batch_count = batch_count;
            local_handle.local_runtime_health_active = true;
            result_json = evoke_runtime_service_wait_async(&local_handle);
        }
        PG_CATCH();
        {
            if (local_handle.local_runtime_health_active)
            {
                evoke_runtime_accelerator_note_local_abandoned();
                local_handle.local_runtime_health_active = false;
            }
            evoke_runtime_service_cancel_async(&local_handle);
            PG_RE_THROW();
        }
        PG_END_TRY();
        return result_json;
    }

    checkout_signature =
        evoke_runtime_accelerator_checkout_signature(model_path);
    initStringInfo(&merged);
    appendStringInfoString(&merged, "{\"checkout_signature\":");
    evoke_append_json_string(&merged, checkout_signature);
    appendStringInfoString(&merged, ",\"runtime_precision\":");
    evoke_append_json_string(&merged, runtime_precision);
    appendStringInfoString(&merged, ",\"results\":[");

    for (uint32 offset = 0; offset < batch_count;)
    {
        EvokeRuntimeRequestHandle local_handle = {0};
        uint32 chunk_count = Min(local_max_batch_size, batch_count - offset);
        char *chunk_json = NULL;

        local_handle.response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
        local_handle.accelerator_fd = -1;
        PG_TRY();
        {
            evoke_runtime_service_enqueue(
                model_path,
                runtime_precision,
                texts + offset,
                chunk_count,
                EVOKE_RUNTIME_REQUEST_DOCUMENT,
                true,
                &local_handle
            );
            local_handle.local_runtime_started_at =
                evoke_runtime_accelerator_note_local_start(chunk_count);
            local_handle.local_runtime_batch_count = chunk_count;
            local_handle.local_runtime_health_active = true;
            chunk_json = evoke_runtime_service_wait_async(&local_handle);
            evoke_runtime_service_append_document_chunk_results(
                &merged,
                chunk_json,
                checkout_signature,
                runtime_precision,
                chunk_count,
                &first_result
            );
        }
        PG_CATCH();
        {
            if (chunk_json != NULL)
            {
                pfree(chunk_json);
            }
            if (local_handle.local_runtime_health_active)
            {
                evoke_runtime_accelerator_note_local_abandoned();
                local_handle.local_runtime_health_active = false;
            }
            evoke_runtime_service_cancel_async(&local_handle);
            pfree(checkout_signature);
            PG_RE_THROW();
        }
        PG_END_TRY();
        if (chunk_json != NULL)
        {
            pfree(chunk_json);
        }
        offset += chunk_count;
    }
    appendStringInfoString(&merged, "]}");
    pfree(checkout_signature);
    return merged.data;
}

static void
evoke_runtime_accelerator_close_failed_fd(int fd)
{
    struct linger linger_option;

    if (fd < 0)
    {
        return;
    }
    memset(&linger_option, 0, sizeof(linger_option));
    linger_option.l_onoff = 1;
    linger_option.l_linger = 0;
    (void) setsockopt(
        fd,
        SOL_SOCKET,
        SO_LINGER,
        &linger_option,
        sizeof(linger_option)
    );
    close(fd);
}

#ifdef EVOKE_ENABLE_ONNXRUNTIME
static void
evoke_runtime_service_free_accelerator_saved_request(
    char *url,
    char *checkout_signature,
    char *model_path,
    char *runtime_precision,
    char **texts,
    uint32 batch_count
)
{
    if (url != NULL)
    {
        pfree(url);
    }
    if (checkout_signature != NULL)
    {
        pfree(checkout_signature);
    }
    if (model_path != NULL)
    {
        pfree(model_path);
    }
    if (runtime_precision != NULL)
    {
        pfree(runtime_precision);
    }
    if (texts != NULL)
    {
        for (uint32 i = 0; i < batch_count; i++)
        {
            if (texts[i] != NULL)
            {
                pfree(texts[i]);
            }
        }
        pfree(texts);
    }
}
#endif

static bool
evoke_runtime_service_enqueue_local_failover_async(
    EvokeRuntimeRequestHandle *handle
)
{
#ifdef EVOKE_ENABLE_ONNXRUNTIME
    Size text_lengths[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE];
    Size query_len = 0;
    char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
    EvokeRuntimeRequestHandle local_handle;
    char *saved_url;
    char *saved_checkout_signature;
    char *saved_model_path;
    char *saved_runtime_precision;
    char **saved_texts;
    uint32 saved_batch_count;

    if (handle == NULL || handle->accelerator_model_path == NULL ||
        handle->accelerator_runtime_precision == NULL ||
        handle->accelerator_texts == NULL ||
        handle->accelerator_batch_count == 0)
    {
        return false;
    }

    saved_url = handle->accelerator_url;
    saved_checkout_signature = handle->accelerator_checkout_signature;
    saved_model_path = handle->accelerator_model_path;
    saved_runtime_precision = handle->accelerator_runtime_precision;
    saved_texts = handle->accelerator_texts;
    saved_batch_count = handle->accelerator_batch_count;

    if (!evoke_runtime_service_available() ||
        !evoke_runtime_service_worker_is_ready())
    {
        return false;
    }
    evoke_runtime_precision_normalize(
        saved_runtime_precision,
        runtime_precision
    );
    for (uint32 i = 0; i < saved_batch_count; i++)
    {
        Size text_len;

        if (saved_texts[i] == NULL || saved_texts[i][0] == '\0')
        {
            ereport(
                ERROR,
                (errmsg("runtime batch text %u must not be empty", i + 1))
            );
        }
        text_len = strlen(saved_texts[i]);
        if (text_len >= EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES ||
            query_len > EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES - text_len - 1)
        {
            ereport(
                ERROR,
                (
                    errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                    errmsg(
                        "evoke runtime service text batch exceeds the "
                        "transport limit"
                    ),
                    errdetail(
                        "batch_count=%u total_limit=%u bytes",
                        saved_batch_count,
                        EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES - 1
                    )
                )
            );
        }
        text_lengths[i] = text_len;
        query_len += text_len + 1;
    }
    evoke_runtime_service_drop_orphan_responses();
    evoke_runtime_service_init_handle(&local_handle);
    if (!evoke_runtime_service_try_enqueue_prepared(
            saved_model_path,
            runtime_precision,
            (const char *const *) saved_texts,
            text_lengths,
            query_len,
            saved_batch_count,
            EVOKE_RUNTIME_REQUEST_DOCUMENT,
            true,
            &local_handle))
    {
        return false;
    }

    PG_TRY();
    {
        *handle = local_handle;
        handle->local_runtime_started_at =
            evoke_runtime_accelerator_note_local_start(saved_batch_count);
        handle->local_runtime_batch_count = saved_batch_count;
        handle->local_runtime_health_active = true;
    }
    PG_CATCH();
    {
        evoke_runtime_service_free_accelerator_saved_request(
            saved_url,
            saved_checkout_signature,
            saved_model_path,
            saved_runtime_precision,
            saved_texts,
            saved_batch_count
        );
        PG_RE_THROW();
    }
    PG_END_TRY();

    evoke_runtime_service_free_accelerator_saved_request(
        saved_url,
        saved_checkout_signature,
        saved_model_path,
        saved_runtime_precision,
        saved_texts,
        saved_batch_count
    );
    return true;
#else
    return false;
#endif
}

static bool
evoke_runtime_service_failover_accelerator_async(
    EvokeRuntimeRequestHandle *handle,
    int status,
    const char *status_body
)
{
    char *retry_url = NULL;
    TimestampTz now;
    TimestampTz retry_started_at = 0;
    bool deadline_reached;
    bool local_fallback_allowed;
    int retry_fd;

    if (handle == NULL || !handle->accelerator_active)
    {
        return false;
    }
    if (handle->accelerator_fd >= 0)
    {
        evoke_runtime_accelerator_close_failed_fd(handle->accelerator_fd);
        handle->accelerator_fd = -1;
    }
    if (handle->accelerator_health_active && handle->accelerator_url != NULL)
    {
        if (status == 503)
        {
            evoke_runtime_accelerator_note_backpressure(handle->accelerator_url);
        }
        else
        {
            evoke_runtime_accelerator_note_request_failure(
                handle->accelerator_url,
                status == 0
                    ? EVOKE_RUNTIME_ACCELERATOR_FAILURE_READ
                    : status < 0
                        ? EVOKE_RUNTIME_ACCELERATOR_FAILURE_BODY
                    : EVOKE_RUNTIME_ACCELERATOR_FAILURE_STATUS,
                status,
                status_body
            );
        }
    }
    handle->accelerator_health_active = false;
    evoke_runtime_accelerator_http_clear_response(handle);

    if (handle->accelerator_retry_count < UINT32_MAX)
    {
        handle->accelerator_retry_count++;
    }
    now = GetCurrentTimestamp();
    deadline_reached = handle->accelerator_deadline_at > 0 &&
        now >= handle->accelerator_deadline_at;
    local_fallback_allowed = deadline_reached ||
        handle->accelerator_retry_count >=
            EVOKE_RUNTIME_ACCELERATOR_ASYNC_RETRY_LIMIT;
    if (local_fallback_allowed &&
        evoke_runtime_service_enqueue_local_failover_async(handle))
    {
        return true;
    }
    if (deadline_reached)
    {
#ifdef EVOKE_ENABLE_ONNXRUNTIME
        if (handle->accelerator_local_fallback_deadline_at == 0)
        {
            handle->accelerator_local_fallback_deadline_at = now +
                (TimestampTz)
                    EVOKE_RUNTIME_ACCELERATOR_LOCAL_FAILOVER_WAIT_MS * 1000;
        }
        if (now < handle->accelerator_local_fallback_deadline_at)
        {
            handle->accelerator_retry_after = now +
                (TimestampTz)
                    EVOKE_RUNTIME_ACCELERATOR_ASYNC_RETRY_WAIT_MS * 1000;
            return true;
        }
#endif
        evoke_runtime_request_clear_accelerator(handle, false);
        handle->active = false;
        return false;
    }
    retry_fd = evoke_runtime_accelerator_start_document_batch(
        handle->accelerator_model_path,
        handle->accelerator_checkout_signature,
        handle->accelerator_runtime_precision,
        (const char *const *) handle->accelerator_texts,
        handle->accelerator_batch_count,
        &retry_url,
        &retry_started_at
    );
    if (retry_fd >= 0)
    {
        if (handle->accelerator_url != NULL)
        {
            pfree(handle->accelerator_url);
        }
        handle->accelerator_url = retry_url;
        retry_url = NULL;
        handle->accelerator_fd = retry_fd;
        handle->accelerator_started_at = retry_started_at;
        handle->accelerator_retry_after = 0;
        handle->accelerator_health_active = true;
        return true;
    }
    if (retry_url != NULL)
    {
        pfree(retry_url);
        retry_url = NULL;
    }

    handle->accelerator_retry_after = now +
        (TimestampTz) EVOKE_RUNTIME_ACCELERATOR_ASYNC_RETRY_WAIT_MS * 1000;
    return true;
}

static char *
evoke_runtime_service_wait_accelerator(EvokeRuntimeRequestHandle *handle)
{
    char *result_json = NULL;
    char *body = NULL;
    int status = 0;
    int fd;
    uint32 retry_count = 0;
    uint32 retry_limit = EVOKE_RUNTIME_ACCELERATOR_RETRY_LIMIT;
    TimestampTz retry_started_at = 0;
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    uint32 required_wait_ms = 0;
#endif

#ifndef EVOKE_ENABLE_ONNXRUNTIME
    retry_limit = EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT;
#endif
    if (handle == NULL || !handle->accelerator_active ||
        handle->accelerator_fd < 0)
    {
        ereport(ERROR, (errmsg("evoke accelerator request handle is invalid")));
    }
    for (;;)
    {
        char *retry_url = NULL;

        fd = handle->accelerator_fd;
        if (evoke_runtime_accelerator_http_wait_response(
                handle,
                &status,
                &body) &&
            status == 200 && body != NULL)
        {
            if (fd >= 0)
            {
                if (handle->accelerator_response_keep_alive)
                {
                    evoke_runtime_accelerator_release_idle_connection(
                        handle->accelerator_url,
                        fd
                    );
                }
                else
                {
                    close(fd);
                }
                handle->accelerator_fd = -1;
            }
            evoke_runtime_accelerator_note_success(
                handle->accelerator_url,
                handle->accelerator_batch_count,
                handle->accelerator_started_at
            );
            handle->accelerator_health_active = false;
            evoke_runtime_request_clear_accelerator(handle, false);
            handle->active = false;
            return body;
        }
        if (fd >= 0)
        {
            evoke_runtime_accelerator_close_failed_fd(fd);
            handle->accelerator_fd = -1;
        }
        if (status == 503)
        {
            evoke_runtime_accelerator_note_backpressure(
                handle->accelerator_url
            );
        }
        else
        {
            evoke_runtime_accelerator_note_request_failure(
                handle->accelerator_url,
                status == 0
                    ? EVOKE_RUNTIME_ACCELERATOR_FAILURE_READ
                    : EVOKE_RUNTIME_ACCELERATOR_FAILURE_STATUS,
                status,
                body
            );
        }
        handle->accelerator_health_active = false;
        if (body != NULL)
        {
            pfree(body);
            body = NULL;
        }
        evoke_runtime_accelerator_http_clear_response(handle);
        for (;;)
        {
            if (retry_count >= retry_limit)
            {
                break;
            }
            retry_count++;
            fd = evoke_runtime_accelerator_start_document_batch(
                handle->accelerator_model_path,
                handle->accelerator_checkout_signature,
                handle->accelerator_runtime_precision,
                (const char *const *) handle->accelerator_texts,
                handle->accelerator_batch_count,
                &retry_url,
                &retry_started_at
            );
            if (fd >= 0)
            {
                break;
            }
            if (retry_url != NULL)
            {
                pfree(retry_url);
                retry_url = NULL;
            }
#ifndef EVOKE_ENABLE_ONNXRUNTIME
            if (!evoke_runtime_accelerator_wait_required_retry(
                    &required_wait_ms))
            {
                break;
            }
            continue;
#else
            break;
#endif
        }
        if (fd < 0)
        {
            break;
        }
        if (handle->accelerator_url != NULL)
        {
            pfree(handle->accelerator_url);
        }
        handle->accelerator_url = retry_url;
        handle->accelerator_fd = fd;
        handle->accelerator_started_at = retry_started_at;
        handle->accelerator_health_active = true;
    }

#ifndef EVOKE_ENABLE_ONNXRUNTIME
    evoke_runtime_request_clear_accelerator(handle, false);
    handle->active = false;
    ereport(
        ERROR,
        (
            errmsg("no evoke document runtime accelerator is available"),
            errdetail(
                "All configured remote accelerators failed or remained "
                "unavailable while local ONNX Runtime support is disabled."
            )
        )
    );
#endif

    PG_TRY();
    {
        result_json = evoke_runtime_service_wait_local_document_chunks(
            handle->accelerator_model_path,
            handle->accelerator_runtime_precision,
            (const char *const *) handle->accelerator_texts,
            handle->accelerator_batch_count
        );
    }
    PG_CATCH();
    {
        evoke_runtime_request_clear_accelerator(handle, false);
        handle->active = false;
        PG_RE_THROW();
    }
    PG_END_TRY();

    evoke_runtime_request_clear_accelerator(handle, false);
    handle->active = false;
    return result_json;
}

bool
evoke_runtime_service_try_complete_async(
    EvokeRuntimeRequestHandle *handle,
    char **result_json_out
)
{
    if (result_json_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke runtime result output is missing")));
    }
    *result_json_out = NULL;
    if (handle == NULL || !handle->active)
    {
        return false;
    }
    if (handle->accelerator_active)
    {
        bool complete = false;
        int status = 0;
        char *body = NULL;
        int fd;

        if (handle->accelerator_fd < 0)
        {
            if (!evoke_runtime_service_failover_accelerator_async(
                    handle,
                    0,
                    NULL))
            {
                ereport(ERROR, (errmsg("evoke accelerator failover failed")));
            }
            return false;
        }
        if (!evoke_runtime_accelerator_http_try_read_response(
                handle,
                &complete))
        {
            if (!evoke_runtime_service_failover_accelerator_async(
                    handle,
                    0,
                    NULL))
            {
                ereport(ERROR, (errmsg("evoke accelerator failover failed")));
            }
            return false;
        }
        if (!complete)
        {
            if (evoke_runtime_accelerator_request_timed_out(handle) ||
                (
                    handle->accelerator_deadline_at > 0 &&
                    GetCurrentTimestamp() >=
                        handle->accelerator_deadline_at
                ))
            {
                if (!evoke_runtime_service_failover_accelerator_async(
                    handle,
                    0,
                    NULL))
                {
                    ereport(
                        ERROR,
                        (errmsg("evoke accelerator failover failed"))
                    );
                }
                return false;
            }
            return false;
        }
        status = handle->accelerator_response_status;
        if (status != 200)
        {
            body = evoke_runtime_accelerator_http_take_body(handle, &status);
            if (!evoke_runtime_service_failover_accelerator_async(
                    handle,
                    status,
                    body))
            {
                if (body != NULL)
                {
                    pfree(body);
                }
                ereport(ERROR, (errmsg("evoke accelerator failover failed")));
            }
            if (body != NULL)
            {
                pfree(body);
            }
            return false;
        }
        body = evoke_runtime_accelerator_http_take_body(handle, &status);
        if (body == NULL)
        {
            if (!evoke_runtime_service_failover_accelerator_async(
                    handle,
                    -1,
                    NULL))
            {
                ereport(ERROR, (errmsg("evoke accelerator failover failed")));
            }
            return false;
        }
        fd = handle->accelerator_fd;
        if (fd >= 0)
        {
            if (handle->accelerator_response_keep_alive)
            {
                evoke_runtime_accelerator_release_idle_connection(
                    handle->accelerator_url,
                    fd
                );
            }
            else
            {
                close(fd);
            }
            handle->accelerator_fd = -1;
        }
        evoke_runtime_accelerator_note_success(
            handle->accelerator_url,
            handle->accelerator_batch_count,
            handle->accelerator_started_at
        );
        handle->accelerator_health_active = false;
        evoke_runtime_request_clear_accelerator(handle, false);
        handle->active = false;
        *result_json_out = body;
        return true;
    }
    if (!evoke_runtime_service_request_ready(handle))
    {
        return false;
    }
    *result_json_out = evoke_runtime_service_wait_async(handle);
    return true;
}

char *
evoke_runtime_service_wait_async(EvokeRuntimeRequestHandle *handle)
{
    char *result_json = NULL;
    char *error_message = NULL;

    if (handle == NULL || !handle->active ||
        (
            !handle->accelerator_active &&
            (
                handle->request_id == 0 ||
                handle->response_slot >=
                    EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY
            )
        ))
    {
        ereport(ERROR, (errmsg("evoke runtime request handle is not active")));
    }
    if (handle->accelerator_active)
    {
        return evoke_runtime_service_wait_accelerator(handle);
    }

    PG_TRY();
    {
        for (;;)
        {
            bool ready;
            bool is_error = false;
            Size result_len = 0;
            char local_error[EVOKE_RUNTIME_SERVICE_ERROR_MAX_BYTES];
            evoke_runtime_service_response *response;

            CHECK_FOR_INTERRUPTS();
            ResetLatch(&MyProc->procLatch);
            local_error[0] = '\0';
            LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
            response = &evoke_runtime_service->responses[
                handle->response_slot];
            ready = response->occupied &&
                response->ready &&
                response->request_id == handle->request_id &&
                response->owner_pid == MyProcPid &&
                response->owner_proc_number == MyProcNumber;
            if (ready)
            {
                is_error = response->error;
                if (is_error)
                {
                    strlcpy(
                        local_error,
                        response->error_message,
                        sizeof(local_error)
                    );
                }
                else
                {
                    result_len = response->result_len;
                }
            }
            LWLockRelease(evoke_runtime_service_lock);
            if (ready)
            {
                if (is_error)
                {
                    error_message = pstrdup(local_error);
                }
                else
                {
                    result_json = palloc(result_len + 1);
                    memcpy(
                        result_json,
                        evoke_runtime_service->responses[
                            handle->response_slot].result_json,
                        result_len
                    );
                    result_json[result_len] = '\0';
                }
                LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
                response = &evoke_runtime_service->responses[
                    handle->response_slot];
                if (response->occupied &&
                    response->ready &&
                    response->request_id == handle->request_id &&
                    response->owner_pid == MyProcPid &&
                    response->owner_proc_number == MyProcNumber)
                {
                    evoke_runtime_service_clear_response_locked(
                        handle->response_slot
                    );
                    handle->active = false;
                    LWLockRelease(evoke_runtime_service_lock);
                    if (handle->local_runtime_health_active)
                    {
                        if (is_error)
                        {
                            evoke_runtime_accelerator_note_local_abandoned();
                        }
                        else
                        {
                            evoke_runtime_accelerator_note_local_success(
                                handle->local_runtime_batch_count,
                                handle->local_runtime_started_at
                            );
                        }
                        handle->local_runtime_health_active = false;
                    }
                    evoke_runtime_service_signal_workers();
                    break;
                }
                LWLockRelease(evoke_runtime_service_lock);
                if (result_json != NULL)
                {
                    pfree(result_json);
                    result_json = NULL;
                }
                if (error_message != NULL)
                {
                    pfree(error_message);
                    error_message = NULL;
                }
                continue;
            }

            if (!evoke_runtime_service_worker_is_ready())
            {
                ereport(
                    ERROR,
                    (
                        errmsg("all evoke SAE runtime workers became unavailable"),
                        errhint(
                            "Retry after a shared runtime worker has "
                            "restarted."
                        )
                    )
                );
            }

            (void) WaitLatch(
                &MyProc->procLatch,
                WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                1000,
                0
            );
        }
    }
    PG_CATCH();
    {
        if (handle->local_runtime_health_active)
        {
            evoke_runtime_accelerator_note_local_abandoned();
            handle->local_runtime_health_active = false;
        }
        evoke_runtime_service_cancel_request(handle->request_id);
        handle->active = false;
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (error_message != NULL)
    {
        ereport(
            ERROR,
            (
                errmsg("evoke SAE runtime service failed"),
                errdetail("%s", error_message)
            )
        );
    }

    return result_json;
}

bool
evoke_runtime_service_request_ready(const EvokeRuntimeRequestHandle *handle)
{
    if (handle == NULL || !handle->active)
    {
        return false;
    }
    if (handle->accelerator_active)
    {
        struct pollfd poll_fd;
        int rc;

        if (handle->accelerator_fd < 0)
        {
            TimestampTz now = GetCurrentTimestamp();

            if (
                handle->accelerator_local_fallback_deadline_at > 0 &&
                now >= handle->accelerator_local_fallback_deadline_at
            )
            {
                return true;
            }
            if (handle->accelerator_retry_after > 0)
            {
                return now >= handle->accelerator_retry_after;
            }
            return true;
        }
        if (handle->accelerator_deadline_at > 0 &&
            GetCurrentTimestamp() >= handle->accelerator_deadline_at)
        {
            return true;
        }
        if (evoke_runtime_accelerator_request_timed_out(handle))
        {
            return true;
        }
        if (evoke_runtime_accelerator_http_response_complete(handle))
        {
            return true;
        }
        memset(&poll_fd, 0, sizeof(poll_fd));
        poll_fd.fd = handle->accelerator_fd;
        poll_fd.events = POLLIN | POLLERR | POLLHUP;
        rc = poll(&poll_fd, 1, 0);
        if (rc < 0)
        {
            return errno != EINTR;
        }
        return rc > 0 &&
            (poll_fd.revents & (POLLIN | POLLERR | POLLHUP)) != 0;
    }
    if (handle->request_id == 0 ||
        handle->response_slot >= EVOKE_RUNTIME_SERVICE_RESPONSE_CAPACITY ||
        !evoke_runtime_service_attach_if_possible())
    {
        return false;
    }
    LWLockAcquire(evoke_runtime_service_lock, LW_SHARED);
    {
        const evoke_runtime_service_response *response =
            &evoke_runtime_service->responses[handle->response_slot];
        bool ready = response->occupied &&
            response->ready &&
            response->request_id == handle->request_id &&
            response->owner_pid == MyProcPid &&
            response->owner_proc_number == MyProcNumber;

        LWLockRelease(evoke_runtime_service_lock);
        return ready;
    }
}

void
evoke_runtime_service_cancel_async(EvokeRuntimeRequestHandle *handle)
{
    if (handle == NULL || !handle->active)
    {
        return;
    }
    evoke_runtime_request_clear_accelerator(handle, true);
    if (handle->request_id == 0)
    {
        handle->active = false;
        return;
    }
    evoke_runtime_service_cancel_request(handle->request_id);
    handle->active = false;
}

void
evoke_runtime_service_submit_document_prefix_async(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    uint32 *submitted_count_out,
    EvokeRuntimeRequestHandle *handle
)
{
    evoke_runtime_service_submit_document_prefix_checkout_async(
        model_path,
        NULL,
        runtime_precision,
        texts,
        available_count,
        submitted_count_out,
        handle
    );
}

void
evoke_runtime_service_submit_document_prefix_checkout_async(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 available_count,
    uint32 *submitted_count_out,
    EvokeRuntimeRequestHandle *handle
)
{
    char runtime_precision_normalized[
        EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES
    ];
    char *accelerator_url = NULL;
    int accelerator_fd = -1;
    TimestampTz accelerator_started_at = 0;
    MemoryContext old_context = CurrentMemoryContext;
    uint32 retry_limit = EVOKE_RUNTIME_ACCELERATOR_RETRY_LIMIT;
    uint32 batch_count = 0;
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    uint32 required_wait_ms = 0;
#endif

#ifndef EVOKE_ENABLE_ONNXRUNTIME
    retry_limit = EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT;
#endif
    if (submitted_count_out == NULL)
    {
        ereport(ERROR, (errmsg("evoke submitted batch count output is missing")));
    }
    *submitted_count_out = 0;
    if (available_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke runtime batch size")));
    }
    if (handle == NULL)
    {
        ereport(ERROR, (errmsg("evoke runtime request handle is missing")));
    }
    memset(handle, 0, sizeof(*handle));
    handle->response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    handle->accelerator_fd = -1;
    evoke_runtime_precision_normalize(
        runtime_precision,
        runtime_precision_normalized
    );
    for (uint32 attempt = 0; attempt <= retry_limit; attempt++)
    {
        PG_TRY();
        {
            accelerator_fd =
                evoke_runtime_accelerator_start_document_prefix_batch(
                    model_path,
                    checkout_signature,
                    runtime_precision_normalized,
                    texts,
                    available_count,
                    &batch_count,
                    &accelerator_url,
                    &accelerator_started_at
                );
        }
        PG_CATCH();
        {
            int caught_sqlerrcode = geterrcode();

            MemoryContextSwitchTo(old_context);
            if (accelerator_fd >= 0)
            {
                close(accelerator_fd);
            }
            if (accelerator_url != NULL)
            {
                pfree(accelerator_url);
            }
            accelerator_fd = -1;
            accelerator_url = NULL;
            batch_count = 0;
            if (caught_sqlerrcode == ERRCODE_QUERY_CANCELED)
            {
                PG_RE_THROW();
            }
            FlushErrorState();
        }
        PG_END_TRY();
        if (accelerator_fd >= 0)
        {
            break;
        }
#ifdef EVOKE_ENABLE_ONNXRUNTIME
        break;
#else
        if (!evoke_runtime_accelerator_wait_required_retry(
                &required_wait_ms))
        {
            break;
        }
#endif
    }
    if (accelerator_fd >= 0)
    {
        handle->accelerator_fd = accelerator_fd;
        handle->accelerator_url = accelerator_url;
        accelerator_url = NULL;
        handle->accelerator_started_at = accelerator_started_at;
        handle->accelerator_health_active = true;
        handle->accelerator_active = true;
        PG_TRY();
        {
            evoke_runtime_request_save_accelerator_fallback(
                handle,
                model_path,
                checkout_signature,
                runtime_precision_normalized,
                texts,
                batch_count
            );
        }
        PG_CATCH();
        {
            evoke_runtime_request_clear_accelerator(handle, true);
            PG_RE_THROW();
        }
        PG_END_TRY();
        handle->active = true;
        *submitted_count_out = batch_count;
        return;
    }
    if (accelerator_url != NULL)
    {
        pfree(accelerator_url);
    }
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    ereport(
        ERROR,
        (
            errmsg("no evoke document runtime accelerator is available"),
            errdetail(
                "evoke was built without ONNX Runtime CPU support, so "
                "document builds require a reachable remote accelerator."
            )
        )
    );
#endif
    if (batch_count == 0)
    {
        batch_count = evoke_runtime_accelerator_candidate_batch_size(
            available_count,
            evoke_runtime_accelerator_local_max_batch_size(),
            false
        );
    }
    if (batch_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke runtime batch size")));
    }
#ifdef EVOKE_ENABLE_ONNXRUNTIME
    if (evoke_runtime_accelerator_local_score(batch_count) == DBL_MAX)
    {
        *submitted_count_out = 0;
        return;
    }
#endif
    PG_TRY();
    {
        evoke_runtime_service_enqueue(
            model_path,
            runtime_precision_normalized,
            texts,
            batch_count,
            EVOKE_RUNTIME_REQUEST_DOCUMENT,
            true,
            handle
        );
        handle->local_runtime_started_at =
            evoke_runtime_accelerator_note_local_start(batch_count);
        handle->local_runtime_batch_count = batch_count;
        handle->local_runtime_health_active = true;
    }
    PG_CATCH();
    {
        if (handle->local_runtime_health_active)
        {
            evoke_runtime_accelerator_note_local_abandoned();
            handle->local_runtime_health_active = false;
        }
        PG_RE_THROW();
    }
    PG_END_TRY();
    *submitted_count_out = batch_count;
}

static void
evoke_runtime_service_submit_document_checkout_async(
    const char *model_path,
    const char *checkout_signature,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    EvokeRuntimeRequestHandle *handle
)
{
    char runtime_precision_normalized[
        EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES
    ];
    char *accelerator_url = NULL;
    int accelerator_fd = -1;
    TimestampTz accelerator_started_at = 0;
    MemoryContext old_context = CurrentMemoryContext;
    uint32 retry_limit = EVOKE_RUNTIME_ACCELERATOR_RETRY_LIMIT;
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    uint32 required_wait_ms = 0;
#endif

#ifndef EVOKE_ENABLE_ONNXRUNTIME
    retry_limit = EVOKE_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT;
#endif
    if (handle == NULL)
    {
        ereport(ERROR, (errmsg("evoke runtime request handle is missing")));
    }
    memset(handle, 0, sizeof(*handle));
    handle->response_slot = EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
    handle->accelerator_fd = -1;
    evoke_runtime_precision_normalize(
        runtime_precision,
        runtime_precision_normalized
    );
    for (uint32 attempt = 0; attempt <= retry_limit; attempt++)
    {
        PG_TRY();
        {
            accelerator_fd = evoke_runtime_accelerator_start_document_batch(
                model_path,
                checkout_signature,
                runtime_precision_normalized,
                texts,
                batch_count,
                &accelerator_url,
                &accelerator_started_at
            );
        }
        PG_CATCH();
        {
            int caught_sqlerrcode = geterrcode();

            MemoryContextSwitchTo(old_context);
            if (accelerator_fd >= 0)
            {
                close(accelerator_fd);
            }
            if (accelerator_url != NULL)
            {
                pfree(accelerator_url);
            }
            accelerator_fd = -1;
            accelerator_url = NULL;
            if (caught_sqlerrcode == ERRCODE_QUERY_CANCELED)
            {
                PG_RE_THROW();
            }
            FlushErrorState();
        }
        PG_END_TRY();
        if (accelerator_fd >= 0)
        {
            break;
        }
#ifdef EVOKE_ENABLE_ONNXRUNTIME
        break;
#else
        if (!evoke_runtime_accelerator_wait_required_retry(
                &required_wait_ms))
        {
            break;
        }
#endif
    }
    if (accelerator_fd >= 0)
    {
        handle->accelerator_fd = accelerator_fd;
        handle->accelerator_url = accelerator_url;
        accelerator_url = NULL;
        handle->accelerator_started_at = accelerator_started_at;
        handle->accelerator_health_active = true;
        handle->accelerator_active = true;
        PG_TRY();
        {
            evoke_runtime_request_save_accelerator_fallback(
                handle,
                model_path,
                checkout_signature,
                runtime_precision_normalized,
                texts,
                batch_count
            );
        }
        PG_CATCH();
        {
            evoke_runtime_request_clear_accelerator(handle, true);
            PG_RE_THROW();
        }
        PG_END_TRY();
        handle->active = true;
        return;
    }
    if (accelerator_url != NULL)
    {
        pfree(accelerator_url);
    }
#ifndef EVOKE_ENABLE_ONNXRUNTIME
    ereport(
        ERROR,
        (
            errmsg("no evoke document runtime accelerator is available"),
            errdetail(
                "evoke was built without ONNX Runtime CPU support, so "
                "document builds require a reachable remote accelerator."
            )
        )
    );
#endif
    PG_TRY();
    {
        evoke_runtime_service_enqueue(
            model_path,
            runtime_precision_normalized,
            texts,
            batch_count,
            EVOKE_RUNTIME_REQUEST_DOCUMENT,
            true,
            handle
        );
        handle->local_runtime_started_at =
            evoke_runtime_accelerator_note_local_start(batch_count);
        handle->local_runtime_batch_count = batch_count;
        handle->local_runtime_health_active = true;
    }
    PG_CATCH();
    {
        if (handle->local_runtime_health_active)
        {
            evoke_runtime_accelerator_note_local_abandoned();
            handle->local_runtime_health_active = false;
        }
        PG_RE_THROW();
    }
    PG_END_TRY();
}

void
evoke_runtime_service_submit_document_async(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    EvokeRuntimeRequestHandle *handle
)
{
    evoke_runtime_service_submit_document_checkout_async(
        model_path,
        NULL,
        runtime_precision,
        texts,
        batch_count,
        handle
    );
}

static char *
evoke_runtime_service_submit(
    const char *model_path,
    const char *runtime_precision,
    const char *const *texts,
    uint32 batch_count,
    evoke_runtime_request_kind request_kind,
    bool batch_response
)
{
    EvokeRuntimeRequestHandle handle = {0};
    char *result_json = NULL;

    PG_TRY();
    {
        evoke_runtime_service_enqueue(
            model_path,
            runtime_precision,
            texts,
            batch_count,
            request_kind,
            batch_response,
            &handle
        );
        result_json = evoke_runtime_service_wait_async(&handle);
    }
    PG_CATCH();
    {
        evoke_runtime_service_cancel_async(&handle);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return result_json;
}

PG_FUNCTION_INFO_V1(evoke_runtime_service_query_atoms);
Datum
evoke_runtime_service_query_atoms(PG_FUNCTION_ARGS)
{
    char *model_path;
    char *runtime_precision;
    char *query_text;
    const char *texts[1];
    char *result_json;

    if (PG_NARGS() != 2 && PG_NARGS() != 3)
    {
        ereport(ERROR, (errmsg("invalid evoke runtime query signature")));
    }
    if (PG_ARGISNULL(0) ||
        (PG_NARGS() == 2 && PG_ARGISNULL(1)) ||
        (PG_NARGS() == 3 && (PG_ARGISNULL(1) || PG_ARGISNULL(2))))
    {
        ereport(
            ERROR,
            (errmsg("model path, runtime precision, and query text are required"))
        );
    }

    evoke_checkout_require_superuser();
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(0));
    if (PG_NARGS() == 3)
    {
        runtime_precision = text_to_cstring(PG_GETARG_TEXT_PP(1));
        query_text = text_to_cstring(PG_GETARG_TEXT_PP(2));
    }
    else
    {
        runtime_precision = pstrdup(EVOKE_RUNTIME_PRECISION_FP16);
        query_text = text_to_cstring(PG_GETARG_TEXT_PP(1));
    }
    texts[0] = query_text;
    result_json = evoke_runtime_service_submit(
        model_path,
        runtime_precision,
        texts,
        1,
        EVOKE_RUNTIME_REQUEST_QUERY,
        false
    );

    PG_RETURN_DATUM(DirectFunctionCall1(
        jsonb_in,
        CStringGetDatum(result_json)
    ));
}

static Datum
evoke_runtime_service_atoms_batch(
    FunctionCallInfo fcinfo,
    evoke_runtime_request_kind request_kind
)
{
    char *model_path;
    char *runtime_precision;
    ArrayType *texts_array;
    int texts_argno;
    Datum *text_datums;
    bool *text_nulls;
    int text_count;
    const char *texts[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE];
    char *result_json;

    if (PG_NARGS() != 2 && PG_NARGS() != 3)
    {
        ereport(ERROR, (errmsg("invalid evoke runtime batch signature")));
    }
    texts_argno = PG_NARGS() == 3 ? 2 : 1;
    if (PG_ARGISNULL(0) ||
        (PG_NARGS() == 3 && PG_ARGISNULL(1)) ||
        PG_ARGISNULL(texts_argno))
    {
        ereport(
            ERROR,
            (errmsg("model path, runtime precision, and text batch are required"))
        );
    }
    evoke_checkout_require_superuser();
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(0));
    runtime_precision = PG_NARGS() == 3
        ? text_to_cstring(PG_GETARG_TEXT_PP(1))
        : pstrdup(EVOKE_RUNTIME_PRECISION_FP16);
    texts_array = PG_GETARG_ARRAYTYPE_P(texts_argno);
    if (ARR_NDIM(texts_array) != 1)
    {
        ereport(ERROR, (errmsg("evoke runtime text batch must be one-dimensional")));
    }
    deconstruct_array(
        texts_array,
        TEXTOID,
        -1,
        false,
        TYPALIGN_INT,
        &text_datums,
        &text_nulls,
        &text_count
    );
    if (text_count <= 0 ||
        text_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke runtime text batch size"),
                errdetail(
                    "batch_count=%d max_batch_size=%u",
                    text_count,
                    EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE
                )
            )
        );
    }
    for (int i = 0; i < text_count; i++)
    {
        if (text_nulls[i])
        {
            ereport(
                ERROR,
                (errmsg("evoke runtime text batch must not contain nulls"))
            );
        }
        texts[i] = text_to_cstring(DatumGetTextPP(text_datums[i]));
    }
    if (request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT)
    {
        result_json = evoke_runtime_service_wait_document_pipeline(
            model_path,
            runtime_precision,
            texts,
            (uint32) text_count
        );
    }
    else
    {
        result_json = evoke_runtime_service_submit(
            model_path,
            runtime_precision,
            texts,
            (uint32) text_count,
            request_kind,
            true
        );
    }

    PG_RETURN_DATUM(DirectFunctionCall1(
        jsonb_in,
        CStringGetDatum(result_json)
    ));
}

PG_FUNCTION_INFO_V1(evoke_runtime_service_query_atoms_batch);
Datum
evoke_runtime_service_query_atoms_batch(PG_FUNCTION_ARGS)
{
    return evoke_runtime_service_atoms_batch(
        fcinfo,
        EVOKE_RUNTIME_REQUEST_QUERY
    );
}

PG_FUNCTION_INFO_V1(evoke_runtime_service_document_atoms_batch);
Datum
evoke_runtime_service_document_atoms_batch(PG_FUNCTION_ARGS)
{
    return evoke_runtime_service_atoms_batch(
        fcinfo,
        EVOKE_RUNTIME_REQUEST_DOCUMENT
    );
}

static void
evoke_runtime_worker_shmem_exit(int code, Datum arg)
{
    uint32 worker_id = DatumGetUInt32(arg);
    pid_t interrupted_owner_pid = 0;
    ProcNumber interrupted_owner_proc_number = INVALID_PROC_NUMBER;

    (void) code;
    /* The complete shared service is disappearing during postmaster stop. */
    if (ProcDiePending)
    {
        return;
    }
    if (evoke_runtime_service == NULL ||
        evoke_runtime_service_lock == NULL ||
        worker_id >= EVOKE_RUNTIME_SERVICE_MAX_WORKERS)
    {
        return;
    }

    LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
    interrupted_owner_pid =
        evoke_runtime_service_mark_worker_unavailable_locked(
            worker_id,
            MyProcPid,
            MyProcNumber,
            "evoke SAE runtime worker exited while processing request",
            &interrupted_owner_proc_number
        );
    LWLockRelease(evoke_runtime_service_lock);
    (void) evoke_runtime_service_signal_backend(
        interrupted_owner_pid,
        interrupted_owner_proc_number
    );
}

PGDLLEXPORT void
evoke_runtime_worker_main(Datum main_arg)
{
    uint32 worker_id = DatumGetUInt32(main_arg);
    pid_t interrupted_owner_pid = 0;
    ProcNumber interrupted_owner_proc_number = INVALID_PROC_NUMBER;
    evoke_runtime_service_worker *worker;

    pqsignal(SIGTERM, die);
    pqsignal(SIGHUP, evoke_runtime_worker_sighup);
    BackgroundWorkerUnblockSignals();
    BackgroundWorkerInitializeConnection(
        evoke_control_database,
        NULL,
        BGWORKER_BYPASS_ALLOWCONN | BGWORKER_BYPASS_ROLELOGINCHECK
    );
    pgstat_report_appname("evoke runtime service");

    if (worker_id >= EVOKE_RUNTIME_SERVICE_MAX_WORKERS)
    {
        ereport(
            FATAL,
            (
                errmsg("evoke runtime worker slot is invalid"),
                errdetail("worker_slot=%u", worker_id)
            )
        );
    }
    if (evoke_runtime_service_available())
    {
        LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
        if (worker_id >= evoke_runtime_service_configured_workers())
        {
            LWLockRelease(evoke_runtime_service_lock);
            ereport(
                FATAL,
                (
                    errmsg("evoke runtime worker slot is not configured"),
                    errdetail(
                        "worker_slot=%u configured_workers=%u",
                        worker_id,
                        evoke_runtime_service->configured_worker_count
                    )
                )
            );
        }
        worker = &evoke_runtime_service->workers[worker_id];
        if (worker->started)
        {
            worker->recoveries++;
            evoke_runtime_service->worker_recoveries++;
        }
        interrupted_owner_pid =
            evoke_runtime_service_mark_worker_unavailable_locked(
                worker_id,
                0,
                INVALID_PROC_NUMBER,
                "evoke SAE runtime worker restarted while processing request",
                &interrupted_owner_proc_number
            );
        worker = &evoke_runtime_service->workers[worker_id];
        worker->started = true;
        worker->ready = true;
        worker->pid = MyProcPid;
        worker->proc_number = MyProcNumber;
        worker->starts++;
        worker->started_at = GetCurrentTimestamp();
        snprintf(
            worker->provider,
            sizeof(worker->provider),
            "auto"
        );
        snprintf(
            worker->active_provider,
            sizeof(worker->active_provider),
            "cpu"
        );
        snprintf(
            worker->runtime_precision,
            sizeof(worker->runtime_precision),
            "%s",
            EVOKE_RUNTIME_PRECISION_FP16
        );
        LWLockRelease(evoke_runtime_service_lock);
        before_shmem_exit(
            evoke_runtime_worker_shmem_exit,
            UInt32GetDatum(worker_id)
        );
        (void) evoke_runtime_service_signal_backend(
            interrupted_owner_pid,
            interrupted_owner_proc_number
        );
    }

    for (;;)
    {
        bool has_request = false;
        uint64 request_id = 0;
        pid_t owner_pid = 0;
        ProcNumber owner_proc_number = INVALID_PROC_NUMBER;
        uint32 response_slot =
            EVOKE_RUNTIME_SERVICE_INVALID_RESPONSE_SLOT;
        uint32 batch_count = 0;
        uint32 request_kind = EVOKE_RUNTIME_REQUEST_QUERY;
        bool batch_response = false;
        char model_path[MAXPGPATH];
        char runtime_precision[EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES];
        char text_data[EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES];
        const char *texts[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE] = {0};
        uint32 text_offsets[EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE + 1];
        char *result_json = NULL;
        char error_message[EVOKE_RUNTIME_SERVICE_ERROR_MAX_BYTES];
#ifdef EVOKE_ENABLE_ONNXRUNTIME
        MemoryContext request_context = NULL;
        volatile bool started_xact = false;
        volatile bool pushed_snapshot = false;
#endif
        double runtime_started_ms = 0.0;
        uint64 runtime_us = 0;

        CHECK_FOR_INTERRUPTS();
        if (evoke_runtime_worker_got_sighup)
        {
            evoke_runtime_worker_got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }
        ResetLatch(&MyProc->procLatch);
        if (!evoke_runtime_service_available())
        {
            (void) WaitLatch(
                &MyProc->procLatch,
                WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                1000,
                0
            );
            continue;
        }
        evoke_runtime_service_drop_orphan_responses();

        error_message[0] = '\0';
        LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
        evoke_runtime_service_drop_canceled_requests_locked();
        worker = &evoke_runtime_service->workers[worker_id];
        if (evoke_runtime_service->queue_count > 0 &&
            worker->ready &&
            !worker->processing)
        {
            uint32 slot =
                evoke_runtime_service_select_request_locked(worker_id);

            if (slot < EVOKE_RUNTIME_SERVICE_QUEUE_CAPACITY)
            {
                Size query_len =
                    evoke_runtime_service->queue[slot].query_len;
                double queued_ms;

                has_request = true;
                request_id =
                    evoke_runtime_service->queue[slot].request_id;
                owner_pid =
                    evoke_runtime_service->queue[slot].caller_pid;
                owner_proc_number =
                    evoke_runtime_service->queue[
                        slot].caller_proc_number;
                response_slot =
                    evoke_runtime_service->queue[slot].response_slot;
                batch_count =
                    evoke_runtime_service->queue[slot].batch_count;
                request_kind =
                    evoke_runtime_service->queue[slot].request_kind;
                batch_response =
                    evoke_runtime_service->queue[slot].batch_response;
                worker->processing = true;
                worker->processing_document =
                    request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT;
                pg_atomic_write_u32(&worker->cancel_requested, 0);
                pg_atomic_write_u32(
                    &worker->termination_reason,
                    EVOKE_RUNTIME_TERMINATION_NONE
                );
                worker->terminate_requested = false;
                worker->processing_request_id = request_id;
                worker->processing_owner_pid = owner_pid;
                worker->processing_owner_proc_number =
                    owner_proc_number;
                worker->processing_response_slot = response_slot;
                worker->processing_batch_count = batch_count;
                worker->processing_started_at = GetCurrentTimestamp();
                worker->last_progress_at = worker->processing_started_at;
                worker->last_request_at = worker->processing_started_at;
                if (worker->processing_document)
                {
                    evoke_runtime_service->document_dispatches++;
                }
                else
                {
                    evoke_runtime_service->query_dispatches++;
                }
                strlcpy(
                    model_path,
                    evoke_runtime_service->queue[slot].model_path,
                    sizeof(model_path)
                );
                strlcpy(
                    runtime_precision,
                    evoke_runtime_service->queue[slot].runtime_precision,
                    sizeof(runtime_precision)
                );
                memcpy(
                    text_data,
                    evoke_runtime_service->queue[slot].query_text,
                    query_len
                );
                memcpy(
                    text_offsets,
                    evoke_runtime_service->queue[slot].text_offsets,
                    sizeof(uint32) * (batch_count + 1)
                );
                queued_ms = Max(
                    evoke_monotonic_ms() -
                        evoke_runtime_service->queue[slot].enqueued_ms,
                    0.0
                );
                if (queued_ms >= 1.0)
                {
                    uint64 wait_ms = (uint64) queued_ms;

                    evoke_runtime_service->queue_waits++;
                    evoke_runtime_service->queue_total_wait_ms += wait_ms;
                    if (wait_ms >
                        evoke_runtime_service->queue_max_wait_ms)
                    {
                        evoke_runtime_service->queue_max_wait_ms = wait_ms;
                    }
                }
                memset(
                    &evoke_runtime_service->queue[slot],
                    0,
                    sizeof(evoke_runtime_service->queue[slot])
                );
                evoke_runtime_service->queue_count--;
                evoke_runtime_service->request_pending =
                    evoke_runtime_service->queue_count > 0;
            }
        }
        LWLockRelease(evoke_runtime_service_lock);

        if (!has_request)
        {
            (void) WaitLatch(
                &MyProc->procLatch,
                WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                1000,
                0
            );
            continue;
        }

        if (batch_count == 0 ||
            batch_count > EVOKE_RUNTIME_SERVICE_MAX_BATCH_SIZE ||
            text_offsets[batch_count] == 0 ||
            text_offsets[batch_count] > EVOKE_RUNTIME_SERVICE_TEXT_MAX_BYTES)
        {
            snprintf(
                error_message,
                sizeof(error_message),
                "runtime request contains an invalid text batch"
            );
        }
        else if (
            request_kind != EVOKE_RUNTIME_REQUEST_QUERY &&
            request_kind != EVOKE_RUNTIME_REQUEST_DOCUMENT
        )
        {
            snprintf(
                error_message,
                sizeof(error_message),
                "runtime request contains an invalid compiler role"
            );
        }
        else if (
            request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT &&
            !batch_response
        )
        {
            snprintf(
                error_message,
                sizeof(error_message),
                "document runtime request requires batch output"
            );
        }
        else
        {
            char normalized_precision[
                EVOKE_RUNTIME_SERVICE_PRECISION_MAX_BYTES
            ];

            evoke_runtime_precision_normalize(
                runtime_precision,
                normalized_precision
            );
            strlcpy(
                runtime_precision,
                normalized_precision,
                sizeof(runtime_precision)
            );
            for (uint32 i = 0; i < batch_count; i++)
            {
                if (text_offsets[i] >= text_offsets[i + 1] ||
                    text_data[text_offsets[i + 1] - 1] != '\0')
                {
                    snprintf(
                        error_message,
                        sizeof(error_message),
                        "runtime request contains invalid text offsets"
                    );
                    break;
                }
                texts[i] = text_data + text_offsets[i];
            }
        }
        runtime_started_ms = evoke_monotonic_ms();

#ifdef EVOKE_ENABLE_ONNXRUNTIME
        if (error_message[0] == '\0')
        {
            StartTransactionCommand();
            started_xact = true;
            PushActiveSnapshot(GetTransactionSnapshot());
            pushed_snapshot = true;
            request_context = AllocSetContextCreate(
                TopMemoryContext,
                "evoke runtime request",
                ALLOCSET_DEFAULT_SIZES
            );
            MemoryContextSwitchTo(request_context);
            evoke_runtime_active_worker_id = worker_id;
            evoke_runtime_active_request_id = request_id;
            PG_TRY();
            {
                char *request_result;

                if (request_kind == EVOKE_RUNTIME_REQUEST_DOCUMENT)
                {
                    result_json = evoke_onnxruntime_atoms_batch_json(
                        model_path,
                        runtime_precision,
                        texts,
                        batch_count,
                        true
                    );
                }
                else if (batch_response)
                {
                    result_json = evoke_onnxruntime_atoms_batch_json(
                        model_path,
                        runtime_precision,
                        texts,
                        batch_count,
                        false
                    );
                }
                else
                {
                    if (batch_count != 1)
                    {
                        ereport(
                            ERROR,
                            (errmsg("single runtime request has batch size %u",
                                batch_count))
                        );
                    }
                    result_json = evoke_p2_query_atoms_json(
                        model_path,
                        runtime_precision,
                        texts[0]
                    );
                }
                if (result_json == NULL)
                {
                    ereport(ERROR, (errmsg("runtime returned no result")));
                }
                request_result = result_json;
                if (strlen(request_result) >=
                    EVOKE_RUNTIME_SERVICE_RESULT_MAX_BYTES)
                {
                    snprintf(
                        error_message,
                        sizeof(error_message),
                        "runtime result exceeds shared response buffer"
                    );
                    result_json = NULL;
                }
                else
                {
                    MemoryContextSwitchTo(TopMemoryContext);
                    result_json = pstrdup(request_result);
                }
                PopActiveSnapshot();
                pushed_snapshot = false;
                CommitTransactionCommand();
                started_xact = false;
            }
            PG_CATCH();
            {
                ErrorData *edata;
                const char *error_detail;

                MemoryContextSwitchTo(TopMemoryContext);
                edata = CopyErrorData();
                FlushErrorState();
                if (pushed_snapshot)
                {
                    PopActiveSnapshot();
                    pushed_snapshot = false;
                }
                if (started_xact)
                {
                    AbortCurrentTransaction();
                    started_xact = false;
                }
                error_detail = edata->detail_log != NULL
                    ? edata->detail_log
                    : edata->detail;
                if (error_detail != NULL && error_detail[0] != '\0')
                {
                    snprintf(
                        error_message,
                        sizeof(error_message),
                        "%s: %s",
                        edata->message != NULL
                            ? edata->message
                            : "unknown runtime error",
                        error_detail
                    );
                }
                else
                {
                    snprintf(
                        error_message,
                        sizeof(error_message),
                        "%s",
                        edata->message != NULL
                            ? edata->message
                            : "unknown runtime error"
                    );
                }
                FreeErrorData(edata);
                result_json = NULL;
            }
            PG_END_TRY();
            evoke_runtime_active_request_id = 0;
            evoke_runtime_active_worker_id =
                EVOKE_RUNTIME_SERVICE_INVALID_WORKER_SLOT;
            MemoryContextSwitchTo(TopMemoryContext);
            MemoryContextDelete(request_context);
            request_context = NULL;
        }
#else
        snprintf(
            error_message,
            sizeof(error_message),
            "evoke was built without ONNX Runtime CPU support"
        );
#endif

        runtime_us = (uint64) Max(
            (evoke_monotonic_ms() - runtime_started_ms) * 1000.0,
            0.0
        );

        if (evoke_runtime_service_available())
        {
            LWLockAcquire(evoke_runtime_service_lock, LW_EXCLUSIVE);
            worker = &evoke_runtime_service->workers[worker_id];
#ifdef EVOKE_ENABLE_ONNXRUNTIME
            if (evoke_ort_session_cache.last_requested_provider[0] != '\0')
            {
                strlcpy(
                    evoke_runtime_service->provider,
                    evoke_ort_session_cache.last_requested_provider,
                    sizeof(evoke_runtime_service->provider)
                );
                strlcpy(
                    worker->provider,
                    evoke_ort_session_cache.last_requested_provider,
                    sizeof(worker->provider)
                );
            }
            if (evoke_ort_session_cache.last_active_provider[0] != '\0')
            {
                strlcpy(
                    evoke_runtime_service->active_provider,
                    evoke_ort_session_cache.last_active_provider,
                    sizeof(evoke_runtime_service->active_provider)
                );
                strlcpy(
                    worker->active_provider,
                    evoke_ort_session_cache.last_active_provider,
                    sizeof(worker->active_provider)
                );
            }
            if (evoke_ort_session_cache.last_runtime_precision[0] != '\0')
            {
                strlcpy(
                    evoke_runtime_service->runtime_precision,
                    evoke_ort_session_cache.last_runtime_precision,
                    sizeof(evoke_runtime_service->runtime_precision)
                );
                strlcpy(
                    worker->runtime_precision,
                    evoke_ort_session_cache.last_runtime_precision,
                    sizeof(worker->runtime_precision)
                );
            }
            worker->session_cache_hits = evoke_ort_session_cache.hits;
            worker->session_cache_misses = evoke_ort_session_cache.misses;
            worker->session_cache_loads = evoke_ort_session_cache.loads;
            worker->session_cache_evictions =
                evoke_ort_session_cache.evictions;
            if (error_message[0] == '\0' &&
                evoke_ort_effective_cache_size() > 0)
            {
                worker->affinity_request_kind = request_kind;
                strlcpy(
                    worker->affinity_model_path,
                    model_path,
                    sizeof(worker->affinity_model_path)
                );
                strlcpy(
                    worker->affinity_runtime_precision,
                    runtime_precision,
                    sizeof(worker->affinity_runtime_precision)
                );
            }
            else if (evoke_ort_effective_cache_size() == 0)
            {
                worker->affinity_request_kind = 0;
                worker->affinity_model_path[0] = '\0';
                worker->affinity_runtime_precision[0] = '\0';
            }
#endif
            LWLockRelease(evoke_runtime_service_lock);
        }

        evoke_runtime_service_store_response(
            worker_id,
            request_id,
            response_slot,
            batch_count,
            runtime_us,
            result_json,
            error_message[0] == '\0' ? NULL : error_message
        );
        evoke_runtime_service_signal_workers();
        if (result_json != NULL)
        {
            pfree(result_json);
        }
    }
}

PG_FUNCTION_INFO_V1(evoke_index_checkout_manifest_internal_c);
Datum
evoke_index_checkout_manifest_internal_c(PG_FUNCTION_ARGS)
{
    char *model_path;
    char *manifest_text;
    Datum manifest;

    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("evoke model path is required")));
    }

    evoke_checkout_require_superuser();
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(0));
    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    /* Artifact bytes are verified by the runtime worker before inference. */
    evoke_checkout_validate_manifest_shape(manifest, NULL, model_path);
    PG_RETURN_DATUM(manifest);
}

PG_FUNCTION_INFO_V1(evoke_checkout_manifest_signature_internal_c);
Datum
evoke_checkout_manifest_signature_internal_c(PG_FUNCTION_ARGS)
{
    char signature[EVOKE_CHECKOUT_SIGNATURE_HEX_LEN + 1];

    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("evoke checkout manifest is required")));
    }

    evoke_checkout_manifest_signature(PG_GETARG_DATUM(0), signature);
    PG_RETURN_TEXT_P(cstring_to_text(signature));
}

PG_FUNCTION_INFO_V1(evoke_index_checkout_validate_internal_c);
Datum
evoke_index_checkout_validate_internal_c(PG_FUNCTION_ARGS)
{
    char *model_path;
    char *manifest_text;
    Datum manifest;

    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("evoke model path is required")));
    }

    evoke_checkout_require_superuser();
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(0));
    manifest_text = evoke_checkout_read_manifest_text(model_path, NULL);
    manifest = evoke_checkout_manifest_jsonb_from_text(manifest_text);
    /* Status is the authoritative deep audit, not a runtime cache lookup. */
    evoke_checkout_validation_cache_clear();
    evoke_checkout_validate_manifest(manifest, NULL, model_path);
    PG_RETURN_BOOL(true);
}
