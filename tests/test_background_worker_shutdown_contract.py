from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def function_body(source: str, signature: str) -> str:
    signature_offset = source.index(signature)
    body_offset = source.index('{', signature_offset)
    depth = 0

    for offset in range(body_offset, len(source)):
        character = source[offset]
        if character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                return source[body_offset:offset + 1]

    raise AssertionError(f'unterminated C function {signature}')


def test_background_workers_defer_shutdown_outside_critical_sections() -> None:
    am_source = (REPO_ROOT / 'src' / 'evoke_am.c').read_text(
        encoding='utf-8',
    )
    semantic_source = (REPO_ROOT / 'src' / 'evoke_semantic.c').read_text(
        encoding='utf-8',
    )
    workers = (
        function_body(
            am_source,
            'PGDLLEXPORT void\nevoke_maintenance_worker_main(Datum main_arg)',
        ),
        function_body(
            am_source,
            'PGDLLEXPORT void\nevoke_maintenance_supervisor_main(Datum main_arg)',
        ),
        function_body(
            semantic_source,
            'PGDLLEXPORT void\nevoke_runtime_worker_main(Datum main_arg)',
        ),
    )

    assert '#include "tcop/tcopprot.h"' in am_source
    assert '#include "tcop/tcopprot.h"' in semantic_source
    for worker in workers:
        assert 'pqsignal(SIGTERM, die);' in worker
        assert 'bgworker_die' not in worker


def test_runtime_worker_releases_onnx_sessions_only_on_clean_exit() -> None:
    semantic_source = (REPO_ROOT / 'src' / 'evoke_semantic.c').read_text(
        encoding='utf-8',
    )
    cache_exit = function_body(
        semantic_source,
        'static void\nevoke_ort_cache_shmem_exit(int code, Datum arg)',
    )
    session_loader = function_body(
        semantic_source,
        'static OrtSession *\nevoke_ort_cache_session(',
    )

    assert 'evoke_ort_cache_apply_size_limit(ort, 0);' in cache_exit
    assert 'if (ProcDiePending)' in cache_exit
    assert cache_exit.index('if (ProcDiePending)') < cache_exit.index(
        'evoke_ort_api()'
    )
    assert (
        'before_shmem_exit(evoke_ort_cache_shmem_exit, 0);'
        in session_loader
    )

    worker_exit = function_body(
        semantic_source,
        'static void\nevoke_runtime_worker_shmem_exit(int code, Datum arg)',
    )
    assert 'if (ProcDiePending)' in worker_exit


def test_maintenance_launch_is_bounded_before_database_connection() -> None:
    am_source = (REPO_ROOT / 'src' / 'evoke_am.c').read_text(
        encoding='utf-8',
    )
    scheduler_source = (
        REPO_ROOT / 'src' / 'evoke_am_scheduler.c'
    ).read_text(encoding='utf-8')
    worker = function_body(
        am_source,
        'PGDLLEXPORT void\nevoke_maintenance_worker_main(Datum main_arg)',
    )
    adoption = function_body(
        scheduler_source,
        'void\nevoke_am_scheduler_process_adopt_worker_launch(void)',
    )

    assert worker.index(
        'evoke_am_scheduler_process_adopt_worker_launch();'
    ) < worker.index('BackgroundWorkerInitializeConnectionByOid(')
    assert worker.index(
        'catalog_current = evoke_am_maintenance_catalog_is_current();'
    ) < worker.index('evoke_am_try_maintenance_worker_slot(')
    assert 'evoke_am_scheduler_process_note_started();' in adoption
    assert (
        'before_shmem_exit(evoke_am_scheduler_process_worker_exit, 0);'
        in adoption
    )

    catalog_check = function_body(
        am_source,
        'static bool\nevoke_am_maintenance_catalog_is_current(void)',
    )
    assert 'pg_catalog.pg_extension' in catalog_check
    assert 'pg_catalog.pg_available_extensions' in catalog_check
    assert 'installed.extversion = available.default_version' in catalog_check


def test_maintenance_worker_slot_survives_transaction_boundaries() -> None:
    am_source = (REPO_ROOT / 'src' / 'evoke_am.c').read_text(
        encoding='utf-8',
    )
    acquire = function_body(
        am_source,
        'static bool\nevoke_am_try_maintenance_worker_slot(int *slot_out)',
    )
    release = function_body(
        am_source,
        'static void\nevoke_am_release_maintenance_worker_slot(int slot)',
    )

    assert 'LockAcquire(&tag, ExclusiveLock, true, true)' in acquire
    assert 'LockRelease(&tag, ExclusiveLock, true)' in release


def test_maintenance_worker_batches_with_transaction_and_time_bounds() -> None:
    am_source = (REPO_ROOT / 'src' / 'evoke_am.c').read_text(
        encoding='utf-8',
    )
    worker = function_body(
        am_source,
        'PGDLLEXPORT void\nevoke_maintenance_worker_main(Datum main_arg)',
    )

    assert '#define EVOKE_AM_MAINTENANCE_ACTIONS_PER_WORKER 256' in am_source
    assert '#define EVOKE_AM_MAINTENANCE_ACTION_BUDGET_MS 1000' in am_source
    assert 'maintenance_count > 0' in worker
    assert 'EVOKE_AM_MAINTENANCE_ACTIONS_PER_WORKER' in worker
    assert 'EVOKE_AM_MAINTENANCE_ACTION_BUDGET_MS' in worker
    assert 'CHECK_FOR_INTERRUPTS();' in worker
    execute = worker.index('maintenance_count =')
    commit = worker.index('CommitTransactionCommand();', execute)
    ownership_check = worker.index(
        'evoke_am_session_maintenance_lock_held(',
        commit,
    )
    unlock = worker.index(
        'evoke_am_session_maintenance_unlock(',
        ownership_check,
    )
    batch = worker.index('maintenance_actions += maintenance_count;')
    assert execute < commit < ownership_check < unlock < batch
