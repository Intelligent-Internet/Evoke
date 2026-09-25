#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from test_unified_index_lifecycle_smoke import (
    configure_cluster,
    connect,
    extension_control_root,
    restart_cluster,
    run,
    sql_literal,
    start_cluster,
    stop_cluster,
)


INDEX_COUNT = 4
HOT_INDEX = 0
SINGLE_WORKER_LIMIT = 1
PARALLEL_WORKER_LIMIT = 4
ACCELERATOR_INDEX_COUNT = 2
ACCELERATOR_COLD_INDEX = 0
ACCELERATOR_HOT_INDEX = 1
ACCELERATOR_BUILD_LOCK_TAG = 0x32534247
MAINTENANCE_LOCK_TAG = 0x3253424D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Prove fair eventual semantic maintenance under sustained ingress.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--port', type=int, default=55591)
    parser.add_argument('--single-duration-seconds', type=float, default=18.0)
    parser.add_argument(
        '--parallel-duration-seconds',
        type=float,
        default=28.0,
    )
    parser.add_argument(
        '--drain-timeout-seconds',
        type=float,
        default=150.0,
    )
    parser.add_argument('--restart-timeout-seconds', type=float, default=45.0)
    parser.add_argument('--writer-batch-size', type=int, default=8)
    parser.add_argument('--writer-interval-seconds', type=float, default=0.25)
    parser.add_argument('--accelerator-only', action='store_true')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def index_name(schema: str, ordinal: int) -> str:
    return f'{schema}.docs_{ordinal}_idx'


def table_name(schema: str, ordinal: int) -> str:
    return f'{schema}.docs_{ordinal}'


def fetch_status(
    connection: psycopg.Connection[Any],
    name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_status(%s::regclass)',
            (name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError(f'evoke_index_status failed for {name}')
    return dict(row[0])


def fetch_cache_state(
    connection: psycopg.Connection[Any],
    name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_runtime_state_json(%s::regclass)',
            (name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise RuntimeError(
            f'evoke_index_runtime_state_json failed for {name}'
        )
    return dict(row[0])


def maintenance_metrics(cache: dict[str, Any]) -> dict[str, Any]:
    maintenance = cache.get('maintenance')
    if not isinstance(maintenance, dict):
        raise RuntimeError('cache state is missing maintenance metrics')
    return maintenance


def completion_state(status: dict[str, Any]) -> dict[str, Any]:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        raise RuntimeError('evoke status omitted generation')
    delta = generation.get('delta')
    if not isinstance(delta, dict):
        raise RuntimeError('evoke generation status omitted delta')
    completion = delta.get('semantic_completion')
    if not isinstance(completion, dict):
        raise RuntimeError('evoke status omitted semantic completion')
    return dict(completion)


def completion_total(status: dict[str, Any]) -> int:
    telemetry = completion_state(status).get('telemetry')
    if not isinstance(telemetry, dict):
        raise RuntimeError('evoke status omitted completion telemetry')
    return int(telemetry.get('completed') or 0)


def generation_number(status: dict[str, Any]) -> int:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        raise RuntimeError('evoke status omitted generation')
    return int(generation.get('generation') or 0)


def segment_count(status: dict[str, Any]) -> int:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        raise RuntimeError('evoke status omitted generation')
    primary = generation.get('primary')
    if not isinstance(primary, dict):
        raise RuntimeError('evoke status omitted primary generation')
    return int(primary.get('segment_count') or 0)


def delta_records(status: dict[str, Any]) -> int:
    details = status.get('details')
    if not isinstance(details, dict):
        raise RuntimeError('evoke status omitted index details')
    return int(details.get('delta_records') or 0)


def semantic_accelerator_ready(status: dict[str, Any]) -> bool:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        raise RuntimeError('evoke status omitted generation')
    accelerator = generation.get('semantic_accelerator')
    if not isinstance(accelerator, dict):
        raise RuntimeError('evoke status omitted semantic accelerator')
    return bool(
        accelerator.get('present')
        and accelerator.get('eligible')
        and accelerator.get('state') == 'ready'
        and accelerator.get('forward_complete')
    )


def active_accelerator_horizons(
    connection: psycopg.Connection[Any],
) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                pid,
                backend_xid::text,
                backend_xmin::text,
                query
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND application_name = 'evoke maintenance'
              AND state = 'active'
              AND query LIKE
                  'evoke maintenance:%semantic query accelerator%'
            ORDER BY pid
            """
        )
        rows = cursor.fetchall()
    return [
        {
            'pid': int(pid),
            'backend_xid': backend_xid,
            'backend_xmin': backend_xmin,
            'phase': phase,
        }
        for pid, backend_xid, backend_xmin, phase in rows
    ]


def active_accelerator_locks(
    connection: psycopg.Connection[Any],
    index_oid: int,
) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                build.pid,
                EXISTS (
                    SELECT 1
                    FROM pg_locks AS maintenance
                    WHERE maintenance.locktype = 'advisory'
                      AND maintenance.database = build.database
                      AND maintenance.classid = %s::oid
                      AND maintenance.objid = build.objid
                      AND maintenance.pid = build.pid
                      AND maintenance.granted
                ) AS holds_maintenance_lock
            FROM pg_locks AS build
            WHERE build.locktype = 'advisory'
              AND build.database = (
                  SELECT oid
                  FROM pg_database
                  WHERE datname = current_database()
              )
              AND build.classid = %s::oid
              AND build.objid = %s::oid
              AND build.granted
            ORDER BY build.pid
            """,
            (
                MAINTENANCE_LOCK_TAG,
                ACCELERATOR_BUILD_LOCK_TAG,
                index_oid,
            ),
        )
        rows = cursor.fetchall()
    return [
        {
            'pid': int(pid),
            'holds_maintenance_lock': bool(holds_maintenance_lock),
        }
        for pid, holds_maintenance_lock in rows
    ]


def acquire_guard(
    socket_dir: Path,
    port: int,
    name: str,
) -> psycopg.Connection[Any]:
    connection = connect(socket_dir, port)
    deadline = time.monotonic() + 10.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_try_maintenance_lock(%s::regclass)',
                (name,),
            )
            acquired = bool(cursor.fetchone()[0])
        if acquired:
            return connection
        if time.monotonic() >= deadline:
            connection.close()
            raise TimeoutError(f'could not acquire maintenance guard: {name}')
        time.sleep(0.01)


def release_guard(
    connection: psycopg.Connection[Any],
    name: str,
) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_maintenance_unlock(%s::regclass)',
                (name,),
            )
    finally:
        connection.close()


def configure_worker_limit(
    connection: psycopg.Connection[Any],
    worker_limit: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'ALTER SYSTEM SET evoke.maintenance_worker_limit = '
            f"'{int(worker_limit)}'",
        )
        cursor.execute('SELECT pg_reload_conf()')
    deadline = time.monotonic() + 10.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute('SHOW evoke.maintenance_worker_limit')
            observed = int(cursor.fetchone()[0])
        if observed == worker_limit:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'maintenance worker limit did not reload: '
                f'expected={worker_limit}, observed={observed}'
            )
        time.sleep(0.05)


def configure_maintenance_budget(
    connection: psycopg.Connection[Any],
    value: str | None,
) -> str:
    with connection.cursor() as cursor:
        if value is None:
            cursor.execute(
                'ALTER SYSTEM RESET '
                'evoke.maintenance_rebuild_memory_budget'
            )
        else:
            cursor.execute(
                'ALTER SYSTEM SET '
                'evoke.maintenance_rebuild_memory_budget = '
                f'{sql_literal(value)}'
            )
        cursor.execute('SELECT pg_reload_conf()')
        if cursor.fetchone() != (True,):
            raise RuntimeError('PostgreSQL did not reload maintenance budget')

    deadline = time.monotonic() + 10.0
    observed = ''
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute('SHOW evoke.maintenance_rebuild_memory_budget')
            observed = str(cursor.fetchone()[0])
        if (value is None and observed != '1MB') or observed == value:
            return observed
        time.sleep(0.05)
    raise TimeoutError(
        'maintenance budget did not reload: '
        f'expected={value}, observed={observed}'
    )


def setup_scenario(
    connection: psycopg.Connection[Any],
    schema: str,
    model_path: Path,
    index_count: int = INDEX_COUNT,
    auto_preload_priorities: list[int] | None = None,
) -> list[str]:
    if (
        auto_preload_priorities is not None
        and len(auto_preload_priorities) != index_count
    ):
        raise ValueError('auto-preload priorities do not match index count')
    names: list[str] = []
    with connection.cursor() as cursor:
        cursor.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        cursor.execute(f'CREATE SCHEMA {schema}')
        for ordinal in range(index_count):
            table = table_name(schema, ordinal)
            name = index_name(schema, ordinal)
            auto_preload = (
                0
                if auto_preload_priorities is None
                else auto_preload_priorities[ordinal]
            )
            names.append(name)
            cursor.execute(
                f"""
                CREATE TABLE {table} (
                    id bigint PRIMARY KEY,
                    body text NOT NULL,
                    scope text NOT NULL DEFAULT repeat('scope-', 1024)
                );
                ALTER TABLE {table}
                ALTER COLUMN scope SET STORAGE EXTERNAL;
                INSERT INTO {table} (id, body) VALUES (
                    -1,
                    'semantic fairness baseline {ordinal}'
                );
                CREATE INDEX docs_{ordinal}_idx
                ON {table}
                USING evoke (body)
                INCLUDE (scope)
                WITH (
                    sae = true,
                    model_path = {sql_literal(str(model_path))},
                    consistency = eventual,
                    auto_preload = {auto_preload}
                );
                """
            )
    return names


def insert_seed(
    connection: psycopg.Connection[Any],
    schema: str,
    ordinal: int,
    row_count: int,
) -> None:
    table = table_name(schema, ordinal)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {table} (id, body)
            SELECT
                value,
                'semantic fairness index {ordinal} marker ' || value::text
            FROM generate_series(1, %s::bigint) AS value
            """,
            (row_count,),
        )


def query_rows(
    connection: psycopg.Connection[Any],
    schema: str,
    ordinal: int,
    *,
    oracle: bool,
) -> list[tuple[int, float]]:
    table = table_name(schema, ordinal)
    name = index_name(schema, ordinal)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            f"""
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                %s::regclass,
                %s,
                20
            ) AS hit
            JOIN {table} AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (name, f'semantic fairness index {ordinal} marker'),
        )
        return [
            (int(row[0]), round(float(row[1]), 5))
            for row in cursor.fetchall()
        ]


def assert_oracle_parity(
    connection: psycopg.Connection[Any],
    schema: str,
) -> dict[str, bool]:
    parity: dict[str, bool] = {}
    for ordinal in range(INDEX_COUNT):
        normal = query_rows(
            connection,
            schema,
            ordinal,
            oracle=False,
        )
        oracle = query_rows(
            connection,
            schema,
            ordinal,
            oracle=True,
        )
        parity[index_name(schema, ordinal)] = normal == oracle
    return parity


def writer_loop(
    socket_dir: Path,
    port: int,
    schema: str,
    hot_index: int,
    args: argparse.Namespace,
    stop_event: threading.Event,
    result: dict[str, Any],
) -> None:
    connection: psycopg.Connection[Any] | None = None
    next_id = 1_000_000
    inserted = 0
    errors: list[str] = []
    try:
        connection = connect(socket_dir, port)
        table = table_name(schema, hot_index)
        while not stop_event.is_set():
            last_id = next_id + args.writer_batch_size - 1
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {table} (id, body)
                    SELECT
                        value,
                        'semantic fairness hot stream ' || value::text
                    FROM generate_series(
                        %s::bigint,
                        %s::bigint
                    ) AS value
                    """,
                    (next_id, last_id),
                )
            inserted += args.writer_batch_size
            next_id = last_id + 1
            stop_event.wait(args.writer_interval_seconds)
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
        stop_event.set()
    finally:
        if connection is not None:
            connection.close()
        result['inserted'] = inserted
        result['errors'] = errors


def wait_for_drain(
    connection: psycopg.Connection[Any],
    names: list[str],
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, Any]], float]:
    started = time.monotonic()
    final: dict[str, dict[str, Any]] = {}
    while True:
        final = {
            name: fetch_status(connection, name)
            for name in names
        }
        converged = all(
            bool(completion_state(status).get('converged'))
            and delta_records(status) == 0
            for status in final.values()
        )
        if converged:
            return final, time.monotonic() - started
        if time.monotonic() - started >= timeout_seconds:
            cache = fetch_cache_state(connection, names[0])
            raise TimeoutError(
                'eventual maintenance did not drain: '
                + json.dumps(
                    {
                        'cache': cache,
                        'indexes': final,
                    },
                    sort_keys=True,
                )
            )
        time.sleep(0.1)


def wait_for_accelerator_drain(
    connection: psycopg.Connection[Any],
    names: list[str],
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, Any]], float]:
    started = time.monotonic()
    final: dict[str, dict[str, Any]] = {}
    while True:
        final = {
            name: fetch_status(connection, name)
            for name in names
        }
        if all(
            semantic_accelerator_ready(status)
            for status in final.values()
        ):
            return final, time.monotonic() - started
        if time.monotonic() - started >= timeout_seconds:
            cache = fetch_cache_state(connection, names[0])
            raise TimeoutError(
                'eventual accelerator maintenance did not drain: '
                + json.dumps(
                    {
                        'cache': cache,
                        'indexes': final,
                    },
                    sort_keys=True,
                )
            )
        time.sleep(0.1)


def run_scenario(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    model_path: Path,
    worker_limit: int,
    duration_seconds: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    schema = f'eventual_fairness_w{worker_limit}'
    configure_worker_limit(connection, worker_limit)
    names = setup_scenario(connection, schema, model_path)
    guards = [
        acquire_guard(socket_dir, port, name)
        for name in names
    ]
    # Parallel seeds must outlive the one-second supervisor launch cadence.
    # Short jobs make the concurrency assertion depend on scheduling jitter.
    seed_rows = (
        [2048, 128, 128, 128]
        if worker_limit == SINGLE_WORKER_LIMIT
        else [4096, 2048, 2048, 2048]
    )
    for ordinal, row_count in enumerate(seed_rows):
        insert_seed(connection, schema, ordinal, row_count)

    baseline_status = {
        name: fetch_status(connection, name)
        for name in names
    }
    baseline_cache = fetch_cache_state(connection, names[0])
    pending_parity = assert_oracle_parity(connection, schema)
    for guard, name in zip(guards, names, strict=True):
        release_guard(guard, name)

    stop_event = threading.Event()
    writer_result: dict[str, Any] = {}
    writer = threading.Thread(
        target=writer_loop,
        args=(
            socket_dir,
            port,
            schema,
            HOT_INDEX,
            args,
            stop_event,
            writer_result,
        ),
        daemon=True,
    )
    writer.start()

    started = time.monotonic()
    first_progress_seconds: dict[str, float] = {}
    max_active_workers = 0
    max_hot_delta_records = 0
    structural_progress_while_writing = False
    samples = 0
    try:
        while time.monotonic() - started < duration_seconds:
            statuses = {
                name: fetch_status(connection, name)
                for name in names
            }
            cache = fetch_cache_state(connection, names[0])
            maintenance = maintenance_metrics(cache)
            elapsed = time.monotonic() - started
            samples += 1
            max_active_workers = max(
                max_active_workers,
                int(
                    maintenance.get(
                        'active_index_maintenance_workers',
                    )
                    or 0
                ),
            )
            max_hot_delta_records = max(
                max_hot_delta_records,
                delta_records(statuses[names[HOT_INDEX]]),
            )
            for name, status in statuses.items():
                if (
                    name not in first_progress_seconds
                    and completion_total(status)
                    > completion_total(baseline_status[name])
                ):
                    first_progress_seconds[name] = round(elapsed, 3)
            hot_status = statuses[names[HOT_INDEX]]
            hot_baseline = baseline_status[names[HOT_INDEX]]
            if writer.is_alive() and (
                generation_number(hot_status) >
                    generation_number(hot_baseline) or
                segment_count(hot_status) != segment_count(hot_baseline)
            ):
                structural_progress_while_writing = True
            if writer_result.get('errors'):
                break
            time.sleep(0.05)
    finally:
        stop_event.set()
        writer.join(timeout=10.0)
    if writer.is_alive():
        raise RuntimeError('sustained ingress writer did not stop')

    final_status, drain_seconds = wait_for_drain(
        connection,
        names,
        args.drain_timeout_seconds,
    )
    final_status, accelerator_drain_seconds = wait_for_accelerator_drain(
        connection,
        names,
        args.drain_timeout_seconds,
    )
    final_cache = fetch_cache_state(connection, names[0])
    baseline_maintenance = maintenance_metrics(baseline_cache)
    final_maintenance = maintenance_metrics(final_cache)
    final_parity = assert_oracle_parity(connection, schema)
    overflow_delta = (
        int(final_maintenance.get('work_hint_overflows') or 0)
        - int(baseline_maintenance.get('work_hint_overflows') or 0)
    )
    service_delta = (
        int(final_maintenance.get('maintenance_service_count') or 0)
        - int(baseline_maintenance.get('maintenance_service_count') or 0)
    )

    failures: list[str] = []
    missing_progress = [
        name for name in names
        if name not in first_progress_seconds
    ]
    if missing_progress:
        failures.append(
            'indexes made no semantic progress: '
            + ', '.join(missing_progress)
        )
    # Scalar x86_64 hosts can take roughly 15 seconds to first serve the hot
    # index after the three cold indexes. Keep this bound independent from
    # the longer full-drain allowance so fairness regressions still fail.
    progress_limit = 18.0
    late_progress = {
        name: seconds
        for name, seconds in first_progress_seconds.items()
        if seconds > progress_limit
    }
    if late_progress:
        failures.append(
            'index service exceeded fairness bound: '
            + json.dumps(late_progress, sort_keys=True)
        )
    if worker_limit == 1 and max_active_workers > 1:
        failures.append(
            'single-worker scenario exceeded one active maintenance worker'
        )
    if worker_limit > 1 and max_active_workers < 2:
        failures.append(
            'parallel scenario never observed concurrent maintenance workers'
        )
    if worker_limit > 1 and not structural_progress_while_writing:
        failures.append(
            'parallel scenario made no structural progress during ingress'
        )
    if writer_result.get('errors'):
        failures.append(
            'foreground writer failed: '
            + json.dumps(writer_result['errors'])
        )
    if overflow_delta != 0:
        failures.append(f'work hint overflow delta is {overflow_delta}')
    if service_delta < INDEX_COUNT:
        failures.append(
            'maintenance service counter did not cover every index: '
            f'{service_delta}'
        )
    if not all(pending_parity.values()):
        failures.append('pending normal/oracle parity failed')
    if not all(final_parity.values()):
        failures.append('final normal/oracle parity failed')

    return (
        {
            'worker_limit': worker_limit,
            'duration_seconds': duration_seconds,
            'seed_rows': seed_rows,
            'writer': writer_result,
            'samples': samples,
            'first_progress_seconds': first_progress_seconds,
            'progress_limit_seconds': progress_limit,
            'max_active_index_maintenance_workers': max_active_workers,
            'max_hot_delta_records': max_hot_delta_records,
            'structural_progress_while_writing':
                structural_progress_while_writing,
            'drain_seconds': round(drain_seconds, 3),
            'accelerator_drain_seconds': round(
                accelerator_drain_seconds,
                3,
            ),
            'work_hint_overflow_delta': overflow_delta,
            'maintenance_service_delta': service_delta,
            'pending_oracle_parity': pending_parity,
            'final_oracle_parity': final_parity,
            'baseline_status': baseline_status,
            'final_status': final_status,
            'failures': failures,
            'passed': not failures,
        },
        names,
    )


def run_accelerator_fairness_scenario(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    model_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    schema = 'eventual_accelerator_fairness'
    writer: threading.Thread | None = None
    stop_event = threading.Event()
    writer_result: dict[str, Any] = {}
    failures: list[str] = []
    observations: dict[str, Any] = {}
    maintenance_horizon_samples: list[dict[str, Any]] = []
    accelerator_lock_samples: list[dict[str, Any]] = []
    accelerator_maintenance_probe: str | None = None
    same_root_ingress_maintenance: str | None = None
    same_root_progress_while_building = False

    configure_worker_limit(connection, SINGLE_WORKER_LIMIT)
    configure_maintenance_budget(connection, '1MB')
    try:
        names = setup_scenario(
            connection,
            schema,
            model_path,
            ACCELERATOR_INDEX_COUNT,
            [90, 100],
        )
        guards = [
            acquire_guard(socket_dir, port, name)
            for name in names
        ]
        try:
            for ordinal in range(ACCELERATOR_INDEX_COUNT):
                insert_seed(connection, schema, ordinal, 128)
        finally:
            for guard, name in zip(guards, names, strict=True):
                release_guard(guard, name)

        blocked_deadline = time.monotonic() + 60.0
        blocked_statuses: dict[str, dict[str, Any]] = {}
        while time.monotonic() < blocked_deadline:
            blocked_statuses = {
                name: fetch_status(connection, name)
                for name in names
            }
            if all(
                completion_state(status).get('converged')
                and delta_records(status) == 0
                and not semantic_accelerator_ready(status)
                for status in blocked_statuses.values()
            ):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(
                'accelerator fixture did not reach the budget gate: '
                + json.dumps(blocked_statuses, sort_keys=True)
            )

        hot_name = names[ACCELERATOR_HOT_INDEX]
        cold_name = names[ACCELERATOR_COLD_INDEX]
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT %s::regclass::oid, %s::regclass::oid',
                (hot_name, cold_name),
            )
            hot_oid, cold_oid = (
                int(value)
                for value in cursor.fetchone()
            )
        hot_completion_before = completion_total(
            blocked_statuses[hot_name]
        )

        writer = threading.Thread(
            target=writer_loop,
            args=(
                socket_dir,
                port,
                schema,
                ACCELERATOR_HOT_INDEX,
                args,
                stop_event,
                writer_result,
            ),
            daemon=True,
        )
        writer.start()

        cursor_deadline = time.monotonic() + 30.0
        cursor_state: dict[str, Any] = {}
        cursor_observed = False
        while time.monotonic() < cursor_deadline:
            hot_status = fetch_status(connection, hot_name)
            cursor_state = maintenance_metrics(
                fetch_cache_state(connection, hot_name)
            )
            cursor_observed = (
                completion_total(hot_status) > hot_completion_before
                and int(
                    cursor_state.get(
                        'maintenance_last_served_index_oid'
                    ) or 0
                ) == hot_oid
            )
            if cursor_observed or writer_result.get('errors'):
                break
            time.sleep(0.05)
        if not cursor_observed:
            failures.append(
                'continuous-ingress root did not own the fairness cursor'
            )

        restored_budget = configure_maintenance_budget(connection, None)
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_index_touch_maintenance()')
            maintenance_touch = str(cursor.fetchone()[0])
        accelerator_started = time.monotonic()
        cold_status = blocked_statuses[cold_name]
        while (
            time.monotonic() - accelerator_started
            < args.drain_timeout_seconds
        ):
            maintenance_horizon_samples.extend(
                active_accelerator_horizons(connection)
            )
            lock_samples = active_accelerator_locks(
                connection,
                cold_oid,
            )
            accelerator_lock_samples.extend(lock_samples)
            if lock_samples and accelerator_maintenance_probe is None:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'SELECT evoke_index_try_maintain(%s::regclass)',
                        (cold_name,),
                    )
                    accelerator_maintenance_probe = str(
                        cursor.fetchone()[0]
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO {
                            table_name(schema, ACCELERATOR_COLD_INDEX)
                        } (id, body)
                        VALUES (
                            2000000,
                            'same root ingress during accelerator build'
                        )
                        """
                    )
                    cursor.execute(
                        'SELECT evoke_index_try_maintain(%s::regclass)',
                        (cold_name,),
                    )
                    same_root_ingress_maintenance = str(
                        cursor.fetchone()[0]
                    )
                same_root_progress_while_building = bool(
                    active_accelerator_locks(connection, cold_oid)
                )
            cold_status = fetch_status(connection, cold_name)
            if semantic_accelerator_ready(cold_status):
                break
            if writer_result.get('errors') or not writer.is_alive():
                break
            time.sleep(0.05)
        accelerator_seconds = time.monotonic() - accelerator_started
        cold_ready = semantic_accelerator_ready(cold_status)
        if not cold_ready:
            failures.append(
                'lower-OID accelerator starved behind continuous ingress'
            )
        if not maintenance_horizon_samples:
            failures.append(
                'accelerator build never exposed a prepare/publish sample'
            )
        if not accelerator_lock_samples:
            failures.append(
                'accelerator build lock was never observed'
            )
        locked_builders = [
            sample
            for sample in accelerator_lock_samples
            if sample['holds_maintenance_lock']
        ]
        if locked_builders:
            failures.append(
                'accelerator builder retained the root maintenance lock: '
                + json.dumps(locked_builders, sort_keys=True)
            )
        if (
            accelerator_maintenance_probe is None
            or 'reason=accelerator_build_busy' not in
                accelerator_maintenance_probe
        ):
            failures.append(
                'same-root maintenance remained blocked during accelerator '
                f'build: {accelerator_maintenance_probe}'
            )
        if (
            same_root_ingress_maintenance is None
            or 'maintained=true' not in same_root_ingress_maintenance
            or not same_root_progress_while_building
        ):
            failures.append(
                'same-root ingress did not converge while accelerator build '
                f'remained active: {same_root_ingress_maintenance}'
            )
        unexpected_snapshot_horizons = [
            sample for sample in maintenance_horizon_samples
            if sample['backend_xmin'] is not None
            and 'accelerator scope' not in sample['phase']
        ]
        retained_prepare_xids = [
            sample for sample in maintenance_horizon_samples
            if 'prepare semantic query accelerator' in sample['phase']
            and sample['backend_xid'] is not None
        ]
        if unexpected_snapshot_horizons:
            failures.append(
                'accelerator maintenance retained an unexpected snapshot '
                'horizon: '
                + json.dumps(
                    unexpected_snapshot_horizons,
                    sort_keys=True,
                )
            )
        if retained_prepare_xids:
            failures.append(
                'accelerator prepare retained a transaction ID: '
                + json.dumps(retained_prepare_xids, sort_keys=True)
            )
        if writer_result.get('errors'):
            failures.append(
                'accelerator fairness writer failed: '
                + json.dumps(writer_result['errors'])
            )
        if writer is not None and not writer.is_alive():
            failures.append(
                'continuous-ingress writer stopped before accelerator ready'
            )
        observations = {
            'index_names': names,
            'hot_index_oid': hot_oid,
            'cursor_before_budget_release': cursor_state.get(
                'maintenance_last_served_index_oid'
            ),
            'restored_budget': restored_budget,
            'maintenance_touch': maintenance_touch,
            'cold_accelerator_ready': cold_ready,
            'cold_accelerator_seconds': round(accelerator_seconds, 3),
            'maintenance_horizon_sample_count': len(
                maintenance_horizon_samples
            ),
            'maintenance_horizon_samples': maintenance_horizon_samples,
            'accelerator_lock_sample_count': len(
                accelerator_lock_samples
            ),
            'accelerator_lock_samples': accelerator_lock_samples,
            'accelerator_maintenance_probe':
                accelerator_maintenance_probe,
            'same_root_ingress_maintenance':
                same_root_ingress_maintenance,
            'same_root_progress_while_building':
                same_root_progress_while_building,
        }
    finally:
        stop_event.set()
        if writer is not None:
            writer.join(timeout=10.0)
            if writer.is_alive():
                failures.append('accelerator fairness writer did not stop')
        configure_maintenance_budget(connection, None)
        configure_worker_limit(connection, PARALLEL_WORKER_LIMIT)

    return {
        **observations,
        'writer': writer_result,
        'failures': failures,
        'passed': not failures,
    }


def run_restart_reconciliation(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    pg_ctl: Path,
    data_dir: Path,
    log_path: Path,
    schema: str,
    names: list[str],
    timeout_seconds: float,
) -> tuple[dict[str, Any], psycopg.Connection[Any]]:
    realtime_bm25_table = f'{schema}.realtime_bm25_docs'
    realtime_bm25_index = f'{schema}.realtime_bm25_docs_idx'
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE {realtime_bm25_table} (
                id bigint PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO {realtime_bm25_table}
            VALUES (-1, 'realtime lexical restart baseline');
            CREATE INDEX realtime_bm25_docs_idx
            ON {realtime_bm25_table}
            USING evoke (body)
            WITH (
                consistency = realtime,
                auto_preload = 0
            )
            """
        )
    realtime_names = [
        realtime_bm25_index,
    ]
    guarded_names = names + realtime_names
    guards = [
        acquire_guard(socket_dir, port, name)
        for name in guarded_names
    ]
    for ordinal in range(INDEX_COUNT):
        table = table_name(schema, ordinal)
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {table} (id, body)
                VALUES (
                    %s,
                    %s
                )
                """,
                (
                    9_000_000 + ordinal,
                    f'semantic fairness restart marker {ordinal}',
                ),
            )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {realtime_bm25_table} (id, body)
            VALUES (
                9000200,
                'realtime lexical restart marker'
            )
            """
        )
    pending_before = {
        name: completion_state(fetch_status(connection, name))
        for name in names
    }
    realtime_pending_before = {
        name: delta_records(fetch_status(connection, name))
        for name in realtime_names
    }

    restart_cluster(pg_ctl, data_dir, log_path)
    for guard in guards:
        try:
            guard.close()
        except Exception:
            pass
    try:
        connection.close()
    except Exception:
        pass
    connection = connect(socket_dir, port)

    started = time.monotonic()
    first_converged_seconds: dict[str, float] = {}
    while True:
        statuses = {
            name: fetch_status(connection, name)
            for name in names
        }
        realtime_statuses = {
            name: fetch_status(connection, name)
            for name in realtime_names
        }
        elapsed = time.monotonic() - started
        for name, status in statuses.items():
            if (
                name not in first_converged_seconds
                and completion_state(status).get('converged')
                and delta_records(status) == 0
            ):
                first_converged_seconds[name] = round(elapsed, 3)
        for name, status in realtime_statuses.items():
            if (
                name not in first_converged_seconds
                and delta_records(status) == 0
            ):
                first_converged_seconds[name] = round(elapsed, 3)
        if len(first_converged_seconds) == len(guarded_names):
            break
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                'restart reconciliation did not converge: '
                + json.dumps(
                    {
                        'eventual': statuses,
                        'realtime': realtime_statuses,
                    },
                    sort_keys=True,
                )
            )
        time.sleep(0.1)

    parity = assert_oracle_parity(connection, schema)
    failures: list[str] = []
    if not all(
        int(state.get('pending') or 0) > 0
        for state in pending_before.values()
    ):
        failures.append('restart fixture did not persist pending work')
    if not all(
        records > 0
        for records in realtime_pending_before.values()
    ):
        failures.append(
            'realtime BM25 restart fixture did not persist delta work'
        )
    if not all(parity.values()):
        failures.append('restart normal/oracle parity failed')
    return (
        {
            'pending_before_restart': pending_before,
            'realtime_delta_before_restart': realtime_pending_before,
            'first_converged_seconds': first_converged_seconds,
            'oracle_parity': parity,
            'failures': failures,
            'passed': not failures,
        },
        connection,
    )


def main() -> None:
    args = parse_args()
    args.pg_bin = args.pg_bin.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.extension_libdir is not None:
        args.extension_libdir = (
            args.extension_libdir.expanduser().resolve()
        )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir.expanduser().resolve()
        )
    if not (args.model_path / 'manifest.json').is_file():
        raise FileNotFoundError(
            f'model checkout is missing: {args.model_path}'
        )
    if args.writer_batch_size <= 0:
        raise ValueError('writer-batch-size must be positive')
    if args.writer_interval_seconds <= 0:
        raise ValueError('writer-interval-seconds must be positive')

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    temporary = tempfile.TemporaryDirectory(delete=not args.keep)
    root = Path(temporary.name)
    data_dir = root / 'data'
    socket_dir = root / 'socket'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
    if args.keep:
        print(f'using temporary cluster at {root}', flush=True)
    run(
        [
            str(initdb),
            '-D',
            str(data_dir),
            '-A',
            'trust',
            '-U',
            'postgres',
        ]
    )
    configure_cluster(
        data_dir,
        socket_dir,
        args.port,
        args.extension_libdir,
        args.extension_control_dir,
    )
    with (data_dir / 'postgresql.conf').open(
        'a',
        encoding='utf-8',
    ) as handle:
        handle.write('evoke.runtime_worker_count = 4\n')
        handle.write('evoke.maintenance_worker_limit = 1\n')
    start_cluster(pg_ctl, data_dir, log_path)

    connection = connect(socket_dir, args.port)
    scenarios: list[dict[str, Any]] = []
    restart_result: dict[str, Any] = {}
    accelerator_result: dict[str, Any] = {}
    try:
        with connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION evoke')
        if args.accelerator_only:
            accelerator_result = run_accelerator_fairness_scenario(
                connection,
                socket_dir,
                args.port,
                args.model_path,
                args,
            )
            report = {
                'accelerator_fairness': accelerator_result,
                'failures': accelerator_result['failures'],
                'passed': accelerator_result['passed'],
            }
            rendered = json.dumps(report, indent=2, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    rendered + '\n',
                    encoding='utf-8',
                )
            print(rendered)
            if accelerator_result['failures']:
                raise SystemExit(1)
            return
        single, _ = run_scenario(
            connection,
            socket_dir,
            args.port,
            args.model_path,
            SINGLE_WORKER_LIMIT,
            args.single_duration_seconds,
            args,
        )
        scenarios.append(single)
        parallel, names = run_scenario(
            connection,
            socket_dir,
            args.port,
            args.model_path,
            PARALLEL_WORKER_LIMIT,
            args.parallel_duration_seconds,
            args,
        )
        scenarios.append(parallel)
        restart_result, connection = run_restart_reconciliation(
            connection,
            socket_dir,
            args.port,
            pg_ctl,
            data_dir,
            log_path,
            'eventual_fairness_w4',
            names,
            args.restart_timeout_seconds,
        )
        accelerator_result = run_accelerator_fairness_scenario(
            connection,
            socket_dir,
            args.port,
            args.model_path,
            args,
        )
        failures = [
            f'workers={scenario["worker_limit"]}: {failure}'
            for scenario in scenarios
            for failure in scenario['failures']
        ]
        failures.extend(
            f'restart: {failure}'
            for failure in restart_result['failures']
        )
        failures.extend(
            f'accelerator: {failure}'
            for failure in accelerator_result['failures']
        )
        report = {
            'scenarios': scenarios,
            'restart_reconciliation': restart_result,
            'accelerator_fairness': accelerator_result,
            'failures': failures,
            'passed': not failures,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + '\n', encoding='utf-8')
        print(rendered)
        if failures:
            raise SystemExit(1)
    finally:
        if not connection.closed:
            connection.close()
        stop_cluster(pg_ctl, data_dir)
        if args.keep:
            print(f'kept temporary cluster at {root}')
        else:
            temporary.cleanup()


if __name__ == '__main__':
    main()
