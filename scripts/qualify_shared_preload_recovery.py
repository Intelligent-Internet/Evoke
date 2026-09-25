#!/usr/bin/env python3
"""Qualify shared-residency eviction and worker-driven re-admission."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import psycopg
from psycopg import sql


RSS_PATTERN = re.compile(r'^VmRSS:\s+(\d+)\s+kB$', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Clear one isolated Evoke shared runtime, wait for worker-driven '
            're-admission, and qualify exact concurrent first/warm queries.'
        ),
    )
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--expected-port', type=int, required=True)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--warm-repetitions', type=int, default=4)
    parser.add_argument('--readmission-timeout-seconds', type=float, default=120)
    parser.add_argument('--first-max-ms', type=float, default=2_000)
    parser.add_argument('--warm-p50-max-ms', type=float, default=500)
    parser.add_argument('--warm-p95-max-ms', type=float, default=1_000)
    parser.add_argument('--rss-slack-mib', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--allow-cache-clear',
        action='store_true',
        help='required acknowledgement that this clears the isolated runtime',
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.allow_cache_clear:
        raise ValueError('--allow-cache-clear is required')
    if not 1 <= args.expected_port <= 65535:
        raise ValueError('--expected-port is invalid')
    if args.k <= 0 or args.clients <= 0:
        raise ValueError('--k and --clients must be positive')
    if args.warm_repetitions < 2:
        raise ValueError('--warm-repetitions must be at least 2')
    if args.readmission_timeout_seconds <= 0:
        raise ValueError('--readmission-timeout-seconds must be positive')
    if min(
        args.first_max_ms,
        args.warm_p50_max_ms,
        args.warm_p95_max_ms,
    ) <= 0:
        raise ValueError('latency limits must be positive')
    if args.rss_slack_mib < 0:
        raise ValueError('--rss-slack-mib must be non-negative')


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError('percentile requires samples')
    offset = max(math.ceil(len(ordered) * fraction) - 1, 0)
    return ordered[offset]


def extension_schema(
    connection: psycopg.Connection[Any],
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT namespace.nspname
            FROM pg_extension AS extension
            JOIN pg_namespace AS namespace
              ON namespace.oid = extension.extnamespace
            WHERE extension.extname = 'evoke'
            """
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('evoke is not installed')
    return str(row[0])


def call_json(
    connection: psycopg.Connection[Any],
    schema: str,
    function: str,
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT {}(%s::regclass)::jsonb').format(
                sql.Identifier(schema, function)
            ),
            (index_name,),
        )
        value = cursor.fetchone()[0]
    if not isinstance(value, dict):
        raise RuntimeError(f'{function} returned invalid JSON')
    return value


def generation_status(
    connection: psycopg.Connection[Any],
    schema: str,
    index_name: str,
) -> dict[str, Any]:
    return call_json(
        connection,
        schema,
        'evoke_index_generation_status_internal',
        index_name,
    )


def runtime_state(
    connection: psycopg.Connection[Any],
    schema: str,
    index_name: str,
) -> dict[str, Any]:
    return call_json(
        connection,
        schema,
        'evoke_index_runtime_state_json',
        index_name,
    )


def generation_identity(status: dict[str, Any]) -> dict[str, Any]:
    accelerator = status.get('semantic_accelerator') or {}
    primary = status.get('primary') or {}
    posting = status.get('posting') or {}
    return {
        'generation_id': status.get('generation_id'),
        'contract_signature': status.get('contract_signature'),
        'posting_signature': posting.get('signature'),
        'manifest_start_block': primary.get('manifest_start_block'),
        'accelerator_source_manifest_id': accelerator.get(
            'source_manifest_id'
        ),
        'accelerator_builder_policy_id': accelerator.get(
            'builder_policy_id'
        ),
    }


def runtime_ready(state: dict[str, Any]) -> bool:
    generation = state.get('generation') or {}
    preload = state.get('shared_preload') or {}
    resident_fold_ready = bool(
        preload.get('resident_fold_current') is True
        and preload.get('resident_fold_loading') is not True
    )
    page_metadata_ready = bool(
        preload.get('query_warm_marker_valid') is True
        and preload.get('query_metadata_warm') is True
    )
    return bool(
        int(generation.get('auto_preload_priority') or 0) > 0
        and preload.get('available') is True
        and preload.get('loading') is not True
        and (resident_fold_ready or page_metadata_ready)
    )


def requires_query_trace(state: dict[str, Any]) -> bool:
    return state.get('sae_enabled') is True


def target_eviction_observed(state: dict[str, Any]) -> bool:
    return not runtime_ready(state)


def query_route(sample: dict[str, Any]) -> str | None:
    trace = sample.get('trace')
    if not isinstance(trace, dict):
        return None
    route = trace.get('query_route')
    if not isinstance(route, str) or not route:
        return None
    return route


def query_route_stable(
    baseline: dict[str, Any],
    clients: list[dict[str, Any]],
    required: bool,
) -> bool:
    if not required:
        return True
    expected = query_route(baseline)
    return bool(
        expected is not None
        and all(
            query_route(sample) == expected
            for client in clients
            for sample in client['samples']
        )
    )


def backend_rss_bytes(
    connection: psycopg.Connection[Any],
    pid: int,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_read_file('/proc/' || %s || '/status')",
            (pid,),
        )
        status = str(cursor.fetchone()[0])
    match = RSS_PATTERN.search(status)
    if match is None:
        raise RuntimeError(f'could not read VmRSS for backend {pid}')
    return int(match.group(1)) * 1024


def backend_memory_bytes(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT COALESCE(sum(total_bytes), 0)::int8 '
            'FROM pg_backend_memory_contexts'
        )
        return int(cursor.fetchone()[0])


def query_once(
    connection: psycopg.Connection[Any],
    schema: str,
    index_name: str,
    query_text: str,
    k: int,
    require_trace: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                SELECT
                    hit.doc_id,
                    encode(pg_catalog.float4send(hit.score), 'hex')
                FROM {}(%s::regclass, %s, %s)
                    WITH ORDINALITY
                    AS hit(ctid, doc_id, score, rank_position)
                ORDER BY hit.rank_position
                """
            ).format(sql.Identifier(schema, 'evoke_query')),
            (index_name, query_text, k),
        )
        hits = [(int(row[0]), str(row[1])) for row in cursor.fetchall()]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT {}()::jsonb').format(
                sql.Identifier(schema, 'evoke_query_trace_internal')
            )
        )
        trace = cursor.fetchone()[0]
    if require_trace and not isinstance(trace, dict):
        raise RuntimeError('evoke query trace is unavailable')
    if not isinstance(trace, dict):
        trace = None
    pid = connection.info.backend_pid
    return {
        'elapsed_ms': elapsed_ms,
        'hits': hits,
        'trace': trace,
        'backend_pid': pid,
        'backend_memory_bytes': backend_memory_bytes(connection),
        'backend_rss_bytes': backend_rss_bytes(connection, pid),
    }


def worker(
    dsn: str,
    barrier: threading.Barrier,
    schema: str,
    index_name: str,
    query_text: str,
    k: int,
    warm_repetitions: int,
    require_trace: bool,
) -> dict[str, Any]:
    with psycopg.connect(dsn, autocommit=True) as connection:
        pid = connection.info.backend_pid
        baseline = {
            'backend_pid': pid,
            'backend_memory_bytes': backend_memory_bytes(connection),
            'backend_rss_bytes': backend_rss_bytes(connection, pid),
        }
        barrier.wait(timeout=60.0)
        samples = [
            query_once(
                connection,
                schema,
                index_name,
                query_text,
                k,
                require_trace,
            )
            for _ in range(warm_repetitions + 1)
        ]
        return {'baseline': baseline, 'samples': samples}


def memory_plateau(
    client: dict[str, Any],
    slack_bytes: int,
) -> dict[str, Any]:
    samples = client['samples']
    split = max(2, len(samples) // 2)
    prefix = samples[:split]
    tail = samples[split:]
    rss_ceiling = max(
        client['baseline']['backend_rss_bytes'],
        *(sample['backend_rss_bytes'] for sample in prefix),
    ) + slack_bytes
    memory_ceiling = max(
        client['baseline']['backend_memory_bytes'],
        *(sample['backend_memory_bytes'] for sample in prefix),
    ) + slack_bytes
    return {
        'rss_ceiling_bytes': rss_ceiling,
        'memory_ceiling_bytes': memory_ceiling,
        'passed': bool(
            tail
            and max(sample['backend_rss_bytes'] for sample in tail)
                <= rss_ceiling
            and max(sample['backend_memory_bytes'] for sample in tail)
                <= memory_ceiling
        ),
    }


def qualification_guard(
    connection: psycopg.Connection[Any],
    expected_port: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_setting('port')::int4,
                current_user,
                current_setting('data_directory'),
                pg_postmaster_start_time(),
                (SELECT count(*)
                 FROM pg_stat_progress_create_index),
                (SELECT count(*)
                 FROM pg_stat_activity
                 WHERE backend_type = 'client backend'
                   AND pid <> pg_backend_pid()
                   AND state <> 'idle')
            """
        )
        row = cursor.fetchone()
        cursor.execute('SELECT rolsuper FROM pg_roles WHERE rolname = current_user')
        superuser = bool(cursor.fetchone()[0])
    if int(row[0]) != expected_port:
        raise RuntimeError(
            f'connected port {row[0]} differs from {expected_port}'
        )
    if not superuser:
        raise RuntimeError('qualification requires a superuser connection')
    if int(row[4]) != 0:
        raise RuntimeError('an index build is active')
    if int(row[5]) != 0:
        raise RuntimeError('another active client backend is present')
    return {
        'port': int(row[0]),
        'user': str(row[1]),
        'data_directory': str(row[2]),
        'postmaster_started_at': str(row[3]),
        'active_index_builds': int(row[4]),
        'other_active_clients': int(row[5]),
    }


def clear_runtime(
    connection: psycopg.Connection[Any],
    schema: str,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT {}()').format(
                sql.Identifier(schema, 'evoke_runtime_cache_clear')
            )
        )
        return int(cursor.fetchone()[0])


def wait_for_readmission(
    connection: psycopg.Connection[Any],
    schema: str,
    index_name: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = runtime_state(connection, schema, index_name)
        if runtime_ready(last_state):
            return last_state, time.monotonic() - started
        time.sleep(0.1)
    raise RuntimeError(
        'worker did not restore query metadata before timeout: '
        f'{last_state}'
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    started_at = time.time()
    with psycopg.connect(args.dsn, autocommit=True) as control:
        guard = qualification_guard(control, args.expected_port)
        schema = extension_schema(control)
        generation_before = generation_status(
            control, schema, args.index
        )
        identity_before = generation_identity(generation_before)
        runtime_before = runtime_state(control, schema, args.index)
        if not runtime_ready(runtime_before):
            raise RuntimeError('target index is not query-metadata warm')
        require_trace = requires_query_trace(runtime_before)
        baseline = query_once(
            control,
            schema,
            args.index,
            args.query,
            args.k,
            require_trace,
        )
        cleared_entries = clear_runtime(control, schema)
        state_after_clear = runtime_state(control, schema, args.index)
        eviction_observed = target_eviction_observed(state_after_clear)
        state_after_readmission, readmission_seconds = wait_for_readmission(
            control,
            schema,
            args.index,
            args.readmission_timeout_seconds,
        )

        barrier = threading.Barrier(args.clients + 1)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.clients
        ) as executor:
            futures = [
                executor.submit(
                    worker,
                    args.dsn,
                    barrier,
                    schema,
                    args.index,
                    args.query,
                    args.k,
                    args.warm_repetitions,
                    require_trace,
                )
                for _ in range(args.clients)
            ]
            barrier.wait(timeout=60.0)
            clients = [future.result() for future in futures]

        generation_after = generation_status(control, schema, args.index)
        identity_after = generation_identity(generation_after)

    expected_hits = baseline['hits']
    exact = all(
        sample['hits'] == expected_hits
        for client in clients
        for sample in client['samples']
    )
    route_stable = query_route_stable(
        baseline,
        clients,
        require_trace,
    )
    first_latencies = [
        float(client['samples'][0]['elapsed_ms']) for client in clients
    ]
    warm_latencies = [
        float(sample['elapsed_ms'])
        for client in clients
        for sample in client['samples'][1:]
    ]
    slack_bytes = args.rss_slack_mib * 1024 * 1024
    plateaus = [memory_plateau(client, slack_bytes) for client in clients]
    summary = {
        'first_max_ms': max(first_latencies),
        'first_p50_ms': statistics.median(first_latencies),
        'first_p95_ms': percentile(first_latencies, 0.95),
        'warm_p50_ms': statistics.median(warm_latencies),
        'warm_p95_ms': percentile(warm_latencies, 0.95),
        'warm_max_ms': max(warm_latencies),
    }
    passed = bool(
        cleared_entries > 0
        and eviction_observed
        and runtime_ready(state_after_readmission)
        and identity_before == identity_after
        and exact
        and route_stable
        and summary['first_max_ms'] <= args.first_max_ms
        and summary['warm_p50_ms'] <= args.warm_p50_max_ms
        and summary['warm_p95_ms'] <= args.warm_p95_max_ms
        and all(plateau['passed'] for plateau in plateaus)
    )
    output = {
        'schema': 'evoke_shared_preload_recovery_v1',
        'started_at_epoch': started_at,
        'finished_at_epoch': time.time(),
        'guard': guard,
        'index': args.index,
        'query': args.query,
        'k': args.k,
        'clients': args.clients,
        'warm_repetitions': args.warm_repetitions,
        'cleared_entries': cleared_entries,
        'target_eviction_observed': eviction_observed,
        'readmission_seconds': readmission_seconds,
        'generation_before': generation_before,
        'generation_after': generation_after,
        'generation_identity_before': identity_before,
        'generation_identity_after': identity_after,
        'runtime_before': runtime_before,
        'runtime_after_clear': state_after_clear,
        'runtime_after_readmission': state_after_readmission,
        'baseline': baseline,
        'client_results': clients,
        'memory_plateaus': plateaus,
        'rank_and_score_bits_exact': exact,
        'query_route_stable': route_stable,
        'latency': summary,
        'limits': {
            'first_max_ms': args.first_max_ms,
            'warm_p50_max_ms': args.warm_p50_max_ms,
            'warm_p95_max_ms': args.warm_p95_max_ms,
            'rss_slack_mib': args.rss_slack_mib,
        },
        'passed': passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(args.output)
    print(json.dumps({
        'output': str(args.output),
        'passed': passed,
        'readmission_seconds': readmission_seconds,
        **summary,
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
