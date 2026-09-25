#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from ii42_test_support import extension_control_root


PG_BIN_DEFAULT = Path('/opt/homebrew/opt/postgresql@18/bin')
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Soak the shared model runtime worker pool and reject linear '
            'aggregate or per-worker RSS growth.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=PG_BIN_DEFAULT)
    parser.add_argument('--port', type=int, default=55451)
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument(
        '--worker-count',
        type=int,
        default=2,
        choices=range(1, 17),
    )
    parser.add_argument(
        '--intra-op-threads',
        type=int,
        default=0,
        choices=range(0, 65),
    )
    parser.add_argument(
        '--client-counts',
        default='',
        help=(
            'Comma-separated concurrent client counts. The default uses the '
            'configured worker count.'
        ),
    )
    parser.add_argument('--max-growth-kb', type=int, default=16384)
    parser.add_argument('--max-slope-kb', type=float, default=64.0)
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Current model checkout.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/ii42.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        result.check_returncode()
    return result


def process_rss_kb(pid: int) -> int:
    result = run(['ps', '-o', 'rss=', '-p', str(pid)])
    return int(result.stdout.strip())


def ready_worker_pids(status: dict[str, Any]) -> list[int]:
    pids = []
    for worker in status.get('workers', []):
        pid = int(worker.get('pid', 0))
        if worker.get('ready') is True and pid > 0:
            pids.append(pid)
    return sorted(pids)


def pool_rss_sample(
    worker_pids: list[int],
    iteration: int,
) -> dict[str, Any]:
    worker_rss_kb = {
        str(pid): process_rss_kb(pid)
        for pid in worker_pids
    }
    return {
        'iteration': iteration,
        'rss_kb': sum(worker_rss_kb.values()),
        'worker_rss_kb': worker_rss_kb,
    }


def runtime_status(dsn: str) -> dict[str, Any]:
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute('SELECT ii42_runtime_service_status()')
            return cursor.fetchone()[0]


def tail_slope_kb(samples: list[dict[str, Any]]) -> float:
    tail = samples[len(samples) // 2 :]
    if len(tail) < 2:
        return 0.0
    xs = [float(sample['iteration']) for sample in tail]
    ys = [float(sample['rss_kb']) for sample in tail]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys, strict=True)
    )
    return numerator / denominator


def worker_tail_slopes_kb(
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    if not samples:
        return {}
    return {
        pid: round(
            tail_slope_kb([
                {
                    'iteration': sample['iteration'],
                    'rss_kb': sample['worker_rss_kb'][pid],
                }
                for sample in samples
            ]),
            4,
        )
        for pid in samples[0]['worker_rss_kb']
    }


def phase_tail_is_bounded(
    phase: dict[str, Any],
    max_slope_kb: float,
) -> bool:
    return (
        phase['tail_slope_kb_per_iteration'] <= max_slope_kb
        and all(
            slope <= max_slope_kb
            for slope in phase[
                'worker_tail_slopes_kb_per_iteration'
            ].values()
        )
    )


def phase_memory_is_bounded(
    phase: dict[str, Any],
    max_growth_kb: int,
    max_slope_kb: float,
) -> bool:
    return (
        phase['growth_kb'] <= max_growth_kb
        and all(
            growth <= max_growth_kb
            for growth in phase['worker_growth_kb'].values()
        )
        and phase_tail_is_bounded(phase, max_slope_kb)
    )


def phase_completed_without_churn(phase: dict[str, Any]) -> bool:
    return (
        phase['successful_rows']
        == phase['iterations'] * phase['clients']
        and phase['session_cache_activity']['load_delta'] == 0
        and phase['session_cache_activity']['eviction_delta'] == 0
    )


def validate_model(model_path: Path) -> None:
    manifest_path = model_path / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'model manifest is missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            '--model-path must use the current Evoke model contract'
        )


def run_phase(
    dsn: str,
    worker_pids: list[int],
    model_path: Path,
    client_count: int,
    iterations: int,
    sample_every: int,
    workload: str,
) -> dict[str, Any]:
    runtime_before = runtime_status(dsn)
    samples = [pool_rss_sample(worker_pids, 0)]
    atom_rows = 0
    query_requests = 0
    document_requests = 0
    request_latency_ms = []
    round_latency_ms = []
    started_at = time.monotonic()

    def request_once(
        lane: int,
        iteration: int,
    ) -> tuple[bool, float, str]:
        request_started_at = time.monotonic()
        request_kind = (
            'document'
            if workload == 'mixed' and lane > 0
            else 'query'
        )
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                if request_kind == 'document':
                    texts = [
                        (
                            f'shared runtime soak lane {lane} '
                            f'document {iteration} row {row} '
                            + ('semantic completion payload ' * 12)
                        )
                        for row in range(16)
                    ]
                    cursor.execute(
                        'SELECT '
                        'ii42_runtime_service_document_atoms_batch'
                        '(%s, %s)',
                        (str(model_path), texts),
                    )
                else:
                    cursor.execute(
                        'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                        (
                            str(model_path),
                            (
                                f'shared runtime soak lane {lane} '
                                f'query {iteration}'
                            ),
                        ),
                    )
                result = cursor.fetchone()[0]
        return (
            isinstance(result, dict) and bool(
                result.get(
                    'results' if request_kind == 'document' else 'atoms'
                )
            ),
            (time.monotonic() - request_started_at) * 1000.0,
            request_kind,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=client_count,
    ) as executor:
        for iteration in range(1, iterations + 1):
            round_started_at = time.monotonic()
            futures = [
                executor.submit(request_once, lane, iteration)
                for lane in range(client_count)
            ]
            results = [future.result(timeout=30) for future in futures]
            if not all(success for success, _, _ in results):
                raise AssertionError(
                    f'shared runtime returned invalid result: {results}'
                )
            request_latency_ms.extend(
                latency_ms
                for _, latency_ms, _ in results
            )
            query_requests += sum(
                request_kind == 'query'
                for _, _, request_kind in results
            )
            document_requests += sum(
                request_kind == 'document'
                for _, _, request_kind in results
            )
            round_latency_ms.append(
                (time.monotonic() - round_started_at) * 1000.0
            )
            atom_rows += len(results)
            if iteration % sample_every == 0 or iteration == iterations:
                samples.append(pool_rss_sample(worker_pids, iteration))

    runtime_after = runtime_status(dsn)
    growth_kb = samples[-1]['rss_kb'] - samples[0]['rss_kb']
    elapsed_seconds = time.monotonic() - started_at
    worker_growth_kb = {
        str(pid): (
            samples[-1]['worker_rss_kb'][str(pid)]
            - samples[0]['worker_rss_kb'][str(pid)]
        )
        for pid in worker_pids
    }
    return {
        'workload': workload,
        'clients': client_count,
        'iterations': iterations,
        'successful_rows': atom_rows,
        'query_requests': query_requests,
        'document_requests': document_requests,
        'elapsed_seconds': round(elapsed_seconds, 6),
        'requests_per_second': round(atom_rows / elapsed_seconds, 4),
        'request_latency_ms': {
            'mean': round(statistics.fmean(request_latency_ms), 4),
            'p50': round(statistics.median(request_latency_ms), 4),
            'p95': round(
                sorted(request_latency_ms)[
                    max(0, int(len(request_latency_ms) * 0.95) - 1)
                ],
                4,
            ),
            'maximum': round(max(request_latency_ms), 4),
        },
        'round_latency_ms': {
            'mean': round(statistics.fmean(round_latency_ms), 4),
            'p50': round(statistics.median(round_latency_ms), 4),
            'p95': round(
                sorted(round_latency_ms)[
                    max(0, int(len(round_latency_ms) * 0.95) - 1)
                ],
                4,
            ),
            'maximum': round(max(round_latency_ms), 4),
        },
        'growth_kb': growth_kb,
        'worker_growth_kb': worker_growth_kb,
        'tail_slope_kb_per_iteration': round(
            tail_slope_kb(samples),
            4,
        ),
        'worker_tail_slopes_kb_per_iteration': (
            worker_tail_slopes_kb(samples)
        ),
        'session_cache_activity': {
            'loads_before': int(
                runtime_before.get('session_cache_loads', 0)
            ),
            'loads_after': int(
                runtime_after.get('session_cache_loads', 0)
            ),
            'load_delta': (
                int(runtime_after.get('session_cache_loads', 0))
                - int(runtime_before.get('session_cache_loads', 0))
            ),
            'evictions_before': int(
                runtime_before.get('session_cache_evictions', 0)
            ),
            'evictions_after': int(
                runtime_after.get('session_cache_evictions', 0)
            ),
            'eviction_delta': (
                int(runtime_after.get('session_cache_evictions', 0))
                - int(runtime_before.get('session_cache_evictions', 0))
            ),
        },
        'affinity_roles_before': {
            str(worker.get('pid', 0)): worker.get('affinity_role')
            for worker in runtime_before.get('workers', [])
            if int(worker.get('pid', 0)) > 0
        },
        'affinity_roles_after': {
            str(worker.get('pid', 0)): worker.get('affinity_role')
            for worker in runtime_after.get('workers', [])
            if int(worker.get('pid', 0)) > 0
        },
        'samples': samples,
    }


def main() -> None:
    args = parse_args()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    if args.iterations < 20 or args.sample_every < 1:
        raise ValueError(
            'iterations must be >= 20 and sample_every must be positive'
        )
    model_path = args.model_path.expanduser().resolve()
    validate_model(model_path)
    client_counts = (
        [args.worker_count]
        if not args.client_counts.strip()
        else [
            int(value)
            for value in args.client_counts.split(',')
            if value.strip()
        ]
    )
    if (
        not client_counts
        or any(value < 1 or value > 256 for value in client_counts)
        or len(set(client_counts)) != len(client_counts)
    ):
        raise ValueError(
            '--client-counts must contain unique integers from 1 through 256'
        )

    with tempfile.TemporaryDirectory(prefix='ii42_ort_soak_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()
        initdb = args.pg_bin / 'initdb'
        pg_ctl = args.pg_bin / 'pg_ctl'
        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as config:
            config.write("\nshared_preload_libraries = 'ii42'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                config.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(
                    args.extension_control_dir
                ).replace("'", "''")
                config.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            config.write("ii42.shared_runtime_size = '64MB'\n")
            config.write(
                f'ii42.runtime_worker_count = {args.worker_count}\n'
            )
            config.write(
                'ii42.onnxruntime_intra_op_threads = '
                f'{args.intra_op_threads}\n'
            )
            config.write("listen_addresses = ''\n")
            config.write('max_worker_processes = 16\n')

        started = False
        try:
            run([
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-l',
                str(log_path),
                '-o',
                f'-k {socket_dir} -p {args.port}',
                'start',
                '-w',
            ])
            started = True
            with psycopg.connect(
                host=str(socket_dir),
                port=args.port,
                dbname='postgres',
                autocommit=True,
            ) as connection:
                dsn = (
                    f'host={socket_dir} port={args.port} '
                    'dbname=postgres'
                )
                with connection.cursor() as cursor:
                    cursor.execute('CREATE EXTENSION ii42')

                    def warm_pool(
                        workload: str,
                        document_repetitions: int = 96,
                    ) -> None:
                        warm_barrier = threading.Barrier(
                            args.worker_count
                        )

                        def warm_lane(
                            lane: int,
                            warmup_round: int,
                        ) -> bool:
                            request_kind = (
                                'document'
                                if workload == 'mixed' and lane > 0
                                else 'query'
                            )
                            payload_repetitions = (
                                document_repetitions
                                if request_kind == 'document'
                                else 8
                            )
                            row_count = (
                                16 if request_kind == 'document' else 32
                            )
                            texts = [
                                (
                                    f'{workload} warmup round '
                                    f'{warmup_round} lane {lane} row {row} '
                                    + (
                                        'semantic runtime warmup '
                                        * payload_repetitions
                                    )
                                )
                                for row in range(row_count)
                            ]
                            function_name = (
                                'ii42_runtime_service_'
                                f'{request_kind}_atoms_batch'
                            )
                            with psycopg.connect(
                                dsn,
                                autocommit=True,
                            ) as warm_connection:
                                warm_barrier.wait(timeout=10)
                                with warm_connection.cursor() as warm_cursor:
                                    warm_cursor.execute(
                                        f'SELECT {function_name}(%s, %s)',
                                        (str(model_path), texts),
                                    )
                                    return (
                                        warm_cursor.fetchone() is not None
                                    )

                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=args.worker_count,
                        ) as executor:
                            for warmup_round in range(10):
                                futures = [
                                    executor.submit(
                                        warm_lane,
                                        lane,
                                        warmup_round,
                                    )
                                    for lane in range(args.worker_count)
                                ]
                                warm_results = [
                                    future.result(timeout=30)
                                    for future in futures
                                ]
                                if warm_results != (
                                    [True] * args.worker_count
                                ):
                                    raise AssertionError(
                                        f'{workload} worker pool warmup '
                                        f'failed: {warm_results}'
                                    )

                    warm_pool('query')
                    before = runtime_status(dsn)

                worker_pids = ready_worker_pids(before)
                loaded_workers = [
                    worker
                    for worker in before.get('workers', [])
                    if worker.get('affinity_model_loaded') is True
                ]
                if (
                    worker_pids == []
                    or int(before.get('worker_count_ready', 0))
                    != args.worker_count
                    or len(worker_pids) != args.worker_count
                    or len(loaded_workers) != args.worker_count
                ):
                    raise AssertionError(
                        f'shared runtime worker pool is not ready: {before}'
                    )
                phases = {
                    f'query_{client_count}': run_phase(
                        dsn,
                        worker_pids,
                        model_path,
                        client_count,
                        args.iterations,
                        args.sample_every,
                        'query',
                    )
                    for client_count in client_counts
                }
                stabilization_phase = None
                if args.worker_count >= 2:
                    warm_pool('mixed')
                    warm_pool('mixed', document_repetitions=12)
                    stabilization_phase = run_phase(
                        dsn,
                        worker_pids,
                        model_path,
                        args.worker_count,
                        min(args.iterations, 500),
                        args.sample_every,
                        'mixed',
                    )
                    phases['mixed_query_document'] = run_phase(
                        dsn,
                        worker_pids,
                        model_path,
                        args.worker_count,
                        args.iterations,
                        args.sample_every,
                        'mixed',
                    )
                with connection.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    after = cursor.fetchone()[0]

            gates = {
                'worker_growth_bounded': all(
                    phase_memory_is_bounded(
                        phase,
                        args.max_growth_kb,
                        args.max_slope_kb,
                    )
                    for phase in phases.values()
                ),
                'stabilization_completed': (
                    stabilization_phase is None
                    or phase_completed_without_churn(
                        stabilization_phase
                    )
                ),
                'all_requests_succeeded': (
                    all(
                        phase['successful_rows']
                        == args.iterations * phase['clients']
                        for phase in phases.values()
                    )
                    and int(after.get('failures', 0))
                    == int(before.get('failures', 0))
                ),
                'mixed_document_completion_exercised': (
                    args.worker_count < 2
                    or (
                        phases['mixed_query_document'][
                            'query_requests'
                        ] == args.iterations
                        and phases['mixed_query_document'][
                            'document_requests'
                        ] == args.iterations *
                            (args.worker_count - 1)
                    )
                ),
                'worker_pool_remained_ready': (
                    int(after.get('worker_count_ready', 0))
                    == args.worker_count
                    and ready_worker_pids(after) == worker_pids
                ),
                'measured_phases_avoid_session_churn': all(
                    phase['session_cache_activity']['load_delta'] == 0
                    and phase['session_cache_activity'][
                        'eviction_delta'
                    ] == 0
                    for phase in phases.values()
                ),
            }
            report = {
                'extension_binding': {
                    'mode': (
                        'staged'
                        if args.extension_libdir is not None
                        else 'system'
                    ),
                    'library_dir': (
                        str(args.extension_libdir)
                        if args.extension_libdir is not None
                        else None
                    ),
                    'control_root': (
                        str(args.extension_control_dir)
                        if args.extension_control_dir is not None
                        else None
                    ),
                },
                'worker_count': args.worker_count,
                'intra_op_threads': args.intra_op_threads,
                'client_counts': client_counts,
                'worker_pids': worker_pids,
                'limits': {
                    'max_growth_kb': args.max_growth_kb,
                    'max_slope_kb_per_iteration': args.max_slope_kb,
                },
                'shared_runtime_phases': phases,
                'mixed_stabilization_phase': stabilization_phase,
                'runtime_before': before,
                'runtime_after': after,
                'gates': gates,
                'all_gates_passed': all(gates.values()),
            }
        finally:
            if started:
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    'stop',
                    '-m',
                    'fast',
                    '-w',
                ], check=False)

    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not report['all_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
