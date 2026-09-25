#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from ii42_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
DEFAULT_OUTPUT = (
    Path(tempfile.gettempdir()) / 'ii42_product_path_benchmark.json'
)
DEFAULT_QUERIES = [
    'CUDA graph neural network optimization',
    'policy climate graph retrieval',
    'semantic sparse atom ranking',
    'genomics vaccine trial evidence',
]
RUNTIME_COUNTERS = (
    'requests',
    'successes',
    'failures',
    'runtime_runs',
    'runtime_total_us',
    'encoded_texts',
    'batch_successes',
    'worker_recoveries',
)


def process_rss_snapshot(pids: list[int]) -> dict[int, int]:
    if not pids:
        return {}
    result = run([
        'ps',
        '-o',
        'pid=,rss=',
        '-p',
        ','.join(str(pid) for pid in pids),
    ], check=False)
    if result.returncode != 0:
        return {}
    snapshot: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            snapshot[int(fields[0])] = int(fields[1]) * 1024
        except ValueError:
            continue
    return snapshot


def runtime_worker_pids(status: dict[str, Any]) -> list[int]:
    workers = status.get('workers')
    if not isinstance(workers, list):
        return []
    return sorted({
        int(worker['pid'])
        for worker in workers
        if isinstance(worker, dict) and int(worker.get('pid', 0)) > 0
    })


def sample_process_rss(
    pids: list[int],
    stop: threading.Event,
    samples: dict[int, list[int]],
) -> None:
    while not stop.wait(2.0):
        for pid, rss in process_rss_snapshot(pids).items():
            samples[pid].append(rss)


def summarize_process_rss(
    samples: dict[int, list[int]],
    runtime_pids: set[int],
    backend_pid: int,
) -> dict[str, Any]:
    processes: dict[str, Any] = {}
    combined_peak = 0

    for pid, values in samples.items():
        if not values:
            continue
        role = 'backend' if pid == backend_pid else 'runtime_worker'
        peak = max(values)
        combined_peak += peak
        processes[str(pid)] = {
            'role': role,
            'samples': len(values),
            'initial_bytes': values[0],
            'peak_bytes': peak,
            'final_bytes': values[-1],
        }
    return {
        'backend_pid': backend_pid,
        'runtime_worker_pids': sorted(runtime_pids),
        'combined_individual_peak_upper_bound_bytes': combined_peak,
        'processes': processes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark Evoke BM25 and semantic-enabled index paths.',
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=Path(tempfile.gettempdir()),
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--port', type=int, default=55452)
    parser.add_argument('--docs', type=int, default=10_000)
    parser.add_argument('--corpus-jsonl', type=Path)
    parser.add_argument('--corpus-limit', type=int)
    parser.add_argument('--query-jsonl', type=Path)
    parser.add_argument('--qrels-tsv', type=Path)
    parser.add_argument('--query-limit', type=int, default=64)
    parser.add_argument(
        '--accelerator-heap-factor',
        type=float,
        action='append',
        dest='accelerator_heap_factors',
        help=(
            'Hidden semantic-accelerator A/B factor. Repeat for multiple '
            'routes. Zero opens every cluster and must remain exact.'
        ),
    )
    parser.add_argument(
        '--accelerator-candidate-multiplier',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--accelerator-bound-residual-candidates',
        action='store_true',
        help=(
            'Add A/B routes with a bounded residual per-term candidate '
            'surface. Full-query candidate scoring remains exact.'
        ),
    )
    parser.add_argument(
        '--accelerator-accumulate-residual-candidates',
        action='store_true',
        help=(
            'Add the hidden exact cross-term residual candidate oracle. '
            'This is a diagnostic route, not a product memory model.'
        ),
    )
    parser.add_argument(
        '--accelerator-telemetry-query-limit',
        type=int,
        help=(
            'Collect expensive per-route accelerator telemetry for only '
            'the first N queries. Latency and quality still cover all '
            'queries.'
        ),
    )
    parser.add_argument(
        '--accelerator-summarize-residual-candidates',
        action='store_true',
        help=(
            'Replace the corpus-sized exact residual accumulator with the '
            'hidden weighted Space-Saving A/B summary.'
        ),
    )
    parser.add_argument(
        '--accelerator-summary-multiplier',
        type=int,
        default=4,
    )
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--shared-cache-mb', type=int, default=512)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument(
        '--onnxruntime-intra-op-threads',
        type=int,
        default=0,
        help=(
            'CPU intra-op thread cap for the isolated PostgreSQL instance. '
            'Zero preserves the ONNX Runtime default.'
        ),
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        help=(
            'Current semantic model checkout. Required '
            'unless --skip-semantic is used.'
        ),
    )
    parser.add_argument('--skip-semantic', action='store_true')
    parser.add_argument('--keep-pg', action='store_true')
    return parser.parse_args()


def run(
    cmd: list[str | Path],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(item) for item in cmd],
        input=input_text,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def timed(cur: psycopg.Cursor[Any], label: str, sql_text: str, params=()) -> float:
    start = time.perf_counter()
    cur.execute(sql_text, params)
    elapsed = time.perf_counter() - start
    print(f'{label}: {elapsed:.3f}s', flush=True)
    return elapsed


def summarize_ms(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        'count': len(ordered),
        'mean_ms': statistics.fmean(ordered),
        'p50_ms': ordered[len(ordered) // 2],
        'p95_ms': ordered[p95_index],
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
    }


def runtime_phase_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    counters = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in RUNTIME_COUNTERS
    }
    runs = counters['runtime_runs']
    encoded_texts = counters['encoded_texts']
    runtime_seconds = counters['runtime_total_us'] / 1_000_000.0
    return {
        **counters,
        'wall_seconds': wall_seconds,
        'runtime_seconds': runtime_seconds,
        'non_runtime_seconds': max(0.0, wall_seconds - runtime_seconds),
        'runtime_wall_share': (
            runtime_seconds / wall_seconds if wall_seconds > 0.0 else 0.0
        ),
        'average_batch_size': encoded_texts / runs if runs else 0.0,
        'single_request_execution': runs > 0 and encoded_texts == runs,
        'last_batch_size': int(after.get('last_batch_size', 0)),
        'max_observed_batch_size': int(
            after.get('max_observed_batch_size', 0)
        ),
        'max_supported_batch_size': int(
            after.get('max_supported_batch_size', 1)
        ),
        'runtime_max_us': int(after.get('runtime_max_us', 0)),
    }


def query_latency(
    cur: psycopg.Cursor[Any],
    sql_text: str,
    params: tuple[Any, ...],
    repeats: int,
) -> dict[str, Any]:
    values: list[float] = []
    rows = None
    for _ in range(repeats):
        start = time.perf_counter()
        cur.execute(sql_text, params)
        rows = cur.fetchall()
        values.append((time.perf_counter() - start) * 1000.0)
    summary = summarize_ms(values)
    summary['sample_rows'] = rows[:3] if rows else []
    return summary


def load_query_texts(path: Path, limit: int) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            if len(queries) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'invalid query JSONL at {path}:{line_no}: {exc}'
                ) from exc
            query_id = pick_text(
                record,
                ('id', '_id', 'query_id', 'qid'),
            )
            query_text = pick_text(
                record,
                ('text', 'query', 'question', 'title'),
            )
            if query_id is not None and query_text is not None:
                queries.append((query_id, query_text))
    if not queries:
        raise ValueError(f'no usable queries loaded from {path}')
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with path.open(encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n').split('\t')
        expected = ['query-id', 'corpus-id', 'score']
        if header != expected:
            raise ValueError(
                f'unsupported qrels header in {path}: {header!r}'
            )
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 3:
                raise ValueError(
                    f'invalid qrels row {path}:{line_number}'
                )
            query_id, document_id, score_text = fields
            score = float(score_text)
            if score > 0.0:
                qrels.setdefault(query_id, {})[document_id] = score
    if not qrels:
        raise ValueError(f'no positive qrels loaded from {path}')
    return qrels


def retrieval_metrics_for_query(
    ranking: list[str],
    relevant: dict[str, float],
) -> dict[str, float]:
    dcg = 0.0
    for rank, document_id in enumerate(ranking[:10], start=1):
        relevance = relevant.get(document_id, 0.0)
        dcg += (2.0 ** relevance - 1.0) / math.log2(rank + 1.0)
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum(
        (2.0 ** relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(ideal, start=1)
    )

    hits = 0
    precision_sum = 0.0
    first_relevant_rank: int | None = None
    for rank, document_id in enumerate(ranking[:100], start=1):
        if document_id not in relevant:
            continue
        hits += 1
        precision_sum += hits / rank
        if first_relevant_rank is None and rank <= 20:
            first_relevant_rank = rank
    return {
        'ndcg_at_10': dcg / idcg if idcg > 0.0 else 0.0,
        'map_at_100': precision_sum / min(len(relevant), 100),
        'recall_at_100': hits / len(relevant),
        'mrr_at_20': (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
    }


def retrieval_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float | int]:
    query_metrics = [
        retrieval_metrics_for_query(ranking, qrels[query_id])
        for query_id, ranking in rankings.items()
        if query_id in qrels and qrels[query_id]
    ]
    if not query_metrics:
        raise ValueError('no benchmark query has matching positive qrels')
    return {
        'query_count': len(query_metrics),
        **{
            key: statistics.fmean(row[key] for row in query_metrics)
            for key in (
                'ndcg_at_10',
                'map_at_100',
                'recall_at_100',
                'mrr_at_20',
            )
        },
    }


def configure_accelerator_route(
    cur: psycopg.Cursor[Any],
    heap_factor: float | None,
    candidate_multiplier: int,
    bound_residual_candidates: bool = False,
    accumulate_residual_candidates: bool = False,
    summarize_residual_candidates: bool = False,
    summary_multiplier: int = 4,
) -> None:
    cur.execute(
        "SELECT pg_catalog.set_config("
        "'ii42.test_disable_semantic_accelerator', %s, false)",
        ('on' if heap_factor is None else 'off',),
    )
    cur.execute(
        "SELECT pg_catalog.set_config("
        "'ii42.test_semantic_accelerator_bound_residual_candidates', "
        "%s, false)",
        ('on' if bound_residual_candidates else 'off',),
    )
    cur.execute(
        "SELECT pg_catalog.set_config("
        "'ii42.test_semantic_accelerator_accumulate_residual_candidates', "
        "%s, false)",
        ('on' if accumulate_residual_candidates else 'off',),
    )
    cur.execute(
        "SELECT pg_catalog.set_config("
        "'ii42.test_semantic_accelerator_summarize_residual_candidates', "
        "%s, false)",
        ('on' if summarize_residual_candidates else 'off',),
    )
    cur.execute(
        "SELECT pg_catalog.set_config("
        "'ii42.test_semantic_accelerator_summary_multiplier', %s, false)",
        (str(summary_multiplier),),
    )
    if heap_factor is not None:
        cur.execute(
            "SELECT pg_catalog.set_config("
            "'ii42.test_semantic_accelerator_heap_factor', %s, false)",
            (str(heap_factor),),
        )
        cur.execute(
            "SELECT pg_catalog.set_config("
            "'ii42.test_semantic_accelerator_candidate_multiplier', "
            "%s, false)",
            (str(candidate_multiplier),),
        )


def fetch_accelerator_hits(
    cur: psycopg.Cursor[Any],
    query_text: str,
    top_k: int,
) -> tuple[list[tuple[str, float]], float]:
    started = time.perf_counter()
    cur.execute(
        """
        SELECT source.id, hit.score::float8
        FROM ii42_query(
            'bench.docs_body_semantic_idx'::regclass,
            %s,
            %s
        ) AS hit
        JOIN bench.docs AS source ON source.ctid = hit.ctid
        ORDER BY hit.score DESC, source.id
        LIMIT %s
        """,
        (query_text, top_k, top_k),
    )
    rows = [(str(row[0]), float(row[1])) for row in cur.fetchall()]
    return rows, (time.perf_counter() - started) * 1000.0


def fetch_accelerator_telemetry(
    cur: psycopg.Cursor[Any],
    query_text: str,
    top_k: int,
    skip_oracle: bool,
) -> dict[str, Any]:
    cur.execute(
        """
        WITH encoded AS (
            SELECT ii42_encode_text_internal(
                'bench.docs_body_semantic_idx'::regclass,
                %s
            ) AS value
        ), atoms AS (
            SELECT
                array_agg(atom.value::int4 ORDER BY atom.ordinality) AS ids,
                array_agg(weight.value::real ORDER BY weight.ordinality)
                    AS weights
            FROM encoded
            CROSS JOIN LATERAL jsonb_array_elements_text(
                encoded.value->'atoms'
            ) WITH ORDINALITY AS atom(value, ordinality)
            JOIN LATERAL jsonb_array_elements_text(
                encoded.value->'weights'
            ) WITH ORDINALITY AS weight(value, ordinality)
              ON weight.ordinality = atom.ordinality
        )
        SELECT bench.test_query_page_native_topk(
            'bench.docs_body_semantic_idx'::regclass,
            atoms.ids,
            atoms.weights,
            %s,
            %s
        )
        FROM atoms
        """,
        (query_text, top_k, skip_oracle),
    )
    row = cur.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid semantic accelerator telemetry')
    telemetry = dict(row[0])
    telemetry.pop('hits', None)
    return telemetry


def result_overlap(
    baseline: list[tuple[str, float]],
    candidate: list[tuple[str, float]],
) -> float:
    baseline_ids = {row[0] for row in baseline}
    candidate_ids = {row[0] for row in candidate}
    if not baseline_ids:
        return 1.0 if not candidate_ids else 0.0
    return len(baseline_ids & candidate_ids) / len(baseline_ids)


def benchmark_semantic_accelerator(
    cur: psycopg.Cursor[Any],
    query_path: Path,
    query_limit: int,
    repeats: int,
    heap_factors: list[float],
    candidate_multiplier: int,
    include_bounded_residual: bool,
    include_accumulated_residual: bool = False,
    include_summarized_residual: bool = False,
    include_full_accelerator: bool = True,
    include_per_term_residual: bool = True,
    qrels: dict[str, dict[str, float]] | None = None,
    telemetry_query_limit: int | None = None,
    summary_multiplier: int = 4,
    top_k: int = 100,
) -> dict[str, Any]:
    queries = load_query_texts(query_path, query_limit)
    cur.execute(
        "SELECT ii42_index_status("
        "'bench.docs_body_semantic_idx'::regclass)"
    )
    status = cur.fetchone()[0]
    accelerator_status = status['generation']['semantic_accelerator']
    if (
        accelerator_status.get('state') != 'ready'
        or not accelerator_status.get('forward_complete')
    ):
        raise RuntimeError(
            'semantic accelerator is not ready after index publication: '
            f'{accelerator_status}'
        )
    cur.execute(
        """
        CREATE OR REPLACE FUNCTION bench.test_query_page_native_topk(
            regclass,
            int4[],
            real[],
            int4,
            bool
        ) RETURNS jsonb
        AS '$libdir/ii42', 'ii42_test_query_page_native_topk'
        LANGUAGE C STRICT
        """
    )
    route_options: dict[str, tuple[float | None, bool, bool, bool]] = {
        'exact': (None, False, False, False),
    }
    for heap_factor in heap_factors:
        if include_full_accelerator:
            route_options[f'accelerator_h{heap_factor:g}'] = (
                heap_factor,
                False,
                False,
                False,
            )
        if include_bounded_residual and heap_factor > 0.0:
            if include_per_term_residual:
                route_options[
                    f'accelerator_h{heap_factor:g}_bounded_residual'
                ] = (heap_factor, True, False, False)
            if include_accumulated_residual:
                route_options[
                    f'accelerator_h{heap_factor:g}_accumulated_residual'
                ] = (heap_factor, True, True, False)
            if include_summarized_residual:
                route_options[
                    f'accelerator_h{heap_factor:g}_summarized_residual'
                ] = (heap_factor, True, True, True)
    route_latencies: dict[str, list[float]] = {
        route: [] for route in route_options
    }
    route_overlaps: dict[str, list[float]] = {
        route: [] for route in route_options if route != 'exact'
    }
    route_exact: dict[str, bool] = {
        route: True for route in route_options if route != 'exact'
    }
    reference_routes = {
        route: route.replace(
            '_summarized_residual',
            '_accumulated_residual',
        )
        for route in route_options
        if route.endswith('_summarized_residual')
    }
    route_reference_overlaps: dict[str, list[float]] = {
        route: [] for route in reference_routes
    }
    route_rankings: dict[str, dict[str, list[str]]] = {
        route: {} for route in route_options
    }
    query_rows: list[dict[str, Any]] = []

    for query_index, (query_id, query_text) in enumerate(queries):
        route_results: dict[str, list[tuple[str, float]]] = {}
        per_query_latencies: dict[str, list[float]] = {
            route: [] for route in route_options
        }
        route_names = list(route_options)
        for route, route_option in route_options.items():
            (
                heap_factor,
                bound_residual,
                accumulate_residual,
                summarize_residual,
            ) = route_option
            configure_accelerator_route(
                cur,
                heap_factor,
                candidate_multiplier,
                bound_residual,
                accumulate_residual,
                summarize_residual,
                summary_multiplier,
            )
            fetch_accelerator_hits(cur, query_text, top_k)
        for repeat_index in range(repeats):
            offset = (query_index + repeat_index) % len(route_names)
            ordered_routes = route_names[offset:] + route_names[:offset]
            for route in ordered_routes:
                (
                    heap_factor,
                    bound_residual,
                    accumulate_residual,
                    summarize_residual,
                ) = route_options[route]
                configure_accelerator_route(
                    cur,
                    heap_factor,
                    candidate_multiplier,
                    bound_residual,
                    accumulate_residual,
                    summarize_residual,
                    summary_multiplier,
                )
                rows, latency_ms = fetch_accelerator_hits(
                    cur,
                    query_text,
                    top_k,
                )
                route_results[route] = rows
                route_latencies[route].append(latency_ms)
                per_query_latencies[route].append(latency_ms)

        baseline = route_results['exact']
        query_route_rows: dict[str, Any] = {}
        for route, rows in route_results.items():
            route_rankings[route][query_id] = [row[0] for row in rows]
            overlap = 1.0 if route == 'exact' else result_overlap(
                baseline,
                rows,
            )
            if route != 'exact':
                route_overlaps[route].append(overlap)
                route_exact[route] = route_exact[route] and rows == baseline
            (
                heap_factor,
                bound_residual,
                accumulate_residual,
                summarize_residual,
            ) = route_options[route]
            telemetry = None
            if (
                telemetry_query_limit is None or
                query_index < telemetry_query_limit
            ):
                configure_accelerator_route(
                    cur,
                    heap_factor,
                    candidate_multiplier,
                    bound_residual,
                    accumulate_residual,
                    summarize_residual,
                    summary_multiplier,
                )
                try:
                    telemetry = fetch_accelerator_telemetry(
                        cur,
                        query_text,
                        top_k,
                        heap_factor not in (None, 0.0),
                    )
                except Exception as error:
                    raise RuntimeError(
                        'semantic accelerator telemetry failed for '
                        f'query={query_id!r} route={route!r} '
                        f'text={query_text!r}'
                    ) from error
            query_route_rows[route] = {
                'latency': summarize_ms(per_query_latencies[route]),
                'overlap_at_k': overlap,
                'exact': rows == baseline,
                'telemetry': telemetry,
            }
            reference_route = reference_routes.get(route)
            if reference_route is not None:
                reference_overlap = result_overlap(
                    route_results[reference_route],
                    rows,
                )
                route_reference_overlaps[route].append(reference_overlap)
                query_route_rows[route].update({
                    'reference_route': reference_route,
                    'overlap_at_k_vs_reference': reference_overlap,
                })
            relevant = qrels.get(query_id) if qrels is not None else None
            if relevant:
                query_route_rows[route]['retrieval_metrics'] = (
                    retrieval_metrics_for_query(
                        route_rankings[route][query_id],
                        relevant,
                    )
                )
        query_rows.append({
            'query_id': query_id,
            'query_text': query_text,
            'routes': query_route_rows,
        })

    configure_accelerator_route(
        cur,
        None,
        candidate_multiplier,
        False,
        False,
        False,
        summary_multiplier,
    )
    exact_p50 = summarize_ms(route_latencies['exact'])['p50_ms']
    routes: dict[str, Any] = {}
    for route, values in route_latencies.items():
        latency = summarize_ms(values)
        routes[route] = {
            'latency': latency,
            'speedup_vs_exact': exact_p50 / latency['p50_ms'],
        }
        if route != 'exact':
            overlaps = route_overlaps[route]
            routes[route].update({
                'mean_overlap_at_k': statistics.fmean(overlaps),
                'minimum_overlap_at_k': min(overlaps),
                'all_results_exact': route_exact[route],
            })
        reference_route = reference_routes.get(route)
        if reference_route is not None:
            reference_overlaps = route_reference_overlaps[route]
            routes[route].update({
                'reference_route': reference_route,
                'mean_overlap_at_k_vs_reference': statistics.fmean(
                    reference_overlaps
                ),
                'minimum_overlap_at_k_vs_reference': min(
                    reference_overlaps
                ),
            })
        if qrels is not None:
            routes[route]['retrieval_metrics'] = retrieval_metrics(
                route_rankings[route],
                qrels,
            )
    return {
        'query_jsonl': str(query_path),
        'query_count': len(queries),
        'top_k': top_k,
        'repeats': repeats,
        'candidate_multiplier': candidate_multiplier,
        'includes_bounded_residual_routes': include_bounded_residual,
        'includes_accumulated_residual_routes': include_accumulated_residual,
        'includes_summarized_residual_routes': include_summarized_residual,
        'includes_full_accelerator_routes': include_full_accelerator,
        'includes_per_term_residual_routes': include_per_term_residual,
        'telemetry_query_limit': telemetry_query_limit,
        'summary_multiplier': summary_multiplier,
        'status': accelerator_status,
        'routes': routes,
        'queries': query_rows,
    }


def explain(
    cur: psycopg.Cursor[Any],
    sql_text: str,
    params: tuple[Any, ...],
) -> str:
    cur.execute('EXPLAIN (ANALYZE, BUFFERS, VERBOSE) ' + sql_text, params)
    return '\n'.join(row[0] for row in cur.fetchall())


def atom_outputs_match(
    single: dict[str, Any],
    batched: dict[str, Any],
) -> bool:
    if single.get('atoms') != batched.get('atoms'):
        return False
    single_weights = single.get('weights')
    batch_weights = batched.get('weights')
    if not isinstance(single_weights, list) or not isinstance(
        batch_weights,
        list,
    ):
        return False
    if len(single_weights) != len(batch_weights):
        return False
    return all(
        math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-6)
        for left, right in zip(single_weights, batch_weights, strict=True)
    )


def benchmark_runtime_batch_encoding(
    cur: psycopg.Cursor[Any],
    model_path: Path,
    text_count: int = 64,
) -> dict[str, Any]:
    texts = [
        f'{DEFAULT_QUERIES[i % len(DEFAULT_QUERIES)]} document {i}'
        for i in range(text_count)
    ]

    cur.execute('SELECT ii42_runtime_service_status()')
    single_before = cur.fetchone()[0]
    batch_size = max(
        1,
        min(128, int(single_before.get('max_supported_batch_size', 128))),
    )
    started = time.perf_counter()
    single_results: list[dict[str, Any]] = []
    for text_value in texts:
        cur.execute(
            'SELECT ii42_runtime_service_query_atoms(%s, %s)',
            (str(model_path), text_value),
        )
        single_results.append(cur.fetchone()[0])
    single_seconds = time.perf_counter() - started
    cur.execute('SELECT ii42_runtime_service_status()')
    single_after = cur.fetchone()[0]

    cur.execute('SELECT ii42_runtime_service_status()')
    batch_before = cur.fetchone()[0]
    started = time.perf_counter()
    batch_results: list[dict[str, Any]] = []
    for offset in range(0, len(texts), batch_size):
        cur.execute(
            'SELECT ii42_runtime_service_query_atoms_batch(%s, %s)',
            (str(model_path), texts[offset:offset + batch_size]),
        )
        envelope = cur.fetchone()[0]
        batch_results.extend(envelope['results'])
    batch_seconds = time.perf_counter() - started
    cur.execute('SELECT ii42_runtime_service_status()')
    batch_after = cur.fetchone()[0]

    mismatches = sum(
        not atom_outputs_match(single, batched)
        for single, batched in zip(
            single_results,
            batch_results,
            strict=True,
        )
    )
    return {
        'text_count': text_count,
        'batch_size': batch_size,
        'parity_mismatches': mismatches,
        'single': runtime_phase_summary(
            single_before,
            single_after,
            single_seconds,
        ),
        'batch': runtime_phase_summary(
            batch_before,
            batch_after,
            batch_seconds,
        ),
        'wall_speedup': (
            single_seconds / batch_seconds if batch_seconds > 0.0 else None
        ),
    }


def pick_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def load_jsonl_docs(
    cur: psycopg.Cursor[Any],
    path: Path,
    limit: int | None,
) -> dict[str, Any]:
    loaded = 0
    skipped = 0
    started = time.perf_counter()
    copy_sql = 'COPY bench.docs (id, title, body) FROM STDIN'
    with path.open('r', encoding='utf-8') as handle:
        with cur.copy(copy_sql) as copy:
            for line_no, line in enumerate(handle, start=1):
                if limit is not None and loaded >= limit:
                    break
                stripped = line.strip()
                if not stripped:
                    skipped += 1
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'invalid JSONL at {path}:{line_no}: {exc}'
                    ) from exc
                doc_id = pick_text(
                    record,
                    ('id', '_id', 'doc_id', 'paper_id', 'pmid'),
                )
                title = pick_text(record, ('title', 'name', 'heading')) or ''
                body = pick_text(
                    record,
                    (
                        'body',
                        'abstract',
                        'text',
                        'content',
                        'markdown',
                        'document',
                    ),
                )
                if doc_id is None or body is None:
                    skipped += 1
                    continue
                copy.write_row((doc_id, title, body))
                loaded += 1
    if loaded == 0:
        raise ValueError(f'no usable documents loaded from {path}')
    return {
        'path': str(path),
        'limit': limit,
        'loaded': loaded,
        'skipped': skipped,
        'seconds': time.perf_counter() - started,
    }


def configure_postgres(data_dir: Path, args: argparse.Namespace) -> None:
    with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'ii42'\n")
        if args.extension_libdir is not None:
            libdir = str(args.extension_libdir).replace("'", "''")
            handle.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if args.extension_control_dir is not None:
            control_dir = str(args.extension_control_dir).replace("'", "''")
            handle.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
        handle.write("listen_addresses = ''\n")
        handle.write('max_worker_processes = 16\n')
        handle.write(
            f"ii42.shared_runtime_size = '{args.shared_cache_mb}MB'\n"
        )
        handle.write(
            'ii42.onnxruntime_intra_op_threads = '
            f'{args.onnxruntime_intra_op_threads}\n'
        )
        handle.write('ii42.preload_timer_interval_ms = 1000\n')
        handle.write('ii42.maintenance_timer_interval_ms = 60000\n')


def setup_bm25(
    cur: psycopg.Cursor[Any],
    docs: int,
    repeats: int,
    corpus_jsonl: Path | None,
    corpus_limit: int | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        'requested_docs': docs,
        'corpus_jsonl': str(corpus_jsonl) if corpus_jsonl else None,
    }
    timed(cur, 'create bench schema/table', """
        CREATE SCHEMA bench;
        CREATE TABLE bench.docs (
            id text PRIMARY KEY,
            title text NOT NULL,
            body text NOT NULL
        )
    """)
    if corpus_jsonl is None:
        results['insert_seconds'] = timed(cur, 'insert synthetic docs', """
            INSERT INTO bench.docs (id, title, body)
            SELECT
                i::text,
                concat_ws(' ',
                    CASE WHEN i %% 2 = 0 THEN 'cuda' ELSE 'policy' END,
                    CASE WHEN i %% 3 = 0 THEN 'graph' ELSE 'language' END,
                    CASE WHEN i %% 5 = 0 THEN 'neural' ELSE 'evidence' END,
                    'document', i::text
                ),
                concat_ws(' ',
                    CASE WHEN i %% 2 = 0 THEN 'cuda graph optimization'
                        ELSE 'public policy regulation' END,
                    CASE WHEN i %% 3 = 0 THEN 'semantic retrieval ranking'
                        ELSE 'climate health evidence' END,
                    CASE WHEN i %% 7 = 0 THEN 'sparse atom model'
                        ELSE 'dense vector baseline' END,
                    CASE WHEN i %% 11 = 0 THEN 'neural network gpu'
                        ELSE 'journal abstract text' END,
                    md5(i::text), md5((i * 17)::text)
                )
            FROM generate_series(1, %s) AS g(i)
        """, (docs,))
        results['docs'] = docs
    else:
        loaded = load_jsonl_docs(cur, corpus_jsonl, corpus_limit)
        results['load_jsonl'] = loaded
        results['insert_seconds'] = loaded['seconds']
        results['docs'] = loaded['loaded']
    results['body_index_seconds'] = timed(cur, 'create body index', """
        CREATE INDEX docs_body_ii42_idx
        ON bench.docs USING ii42 (body)
        WITH (method = 'lucene', idf_method = 'lucene', consistency = 'manual')
    """)
    results['field_index_seconds'] = timed(cur, 'create field-aware index', """
        CREATE INDEX docs_title_body_field_ii42_idx
        ON bench.docs USING ii42 (title, body)
        WITH (
            method = 'lucene',
            idf_method = 'lucene',
            consistency = 'manual',
            field_aware = true
        )
    """)
    timed(cur, 'analyze docs', 'ANALYZE bench.docs')
    cur.execute("""
        SELECT relname, pg_relation_size(oid)
        FROM pg_class
        WHERE relname IN (
            'docs',
            'docs_body_ii42_idx',
            'docs_title_body_field_ii42_idx'
        )
        ORDER BY relname
    """)
    results['relation_sizes'] = cur.fetchall()

    body_sql = """
        SELECT d.id, h.score
        FROM ii42_query(
            'bench.docs_body_ii42_idx'::regclass,
            %s,
            20
        ) AS h
        JOIN bench.docs AS d ON d.ctid = h.ctid
        ORDER BY h.score DESC, d.id
        LIMIT 20
    """
    body_direct_sql = """
        SELECT d.id, h.score
        FROM ii42_query(
            'bench.docs_body_ii42_idx'::regclass,
            %s,
            20,
            NULL
        ) AS h
        JOIN bench.docs AS d ON d.ctid = h.ctid
        ORDER BY h.score DESC, d.id
        LIMIT 20
    """
    field_sql = """
        SELECT d.id, h.score
        FROM ii42_field_aware_query(
            'bench.docs_title_body_field_ii42_idx'::regclass,
            %s,
            ARRAY['title', 'body']::text[],
            ARRAY[2.0, 1.0]::real[],
            20
        ) AS h
        JOIN bench.docs AS d ON d.ctid = h.ctid
        ORDER BY h.score DESC, d.id
        LIMIT 20
    """
    results['queries'] = {}
    for query_text in DEFAULT_QUERIES:
        results['queries'][query_text] = {
            'body': query_latency(cur, body_sql, (query_text,), repeats),
            'body_direct_diagnostic': query_latency(
                cur,
                body_direct_sql,
                (query_text,),
                repeats,
            ),
            'field_aware': query_latency(cur, field_sql, (query_text,), repeats),
        }
    results['body_explain'] = explain(cur, body_sql, (DEFAULT_QUERIES[0],))
    results['field_explain'] = explain(cur, field_sql, (DEFAULT_QUERIES[0],))
    results['body_source_lookup'] = summarize_source_lookup_plan(
        results['body_explain']
    )
    results['field_source_lookup'] = summarize_source_lookup_plan(
        results['field_explain']
    )
    return results


def setup_semantic(
    cur: psycopg.Cursor[Any],
    model_path: Path,
    repeats: int,
) -> dict[str, Any]:
    cur.execute('SELECT count(*) FROM bench.docs')
    results: dict[str, Any] = {'docs': int(cur.fetchone()[0])}
    escaped_path = str(model_path).replace("'", "''")
    cur.execute('SELECT ii42_runtime_service_status()')
    runtime_before_index = cur.fetchone()[0]
    cur.execute('SELECT pg_backend_pid()')
    backend_pid = int(cur.fetchone()[0])
    runtime_pids = set(runtime_worker_pids(runtime_before_index))
    monitored_pids = sorted(runtime_pids | {backend_pid})
    rss_samples = {pid: [] for pid in monitored_pids}
    for pid, rss in process_rss_snapshot(monitored_pids).items():
        rss_samples[pid].append(rss)
    rss_stop = threading.Event()
    rss_thread = threading.Thread(
        target=sample_process_rss,
        args=(monitored_pids, rss_stop, rss_samples),
        daemon=True,
    )
    rss_thread.start()
    try:
        results['index_seconds'] = timed(
            cur,
            'create unified SAE index',
            f"""
                CREATE INDEX docs_body_semantic_idx
                ON bench.docs
                USING ii42 (body)
                WITH (
                    sae = true,
                    model_path = '{escaped_path}'
                )
            """,
        )
    finally:
        rss_stop.set()
        rss_thread.join()
        for pid, rss in process_rss_snapshot(monitored_pids).items():
            rss_samples[pid].append(rss)
    results['index_process_rss'] = summarize_process_rss(
        rss_samples,
        runtime_pids,
        backend_pid,
    )
    cur.execute('SELECT ii42_runtime_service_status()')
    runtime_after_index = cur.fetchone()[0]
    results['index_runtime_phase'] = runtime_phase_summary(
        runtime_before_index,
        runtime_after_index,
        results['index_seconds'],
    )
    cur.execute("""
        SELECT pg_relation_size('bench.docs_body_semantic_idx'::regclass)
    """)
    results['index_bytes'] = int(cur.fetchone()[0])
    cur.execute("""
        SELECT ii42_index_status('bench.docs_body_semantic_idx'::regclass)
    """)
    results['status'] = cur.fetchone()[0]
    cur.execute("""
        SELECT ii42_index_audit('bench.docs_body_semantic_idx'::regclass)
    """)
    results['audit'] = cur.fetchone()[0]
    cur.execute("""
        SELECT ii42_index_preload(
            'bench.docs_body_semantic_idx'::regclass
        )
    """)
    results['generation_cache_preload'] = cur.fetchone()[0]
    cur.execute("""
        SELECT ii42_index_runtime_state_json(
            'bench.docs_body_semantic_idx'::regclass
        )
    """)
    results['generation_cache_state'] = cur.fetchone()[0]

    unified_query_sql = """
        SELECT source.id, hit.score
        FROM ii42_query(
            'bench.docs_body_semantic_idx'::regclass,
            %s,
            20
        ) AS hit
        JOIN bench.docs AS source ON source.ctid = hit.ctid
        ORDER BY hit.score DESC, source.id
        LIMIT 20
    """
    results['queries'] = {}
    cur.execute('SELECT ii42_runtime_service_status()')
    runtime_before_queries = cur.fetchone()[0]
    query_phase_started = time.perf_counter()
    for query_text in DEFAULT_QUERIES:
        results['queries'][query_text] = query_latency(
            cur,
            unified_query_sql,
            (query_text,),
            repeats,
        )
    results['explain'] = explain(
        cur,
        unified_query_sql,
        (DEFAULT_QUERIES[0],),
    )
    results['source_lookup'] = summarize_source_lookup_plan(
        results['explain']
    )
    query_phase_seconds = time.perf_counter() - query_phase_started
    cur.execute('SELECT ii42_runtime_service_status()')
    runtime_after_queries = cur.fetchone()[0]
    results['query_runtime_phase'] = runtime_phase_summary(
        runtime_before_queries,
        runtime_after_queries,
        query_phase_seconds,
    )
    results['runtime_status'] = runtime_after_queries
    cur.execute(
        """
        SELECT candidate_docs, candidate_postings,
               rerank_doc_terms, memory_bytes
        FROM ii42_query_semantic_internal(
            'bench.docs_body_semantic_idx'::regclass,
            %s,
            20
        )
        ORDER BY rank
        LIMIT 1
        """,
        ('cuda graph runtime sae query',),
    )
    diagnostic_row = cur.fetchone()
    results['internal_semantic_diagnostics'] = diagnostic_row
    results['batch_encoding_diagnostic'] = benchmark_runtime_batch_encoding(
        cur,
        model_path,
    )
    return results


def run_concurrent_semantic_queries(
    dsn: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    sql_text = """
        SELECT count(*)
        FROM ii42_query(
            'bench.docs_body_semantic_idx'::regclass,
            %s,
            20
        )
    """

    ready = threading.Barrier(concurrency, timeout=30.0)

    def query_once(worker_id: int) -> dict[str, Any]:
        total_started = time.perf_counter()
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                connected_at = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(sql_text, ('semantic backend warmup',))
                    cur.fetchone()
                warmed_at = time.perf_counter()
                ready.wait()
                query_started = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(sql_text, (f'concurrent query {worker_id}',))
                    row = cur.fetchone()
                query_finished = time.perf_counter()
        except Exception:
            ready.abort()
            raise

        query_ms = (query_finished - query_started) * 1000.0
        return {
            'worker': worker_id,
            'ok': True,
            # Keep ms as the engine-query measurement used by older readers.
            'ms': query_ms,
            'query_ms': query_ms,
            'connection_ms': (connected_at - total_started) * 1000.0,
            'warmup_ms': (warmed_at - connected_at) * 1000.0,
            'barrier_wait_ms': (query_started - warmed_at) * 1000.0,
            'total_ms': (query_finished - total_started) * 1000.0,
            'rows': int(row[0]) if row else 0,
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(query_once, i) for i in range(concurrency)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({'ok': False, 'error': str(exc)})
    return sorted(results, key=lambda item: item.get('worker', -1))


def summarize_concurrent_queries(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get('ok')]

    def metric(name: str) -> dict[str, float] | None:
        values = [float(row[name]) for row in successful]
        return summarize_ms(values) if values else None

    return {
        'workers': len(rows),
        'successes': len(successful),
        'failures': len(rows) - len(successful),
        'query_ms': metric('query_ms'),
        'connection_ms': metric('connection_ms'),
        'warmup_ms': metric('warmup_ms'),
        'barrier_wait_ms': metric('barrier_wait_ms'),
        'total_ms': metric('total_ms'),
    }


def summarize_source_lookup_plan(plan: str) -> dict[str, bool]:
    return {
        'tid_scan': 'Tid Scan on bench.docs' in plan,
        'sequential_scan': 'Seq Scan on bench.docs' in plan,
    }


def validate_benchmark_results(
    results: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    bm25 = results.get('bm25')
    if not isinstance(bm25, dict):
        errors.append('BM25 benchmark results are missing')
    else:
        require_tid_lookup = int(bm25.get('docs', 0)) >= 10_000
        for name in ('body_source_lookup', 'field_source_lookup'):
            lookup = bm25.get(name)
            if not isinstance(lookup, dict):
                errors.append(f'{name} evidence is missing')
                continue
            if require_tid_lookup and not lookup.get('tid_scan'):
                errors.append(f'{name} did not use TID source lookup')
            if require_tid_lookup and lookup.get('sequential_scan'):
                errors.append(f'{name} scanned the complete source relation')

    semantic = results.get('semantic_enabled')
    if semantic is not None:
        status = semantic.get('status', {})
        audit = semantic.get('audit', {})
        if not status.get('query_ready'):
            errors.append('semantic index is not query ready')
        if not audit.get('model_artifacts_valid'):
            errors.append('semantic model artifacts are invalid')
        if not audit.get('passed'):
            errors.append('semantic index deep audit failed')
        if not status.get('runtime_signature_matches'):
            errors.append('semantic runtime signature does not match')

        runtime_phase = semantic.get('index_runtime_phase', {})
        if int(runtime_phase.get('failures', 0)) != 0:
            errors.append('semantic index build reported runtime failures')
        if int(runtime_phase.get('encoded_texts', 0)) != int(
            semantic.get('docs', 0)
        ):
            errors.append('semantic index build did not encode every document')
        if (
            int(semantic.get('docs', 0)) > 1
            and runtime_phase.get('single_request_execution')
        ):
            errors.append('semantic index build did not use batch encoding')

        batch = semantic.get('batch_encoding_diagnostic', {})
        if int(batch.get('parity_mismatches', -1)) != 0:
            errors.append('batch encoding does not match single-text output')

        require_tid_lookup = int(semantic.get('docs', 0)) >= 10_000
        lookup = semantic.get('source_lookup')
        if not isinstance(lookup, dict):
            errors.append('semantic source lookup evidence is missing')
        elif require_tid_lookup:
            if not lookup.get('tid_scan'):
                errors.append('semantic source join did not use TID lookup')
            if lookup.get('sequential_scan'):
                errors.append(
                    'semantic source join scanned the complete relation'
                )

        concurrency = results.get('semantic_concurrency_summary', {})
        if int(concurrency.get('workers', 0)) <= 0:
            errors.append('semantic concurrency evidence is missing')
        if int(concurrency.get('failures', -1)) != 0:
            errors.append('semantic concurrent queries reported failures')
        if int(concurrency.get('successes', 0)) != int(
            concurrency.get('workers', -1)
        ):
            errors.append('not every semantic concurrent query succeeded')

        runtime_after = results.get('runtime_after_concurrency', {})
        if not runtime_after.get('worker_ready'):
            errors.append('runtime worker is not ready after concurrency')
        if int(runtime_after.get('queue_depth', -1)) != 0:
            errors.append('runtime queue did not drain after concurrency')
        for counter in (
            'failures',
            'busy_rejections',
            'orphan_responses',
            'canceled_requests',
        ):
            if int(runtime_after.get(counter, 0)) != 0:
                errors.append(
                    f'runtime reported nonzero {counter} after concurrency'
                )

        accelerator = results.get('semantic_accelerator')
        if accelerator is not None:
            full_cluster = accelerator.get('routes', {}).get(
                'accelerator_h0'
            )
            if full_cluster is None:
                errors.append(
                    'semantic accelerator exact-route evidence is missing'
                )
            elif not full_cluster.get('all_results_exact'):
                errors.append(
                    'semantic accelerator full-cluster route is not exact'
                )

    return {
        'passed': not errors,
        'errors': errors,
    }


def cleanup_product_cluster(
    root: Path,
    pg_bin: Path,
    data_dir: Path,
    *,
    started: bool,
    keep_pg: bool,
) -> None:
    if keep_pg:
        return
    if started:
        stopped = run(
            [pg_bin / 'pg_ctl', '-D', data_dir, 'stop', '-m', 'fast'],
            check=False,
        )
        if stopped.returncode != 0:
            print(
                'benchmark PostgreSQL did not stop; preserving workdir: '
                f'{root}',
                file=sys.stderr,
            )
            return
    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    if not args.skip_semantic and args.model_path is None:
        raise ValueError(
            '--model-path is required unless --skip-semantic is used'
        )
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
        manifest_path = args.model_path / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {manifest_path}'
            )
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                '--model-path must use the current Evoke model contract'
            )
    if not 0 <= args.onnxruntime_intra_op_threads <= 256:
        raise ValueError(
            '--onnxruntime-intra-op-threads must be between 0 and 256'
        )
    if args.query_limit <= 0:
        raise ValueError('--query-limit must be positive')
    if args.accelerator_candidate_multiplier <= 0:
        raise ValueError(
            '--accelerator-candidate-multiplier must be positive'
        )
    if args.query_jsonl is not None:
        args.query_jsonl = args.query_jsonl.expanduser().resolve()
        if not args.query_jsonl.is_file():
            raise FileNotFoundError(
                f'query JSONL was not found: {args.query_jsonl}'
            )
        if args.accelerator_heap_factors is None:
            args.accelerator_heap_factors = [0.0, 0.7, 1.0]
        if 0.0 not in args.accelerator_heap_factors:
            args.accelerator_heap_factors.insert(0, 0.0)
        if any(
            factor < 0.0 or factor > 1.0
            for factor in args.accelerator_heap_factors
        ):
            raise ValueError(
                '--accelerator-heap-factor must be between zero and one'
            )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
        extension_libraries = [
            args.extension_libdir / name
            for name in ('ii42.so', 'ii42.dylib')
        ]
        if not any(path.is_file() for path in extension_libraries):
            raise FileNotFoundError(
                'ii42 extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='ii42_product_pg_', dir=args.work_dir))
    data_dir = root / 'data'
    socket_dir = root / 'socket'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
    started = False
    results: dict[str, Any] = {
        'workdir': str(root),
        'workdir_retained': args.keep_pg,
        'onnxruntime_intra_op_threads': (
            args.onnxruntime_intra_op_threads
        ),
        'extension_libdir': (
            str(args.extension_libdir)
            if args.extension_libdir is not None
            else None
        ),
        'extension_control_dir': (
            str(args.extension_control_dir)
            if args.extension_control_dir is not None
            else None
        ),
    }

    print(f'workdir={root}', flush=True)
    run([args.pg_bin / 'initdb', '-D', data_dir, '-A', 'trust'])
    configure_postgres(data_dir, args)
    try:
        run([
            args.pg_bin / 'pg_ctl',
            '-D',
            data_dir,
            '-l',
            log_path,
            '-o',
            f'-k {socket_dir} -p {args.port}',
            'start',
            '-w',
        ])
        started = True
        dsn = f'host={socket_dir} port={args.port} dbname=postgres'
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                timed(cur, 'create extension', 'CREATE EXTENSION ii42')
                cur.execute('SELECT ii42_onnxruntime_build_info()')
                results['onnx_build_info'] = cur.fetchone()[0]
                if (
                    not args.skip_semantic
                    and not results['onnx_build_info'].startswith('enabled:')
                ):
                    raise RuntimeError(
                        'semantic-enabled product benchmark requires an '
                        'ONNX-enabled ii42 build; rebuild with '
                        'II42_ENABLE_ONNXRUNTIME=1 or pass --skip-semantic for '
                        'the BM25-only benchmark surface'
                    )
                cur.execute('SELECT ii42_runtime_service_status()')
                results['runtime_initial'] = cur.fetchone()[0]
                results['bm25'] = setup_bm25(
                    cur,
                    docs=args.docs,
                    repeats=args.repeats,
                    corpus_jsonl=args.corpus_jsonl,
                    corpus_limit=args.corpus_limit,
                )
                if not args.skip_semantic:
                    results['semantic_enabled'] = setup_semantic(
                        cur,
                        model_path=args.model_path,
                        repeats=args.repeats,
                    )
                    if args.query_jsonl is not None:
                        results['semantic_accelerator'] = (
                            benchmark_semantic_accelerator(
                                cur,
                                query_path=args.query_jsonl,
                                query_limit=args.query_limit,
                                repeats=args.repeats,
                                heap_factors=(
                                    args.accelerator_heap_factors
                                ),
                                candidate_multiplier=(
                                    args.accelerator_candidate_multiplier
                                ),
                                include_bounded_residual=(
                                    args.accelerator_bound_residual_candidates
                                ),
                                include_accumulated_residual=(
                                    args.accelerator_accumulate_residual_candidates
                                ),
                                include_summarized_residual=(
                                    args.accelerator_summarize_residual_candidates
                                ),
                                qrels=(
                                    load_qrels(args.qrels_tsv)
                                    if args.qrels_tsv is not None
                                    else None
                                ),
                                telemetry_query_limit=(
                                    args.accelerator_telemetry_query_limit
                                ),
                                summary_multiplier=(
                                    args.accelerator_summary_multiplier
                                ),
                            )
                        )
        if not args.skip_semantic:
            concurrency_rows = run_concurrent_semantic_queries(
                dsn,
                args.concurrency,
            )
            results['semantic_concurrency'] = concurrency_rows
            results['semantic_concurrency_summary'] = (
                summarize_concurrent_queries(concurrency_rows)
            )
            with psycopg.connect(dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT ii42_runtime_service_status()')
                    results['runtime_after_concurrency'] = cur.fetchone()[0]
        results['validation'] = validate_benchmark_results(results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, default=str) + '\n',
            encoding='utf-8',
        )
        print(f'results={args.output}', flush=True)
        if not results['validation']['passed']:
            for error in results['validation']['errors']:
                print(f'benchmark validation failed: {error}', file=sys.stderr)
            return 1
        return 0
    except Exception:
        if log_path.exists():
            print(log_path.read_text(encoding='utf-8')[-12000:], file=sys.stderr)
        raise
    finally:
        cleanup_product_cluster(
            root,
            args.pg_bin,
            data_dir,
            started=started,
            keep_pg=args.keep_pg,
        )


if __name__ == '__main__':
    raise SystemExit(main())
