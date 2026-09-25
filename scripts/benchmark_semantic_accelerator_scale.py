#!/usr/bin/env python3
"""Benchmark bounded semantic-accelerator routes on a native Evoke index."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import psycopg

from benchmark_page_native_block_cost import DEFAULT_QUERIES, expand_query


class Route(NamedTuple):
    name: str
    heap_factor: float | None
    candidate_multiplier: int
    bound_residual: bool
    accumulate_residual: bool
    summarize_residual: bool
    summary_multiplier: int
    seed_bmp: bool
    error_budget: float = 0.0
    use_session_defaults: bool = False


ROUTES = (
    Route('exact', None, 16, False, False, False, 2, False),
    Route(
        'default',
        0.7,
        64,
        True,
        True,
        False,
        2,
        False,
        use_session_defaults=True,
    ),
    Route('forced_bmp', None, 16, False, False, False, 2, False),
    Route('term_budget001', None, 16, False, False, False, 2, False),
    Route('term_budget0025', None, 16, False, False, False, 2, False),
    Route('term_budget005', None, 16, False, False, False, 2, False),
    Route('accelerator_h0', 0.0, 16, False, False, False, 2, False),
    Route('per_term', 0.7, 72, True, False, False, 2, False),
    Route('cross_term16', 0.7, 16, True, True, False, 2, False),
    Route('cross_term24', 0.7, 24, True, True, False, 2, False),
    Route('cross_term32', 0.7, 32, True, True, False, 2, False),
    Route(
        'cross_term32_seed_bmp',
        0.7,
        32,
        True,
        True,
        False,
        2,
        True,
    ),
    Route('cross_term64', 0.7, 64, True, True, False, 2, False),
    Route('cross_term', 0.7, 72, True, True, False, 2, False),
    Route('balanced', 0.7, 72, True, True, True, 2, False),
    Route('quality', 0.7, 144, True, True, True, 2, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument('--queries-jsonl', type=Path)
    parser.add_argument('--query-limit', type=int, default=20)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--telemetry-limit', type=int, default=20)
    parser.add_argument(
        '--route',
        action='append',
        choices=tuple(route.name for route in ROUTES),
        dest='routes',
        help='Route to benchmark. Repeat to select multiple routes.',
    )
    parser.add_argument(
        '--error-budget',
        action='append',
        type=float,
        dest='error_budgets',
        help=(
            'Custom semantic error-budget ratio. Repeat to benchmark '
            'multiple ratios; for example, 0.01 means 1%%.'
        ),
    )
    parser.add_argument(
        '--combine-error-budget-with-route',
        action='store_true',
        help=(
            'Clone each explicitly selected non-exact route for every '
            'error budget instead of adding standalone term-budget routes.'
        ),
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def read_queries(path: Path, limit: int) -> list[str]:
    result: list[str] = []

    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get('text') or row.get('query')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('query JSONL row has no text/query string')
            result.append(text)
            if len(result) >= limit:
                break
    return result


def bind_probe(cursor: psycopg.Cursor[Any]) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_query_topk(
            regclass,
            int4[],
            real[],
            int4,
            bool
        ) RETURNS jsonb
        AS '$libdir/evoke', 'evoke_test_query_page_native_topk'
        LANGUAGE C STRICT
        """
    )


def configure_route(cursor: psycopg.Cursor[Any], route: Route) -> None:
    error_budget = effective_error_budget(route)
    settings = {
        'evoke.test_disable_semantic_accelerator': (
            'on' if route.heap_factor is None else 'off'
        ),
        'evoke.test_semantic_accelerator_heap_factor': str(
            route.heap_factor or 0.0
        ),
        'evoke.test_semantic_accelerator_candidate_multiplier': str(
            route.candidate_multiplier
        ),
        'evoke.test_semantic_accelerator_bound_residual_candidates': (
            'on' if route.bound_residual else 'off'
        ),
        'evoke.test_semantic_accelerator_accumulate_residual_candidates': (
            'on' if route.accumulate_residual else 'off'
        ),
        'evoke.test_semantic_accelerator_summarize_residual_candidates': (
            'on' if route.summarize_residual else 'off'
        ),
        'evoke.test_semantic_accelerator_summary_multiplier': str(
            route.summary_multiplier
        ),
        'evoke.test_semantic_accelerator_seed_bmp': (
            'on' if route.seed_bmp else 'off'
        ),
        'evoke.test_force_semantic_bmp': (
            'on' if route.seed_bmp or route.name == 'forced_bmp' else 'off'
        ),
        'evoke.test_disable_semantic_bmp': (
            'on' if error_budget > 0.0 else 'off'
        ),
        'evoke.test_query_max_df_ratio': '1',
        'evoke.test_query_semantic_work_target_postings': '0',
        'evoke.test_query_semantic_error_budget_ratio': str(error_budget),
        'evoke.test_query_semantic_impact_floor_ratio': '0',
        'evoke.test_query_semantic_min_support_ratio': '0',
    }
    if route.use_session_defaults:
        for name in settings:
            cursor.execute(f'RESET {name}')
        return
    for name, value in settings.items():
        cursor.execute('SELECT set_config(%s, %s, false)', (name, value))


def effective_error_budget(route: Route) -> float:
    return route.error_budget or {
        'term_budget001': 0.001,
        'term_budget0025': 0.0025,
        'term_budget005': 0.005,
    }.get(route.name, 0.0)


def select_routes(
    selected_names: set[str],
    error_budgets: list[float] | None,
    combine_error_budget: bool,
) -> tuple[Route, ...]:
    routes = [route for route in ROUTES if route.name in selected_names]
    seen_budgets = {effective_error_budget(route) for route in routes}

    for budget in error_budgets or ():
        if not 0.0 < budget <= 1.0:
            raise ValueError('error budgets must be in the interval (0, 1]')
        label = format(budget, '.8g').replace('.', 'p')
        if combine_error_budget:
            bases = [
                route
                for route in routes
                if route.name != 'exact' and
                not route.use_session_defaults and
                effective_error_budget(route) == 0
            ]
            for base in bases:
                routes.append(
                    base._replace(
                        name=f'{base.name}_budget_{label}',
                        error_budget=budget,
                    )
                )
            continue
        if budget in seen_budgets:
            continue
        routes.append(
            Route(
                f'term_budget_{label}',
                None,
                16,
                False,
                False,
                False,
                2,
                False,
                budget,
            )
        )
        seen_budgets.add(budget)
    return tuple(routes)


def encode_query(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    text: str,
    lexical_dims: int,
    semantic_dims: int,
    field_count: int,
) -> dict[str, Any]:
    cursor.execute(
        'SELECT evoke_encode_text_internal(%s::regclass, %s)',
        (index_name, text),
    )
    encoded = cursor.fetchone()[0]
    ids, weights = expand_query(
        encoded,
        lexical_dims,
        semantic_dims,
        field_count,
    )
    return {
        'text': text,
        'ids': ids,
        'weights': weights,
        'base_atom_count': len(encoded['atoms']),
    }


def fetch_hits(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    text: str,
    k: int,
) -> tuple[list[tuple[str, float]], float]:
    started = time.perf_counter()
    cursor.execute(
        """
        SELECT hit.ctid::text, hit.score::float8
        FROM evoke_query(%s::regclass, %s, %s) AS hit
        ORDER BY hit.score DESC, hit.ctid
        """,
        (index_name, text, k),
    )
    rows = [(str(row[0]), float(row[1])) for row in cursor.fetchall()]
    return rows, (time.perf_counter() - started) * 1000.0


def fetch_telemetry(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    query: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT pg_temp.evoke_query_topk(
            %s::regclass,
            %s::int4[],
            %s::real[],
            %s,
            true
        )
        """,
        (index_name, query['ids'], query['weights'], k),
    )
    row = cursor.fetchone()[0]
    if not isinstance(row, dict):
        raise RuntimeError('page-native query probe returned no JSON object')
    row.pop('page_native_doc_ids', None)
    return row


def overlap(
    reference: list[tuple[str, float]],
    candidate: list[tuple[str, float]],
) -> float:
    reference_ids = {doc_id for doc_id, _score in reference}
    candidate_ids = {doc_id for doc_id, _score in candidate}

    if not reference_ids:
        return 1.0 if not candidate_ids else 0.0
    return len(reference_ids & candidate_ids) / len(reference_ids)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        'p50': statistics.median(values),
        'p95': percentile(values, 0.95),
        'p99': percentile(values, 0.99),
        'samples': values,
    }


def main() -> int:
    args = parse_args()
    if args.k <= 0 or args.repetitions <= 0 or args.query_limit <= 0:
        raise SystemExit('k, repetitions, and query limit must be positive')
    if args.warmups < 0:
        raise SystemExit('warmups must be nonnegative')
    if args.lexical_dims <= 0 or args.semantic_dims <= 0:
        raise SystemExit('dimension counts must be positive')
    if args.field_count <= 0:
        raise SystemExit('field count must be positive')
    if args.queries_jsonl is not None:
        texts = read_queries(args.queries_jsonl, args.query_limit)
    else:
        texts = list(args.queries or DEFAULT_QUERIES)[:args.query_limit]
    if not texts:
        raise SystemExit('no queries selected')
    if args.combine_error_budget_with_route and args.routes is None:
        raise SystemExit('combined budgets require explicit routes')
    selected_names = set(args.routes or (route.name for route in ROUTES))
    try:
        routes = select_routes(
            selected_names,
            args.error_budgets,
            args.combine_error_budget_with_route,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not routes or routes[0].name != 'exact':
        raise SystemExit('the exact route must be selected as the reference')

    route_latencies = {route.name: [] for route in routes}
    route_overlaps = {
        route.name: [] for route in routes if route.name != 'exact'
    }
    query_rows: list[dict[str, Any]] = []
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            bind_probe(cursor)
            queries = [
                encode_query(
                    cursor,
                    args.index,
                    text,
                    args.lexical_dims,
                    args.semantic_dims,
                    args.field_count,
                )
                for text in texts
            ]
            for query_index, query in enumerate(queries):
                results: dict[str, list[tuple[str, float]]] = {}
                per_route: dict[str, Any] = {}
                for _warmup in range(args.warmups):
                    for route in routes:
                        configure_route(cursor, route)
                        fetch_hits(cursor, args.index, query['text'], args.k)
                for repetition in range(args.repetitions):
                    offset = (query_index + repetition) % len(routes)
                    ordered = routes[offset:] + routes[:offset]
                    for route in ordered:
                        configure_route(cursor, route)
                        hits, elapsed = fetch_hits(
                            cursor,
                            args.index,
                            query['text'],
                            args.k,
                        )
                        results[route.name] = hits
                        route_latencies[route.name].append(elapsed)
                reference = results['exact']
                for route in routes:
                    route_overlap = (
                        1.0
                        if route.name == 'exact'
                        else overlap(reference, results[route.name])
                    )
                    if route.name != 'exact':
                        route_overlaps[route.name].append(route_overlap)
                    telemetry = None
                    if query_index < args.telemetry_limit:
                        configure_route(cursor, route)
                        telemetry = fetch_telemetry(
                            cursor,
                            args.index,
                            query,
                            args.k,
                        )
                    per_route[route.name] = {
                        'overlap_at_k': route_overlap,
                        'telemetry': telemetry,
                    }
                query_rows.append(
                    {
                        'query_text': query['text'],
                        'base_atom_count': query['base_atom_count'],
                        'routes': per_route,
                    }
                )

    result = {
        'schema': 'evoke_semantic_accelerator_scale_v2',
        'index_name': args.index,
        'k': args.k,
        'query_count': len(query_rows),
        'repetitions': args.repetitions,
        'warmups': args.warmups,
        'routes': {
            route.name: {
                'error_budget_ratio': effective_error_budget(route),
                'latency_ms': summarize(route_latencies[route.name]),
                'mean_overlap_at_k': (
                    1.0
                    if route.name == 'exact'
                    else statistics.fmean(route_overlaps[route.name])
                ),
                'minimum_overlap_at_k': (
                    1.0
                    if route.name == 'exact'
                    else min(route_overlaps[route.name])
                ),
            }
            for route in routes
        },
        'queries': query_rows,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
