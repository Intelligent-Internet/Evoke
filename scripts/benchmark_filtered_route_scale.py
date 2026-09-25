#!/usr/bin/env python3
"""Compare exact filtered query routes across small native Evoke roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import psycopg
from psycopg import sql


DEFAULT_QUERIES = (
    'cancer immunotherapy biomarkers',
    'graph neural network optimization',
    'protein structure prediction',
)
FORWARD_WORKING_SET_LIMIT = 64 * 1024 * 1024

TRACE_COUNTERS = (
    'accelerator_forward_chunk_reads',
    'accelerator_forward_bytes',
    'accelerator_forward_postings_examined',
    'accelerator_forward_row_reads',
    'filtered_forward_sparse_oracle_full_bytes',
    'filtered_forward_sparse_oracle_selected_bytes',
    'filtered_forward_sparse_oracle_ranges',
    'filtered_forward_sparse_oracle_full_pages',
    'filtered_forward_sparse_oracle_selected_pages',
    'filtered_forward_direct_work',
    'filtered_forward_transpose_work',
    'filtered_forward_direct_estimated_bytes',
    'filtered_forward_transpose_estimated_bytes',
    'filtered_forward_bound_estimated_bytes',
    'filtered_forward_transpose_planning_bytes',
    'filtered_forward_transpose_stream_bytes',
    'filtered_forward_transpose_lane_bytes',
    'filtered_forward_transpose_budget_exceeded',
    'filtered_forward_bound_bytes',
    'filtered_forward_bound_entries',
    'filtered_forward_bound_blocks_considered',
    'filtered_forward_bound_blocks_scored',
    'filtered_forward_bound_budget_exceeded',
    'filtered_forward_bound_probe_fallback',
    'accelerator_membership_bytes',
    'accelerator_owned_index_bytes',
    'accelerator_candidate_scratch_bytes',
    'accelerator_forward_scratch_bytes',
    'accelerator_residual_documents',
    'accelerator_residual_postings',
    'accelerator_residual_scratch_bytes',
    'documents_examined',
    'blocks_considered',
    'blocks_scored',
    'blocks_skipped',
    'posting_block_metadata_reads',
    'document_block_reads',
    'memory_bytes',
    'ranked_prefix_probe_attempts',
    'ranked_prefix_probe_documents_examined',
    'ranked_prefix_probe_memory_bytes',
    'ranked_prefix_probe_postings_examined',
    'ranked_prefix_probe_query_term_count',
    'ranked_prefix_probe_directory_term_count',
    'ranked_prefix_probe_matched_term_count',
    'ranked_prefix_probe_fallback',
    'semantic_accelerator_query_bytes',
    'semantic_bmp_query_bytes',
    'semantic_bmp_ref_reads',
    'semantic_bmp_query_super_ref_count',
    'semantic_bmp_filtered_sequential_scans',
    'filtered_bmp_matching_ref_count',
    'filtered_bmp_matching_super_ref_count',
    'filtered_bmp_allowed_blocks',
    'filtered_bmp_allowed_superblocks',
    'semantic_bmp_record_reads',
    'semantic_bmp_super_ref_reads',
    'semantic_bmp_postings_examined',
    'positive_document_count',
    'zero_score_documents_added',
    'zero_score_cow_objects_loaded',
    'zero_score_cow_records_examined',
    'zero_score_heap_peak',
)


class FilterSpec(NamedTuple):
    name: str
    modulus: int | None
    remainder: int = 0
    range_column: str | None = None
    range_lower: int | None = None


class Route(NamedTuple):
    name: str
    disable_accelerator: bool
    force_bmp: bool
    forward_route: str


ROUTES = (
    Route('exact_bmp', True, True, 'auto'),
    Route('auto', False, False, 'auto'),
    Route('direct', False, False, 'direct'),
    Route('transpose', False, False, 'transpose'),
    Route('hybrid', False, False, 'hybrid'),
    Route('bound', False, False, 'bound'),
)


def parse_filter(value: str) -> FilterSpec:
    if value == 'unfiltered':
        return FilterSpec(value, None)

    parts = value.split(':')
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            'filter must be unfiltered or NAME:MODULUS[:REMAINDER]'
        )
    name = parts[0].strip()
    try:
        modulus = int(parts[1])
        remainder = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'filter modulus and remainder must be integers'
        ) from error
    if not name or modulus <= 0 or not 0 <= remainder < modulus:
        raise argparse.ArgumentTypeError('invalid filter name or modulus')
    return FilterSpec(name, modulus, remainder)


def parse_range_filter(value: str) -> FilterSpec:
    parts = value.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            'range filter must be NAME:COLUMN:LOWER_BOUND'
        )
    name, column, raw_lower = (part.strip() for part in parts)
    valid_column = (
        bool(column)
        and column[0].isalpha()
        and column.replace('_', '').isalnum()
    )
    if not name or not valid_column:
        raise argparse.ArgumentTypeError('invalid range filter name or column')
    try:
        lower = int(raw_lower)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'range filter lower bound must be an integer'
        ) from error
    return FilterSpec(name, None, 0, column, lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--table', required=True)
    parser.add_argument('--id-column', default='pmid')
    parser.add_argument(
        '--field',
        action='append',
        dest='fields',
        help='Field-aware index column; omit for a scalar index.',
    )
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument(
        '--query-vector-table',
        help=(
            'Schema-qualified table containing exactly one atoms, weights, '
            'signature row. This isolates scorer qualification from the '
            'runtime encoder and therefore requires exactly one query.'
        ),
    )
    parser.add_argument(
        '--filter',
        action='append',
        dest='filters',
        type=parse_filter,
    )
    parser.add_argument(
        '--range-filter',
        action='append',
        dest='range_filters',
        type=parse_range_filter,
        help='Contiguous integer filter NAME:COLUMN:LOWER_BOUND.',
    )
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument(
        '--skip-exact',
        action='store_true',
        help=(
            'Reuse direct as the comparison reference and omit the expensive '
            'exact BMP oracle. Use only after exactness has been qualified on '
            'the same root and binary.'
        ),
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--candidate-library',
        type=Path,
        help=(
            'Load native scorer/status/trace symbols from this exact shared '
            'library instead of the installed $libdir/evoke binary.'
        ),
    )
    return parser.parse_args()


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def split_qualified_name(value: str) -> tuple[str, str]:
    parts = value.split('.')
    if len(parts) != 2 or not all(parts):
        raise ValueError('table must be a schema-qualified name')
    return parts[0], parts[1]


def configure_route(cursor: psycopg.Cursor[Any], route: Route) -> None:
    settings = {
        'evoke.test_disable_semantic_accelerator': (
            'on' if route.disable_accelerator else 'off'
        ),
        'evoke.test_force_semantic_bmp': 'on' if route.force_bmp else 'off',
        'evoke.test_disable_semantic_bmp': 'off',
        'evoke.test_filtered_forward_route': route.forward_route,
    }
    for name, value in settings.items():
        cursor.execute('SELECT set_config(%s, %s, false)', (name, value))


def bind_candidate_probes(
    cursor: psycopg.Cursor[Any],
    candidate_library: Path | None,
) -> tuple[str, str]:
    library = (
        str(candidate_library.resolve())
        if candidate_library is not None
        else '$libdir/evoke'
    )
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_filtered_route_trace()
        RETURNS jsonb
        AS {}, 'evoke_query_trace_internal'
        LANGUAGE C VOLATILE PARALLEL UNSAFE
        """
        ).format(sql.Literal(library))
    )
    if candidate_library is None:
        return (
            'evoke_index_semantic_query_native_internal',
            'evoke_index_generation_status_internal',
        )
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_filtered_route_query(
            regclass, int4[], real[], text[], real[], int4, text,
            int4[], tid[], jsonb
        )
        RETURNS TABLE(
            rank int4, ctid tid, doc_ord int4, score float8,
            selected_terms int4, candidate_docs int4, scored_docs int4,
            generator_entry_visits int8, generator_decoded_postings int8,
            candidate_postings int8, rerank_binary_steps int8,
            rerank_slice_hits int8, rerank_score_terms int8,
            rerank_doc_terms int8, memory_bytes int8
        )
        AS {}, 'evoke_index_semantic_query_native_internal'
        LANGUAGE C VOLATILE PARALLEL UNSAFE
        """
        ).format(sql.Literal(library))
    )
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_filtered_route_status(
            regclass
        )
        RETURNS jsonb
        AS {}, 'evoke_index_generation_readiness_internal_c'
        LANGUAGE C STABLE PARALLEL SAFE STRICT
        """
        ).format(sql.Literal(library))
    )
    return (
        'pg_temp.evoke_filtered_route_query',
        'pg_temp.evoke_filtered_route_status',
    )


def build_filters(
    cursor: psycopg.Cursor[Any],
    table_name: str,
    id_column: str,
    specs: Sequence[FilterSpec],
) -> dict[str, int]:
    schema_name, relation_name = split_qualified_name(table_name)
    relation = sql.Identifier(schema_name, relation_name)
    identifier = sql.Identifier(id_column)
    result: dict[str, int] = {}

    cursor.execute(
        'CREATE TEMP TABLE evoke_filtered_route_sets ('
        'name text PRIMARY KEY, tids tid[], allowed_count bigint NOT NULL'
        ') ON COMMIT PRESERVE ROWS'
    )

    for spec in specs:
        if spec.range_column is not None:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO pg_temp.evoke_filtered_route_sets '
                    'SELECT %s, array_agg(ctid ORDER BY ctid), count(*) '
                    'FROM {} WHERE {} >= %s RETURNING allowed_count'
                ).format(relation, sql.Identifier(spec.range_column)),
                (spec.name, spec.range_lower),
            )
            result[spec.name] = int(cursor.fetchone()[0])
            continue
        if spec.modulus is None:
            cursor.execute(
                sql.SQL(
                    'INSERT INTO pg_temp.evoke_filtered_route_sets '
                    'SELECT %s, NULL::tid[], count(*) FROM {} '
                    'RETURNING allowed_count'
                ).format(relation),
                (spec.name,),
            )
            result[spec.name] = int(cursor.fetchone()[0])
            continue
        cursor.execute(
            sql.SQL(
                'INSERT INTO pg_temp.evoke_filtered_route_sets '
                'SELECT %s, array_agg(ctid ORDER BY ctid), count(*) '
                'FROM {} WHERE mod({}, %s) = %s '
                'RETURNING allowed_count'
            ).format(relation, identifier),
            (spec.name, spec.modulus, spec.remainder),
        )
        result[spec.name] = int(cursor.fetchone()[0])
    return result


def encode_query(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    text: str,
    query_vector_table: str | None = None,
) -> tuple[list[int], list[float], str]:
    if query_vector_table is None:
        cursor.execute(
            'SELECT evoke_encode_text_internal(%s::regclass, %s)',
            (index_name, text),
        )
        encoded = cursor.fetchone()[0]
        return (
            [int(value) for value in encoded['atoms']],
            [float(value) for value in encoded['weights']],
            str(encoded['runtime_signature']),
        )

    schema_name, relation_name = split_qualified_name(query_vector_table)
    cursor.execute(
        sql.SQL(
            'SELECT atoms, weights, signature FROM {} LIMIT 2'
        ).format(sql.Identifier(schema_name, relation_name))
    )
    rows = cursor.fetchall()
    if len(rows) != 1:
        raise RuntimeError(
            'query vector fixture must contain exactly one row'
        )
    atoms, weights, signature = rows[0]
    return (
        [int(value) for value in atoms],
        [float(value) for value in weights],
        str(signature),
    )


def execute_query(
    cursor: psycopg.Cursor[Any],
    query_function: str,
    index_name: str,
    fields: Sequence[str] | None,
    atoms: Sequence[int],
    weights: Sequence[float],
    signature: str,
    filter_name: str,
    k: int,
) -> tuple[list[tuple[int, str]], float, dict[str, Any]]:
    started = time.perf_counter()
    cursor.execute(
        sql.SQL(
            """
        SELECT
            hit.doc_ord,
            encode(float8send(hit.score::float8), 'hex')
        FROM {}(
            %s::regclass,
            %s::int4[],
            %s::real[],
            %s::text[],
            %s::real[],
            %s,
            %s,
            NULL,
            (
                SELECT tids
                FROM pg_temp.evoke_filtered_route_sets
                WHERE name = %s
            ),
            NULL
        ) AS hit
        ORDER BY hit.rank
        """
        ).format(sql.SQL(query_function)),
        (
            index_name,
            list(atoms),
            list(weights),
            list(fields) if fields is not None else None,
            [1.0] * len(fields) if fields is not None else None,
            k,
            signature,
            filter_name,
        ),
    )
    hits = [(int(row[0]), str(row[1])) for row in cursor.fetchall()]
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    cursor.execute('SELECT pg_temp.evoke_filtered_route_trace()')
    trace = cursor.fetchone()[0]
    if not isinstance(trace, dict):
        raise RuntimeError('query trace is not a JSON object')
    return hits, elapsed_ms, trace


def summarize_attempts(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(attempt['elapsed_ms']) for attempt in attempts]
    traces = [attempt['trace'] for attempt in attempts]
    routes = sorted({str(trace.get('query_route')) for trace in traces})
    residual_strategies = sorted(
        {
            (
                'dense'
                if trace.get(
                    'semantic_accelerator_residual_dense_accumulation'
                )
                else 'merge'
            )
            for trace in traces
            if int(trace.get('accelerator_residual_postings', 0) or 0) > 0
        }
    )
    forward_modes = sorted(
        mode
        for mode, key in (
            ('direct', 'semantic_accelerator_forward_direct_rows'),
            ('transpose', 'semantic_accelerator_forward_transposed'),
            ('bound', 'semantic_accelerator_forward_bounded'),
        )
        if any(bool(trace.get(key)) for trace in traces)
    )
    counters = {
        key: max(int(trace.get(key, 0) or 0) for trace in traces)
        for key in TRACE_COUNTERS
    }
    return {
        'p50_ms': statistics.median(latencies),
        'p95_ms': percentile(latencies, 0.95),
        'minimum_ms': min(latencies),
        'maximum_ms': max(latencies),
        'samples_ms': latencies,
        'query_routes': routes,
        'forward_modes': forward_modes,
        'residual_strategies': residual_strategies,
        'counters': counters,
    }


def audit_auto_route(trace: dict[str, Any]) -> dict[str, Any]:
    query_route = str(trace.get('query_route') or '')
    if query_route not in {
        'forward_rows',
        'forward_transpose',
        'forward_bound',
    }:
        return {
            'applicable': False,
            'query_route': query_route or None,
            'passed': True,
            'reason': 'auto route did not select a forward executor',
        }

    direct_bytes = int(
        trace.get('filtered_forward_direct_estimated_bytes', 0) or 0
    )
    transpose_bytes = int(
        trace.get('filtered_forward_transpose_estimated_bytes', 0) or 0
    )
    bound_bytes = int(
        trace.get('filtered_forward_bound_estimated_bytes', 0) or 0
    )
    transpose_budget_exceeded = bool(
        trace.get('filtered_forward_transpose_budget_exceeded')
    )
    if direct_bytes <= 0 or transpose_bytes <= 0:
        return {
            'applicable': True,
            'query_route': query_route,
            'passed': False,
            'reason': 'forward route omitted physical byte estimates',
        }

    if direct_bytes <= transpose_bytes or transpose_budget_exceeded:
        expected_mode = 'direct'
    elif (
        transpose_bytes > FORWARD_WORKING_SET_LIMIT
        and 0 < bound_bytes < transpose_bytes
    ):
        expected_mode = 'bound'
    else:
        expected_mode = 'transpose'

    mode_flags = {
        'direct': bool(
            trace.get('semantic_accelerator_forward_direct_rows')
        ),
        'transpose': bool(
            trace.get('semantic_accelerator_forward_transposed')
        ),
        'bound': bool(
            trace.get('semantic_accelerator_forward_bounded')
        ),
    }
    observed_modes = [
        mode for mode, selected in mode_flags.items() if selected
    ]
    observed_mode = observed_modes[0] if len(observed_modes) == 1 else None
    expected_route = {
        'direct': 'forward_rows',
        'transpose': 'forward_transpose',
        'bound': 'forward_bound',
    }[expected_mode]
    passed = (
        observed_mode == expected_mode
        and query_route == expected_route
    )
    return {
        'applicable': True,
        'query_route': query_route,
        'expected_route': expected_route,
        'expected_mode': expected_mode,
        'observed_modes': observed_modes,
        'direct_estimated_bytes': direct_bytes,
        'transpose_estimated_bytes': transpose_bytes,
        'bound_estimated_bytes': bound_bytes,
        'transpose_budget_exceeded': transpose_budget_exceeded,
        'passed': passed,
    }


def compare_hits(
    reference: Sequence[tuple[int, str]],
    candidate: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    reference_ids = {doc_ord for doc_ord, _score in reference}
    candidate_ids = {doc_ord for doc_ord, _score in candidate}
    return {
        'rank_and_score_bits_equal': list(candidate) == list(reference),
        'identity_equal': [row[0] for row in candidate] == [
            row[0] for row in reference
        ],
        'overlap_at_k': (
            len(reference_ids & candidate_ids) / len(reference_ids)
            if reference_ids
            else 1.0
        ),
    }


def hit_digest(hits: Sequence[tuple[int, str]]) -> str:
    digest = hashlib.sha256()

    for document, score_bits in hits:
        digest.update(f'{document}:{score_bits}\n'.encode('ascii'))
    return digest.hexdigest()


def root_identity(status: dict[str, Any]) -> dict[str, Any]:
    posting = status.get('posting') or {}
    primary = status.get('primary') or {}
    accelerator = status.get('semantic_accelerator') or {}
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


def routes_for_run(skip_exact: bool) -> tuple[Route, ...]:
    return ROUTES[1:] if skip_exact else ROUTES


def main() -> int:
    args = parse_args()
    if args.k <= 0 or args.repetitions <= 0 or args.warmups < 0:
        raise SystemExit('k and repetitions must be positive; warmups nonnegative')
    filters = (args.filters or []) + (args.range_filters or [])
    if not filters:
        filters = [
            FilterSpec('p006', 161),
            FilterSpec('p100', 10),
            FilterSpec('p500', 2),
            FilterSpec('unfiltered', None),
        ]
    queries = args.queries or list(DEFAULT_QUERIES)
    if args.query_vector_table is not None and len(queries) != 1:
        raise SystemExit(
            '--query-vector-table requires exactly one --query value'
        )
    rows: list[dict[str, Any]] = []

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            query_function, status_function = bind_candidate_probes(
                cursor,
                args.candidate_library,
            )
            schema_name, relation_name = split_qualified_name(args.table)
            cursor.execute(
                sql.SQL(
                    'SELECT count(*), pg_relation_size(%s::regclass), '
                    '{}(%s::regclass) '
                    'FROM {}'
                ).format(
                    sql.SQL(status_function),
                    sql.Identifier(schema_name, relation_name),
                ),
                (args.table, args.index),
            )
            document_count, table_bytes, root_before = cursor.fetchone()
            filter_sets = build_filters(
                cursor,
                args.table,
                args.id_column,
                filters,
            )
            for query_text in queries:
                atoms, weights, signature = encode_query(
                    cursor,
                    args.index,
                    query_text,
                    args.query_vector_table,
                )
                for filter_spec in filters:
                    route_rows: dict[str, Any] = {}
                    final_hits: dict[str, list[tuple[int, str]]] = {}
                    executed_routes = routes_for_run(args.skip_exact)
                    route_attempts: dict[str, list[dict[str, Any]]] = {
                        route.name: [] for route in executed_routes
                    }
                    exact_route = ROUTES[0]
                    fast_routes = ROUTES[1:]

                    if not args.skip_exact:
                        configure_route(cursor, exact_route)
                        for _warmup in range(args.warmups):
                            execute_query(
                                cursor,
                                query_function,
                                args.index,
                                args.fields,
                                atoms,
                                weights,
                                signature,
                                filter_spec.name,
                                args.k,
                            )
                        for _attempt in range(args.repetitions):
                            hits, elapsed_ms, trace = execute_query(
                                cursor,
                                query_function,
                                args.index,
                                args.fields,
                                atoms,
                                weights,
                                signature,
                                filter_spec.name,
                                args.k,
                            )
                            final_hits[exact_route.name] = hits
                            route_attempts[exact_route.name].append(
                                {
                                    'elapsed_ms': elapsed_ms,
                                    'trace': trace,
                                }
                            )
                    for route in fast_routes:
                        configure_route(cursor, route)
                        for _warmup in range(args.warmups):
                            execute_query(
                                cursor,
                                query_function,
                                args.index,
                                args.fields,
                                atoms,
                                weights,
                                signature,
                                filter_spec.name,
                                args.k,
                            )
                        for _attempt in range(args.repetitions):
                            hits, elapsed_ms, trace = execute_query(
                                cursor,
                                query_function,
                                args.index,
                                args.fields,
                                atoms,
                                weights,
                                signature,
                                filter_spec.name,
                                args.k,
                            )
                            final_hits[route.name] = hits
                            route_attempts[route.name].append(
                                {
                                    'elapsed_ms': elapsed_ms,
                                    'trace': trace,
                                }
                            )
                    for route in executed_routes:
                        route_rows[route.name] = summarize_attempts(
                            route_attempts[route.name]
                        )
                    reference_route = (
                        'direct' if args.skip_exact else exact_route.name
                    )
                    reference = final_hits[reference_route]
                    for route in executed_routes:
                        route_rows[route.name]['exactness'] = compare_hits(
                            reference,
                            final_hits[route.name],
                        )
                        route_rows[route.name]['result_sha256'] = hit_digest(
                            final_hits[route.name]
                        )
                    auto_route_decisions = [
                        audit_auto_route(attempt['trace'])
                        for attempt in route_attempts['auto']
                    ]
                    route_rows['auto']['route_decisions'] = (
                        auto_route_decisions
                    )
                    route_equivalence = {
                        'auto_vs_direct': compare_hits(
                            final_hits['direct'],
                            final_hits['auto'],
                        ),
                        'direct_vs_transpose': compare_hits(
                            final_hits['direct'],
                            final_hits['transpose'],
                        ),
                        'direct_vs_hybrid': compare_hits(
                            final_hits['direct'],
                            final_hits['hybrid'],
                        ),
                        'direct_vs_bound': compare_hits(
                            final_hits['direct'],
                            final_hits['bound'],
                        ),
                    }
                    rows.append(
                        {
                            'query': query_text,
                            'atom_count': len(atoms),
                            'filter': filter_spec.name,
                            'allowed_documents': (
                                filter_sets[filter_spec.name]
                            ),
                            'routes': route_rows,
                            'reference_route': reference_route,
                            'route_equivalence': route_equivalence,
                        }
                    )
            cursor.execute(
                sql.SQL('SELECT {}(%s::regclass)').format(
                    sql.SQL(status_function)
                ),
                (args.index,),
            )
            root_after = cursor.fetchone()[0]

    root_identity_before = root_identity(root_before)
    root_identity_after = root_identity(root_after)
    route_decisions = [
        decision
        for row in rows
        for decision in row['routes']['auto']['route_decisions']
    ]
    applicable_route_decisions = [
        decision
        for decision in route_decisions
        if decision['applicable']
    ]

    result = {
        'schema': 'evoke_filtered_route_scale_v2',
        'index': args.index,
        'table': args.table,
        'document_count': int(document_count),
        'table_bytes': int(table_bytes),
        'k': args.k,
        'warmups': args.warmups,
        'repetitions': args.repetitions,
        'exact_bmp_executed': not args.skip_exact,
        'root_before': root_before,
        'root_after': root_after,
        'root_identity_before': root_identity_before,
        'root_identity_after': root_identity_after,
        'root_stable': root_identity_before == root_identity_after,
        'auto_route_decision': {
            'attempts': len(route_decisions),
            'applicable_attempts': len(applicable_route_decisions),
            'passed': bool(
                applicable_route_decisions
                and all(
                    decision['passed']
                    for decision in applicable_route_decisions
                )
            ),
        },
        'rows': rows,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
