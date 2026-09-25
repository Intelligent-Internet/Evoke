#!/usr/bin/env python3
"""Measure fine-bound granularity and addressability over forward rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

import psycopg
from psycopg import sql


DEFAULT_QUERIES = (
    'cancer immunotherapy biomarkers',
    'graph neural network optimization',
    'protein structure prediction',
)


class FilterSpec(NamedTuple):
    name: str
    selector: Callable[[int], list[int] | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument('--query-vector-table')
    parser.add_argument('--library-path', default='$libdir/evoke')
    parser.add_argument('--filter', action='append', dest='filters')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--ceiling-only', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def bind_probe(
    cursor: psycopg.Cursor[Any],
    library_path: str,
) -> None:
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_term_suffix_ceiling(
            regclass,
            int4[]
        ) RETURNS jsonb
        AS {}, 'evoke_test_query_term_suffix_ceiling'
        LANGUAGE C STRICT
        """
        ).format(sql.Literal(library_path))
    )
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_forward_bound_cost(
            regclass,
            int4[],
            real[],
            int4[],
            int4
        ) RETURNS jsonb
        AS {}, 'evoke_test_query_forward_bound_cost'
        LANGUAGE C CALLED ON NULL INPUT
        """
        ).format(sql.Literal(library_path))
    )
    cursor.execute(
        sql.SQL(
            """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_forward_bound_status(
            regclass
        ) RETURNS jsonb
        AS {}, 'evoke_index_generation_readiness_internal_c'
        LANGUAGE C STRICT
        """
        ).format(sql.Literal(library_path))
    )


def qualified_identifier(value: str) -> sql.Composed:
    parts = value.split('.')
    if not parts or any(not part for part in parts):
        raise ValueError('query vector table must be schema-qualified')
    return sql.SQL('.').join(sql.Identifier(part) for part in parts)


def load_frozen_queries(
    cursor: psycopg.Cursor[Any],
    table: str,
) -> list[tuple[str, list[int], list[float]]]:
    cursor.execute(
        sql.SQL('SELECT atoms, weights, signature FROM {}').format(
            qualified_identifier(table)
        )
    )
    rows = cursor.fetchall()
    if not rows:
        raise ValueError('query vector table is empty')
    queries: list[tuple[str, list[int], list[float]]] = []
    for atoms, weights, signature in rows:
        query_ids = [int(value) for value in atoms]
        query_weights = [float(value) for value in weights]
        if len(query_ids) != len(query_weights):
            raise ValueError('frozen query arrays differ in length')
        queries.append((f'frozen:{signature}', query_ids, query_weights))
    return queries


def expand_query(
    encoded: dict[str, Any],
    lexical_dims: int,
    semantic_dims: int,
    field_count: int,
) -> tuple[list[int], list[float]]:
    atoms = encoded.get('atoms')
    weights = encoded.get('weights')
    if not isinstance(atoms, list) or not isinstance(weights, list):
        raise ValueError('encoder response has no atom and weight arrays')
    if len(atoms) != len(weights):
        raise ValueError('encoder atom and weight arrays differ in length')

    lexical_total = lexical_dims * field_count
    source_dims = lexical_dims + semantic_dims
    expanded: list[tuple[int, float]] = []
    for atom_value, weight_value in zip(atoms, weights, strict=True):
        atom = int(atom_value)
        weight = float(weight_value)
        if atom < 0 or atom >= source_dims:
            raise ValueError(f'encoder atom {atom} is outside the contract')
        for field_index in range(field_count):
            if atom < lexical_dims:
                term_id = field_index * lexical_dims + atom
            else:
                term_id = (
                    lexical_total
                    + field_index * semantic_dims
                    + atom
                    - lexical_dims
                )
            expanded.append((term_id, weight))

    expanded.sort(key=lambda pair: pair[0])
    merged: list[tuple[int, float]] = []
    for term_id, weight in expanded:
        if merged and merged[-1][0] == term_id:
            merged[-1] = (term_id, merged[-1][1] + weight)
        else:
            merged.append((term_id, weight))
    return (
        [term_id for term_id, _weight in merged],
        [weight for _term_id, weight in merged],
    )


def scattered_slots(document_count: int, modulus: int) -> list[int]:
    return list(range(0, document_count, modulus))


def clustered_slots(document_count: int, ratio: float) -> list[int]:
    selected = max(1, min(document_count, round(document_count * ratio)))
    return list(range(selected))


def distributed_run_slots(
    document_count: int,
    run_length: int,
    period: int,
) -> list[int]:
    slots: list[int] = []
    for first in range(0, document_count, period):
        slots.extend(range(first, min(first + run_length, document_count)))
    return slots


def filter_specs() -> tuple[FilterSpec, ...]:
    return (
        FilterSpec(
            'scattered_p006',
            lambda count: scattered_slots(count, 167),
        ),
        FilterSpec(
            'scattered_p025',
            lambda count: scattered_slots(count, 40),
        ),
        FilterSpec(
            'distributed_runs_p025',
            lambda count: distributed_run_slots(count, 8, 320),
        ),
        FilterSpec(
            'clustered_p025',
            lambda count: clustered_slots(count, 0.025),
        ),
        FilterSpec(
            'scattered_p100',
            lambda count: scattered_slots(count, 10),
        ),
        FilterSpec('unfiltered', lambda _count: None),
    )


def gate_level(level: dict[str, Any], filter_name: str) -> dict[str, Any]:
    posting_fraction = float(level['posting_fraction'])
    row_byte_fraction = float(level['row_byte_fraction'])
    metadata_bytes = int(level['projected_bound_bytes'])
    addressable_query_bytes = int(level['addressable_query_bytes'])
    selected_row_bytes = int(level['competitive_row_bytes'])
    full_row_bytes = int(level['allowed_row_bytes'])
    current_query_bytes = metadata_bytes + selected_row_bytes
    addressable_total_bytes = addressable_query_bytes + selected_row_bytes
    representation_ratio = (
        addressable_total_bytes / current_query_bytes
        if current_query_bytes > 0
        else 0.0
    )
    return {
        'topk_contained': bool(level['topk_contained']),
        'posting_reduction_at_least_50pct': posting_fraction <= 0.5,
        'row_byte_reduction_at_least_50pct': row_byte_fraction <= 0.5,
        'query_metadata_plus_rows_below_direct_rows': (
            metadata_bytes + selected_row_bytes < full_row_bytes
        ),
        'addressable_total_bytes': addressable_total_bytes,
        'addressable_representation_ratio': representation_ratio,
        'addressable_reduction_at_least_25pct': (
            representation_ratio <= 0.75
        ),
        'addressable_total_below_direct_rows': (
            addressable_total_bytes < full_row_bytes
        ),
        'promotion_gate': (
            filter_name == 'scattered_p025'
            and bool(level['topk_contained'])
            and posting_fraction <= 0.5
            and row_byte_fraction <= 0.5
            and representation_ratio <= 0.75
            and addressable_total_bytes < full_row_bytes
        ),
    }


def prefix_ceiling(audit: dict[str, Any]) -> dict[str, Any]:
    total_work = int(audit['forward_term_work_total'])
    suffix_work = int(audit['forward_term_work_suffix'])
    total_bytes = int(audit['forward_term_bytes_total'])
    suffix_bytes = int(audit['forward_term_bytes_suffix'])
    work_fraction = suffix_work / total_work if total_work > 0 else 0.0
    byte_fraction = suffix_bytes / total_bytes if total_bytes > 0 else 0.0

    return {
        'maximum_query_term': int(audit['maximum_query_term']),
        'global_suffix_work_fraction': work_fraction,
        'global_suffix_byte_fraction': byte_fraction,
        'can_reach_15pct_physical_reduction': (
            work_fraction >= 0.15 and byte_fraction >= 0.15
        ),
    }


def main() -> int:
    args = parse_args()
    if (
        args.lexical_dims <= 0
        or args.semantic_dims <= 0
        or args.field_count <= 0
        or args.k <= 0
    ):
        raise SystemExit('dimensions, field count, and k must be positive')

    query_texts = args.queries or list(DEFAULT_QUERIES)
    filters = filter_specs()
    if args.filters:
        requested_filters = set(args.filters)
        known_filters = {filter_spec.name for filter_spec in filters}
        unknown_filters = requested_filters - known_filters
        if unknown_filters:
            raise SystemExit(
                'unknown filters: ' + ', '.join(sorted(unknown_filters))
            )
        filters = tuple(
            filter_spec
            for filter_spec in filters
            if filter_spec.name in requested_filters
        )
    rows: list[dict[str, Any]] = []

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            bind_probe(cursor, args.library_path)
            cursor.execute(
                'SELECT pg_temp.evoke_forward_bound_status(%s::regclass)',
                (args.index,),
            )
            root_before = cursor.fetchone()[0]
            document_count = int(root_before['docs'])
            if args.query_vector_table:
                queries = load_frozen_queries(
                    cursor,
                    args.query_vector_table,
                )
            else:
                queries = []
                for query_text in query_texts:
                    cursor.execute(
                        'SELECT evoke_encode_text_internal('
                        '%s::regclass, %s)',
                        (args.index, query_text),
                    )
                    encoded = cursor.fetchone()[0]
                    query_ids, query_weights = expand_query(
                        encoded,
                        args.lexical_dims,
                        args.semantic_dims,
                        args.field_count,
                    )
                    queries.append((query_text, query_ids, query_weights))
            for query_text, query_ids, query_weights in queries:
                if args.ceiling_only:
                    started = time.perf_counter()
                    cursor.execute(
                        'SELECT pg_temp.evoke_term_suffix_ceiling('
                        '%s::regclass, %s::int4[])',
                        (args.index, query_ids),
                    )
                    audit = cursor.fetchone()[0]
                    rows.append(
                        {
                            'query': query_text,
                            'elapsed_ms': 1000.0 * (
                                time.perf_counter() - started
                            ),
                            'audit': {
                                **audit,
                                'prefix_ceiling': prefix_ceiling(audit),
                            },
                        }
                    )
                    continue
                for filter_spec in filters:
                    slots = filter_spec.selector(document_count)
                    started = time.perf_counter()
                    cursor.execute(
                        'SELECT pg_temp.evoke_forward_bound_cost('
                        '%s::regclass, %s::int4[], %s::real[], '
                        '%s::int4[], %s)',
                        (
                            args.index,
                            query_ids,
                            query_weights,
                            slots,
                            args.k,
                        ),
                    )
                    audit = cursor.fetchone()[0]
                    elapsed_ms = 1000.0 * (time.perf_counter() - started)
                    levels = []
                    for raw_level in audit['levels']:
                        level = dict(raw_level)
                        forward_bytes = int(audit['forward_object_bytes'])
                        level['published_bound_fraction_of_forward'] = (
                            int(level['published_bound_bytes'])
                            / forward_bytes
                            if forward_bytes > 0
                            else 0.0
                        )
                        level['gate'] = gate_level(level, filter_spec.name)
                        levels.append(level)
                    rows.append(
                        {
                            'query': query_text,
                            'filter': filter_spec.name,
                            'elapsed_ms': elapsed_ms,
                            'audit': {
                                **audit,
                                'prefix_ceiling': prefix_ceiling(audit),
                                'levels': levels,
                            },
                        }
                    )
            cursor.execute(
                'SELECT pg_temp.evoke_forward_bound_status(%s::regclass)',
                (args.index,),
            )
            root_after = cursor.fetchone()[0]

    result = {
        'schema': (
            'evoke_term_suffix_ceiling_v1'
            if args.ceiling_only
            else 'evoke_forward_bound_projection_oracle_v2'
        ),
        'index': args.index,
        'document_count': document_count,
        'lexical_dims': args.lexical_dims,
        'semantic_dims': args.semantic_dims,
        'field_count': args.field_count,
        'k': args.k,
        'root_before': root_before,
        'root_after': root_after,
        'root_stable': root_before == root_after,
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
