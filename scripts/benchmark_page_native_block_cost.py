#!/usr/bin/env python3
"""Measure exact block-max selectivity on a page-native Evoke index."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql


DEFAULT_QUERIES = (
    'machine learning',
    'cancer immunotherapy',
    'COVID-19 vaccine efficacy',
    'CRISPR gene editing',
    'randomized controlled trial',
    'protein structure prediction',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--encode-dsn')
    parser.add_argument('--encode-index')
    parser.add_argument('--probe-library', default='$libdir/evoke')
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def bind_probe(cursor: psycopg.Cursor[Any], probe_library: str) -> None:
    cursor.execute(
        sql.SQL(
            """
        CREATE FUNCTION pg_temp.evoke_query_block_cost(
            regclass,
            int4[],
            real[],
            int4
        ) RETURNS jsonb
        AS {}, 'evoke_test_query_block_cost'
        LANGUAGE C STRICT
        """
        ).format(sql.Literal(probe_library))
    )


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

    index_lexical_dims = lexical_dims * field_count
    total_dims = lexical_dims + semantic_dims
    query_ids: list[int] = []
    query_weights: list[float] = []
    for atom_value, weight_value in zip(atoms, weights, strict=True):
        atom = int(atom_value)
        weight = float(weight_value)
        if atom < 0 or atom >= total_dims:
            raise ValueError(f'encoder atom {atom} is outside the contract')
        for field_index in range(field_count):
            if atom < lexical_dims:
                query_id = field_index * lexical_dims + atom
            else:
                query_id = (
                    index_lexical_dims
                    + field_index * semantic_dims
                    + atom
                    - lexical_dims
                )
            query_ids.append(query_id)
            query_weights.append(weight)
    ordered = sorted(zip(query_ids, query_weights), key=lambda pair: pair[0])
    return (
        [query_id for query_id, _ in ordered],
        [weight for _, weight in ordered],
    )


def audit_query(
    audit_cursor: psycopg.Cursor[Any],
    encode_cursor: psycopg.Cursor[Any],
    index_name: str,
    encode_index_name: str,
    query_text: str,
    lexical_dims: int,
    semantic_dims: int,
    field_count: int,
    k: int,
) -> dict[str, Any]:
    encode_cursor.execute(
        'SELECT evoke_encode_text_internal(%s::regclass, %s)',
        (encode_index_name, query_text),
    )
    encoded = encode_cursor.fetchone()[0]
    query_ids, query_weights = expand_query(
        encoded,
        lexical_dims,
        semantic_dims,
        field_count,
    )
    started = time.perf_counter()
    audit_cursor.execute(
        """
        SELECT pg_temp.evoke_query_block_cost(
            %s::regclass,
            %s::int4[],
            %s::real[],
            %s
        )
        """,
        (index_name, query_ids, query_weights, k),
    )
    audit = audit_cursor.fetchone()[0]
    audit['elapsed_ms'] = (time.perf_counter() - started) * 1000.0
    audit['query_text'] = query_text
    audit['encoded_atom_count'] = len(encoded['atoms'])
    return audit


def main() -> int:
    args = parse_args()
    if args.lexical_dims <= 0 or args.semantic_dims <= 0:
        raise SystemExit('dimension counts must be positive')
    if args.field_count <= 0 or args.k <= 0:
        raise SystemExit('field count and k must be positive')

    queries = tuple(args.queries or DEFAULT_QUERIES)
    result: dict[str, Any] = {
        'index_name': args.index,
        'encode_index_name': args.encode_index or args.index,
        'lexical_dims': args.lexical_dims,
        'semantic_dims': args.semantic_dims,
        'field_count': args.field_count,
        'k': args.k,
        'queries': [],
    }
    with ExitStack() as stack:
        audit_connection = stack.enter_context(
            psycopg.connect(args.dsn, autocommit=True)
        )
        if args.encode_dsn is None:
            encode_connection = audit_connection
        else:
            encode_connection = stack.enter_context(
                psycopg.connect(args.encode_dsn, autocommit=True)
            )
        with audit_connection.cursor() as audit_cursor:
            bind_probe(audit_cursor, args.probe_library)
            with encode_connection.cursor() as encode_cursor:
                encode_index_name = args.encode_index or args.index
                for query_text in queries:
                    audit = audit_query(
                        audit_cursor,
                        encode_cursor,
                        args.index,
                        encode_index_name,
                        query_text,
                        args.lexical_dims,
                        args.semantic_dims,
                        args.field_count,
                        args.k,
                    )
                    result['queries'].append(audit)
                    level_16 = next(
                        level
                        for level in audit['levels']
                        if int(level['block_size']) == 16
                    )
                    print(
                        f'{query_text}: '
                        f'postings={int(audit["query_postings"]):,} '
                        f'b16={float(level_16["posting_fraction"]):.4%} '
                        f'bounds={int(level_16["bound_bytes"]):,}B '
                        f'audit={float(audit["elapsed_ms"]):.1f}ms',
                        flush=True,
                    )

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output + '\n', encoding='utf-8')
    else:
        print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
