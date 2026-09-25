#!/usr/bin/env python3
"""Evaluate a Seismic-style accelerator over Evoke semantic postings."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg import sql

from benchmark_semantic_error_budget import load_queries, percentile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--source-table', required=True)
    parser.add_argument('--text-column', default='text_content')
    parser.add_argument('--queries-jsonl', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--doc-limit', type=int, default=0)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--candidate-multiplier', type=int, default=4)
    parser.add_argument('--n-postings', type=int, default=3500)
    parser.add_argument('--centroid-fraction', type=float, default=0.1)
    parser.add_argument('--summary-energy', type=float, default=0.4)
    parser.add_argument('--doc-cut', type=int, default=15)
    parser.add_argument('--query-cut', type=int, action='append')
    parser.add_argument('--heap-factor', type=float, action='append')
    parser.add_argument('--reuse-documents', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def qualified_identifier(name: str) -> sql.Identifier:
    parts = name.split('.')
    if not parts or any(not part for part in parts):
        raise ValueError(f'invalid qualified identifier: {name!r}')
    return sql.Identifier(*parts)


def write_document_vectors(args: argparse.Namespace) -> tuple[Path, int]:
    output_path = args.run_dir / 'documents.jsonl'
    if args.reuse_documents and output_path.is_file():
        with output_path.open('r', encoding='utf-8') as source:
            return output_path, sum(1 for _line in source)

    temporary_path = output_path.with_suffix('.jsonl.tmp')
    query = sql.SQL('SELECT {} FROM {} ORDER BY ctid').format(
        sql.Identifier(args.text_column),
        qualified_identifier(args.source_table),
    )
    if args.doc_limit > 0:
        query += sql.SQL(' LIMIT {}').format(sql.Literal(args.doc_limit))

    document_count = 0
    started = time.perf_counter()
    with (
        psycopg.connect(args.dsn) as source_connection,
        psycopg.connect(args.dsn, autocommit=True) as encode_connection,
        temporary_path.open('w', encoding='utf-8') as output,
    ):
        with source_connection.cursor(name='evoke_seismic_export') as source:
            source.itersize = args.batch_size
            source.execute(query)
            while True:
                rows = source.fetchmany(args.batch_size)
                if not rows:
                    break
                texts = [str(row[0] or '') for row in rows]
                with encode_connection.cursor() as encoder:
                    encoder.execute(
                        """
                        SELECT row_ordinal, atom_ids, atom_weights
                        FROM evoke_encode_document_batch_internal(
                            %s::regclass,
                            %s::text[]
                        )
                        ORDER BY row_ordinal
                        """,
                        (args.index, texts),
                    )
                    encoded_rows = encoder.fetchall()
                if len(encoded_rows) != len(rows):
                    raise RuntimeError(
                        'document encoder returned an incomplete batch'
                    )
                for offset, atom_ids, atom_weights in encoded_rows:
                    expected_offset = document_count % args.batch_size + 1
                    if int(offset) != expected_offset:
                        raise RuntimeError(
                            'document encoder changed batch row order'
                        )
                    vector = {
                        str(int(atom_id)): float(weight)
                        for atom_id, weight in zip(
                            atom_ids,
                            atom_weights,
                            strict=True,
                        )
                    }
                    output.write(
                        json.dumps(
                            {'id': document_count, 'vector': vector},
                            separators=(',', ':'),
                        )
                        + '\n'
                    )
                    document_count += 1
                if document_count % 1024 < args.batch_size:
                    elapsed = time.perf_counter() - started
                    rate = document_count / max(elapsed, 1e-9)
                    print(
                        json.dumps(
                            {
                                'phase': 'export',
                                'documents': document_count,
                                'documents_per_second': rate,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    os.replace(temporary_path, output_path)
    return output_path, document_count


def encode_semantic_queries(
    args: argparse.Namespace,
    raw_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            for query in raw_queries:
                cursor.execute(
                    'SELECT evoke_encode_text_internal(%s::regclass, %s)',
                    (args.index, query['text']),
                )
                payload = cursor.fetchone()[0]
                pairs = [
                    (int(atom_id), float(weight))
                    for atom_id, weight in zip(
                        payload['atoms'],
                        payload['weights'],
                        strict=True,
                    )
                    if int(atom_id) >= args.lexical_dims
                ]
                encoded.append(
                    {
                        'id': str(query['id']),
                        'ids': [atom_id for atom_id, _weight in pairs],
                        'weights': [weight for _atom_id, weight in pairs],
                        'runtime_signature': payload['runtime_signature'],
                    }
                )
    return encoded


def exact_rankings(
    args: argparse.Namespace,
    encoded_queries: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, float]]:
    rankings: dict[str, list[int]] = {}
    latencies: list[float] = []
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            for query in encoded_queries:
                started = time.perf_counter()
                cursor.execute(
                    """
                    SELECT doc_ord
                    FROM evoke_index_semantic_query_native_internal(
                        %s::regclass,
                        %s::int4[],
                        %s::real[],
                        NULL,
                        NULL,
                        %s,
                        %s
                    )
                    ORDER BY rank
                    """,
                    (
                        args.index,
                        query['ids'],
                        query['weights'],
                        args.k,
                        query['runtime_signature'],
                    ),
                )
                rankings[query['id']] = [int(row[0]) for row in cursor]
                latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
    return rankings, {
        'p50': statistics.median(latencies),
        'p95': percentile(latencies, 0.95),
    }


def build_index(args: argparse.Namespace, documents_path: Path) -> Any:
    from seismic import SeismicIndex

    started = time.perf_counter()
    index = SeismicIndex.build(
        str(documents_path),
        n_postings=args.n_postings,
        centroid_fraction=args.centroid_fraction,
        summary_energy=args.summary_energy,
        doc_cut=args.doc_cut,
        load_content=False,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                'phase': 'build',
                'seconds': elapsed,
                'documents': index.len,
                'nnz': index.nnz,
                'dimensions': index.dim,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return index, elapsed


def evaluate_variant(
    args: argparse.Namespace,
    index: Any,
    encoded_queries: list[dict[str, Any]],
    exact: dict[str, list[int]],
    query_cut: int,
    heap_factor: float,
) -> dict[str, Any]:
    from seismic import get_seismic_string

    search_k = args.k * args.candidate_multiplier
    string_type = get_seismic_string()
    latencies: list[float] = []
    topk_overlaps: list[float] = []
    candidate_coverages: list[float] = []
    full_coverage_queries = 0
    for query in encoded_queries:
        components = np.asarray(
            [str(atom_id) for atom_id in query['ids']],
            dtype=string_type,
        )
        values = np.asarray(query['weights'], dtype=np.float32)
        started = time.perf_counter()
        hits = index.search(
            query_id=query['id'],
            query_components=components,
            query_values=values,
            k=search_k,
            query_cut=min(query_cut, len(components)),
            heap_factor=heap_factor,
            sorted=True,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        document_ids = [int(hit[2]) for hit in hits]
        exact_set = set(exact[query['id']])
        topk_set = set(document_ids[: args.k])
        candidate_set = set(document_ids[:search_k])
        topk_overlaps.append(
            len(topk_set & exact_set) / max(1, len(exact_set))
        )
        coverage = len(candidate_set & exact_set) / max(1, len(exact_set))
        candidate_coverages.append(coverage)
        full_coverage_queries += coverage == 1.0
    return {
        'query_cut': query_cut,
        'heap_factor': heap_factor,
        'query_count': len(encoded_queries),
        'latency_ms': {
            'p50': statistics.median(latencies),
            'p95': percentile(latencies, 0.95),
        },
        'mean_topk_overlap': statistics.fmean(topk_overlaps),
        'candidate_k': search_k,
        'mean_exact_topk_candidate_coverage': statistics.fmean(
            candidate_coverages
        ),
        'full_coverage_queries': full_coverage_queries,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('batch size must be positive')
    if args.candidate_multiplier <= 0:
        raise ValueError('candidate multiplier must be positive')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    documents_path, document_count = write_document_vectors(args)
    raw_queries = load_queries(args.queries_jsonl, args.query_limit)
    encoded_queries = encode_semantic_queries(args, raw_queries)
    exact, exact_latency = exact_rankings(args, encoded_queries)
    index, build_seconds = build_index(args, documents_path)
    query_cuts = sorted(set(args.query_cut or [10, 16, 24]))
    heap_factors = sorted(set(args.heap_factor or [0.7, 0.8, 1.0]))
    variants: list[dict[str, Any]] = []
    for query_cut in query_cuts:
        for heap_factor in heap_factors:
            variant = evaluate_variant(
                args,
                index,
                encoded_queries,
                exact,
                query_cut,
                heap_factor,
            )
            variants.append(variant)
            print(json.dumps(variant, sort_keys=True), flush=True)
    result = {
        'index_name': args.index,
        'source_table': args.source_table,
        'document_count': document_count,
        'query_count': len(encoded_queries),
        'exact_latency_ms': exact_latency,
        'build_seconds': build_seconds,
        'build': {
            'n_postings': args.n_postings,
            'centroid_fraction': args.centroid_fraction,
            'summary_energy': args.summary_energy,
            'doc_cut': args.doc_cut,
        },
        'variants': variants,
    }
    output_path = args.output or args.run_dir / 'result.json'
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
