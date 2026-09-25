#!/usr/bin/env python3
"""Benchmark Seismic against exact P2 sparse-vector matrix products."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding='utf-8') as source:
        raw = json.load(source)
    return {
        str(query_id): {
            str(document_id): float(relevance)
            for document_id, relevance in documents.items()
        }
        for query_id, documents in raw.items()
    }


def query_metrics(
    ranking: list[tuple[str, float]],
    qrels: dict[str, float],
) -> dict[str, float]:
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0.0}
    ranked_ids = [doc_id for doc_id, _score in ranking]
    recall_hits = sum(doc_id in relevant for doc_id in ranked_ids[:100])
    precision_sum = 0.0
    ap_hits = 0
    reciprocal_rank = 0.0
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids[:100], start=1):
        grade = qrels.get(doc_id, 0.0)
        if grade > 0.0:
            ap_hits += 1
            precision_sum += ap_hits / rank
            if rank <= 20 and reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
        if rank <= 10 and grade > 0.0:
            dcg += (2.0**grade - 1.0) / math.log2(rank + 1.0)
    ideal_grades = sorted(qrels.values(), reverse=True)[:10]
    ideal_dcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0.0
    )
    denominator = len(relevant)
    return {
        'recall_at_100': recall_hits / denominator if denominator else 0.0,
        'map_at_100': precision_sum / denominator if denominator else 0.0,
        'mrr_at_20': reciprocal_rank,
        'ndcg_at_10': dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def mean_metric(rows: list[dict[str, float]], name: str) -> float:
    return statistics.fmean(row[name] for row in rows) if rows else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--documents-npz', type=Path)
    parser.add_argument('--queries-npz', type=Path)
    parser.add_argument('--document-ids', type=Path)
    parser.add_argument('--query-ids', type=Path)
    parser.add_argument('--qrels-json', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--doc-limit', type=int, default=0)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--candidate-multiplier', type=int, action='append')
    parser.add_argument('--n-postings', type=int)
    parser.add_argument('--target-posting-mass', type=float)
    parser.add_argument('--accelerated-posting-mass', type=float)
    parser.add_argument('--minimum-n-postings', type=int, default=1000)
    parser.add_argument('--centroid-fraction', type=float, default=0.1)
    parser.add_argument('--summary-energy', type=float, default=0.4)
    parser.add_argument('--doc-cut', type=int, default=15)
    parser.add_argument('--build-knn', type=int, default=0)
    parser.add_argument('--search-knn', type=int, action='append')
    parser.add_argument(
        '--index-variant',
        choices=('standard', 'dotvbyte'),
        default='standard',
    )
    parser.add_argument('--query-cut', type=int, action='append')
    parser.add_argument('--heap-factor', type=float, action='append')
    parser.add_argument('--reuse-documents', action='store_true')
    parser.add_argument('--reuse-index', action='store_true')
    parser.add_argument('--exact-cache', type=Path)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_csr(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as payload:
        matrix = sparse.csr_matrix(
            (
                payload['data'],
                payload['indices'],
                payload['indptr'],
            ),
            shape=tuple(int(value) for value in payload['shape']),
        )
    matrix.sort_indices()
    return matrix


def load_ids(path: Path, limit: int) -> list[str]:
    with path.open(encoding='utf-8') as source:
        result = [line.rstrip('\n') for line in source if line.strip()]
    return result[:limit] if limit > 0 else result


def ids_digest(values: list[str]) -> str:
    digest = hashlib.sha256()

    for value in values:
        digest.update(value.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()

    for path in paths:
        digest.update(str(path.name).encode('utf-8'))
        digest.update(b'\0')
        digest.update(file_digest(path).encode('ascii'))
        digest.update(b'\0')
    return digest.hexdigest()


def load_dataset_root(
    root: Path,
    doc_limit: int,
    query_limit: int,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    list[str],
    list[str],
    dict[str, dict[str, float]],
    str,
    str,
]:
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    document_matrices: list[sparse.csr_matrix] = []
    document_ids: list[str] = []
    document_paths: list[Path] = []
    remaining = doc_limit

    for shard in manifest['documents']['shards']:
        if doc_limit > 0 and remaining <= 0:
            break
        matrix_path = root / shard['matrix']
        ids_path = root / shard['ids']
        matrix = load_csr(matrix_path)
        ids = load_ids(ids_path, 0)

        if matrix.shape[0] != len(ids):
            raise ValueError(f'shard matrix and ID counts differ: {matrix_path}')
        if doc_limit > 0 and matrix.shape[0] > remaining:
            matrix = matrix[:remaining].tocsr()
            ids = ids[:remaining]
        document_matrices.append(matrix)
        document_ids.extend(ids)
        document_paths.extend((matrix_path, ids_path))
        if doc_limit > 0:
            remaining -= matrix.shape[0]
    if not document_matrices:
        raise ValueError('dataset root contains no document shards')
    documents = sparse.vstack(
        document_matrices,
        format='csr',
        dtype=np.float32,
    )
    query_matrix_path = root / manifest['queries']['matrix']
    query_ids_path = root / manifest['queries']['ids']
    qrels_path = root / manifest['queries']['qrels']
    queries = load_csr(query_matrix_path)
    query_ids = load_ids(query_ids_path, query_limit)
    if query_limit > 0:
        queries = queries[:query_limit].tocsr()
    if queries.shape[0] != len(query_ids):
        raise ValueError('query matrix and ID counts differ')
    return (
        documents,
        queries,
        document_ids,
        query_ids,
        load_qrels(qrels_path),
        files_digest(document_paths),
        files_digest([query_matrix_path, query_ids_path]),
    )


def save_exact_cache(
    path: Path,
    query_ids: list[str],
    document_ids: list[str],
    documents_sha256: str,
    queries_sha256: str,
    k: int,
    rankings: dict[str, list[int]],
    stats: dict[str, Any],
) -> None:
    ordered_offsets = np.asarray(
        [rankings[query_id] for query_id in query_ids],
        dtype=np.uint32,
    )
    external = stats['external_rankings']
    ordered_scores = np.asarray(
        [
            [score for _document_id, score in external[query_id]]
            for query_id in query_ids
        ],
        dtype=np.float32,
    )
    temporary = path.with_suffix(path.suffix + '.tmp')

    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('wb') as output:
        np.savez_compressed(
            output,
            offsets=ordered_offsets,
            scores=ordered_scores,
            document_count=np.asarray([len(document_ids)], dtype=np.uint64),
            query_count=np.asarray([len(query_ids)], dtype=np.uint64),
            k=np.asarray([k], dtype=np.uint32),
            document_ids_sha256=np.asarray([ids_digest(document_ids)]),
            query_ids_sha256=np.asarray([ids_digest(query_ids)]),
            documents_sha256=np.asarray([documents_sha256]),
            queries_sha256=np.asarray([queries_sha256]),
            latency_p50=np.asarray(
                [stats['latency_ms']['p50']],
                dtype=np.float64,
            ),
            latency_p95=np.asarray(
                [stats['latency_ms']['p95']],
                dtype=np.float64,
            ),
        )
    os.replace(temporary, path)


def load_exact_cache(
    path: Path,
    query_ids: list[str],
    document_ids: list[str],
    documents_sha256: str,
    queries_sha256: str,
    k: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        if (
            int(payload['document_count'][0]) != len(document_ids)
            or int(payload['query_count'][0]) != len(query_ids)
            or int(payload['k'][0]) != k
            or str(payload['document_ids_sha256'][0])
            != ids_digest(document_ids)
            or str(payload['query_ids_sha256'][0]) != ids_digest(query_ids)
            or str(payload['documents_sha256'][0]) != documents_sha256
            or str(payload['queries_sha256'][0]) != queries_sha256
        ):
            raise ValueError('exact cache identity does not match inputs')
        offsets = np.asarray(payload['offsets'], dtype=np.uint32)
        scores = np.asarray(payload['scores'], dtype=np.float32)
        latency_p50 = float(payload['latency_p50'][0])
        latency_p95 = float(payload['latency_p95'][0])
    expected_shape = (len(query_ids), min(k, len(document_ids)))
    if offsets.shape != expected_shape or scores.shape != expected_shape:
        raise ValueError('exact cache ranking shape does not match inputs')
    rankings = {
        query_id: [int(value) for value in offsets[position]]
        for position, query_id in enumerate(query_ids)
    }
    external_rankings = {
        query_id: [
            (document_ids[int(offset)], float(score))
            for offset, score in zip(
                offsets[position],
                scores[position],
                strict=True,
            )
        ]
        for position, query_id in enumerate(query_ids)
    }
    return rankings, {
        'latency_ms': {'p50': latency_p50, 'p95': latency_p95},
        'external_rankings': external_rankings,
    }


def write_document_vectors(
    path: Path,
    documents: sparse.csr_matrix,
    reuse: bool,
) -> float:
    if reuse and path.is_file():
        return 0.0
    temporary_path = path.with_suffix('.jsonl.tmp')
    started = time.perf_counter()
    with temporary_path.open('w', encoding='utf-8') as output:
        for document_id in range(documents.shape[0]):
            start = int(documents.indptr[document_id])
            end = int(documents.indptr[document_id + 1])
            vector = {
                str(int(component)): float(value)
                for component, value in zip(
                    documents.indices[start:end],
                    documents.data[start:end],
                    strict=True,
                )
            }
            output.write(
                json.dumps(
                    {'id': document_id, 'vector': vector},
                    separators=(',', ':'),
                )
                + '\n'
            )
            if document_id > 0 and document_id % 8192 == 0:
                print(
                    json.dumps(
                        {
                            'phase': 'convert',
                            'documents': document_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    os.replace(temporary_path, path)
    return time.perf_counter() - started


def posting_cap_for_mass(
    documents: sparse.csr_matrix,
    target_mass: float,
    minimum_cap: int = 1,
) -> tuple[int, float]:
    if not 0.0 < target_mass <= 1.0:
        raise ValueError('target posting mass must be in (0, 1]')
    if minimum_cap <= 0:
        raise ValueError('minimum posting cap must be positive')
    document_frequencies = np.bincount(
        documents.indices,
        minlength=documents.shape[1],
    ).astype(np.uint64, copy=False)
    posting_count = int(document_frequencies.sum())
    if posting_count == 0:
        return minimum_cap, 1.0
    low = minimum_cap
    high = int(document_frequencies.max(initial=1))
    if low >= high:
        cap = low
        retained_mass = float(
            np.minimum(document_frequencies, cap).sum() / posting_count
        )
        return cap, retained_mass
    while low < high:
        middle = low + (high - low) // 2
        retained = int(np.minimum(document_frequencies, middle).sum())
        if retained / posting_count >= target_mass:
            high = middle
        else:
            low = middle + 1
    retained_mass = float(
        np.minimum(document_frequencies, low).sum() / posting_count
    )
    return low, retained_mass


def retained_posting_mass(
    documents: sparse.csr_matrix,
    cap: int,
) -> float:
    if cap <= 0:
        raise ValueError('n-postings must be positive')
    document_frequencies = np.bincount(
        documents.indices,
        minlength=documents.shape[1],
    ).astype(np.uint64, copy=False)
    posting_count = int(document_frequencies.sum())
    if posting_count == 0:
        return 1.0
    return float(
        np.minimum(document_frequencies, cap).sum() / posting_count
    )


def select_accelerated_terms(
    documents: sparse.csr_matrix,
    target_mass: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select the smallest deterministic high-DF term prefix."""
    if not 0.0 < target_mass <= 1.0:
        raise ValueError('accelerated posting mass must be in (0, 1]')
    document_frequencies = np.bincount(
        documents.indices,
        minlength=documents.shape[1],
    ).astype(np.uint64, copy=False)
    posting_count = int(document_frequencies.sum())
    selected = np.zeros(documents.shape[1], dtype=np.bool_)
    if posting_count == 0:
        return selected, document_frequencies, 1.0
    term_ids = np.flatnonzero(document_frequencies)
    descending_frequencies = -document_frequencies[term_ids].astype(
        np.int64,
    )
    order = np.lexsort((term_ids, descending_frequencies))
    ordered_terms = term_ids[order]
    cumulative = np.cumsum(
        document_frequencies[ordered_terms],
        dtype=np.uint64,
    )
    target_postings = math.ceil(target_mass * posting_count)
    selected_count = int(np.searchsorted(cumulative, target_postings)) + 1
    selected[ordered_terms[:selected_count]] = True
    actual_mass = float(cumulative[selected_count - 1] / posting_count)
    return selected, document_frequencies, actual_mass


def residual_document_offsets(
    documents: sparse.csc_matrix,
    components: np.ndarray,
) -> np.ndarray:
    posting_lists = [
        documents.indices[
            documents.indptr[int(component)] :
            documents.indptr[int(component) + 1]
        ]
        for component in components
    ]
    if not posting_lists:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(posting_lists))


def exact_rankings(
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    document_ids: list[str],
    query_ids: list[str],
    k: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    latencies: list[float] = []
    rankings: dict[str, list[int]] = {}
    external_rankings: dict[str, list[tuple[str, float]]] = {}
    for query_offset, query_id in enumerate(query_ids):
        started = time.perf_counter()
        product = documents @ queries.getrow(query_offset).transpose()
        scores = np.asarray(product.toarray()).reshape(-1)
        candidate_count = min(k, scores.size)
        unordered = np.argpartition(-scores, candidate_count - 1)[
            :candidate_count
        ]
        ordered = sorted(
            (int(offset) for offset in unordered),
            key=lambda offset: (-float(scores[offset]), offset),
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        rankings[query_id] = ordered
        external_rankings[query_id] = [
            (document_ids[offset], float(scores[offset]))
            for offset in ordered
        ]
    return rankings, {
        'latency_ms': {
            'p50': statistics.median(latencies),
            'p95': percentile(latencies, 0.95),
        },
        'external_rankings': external_rankings,
    }


def build_index(
    args: argparse.Namespace,
    documents_path: Path,
) -> tuple[Any, float, int]:
    from seismic import SeismicIndex, SeismicIndexDotVByte

    index_class = {
        'standard': SeismicIndex,
        'dotvbyte': SeismicIndexDotVByte,
    }[args.index_variant]

    index_base = args.run_dir / 'index'
    index_path = index_base.with_name(index_base.name + '.index.seismic')
    if args.reuse_index:
        if not index_path.is_file():
            raise FileNotFoundError(f'no reusable index: {index_path}')
        started = time.perf_counter()
        index = index_class.load(str(index_path))
        return index, time.perf_counter() - started, index_path.stat().st_size

    started = time.perf_counter()
    index = index_class.build(
        str(documents_path),
        n_postings=args.n_postings,
        centroid_fraction=args.centroid_fraction,
        summary_energy=args.summary_energy,
        doc_cut=args.doc_cut,
        load_content=False,
    )
    build_seconds = time.perf_counter() - started
    index.save(str(index_base))
    return index, build_seconds, index_path.stat().st_size


def metrics_summary(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    rows = [
        query_metrics(ranking, qrels.get(query_id, {}))
        for query_id, ranking in rankings.items()
    ]
    return {
        name: mean_metric(rows, name)
        for name in (
            'ndcg_at_10',
            'map_at_100',
            'recall_at_100',
            'mrr_at_20',
        )
    }


def evaluate_variant(
    args: argparse.Namespace,
    index: Any,
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    document_ids: list[str],
    query_ids: list[str],
    exact: dict[str, list[int]],
    qrels: dict[str, dict[str, float]],
    candidate_multiplier: int,
    query_cut: int,
    heap_factor: float,
    search_knn: int,
    accelerated_terms: np.ndarray | None = None,
    document_frequencies: np.ndarray | None = None,
    documents_csc: sparse.csc_matrix | None = None,
) -> dict[str, Any]:
    from seismic import get_seismic_string

    string_type = get_seismic_string()
    search_k = min(len(document_ids), args.k * candidate_multiplier)
    latencies: list[float] = []
    search_latencies: list[float] = []
    completion_latencies: list[float] = []
    accelerator_completion_latencies: list[float] = []
    overlaps: list[float] = []
    accelerator_coverages: list[float] = []
    coverages: list[float] = []
    full_accelerator_coverage_queries = 0
    full_coverage_queries = 0
    accelerated_work_ratios: list[float] = []
    residual_work_ratios: list[float] = []
    accelerator_candidate_counts: list[int] = []
    union_candidate_counts: list[int] = []
    external_rankings: dict[str, list[tuple[str, float]]] = {}
    accelerator_external_rankings: dict[
        str,
        list[tuple[str, float]],
    ] = {}
    for query_offset, query_id in enumerate(query_ids):
        start = int(queries.indptr[query_offset])
        end = int(queries.indptr[query_offset + 1])
        query_components = queries.indices[start:end]
        query_values = np.asarray(
            queries.data[start:end],
            dtype=np.float32,
        )
        residual_offsets = np.empty(0, dtype=np.int64)
        if accelerated_terms is None:
            accelerated_positions = np.ones(
                len(query_components),
                dtype=np.bool_,
            )
        else:
            if document_frequencies is None or documents_csc is None:
                raise ValueError('partial acceleration requires DF and CSC')
            top_count = min(query_cut, len(query_components))
            top_positions = np.argsort(-query_values, kind='stable')[:top_count]
            active_positions = np.zeros(
                len(query_components),
                dtype=np.bool_,
            )
            active_positions[top_positions] = True
            accelerated_positions = np.zeros(
                len(query_components),
                dtype=np.bool_,
            )
            accelerated_positions[top_positions] = accelerated_terms[
                query_components[top_positions]
            ]
            residual_offsets = residual_document_offsets(
                documents_csc,
                query_components[
                    active_positions & ~accelerated_positions
                ],
            )
            exact_work = int(
                document_frequencies[
                    query_components[active_positions]
                ].sum()
            )
            accelerated_work = int(
                document_frequencies[
                    query_components[accelerated_positions]
                ].sum()
            )
            accelerated_ratio = (
                accelerated_work / exact_work if exact_work else 0.0
            )
            accelerated_work_ratios.append(accelerated_ratio)
            residual_work_ratios.append(1.0 - accelerated_ratio)
        accelerated_components = query_components[accelerated_positions]
        components = np.asarray(
            [str(int(value)) for value in accelerated_components],
            dtype=string_type,
        )
        values = query_values[accelerated_positions]
        started = time.perf_counter()
        if len(components):
            search_query_cut = (
                len(components)
                if accelerated_terms is not None
                else min(query_cut, len(components))
            )
            hits = index.search(
                query_id=query_id,
                query_components=components,
                query_values=values,
                k=search_k,
                query_cut=search_query_cut,
                heap_factor=heap_factor,
                n_knn=search_knn,
                sorted=True,
            )
        else:
            hits = []
        searched = time.perf_counter()
        accelerator_offsets = np.asarray(
            list(dict.fromkeys(int(hit[2]) for hit in hits)),
            dtype=np.int64,
        )
        document_offsets = np.union1d(
            accelerator_offsets,
            residual_offsets,
        ).astype(np.int64, copy=False)
        accelerator_candidate_counts.append(len(accelerator_offsets))
        union_candidate_counts.append(len(document_offsets))
        candidate_scores = documents[document_offsets] @ (
            queries.getrow(query_offset).transpose()
        )
        exact_candidate_scores = np.asarray(
            candidate_scores.toarray()
        ).reshape(-1)
        rescored_offsets = sorted(
            range(len(document_offsets)),
            key=lambda offset: (
                -float(exact_candidate_scores[offset]),
                document_offsets[offset],
            ),
        )
        ranked_document_offsets = [
            document_offsets[offset]
            for offset in rescored_offsets
        ]
        exact_set = set(exact[query_id])
        topk_set = set(ranked_document_offsets[: args.k])
        candidate_set = set(document_offsets)
        accelerator_set = set(accelerator_offsets)
        overlaps.append(len(topk_set & exact_set) / len(exact_set))
        accelerator_coverage = len(accelerator_set & exact_set) / len(
            exact_set
        )
        coverage = len(candidate_set & exact_set) / len(exact_set)
        accelerator_coverages.append(accelerator_coverage)
        coverages.append(coverage)
        full_accelerator_coverage_queries += accelerator_coverage == 1.0
        full_coverage_queries += coverage == 1.0
        external_rankings[query_id] = [
            (
                document_ids[document_offsets[offset]],
                float(exact_candidate_scores[offset]),
            )
            for offset in rescored_offsets[: args.k]
        ]
        completed = time.perf_counter()
        accelerator_completion_started = time.perf_counter()
        accelerator_scores = documents[accelerator_offsets] @ (
            queries.getrow(query_offset).transpose()
        )
        exact_accelerator_scores = np.asarray(
            accelerator_scores.toarray()
        ).reshape(-1)
        accelerator_order = sorted(
            range(len(accelerator_offsets)),
            key=lambda offset: (
                -float(exact_accelerator_scores[offset]),
                accelerator_offsets[offset],
            ),
        )
        accelerator_external_rankings[query_id] = [
            (
                document_ids[accelerator_offsets[offset]],
                float(exact_accelerator_scores[offset]),
            )
            for offset in accelerator_order[: args.k]
        ]
        accelerator_completion_latencies.append(
            (time.perf_counter() - accelerator_completion_started) * 1000.0
        )
        search_latencies.append((searched - started) * 1000.0)
        completion_latencies.append((completed - searched) * 1000.0)
        latencies.append((completed - started) * 1000.0)
    return {
        'candidate_multiplier': candidate_multiplier,
        'query_cut': query_cut,
        'heap_factor': heap_factor,
        'search_knn': search_knn,
        'query_count': len(query_ids),
        'latency_ms': {
            'p50': statistics.median(latencies),
            'p95': percentile(latencies, 0.95),
        },
        'search_latency_ms': {
            'p50': statistics.median(search_latencies),
            'p95': percentile(search_latencies, 0.95),
        },
        'exact_completion_latency_ms': {
            'p50': statistics.median(completion_latencies),
            'p95': percentile(completion_latencies, 0.95),
        },
        'accelerator_exact_completion_latency_ms': {
            'p50': statistics.median(accelerator_completion_latencies),
            'p95': percentile(accelerator_completion_latencies, 0.95),
        },
        'mean_topk_overlap': statistics.fmean(overlaps),
        'candidate_k': search_k,
        'mean_exact_topk_candidate_coverage': statistics.fmean(coverages),
        'full_coverage_queries': full_coverage_queries,
        'mean_accelerator_exact_topk_candidate_coverage': (
            statistics.fmean(accelerator_coverages)
        ),
        'full_accelerator_coverage_queries': (
            full_accelerator_coverage_queries
        ),
        'partial_acceleration': accelerated_terms is not None,
        'mean_accelerated_posting_ratio': (
            statistics.fmean(accelerated_work_ratios)
            if accelerated_work_ratios
            else 1.0
        ),
        'mean_residual_posting_ratio': (
            statistics.fmean(residual_work_ratios)
            if residual_work_ratios
            else 0.0
        ),
        'mean_accelerator_candidates': statistics.fmean(
            accelerator_candidate_counts
        ),
        'mean_union_candidates': statistics.fmean(union_candidate_counts),
        'accelerator_exact_completion_metrics': metrics_summary(
            accelerator_external_rankings,
            qrels,
        ),
        'metrics': metrics_summary(external_rankings, qrels),
    }


def main() -> int:
    args = parse_args()
    candidate_multipliers = sorted(
        set(args.candidate_multiplier or [4])
    )
    if any(value <= 0 for value in candidate_multipliers):
        raise ValueError('candidate multiplier must be positive')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    explicit_inputs = (
        args.documents_npz,
        args.queries_npz,
        args.document_ids,
        args.query_ids,
        args.qrels_json,
    )
    if args.dataset_root is not None and any(
        path is not None for path in explicit_inputs
    ):
        raise ValueError(
            'choose dataset-root or explicit NPZ/ID/qrels inputs, not both'
        )
    if args.dataset_root is not None:
        (
            documents,
            queries,
            document_ids,
            query_ids,
            qrels,
            documents_sha256,
            queries_sha256,
        ) = load_dataset_root(
            args.dataset_root,
            args.doc_limit,
            args.query_limit,
        )
    else:
        if any(path is None for path in explicit_inputs):
            raise ValueError(
                'explicit input mode requires NPZ, ID, and qrels paths'
            )
        documents = load_csr(args.documents_npz)
        queries = load_csr(args.queries_npz)
        if args.doc_limit > 0:
            documents = documents[: args.doc_limit]
        if args.query_limit > 0:
            queries = queries[: args.query_limit]
        document_ids = load_ids(args.document_ids, args.doc_limit)
        query_ids = load_ids(args.query_ids, args.query_limit)
        qrels = load_qrels(args.qrels_json)
        documents_sha256 = file_digest(args.documents_npz)
        queries_sha256 = file_digest(args.queries_npz)
    if documents.shape[0] != len(document_ids):
        raise ValueError('document vector and ID counts differ')
    if queries.shape[0] != len(query_ids):
        raise ValueError('query vector and ID counts differ')
    if args.n_postings is not None and args.target_posting_mass is not None:
        raise ValueError(
            'choose n-postings or target-posting-mass, not both'
        )
    if args.target_posting_mass is not None:
        args.n_postings, actual_posting_mass = posting_cap_for_mass(
            documents,
            args.target_posting_mass,
            args.minimum_n_postings,
        )
    else:
        args.n_postings = args.n_postings or 3500
        actual_posting_mass = retained_posting_mass(
            documents,
            args.n_postings,
        )
    accelerated_terms = None
    document_frequencies = None
    documents_csc = None
    accelerated_posting_mass = None
    if args.accelerated_posting_mass is not None:
        (
            accelerated_terms,
            document_frequencies,
            accelerated_posting_mass,
        ) = select_accelerated_terms(
            documents,
            args.accelerated_posting_mass,
        )
        documents_csc = documents.tocsc()
    documents_path = args.run_dir / 'documents.jsonl'
    conversion_seconds = write_document_vectors(
        documents_path,
        documents,
        args.reuse_documents,
    )
    if args.exact_cache is not None and args.exact_cache.is_file():
        exact, exact_stats = load_exact_cache(
            args.exact_cache,
            query_ids,
            document_ids,
            documents_sha256,
            queries_sha256,
            args.k,
        )
    else:
        exact, exact_stats = exact_rankings(
            documents,
            queries,
            document_ids,
            query_ids,
            args.k,
        )
        if args.exact_cache is not None:
            save_exact_cache(
                args.exact_cache,
                query_ids,
                document_ids,
                documents_sha256,
                queries_sha256,
                args.k,
                exact,
                exact_stats,
            )
    exact_metrics = metrics_summary(
        exact_stats.pop('external_rankings'),
        qrels,
    )
    index, build_seconds, index_bytes = build_index(args, documents_path)
    knn_build_seconds = 0.0
    knn_bytes = 0
    if args.build_knn > 0:
        started = time.perf_counter()
        index.build_knn(args.build_knn)
        knn_build_seconds = time.perf_counter() - started
        knn_base = args.run_dir / 'index.knn'
        index.save_knn(str(knn_base))
        knn_path = knn_base.with_name(knn_base.name + '.knn.seismic')
        knn_bytes = knn_path.stat().st_size
    variants: list[dict[str, Any]] = []
    for candidate_multiplier in candidate_multipliers:
        for query_cut in sorted(set(args.query_cut or [10, 16, 24])):
            for heap_factor in sorted(
                set(args.heap_factor or [0.7, 0.8, 1.0])
            ):
                for search_knn in sorted(set(args.search_knn or [0])):
                    if search_knn > args.build_knn:
                        raise ValueError(
                            'search KNN exceeds the built graph degree'
                        )
                    variant = evaluate_variant(
                        args,
                        index,
                        documents,
                        queries,
                        document_ids,
                        query_ids,
                        exact,
                        qrels,
                        candidate_multiplier,
                        query_cut,
                        heap_factor,
                        search_knn,
                        accelerated_terms,
                        document_frequencies,
                        documents_csc,
                    )
                    variants.append(variant)
                    print(json.dumps(variant, sort_keys=True), flush=True)
    result = {
        'schema': 'evoke_seismic_npz_oracle_v2',
        'dataset_root': (
            str(args.dataset_root) if args.dataset_root is not None else None
        ),
        'documents_npz': (
            str(args.documents_npz)
            if args.documents_npz is not None
            else None
        ),
        'queries_npz': (
            str(args.queries_npz)
            if args.queries_npz is not None
            else None
        ),
        'document_count': documents.shape[0],
        'query_count': queries.shape[0],
        'conversion_seconds': conversion_seconds,
        'exact': {**exact_stats, 'metrics': exact_metrics},
        'build_seconds': build_seconds,
        'index_bytes': index_bytes,
        'knn_build_seconds': knn_build_seconds,
        'knn_bytes': knn_bytes,
        'build': {
            'index_variant': args.index_variant,
            'n_postings': args.n_postings,
            'target_posting_mass': args.target_posting_mass,
            'minimum_n_postings': args.minimum_n_postings,
            'actual_posting_mass': actual_posting_mass,
            'centroid_fraction': args.centroid_fraction,
            'summary_energy': args.summary_energy,
            'doc_cut': args.doc_cut,
            'build_knn': args.build_knn,
            'accelerated_posting_mass_target': (
                args.accelerated_posting_mass
            ),
            'accelerated_posting_mass_actual': accelerated_posting_mass,
            'accelerated_term_count': (
                int(accelerated_terms.sum())
                if accelerated_terms is not None
                else None
            ),
        },
        'variants': variants,
    }
    output = args.output or args.run_dir / 'result.json'
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
