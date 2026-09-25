#!/usr/bin/env python3
"""M200 tiny INT4 residual floor over a top256 correction window.

This is a no-training replay. It keeps the frozen M190+BM25 candidate source
and evaluates whether very small PCA INT4 document codes can recover useful
tail ordering when they are allowed to act only inside the baseline top256.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971


SCHEMA = 'm200_top256_tiny_int4_floor_v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate tiny INT4 correction inside baseline top256.',
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dense-documents', type=Path, required=True)
    parser.add_argument('--dense-queries', type=Path, required=True)
    parser.add_argument('--bm25-cache', type=Path, required=True)
    parser.add_argument('--qrels', type=Path, required=True)
    parser.add_argument('--m190-documents', type=Path, required=True)
    parser.add_argument('--m190-queries', type=Path, required=True)
    parser.add_argument('--m190-document-ids', type=Path, required=True)
    parser.add_argument('--m190-query-ids', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dimensions', default='16,32,64,256')
    parser.add_argument('--dense-weight', type=float, default=0.5)
    parser.add_argument('--protected-head', type=int, default=20)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in base.METRIC_NAMES
    }


def top_window_rankings(
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    correction_rows: Sequence[np.ndarray],
    *,
    weight: float,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> list[np.ndarray]:
    rankings: list[np.ndarray] = []
    for candidates, baseline, correction in zip(
        candidate_rows,
        baseline_rows,
        correction_rows,
    ):
        baseline = np.asarray(baseline, dtype=np.float32)
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        head_size = min(protected_head, top_k, len(candidates))
        window_size = min(correction_window, len(candidates))
        protected = baseline_order[:head_size]
        eligible = baseline_order[head_size:window_size]
        if eligible.size:
            corrected = baseline[eligible] + weight * m1971.minmax(
                correction[eligible],
            )
            tail_order = base.top_indices(
                corrected,
                top_k=max(top_k - head_size, 0),
            )
            local = np.concatenate((protected, eligible[tail_order]))
        else:
            local = protected
        rankings.append(np.asarray(candidates, dtype=np.int64)[local])
    return rankings


def score_compact_top256(
    *,
    storage: Any,
    query_values: np.ndarray,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    mode: str,
    correction_window: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    compact_rows: list[np.ndarray] = []
    latency_ms: list[float] = []
    touched_docs: list[float] = []
    for row, (candidates, baseline) in enumerate(
        zip(candidate_rows, baseline_rows),
    ):
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        window_size = min(correction_window, len(candidates))
        selected_local = baseline_order[:window_size]
        selected_docs = np.asarray(candidates, dtype=np.int64)[selected_local]
        before = time.perf_counter_ns()
        selected_scores = base.compact_scores(
            storage,
            query_values[row],
            selected_docs,
            mode=mode,
        )
        latency_ms.append((time.perf_counter_ns() - before) / 1_000_000.0)
        scores = np.zeros(len(candidates), dtype=np.float32)
        scores[selected_local] = selected_scores
        compact_rows.append(scores)
        touched_docs.append(float(window_size))
    return compact_rows, latency_ms, touched_docs


def score_exact_top256(
    *,
    dense_documents: np.ndarray,
    dense_queries: np.ndarray,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    correction_window: int,
) -> list[np.ndarray]:
    dense_rows: list[np.ndarray] = []
    for row, (candidates, baseline) in enumerate(
        zip(candidate_rows, baseline_rows),
    ):
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        window_size = min(correction_window, len(candidates))
        selected_local = baseline_order[:window_size]
        selected_docs = np.asarray(candidates, dtype=np.int64)[selected_local]
        selected_scores = dense_documents[selected_docs] @ dense_queries[row]
        scores = np.zeros(len(candidates), dtype=np.float32)
        scores[selected_local] = selected_scores.astype(np.float32, copy=False)
        dense_rows.append(scores)
    return dense_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.protected_head < 0:
        raise ValueError('--protected-head must be non-negative')
    if args.correction_window <= 0 or args.top_k <= 0 or args.source_depth <= 0:
        raise ValueError('source depth, top-k, and correction window must be positive')
    if args.correction_window < args.top_k:
        raise ValueError('--correction-window must be at least --top-k')
    dimensions = base.parse_int_csv(args.dimensions)

    started = time.perf_counter()
    (
        dense_documents_raw,
        dense_queries_raw,
        m190_documents,
        m190_queries,
        bm25,
        qrels,
    ) = base.prepare_inputs(args)
    dense_documents = base.l2_normalize(dense_documents_raw.vectors)
    dense_queries = base.l2_normalize(dense_queries_raw.vectors)
    semantic, lexical = m1971.build_source_rows(
        m190_documents=m190_documents,
        m190_queries=m190_queries,
        bm25=bm25,
        query_texts=dense_queries_raw.texts,
        source_depth=args.source_depth,
    )
    candidates, baseline_score_rows = m1971.fixed_hybrid_rows(semantic, lexical)
    baseline_rankings = base.rank_candidates(
        candidates,
        baseline_score_rows,
        top_k=args.top_k,
    )
    baseline_metrics = m1971.retrieval_metrics(
        baseline_rankings,
        dense_queries=dense_queries_raw,
        dense_documents=dense_documents_raw,
        qrels=qrels,
    )

    candidate_dense_rows = score_exact_top256(
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        candidate_rows=candidates,
        baseline_rows=baseline_score_rows,
        correction_window=args.correction_window,
    )
    exact_rankings = top_window_rankings(
        candidates,
        baseline_score_rows,
        candidate_dense_rows,
        weight=args.dense_weight,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        top_k=args.top_k,
    )
    exact_metrics = m1971.retrieval_metrics(
        exact_rankings,
        dense_queries=dense_queries_raw,
        dense_documents=dense_documents_raw,
        qrels=qrels,
    )

    basis, explained_energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )
    doc_projected = (dense_documents @ basis).astype(np.float32, copy=False)
    query_projected = (dense_queries @ basis).astype(np.float32, copy=False)

    variants: list[dict[str, Any]] = []
    for dimension in dimensions:
        doc_values = doc_projected[:, :dimension]
        query_values = query_projected[:, :dimension]
        storage, bytes_per_document = m1971.storage_for_mode(
            doc_values,
            mode='int4_row',
        )
        compact_rows, latency_ms, touched_docs = score_compact_top256(
            storage=storage,
            query_values=query_values,
            candidate_rows=candidates,
            baseline_rows=baseline_score_rows,
            mode='int4_row',
            correction_window=args.correction_window,
        )
        rankings = top_window_rankings(
            candidates,
            baseline_score_rows,
            compact_rows,
            weight=args.dense_weight,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        )
        metrics = m1971.retrieval_metrics(
            rankings,
            dense_queries=dense_queries_raw,
            dense_documents=dense_documents_raw,
            qrels=qrels,
        )
        delta = metric_delta(metrics, baseline_metrics)
        exact_delta = metric_delta(exact_metrics, baseline_metrics)
        variants.append(
            {
                'dimension': dimension,
                'quantization': 'int4_row',
                'bytes_per_document_payload': bytes_per_document,
                'mean_payload_bytes_per_query': (
                    statistics.fmean(touched_docs) * bytes_per_document
                ),
                'explained_document_energy': explained_energy[str(dimension)],
                'score_latency_ms': {
                    'p50': float(np.percentile(latency_ms, 50)),
                    'p95': float(np.percentile(latency_ms, 95)),
                },
                'metrics': metrics,
                'delta_vs_fixed_hybrid': delta,
                'retention_vs_exact_top256_delta': {
                    name: (
                        float(delta[name])
                        / float(exact_delta[name])
                        if exact_delta[name] != 0.0
                        else None
                    )
                    for name in base.METRIC_NAMES
                },
                'row_safe_at_0.002': min(delta.values()) >= -0.002,
            },
        )

    payload = {
        'schema': SCHEMA,
        'dataset': args.dataset,
        'qrels_usage': 'final metrics only; no variant or weight selection',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'correction_window': args.correction_window,
            'protected_head': args.protected_head,
            'dense_weight': args.dense_weight,
            'correction_scope': 'baseline ranks protected_head+1 through top256',
        },
        'counts': {
            'documents': len(dense_documents_raw.ids),
            'queries': len(dense_queries_raw.ids),
            'mean_candidates': statistics.fmean(
                float(len(row)) for row in candidates
            ),
        },
        'fixed_hybrid': {
            'metrics': baseline_metrics,
        },
        'exact_dense_top256_correction': {
            'metrics': exact_metrics,
            'delta_vs_fixed_hybrid': metric_delta(exact_metrics, baseline_metrics),
        },
        'compact_dense': {
            'basis': str(args.projection_basis),
            'basis_supervision': 'shared documents only, no qrels',
            'dimensions': list(dimensions),
            'quantization': 'int4_row',
            'explained_document_energy': explained_energy,
        },
        'variants': variants,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (args.output_dir / 'summary.md').write_text(
        markdown_report(payload),
        encoding='utf-8',
    )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    exact_delta = payload['exact_dense_top256_correction']['delta_vs_fixed_hybrid']
    lines = [
        '# M200 top256 tiny INT4 floor',
        '',
        f"- Dataset: `{payload['dataset']}`",
        f"- Correction window: `{payload['policy']['correction_window']}`",
        f"- Protected head: `{payload['policy']['protected_head']}`",
        f"- Dense weight: `{payload['policy']['dense_weight']}`",
        '',
        '## Exact Top256 Ceiling',
        '',
        '| dNDCG | dMAP | dRecall | dMRR |',
        '| ---: | ---: | ---: | ---: |',
        (
            f"| {exact_delta['ndcg@10']:+.6f} | "
            f"{exact_delta['map@100']:+.6f} | "
            f"{exact_delta['recall@100']:+.6f} | "
            f"{exact_delta['mrr@20']:+.6f} |"
        ),
        '',
        '## INT4 Variants',
        '',
        '| Dim | Bytes/doc | dNDCG | dMAP | dRecall | dMRR | p95 ms | Safe |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for variant in payload['variants']:
        delta = variant['delta_vs_fixed_hybrid']
        lines.append(
            f"| {variant['dimension']} | "
            f"{variant['bytes_per_document_payload']} | "
            f"{delta['ndcg@10']:+.6f} | "
            f"{delta['map@100']:+.6f} | "
            f"{delta['recall@100']:+.6f} | "
            f"{delta['mrr@20']:+.6f} | "
            f"{variant['score_latency_ms']['p95']:.4f} | "
            f"{'yes' if variant['row_safe_at_0.002'] else 'no'} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[m200] failed: {exc}', file=sys.stderr)
        raise
