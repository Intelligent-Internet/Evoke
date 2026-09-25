#!/usr/bin/env python3
"""M203 dual-head operator ceiling.

This is a no-training replay for the BM25+SSR/SAE plus mini-vector route.  It
keeps the sparse/BM25 candidate source fixed and asks whether exact dense
scores inside a local candidate window can beat the full-corpus PPLX dense
baseline.  Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m200_top256_tiny_int4_floor as m200
import research_sae_m202_shared_top256_residual_head as m202


SCHEMA = 'm203_dual_head_operator_ceiling_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate dual-head local rerank operator ceilings.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--datasets',
        default='arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,'
        'webis-touche2020',
    )
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--protected-heads', default='0,5,10,20')
    parser.add_argument('--correction-windows', default='100,256,512,1000')
    parser.add_argument('--dense-weights', default='0.25,0.5,0.75,1.0,1.5')
    parser.add_argument('--block-size', type=int, default=64)
    return parser.parse_args()


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('integer list must not be empty')
    return values


def parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('float list must not be empty')
    return values


def metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in base.METRIC_NAMES
    }


def macro_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        name: statistics.fmean(row['metrics'][name] for row in rows)
        for name in base.METRIC_NAMES
    }


def macro_delta(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        name: statistics.fmean(row[key][name] for row in rows)
        for name in base.METRIC_NAMES
    }


def dense_full_rankings(
    *,
    dense_documents: np.ndarray,
    dense_queries: np.ndarray,
    top_k: int,
    block_size: int,
) -> list[np.ndarray]:
    rankings: list[np.ndarray] = []
    documents_t = np.asarray(dense_documents, dtype=np.float32).T
    for start in range(0, dense_queries.shape[0], block_size):
        scores = np.asarray(
            dense_queries[start:start + block_size],
            dtype=np.float32,
        ) @ documents_t
        for row in scores:
            rankings.append(base.top_indices(row, top_k=top_k))
    return rankings


def candidate_dense_rows(bundle: m202.DatasetBundle) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for row, candidates in enumerate(bundle.candidates):
        scores = (
            bundle.dense_documents[np.asarray(candidates, dtype=np.int64)]
            @ bundle.dense_queries[row]
        )
        rows.append(scores.astype(np.float32, copy=False))
    return rows


def row_safe(delta: dict[str, float], *, floor: float = -0.002) -> bool:
    return min(delta[name] for name in PRIMARY_METRICS) >= floor


def evaluate_dataset(
    *,
    bundle: m202.DatasetBundle,
    dense_metrics: dict[str, float],
    dense_rows: Sequence[np.ndarray],
    protected_heads: Sequence[int],
    correction_windows: Sequence[int],
    dense_weights: Sequence[float],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protected_head in protected_heads:
        for correction_window in correction_windows:
            for weight in dense_weights:
                rankings = m200.top_window_rankings(
                    bundle.candidates,
                    bundle.baseline_rows,
                    dense_rows,
                    weight=weight,
                    protected_head=protected_head,
                    correction_window=correction_window,
                    top_k=top_k,
                )
                metrics = m1971.retrieval_metrics(
                    rankings,
                    dense_queries=bundle.dense_queries_raw,
                    dense_documents=bundle.dense_documents_raw,
                    qrels=bundle.qrels,
                )
                delta_vs_dense = metric_delta(metrics, dense_metrics)
                delta_vs_fixed = metric_delta(metrics, bundle.baseline_metrics)
                rows.append(
                    {
                        'dataset': bundle.name,
                        'protected_head': protected_head,
                        'correction_window': correction_window,
                        'dense_weight': weight,
                        'metrics': metrics,
                        'delta_vs_pplx_dense': delta_vs_dense,
                        'delta_vs_fixed_hybrid': delta_vs_fixed,
                        'safe_vs_pplx_dense_at_0.002': row_safe(
                            delta_vs_dense,
                        ),
                        'safe_vs_fixed_hybrid_at_0.002': row_safe(
                            delta_vs_fixed,
                        ),
                    },
                )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.source_depth <= 0 or args.top_k <= 0:
        raise ValueError('--source-depth and --top-k must be positive')
    started = time.perf_counter()
    datasets = m202.parse_csv(args.datasets)
    protected_heads = parse_int_csv(args.protected_heads)
    correction_windows = parse_int_csv(args.correction_windows)
    dense_weights = parse_float_csv(args.dense_weights)
    if min(protected_heads) < 0:
        raise ValueError('--protected-heads must be non-negative')
    if min(correction_windows) < args.top_k:
        raise ValueError('--correction-windows must be at least --top-k')

    loader_args = argparse.Namespace(**vars(args))
    loader_args.correction_window = max(correction_windows)
    loader_args.protected_head = min(protected_heads)
    loader_args.dense_weight = 0.5

    dataset_payloads: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int, float], list[dict[str, Any]]] = {
        (head, window, weight): []
        for head in protected_heads
        for window in correction_windows
        for weight in dense_weights
    }
    for dataset in datasets:
        bundle = m202.load_bundle(dataset=dataset, args=loader_args)
        dense_rankings = dense_full_rankings(
            dense_documents=bundle.dense_documents,
            dense_queries=bundle.dense_queries,
            top_k=args.top_k,
            block_size=args.block_size,
        )
        dense_metrics = m1971.retrieval_metrics(
            dense_rankings,
            dense_queries=bundle.dense_queries_raw,
            dense_documents=bundle.dense_documents_raw,
            qrels=bundle.qrels,
        )
        dense_rows = candidate_dense_rows(bundle)
        config_rows = evaluate_dataset(
            bundle=bundle,
            dense_metrics=dense_metrics,
            dense_rows=dense_rows,
            protected_heads=protected_heads,
            correction_windows=correction_windows,
            dense_weights=dense_weights,
            top_k=args.top_k,
        )
        for row in config_rows:
            key = (
                row['protected_head'],
                row['correction_window'],
                row['dense_weight'],
            )
            grouped[key].append(row)
        dataset_payloads.append(
            {
                'dataset': dataset,
                'pplx_dense': dense_metrics,
                'fixed_hybrid': bundle.baseline_metrics,
                'fixed_hybrid_delta_vs_pplx_dense': metric_delta(
                    bundle.baseline_metrics,
                    dense_metrics,
                ),
            },
        )

    variants: list[dict[str, Any]] = []
    for (protected_head, correction_window, weight), rows in grouped.items():
        metrics = macro_metrics(rows)
        delta_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
        delta_vs_fixed = macro_delta(rows, 'delta_vs_fixed_hybrid')
        variants.append(
            {
                'protected_head': protected_head,
                'correction_window': correction_window,
                'dense_weight': weight,
                'macro_metrics': metrics,
                'macro_delta_vs_pplx_dense': delta_vs_dense,
                'macro_delta_vs_fixed_hybrid': delta_vs_fixed,
                'beats_pplx_dense_primary': all(
                    delta_vs_dense[name] > 0.0 for name in PRIMARY_METRICS
                ),
                'unsafe_rows_vs_pplx_dense': [
                    row['dataset']
                    for row in rows
                    if not row['safe_vs_pplx_dense_at_0.002']
                ],
                'unsafe_rows_vs_fixed_hybrid': [
                    row['dataset']
                    for row in rows
                    if not row['safe_vs_fixed_hybrid_at_0.002']
                ],
                'rows': rows,
            },
        )
    variants.sort(
        key=lambda item: (
            item['beats_pplx_dense_primary'],
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'datasets': list(datasets),
        'qrels_usage': 'final metrics only; no training or qrel selection',
        'candidate_source': 'M190/SSR latent postings plus BM25',
        'target_route': 'BM25+SSR candidates with local mini-vector rerank',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'protected_heads': list(protected_heads),
            'correction_windows': list(correction_windows),
            'dense_weights': list(dense_weights),
        },
        'dataset_baselines': dataset_payloads,
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
    lines = [
        '# M203 dual-head operator ceiling',
        '',
        f"- Candidate source: {payload['candidate_source']}",
        f"- Target route: {payload['target_route']}",
        f"- Source depth: `{payload['policy']['source_depth']}`",
        '',
        '## Top Variants',
        '',
        '| Rank | Head | Window | Weight | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe rows vs dense |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'][:20], start=1):
        delta = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_pplx_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['protected_head']} | "
            f"{variant['correction_window']} | "
            f"{variant['dense_weight']:.3g} | "
            f"{delta['map@100']:+.6f} | "
            f"{delta['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
