#!/usr/bin/env python3
"""M202 shared top256 residual head.

This trains one shared query/document projection head across datasets while
keeping the exported payload small. The teacher is exact dense correction inside
the frozen baseline top256 window. Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m200_top256_tiny_int4_floor as m200
import research_sae_m201_top256_distill_projection as m201


SCHEMA = 'm202_shared_top256_residual_head_v1'


@dataclass
class DatasetBundle:
    name: str
    dense_documents_raw: base.EmbeddingRows
    dense_queries_raw: base.EmbeddingRows
    dense_documents: np.ndarray
    dense_queries: np.ndarray
    qrels: dict[str, dict[str, float]]
    candidates: list[np.ndarray]
    baseline_rows: list[np.ndarray]
    baseline_metrics: dict[str, float]
    exact_metrics: dict[str, float]
    exact_delta: dict[str, float]
    windows: list[np.ndarray]
    teachers: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a shared top256 residual projection head.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimensions', default='32,64')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--softmax-temperature', type=float, default=0.35)
    parser.add_argument('--dense-weight', type=float, default=0.5)
    parser.add_argument('--protected-head', type=int, default=20)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=202)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def parse_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('dataset list must not be empty')
    return values


def dataset_paths(
    *,
    dataset: str,
    run_root: Path,
    dense_root: Path,
    official_root: Path,
) -> argparse.Namespace:
    input_dir = run_root / 'inputs' / dataset
    return argparse.Namespace(
        dense_documents=dense_root / dataset / 'documents.jsonl',
        dense_queries=official_root / dataset / 'queries.jsonl',
        bm25_cache=run_root / 'cache' / f'{dataset}-m1960-docs-bm25.pkl',
        qrels=official_root / dataset / 'quality_qrels.json',
        m190_documents=input_dir / f'{dataset}.latent_bm25.documents.npz',
        m190_queries=input_dir / f'{dataset}.latent_bm25.queries.npz',
        m190_document_ids=input_dir / f'{dataset}.document_ids.txt',
        m190_query_ids=input_dir / f'{dataset}.query_ids.txt',
    )


def load_bundle(
    *,
    dataset: str,
    args: argparse.Namespace,
) -> DatasetBundle:
    paths = dataset_paths(
        dataset=dataset,
        run_root=args.run_root,
        dense_root=args.dense_root,
        official_root=args.official_root,
    )
    input_args = argparse.Namespace(**vars(args), **vars(paths))
    (
        dense_documents_raw,
        dense_queries_raw,
        m190_documents,
        m190_queries,
        bm25,
        qrels,
    ) = base.prepare_inputs(input_args)
    dense_documents = base.l2_normalize(dense_documents_raw.vectors)
    dense_queries = base.l2_normalize(dense_queries_raw.vectors)
    semantic, lexical = m1971.build_source_rows(
        m190_documents=m190_documents,
        m190_queries=m190_queries,
        bm25=bm25,
        query_texts=dense_queries_raw.texts,
        source_depth=args.source_depth,
    )
    candidates, baseline_rows = m1971.fixed_hybrid_rows(semantic, lexical)
    baseline_rankings = base.rank_candidates(
        candidates,
        baseline_rows,
        top_k=args.top_k,
    )
    baseline_metrics = m1971.retrieval_metrics(
        baseline_rankings,
        dense_queries=dense_queries_raw,
        dense_documents=dense_documents_raw,
        qrels=qrels,
    )
    dense_rows = m200.score_exact_top256(
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        candidate_rows=candidates,
        baseline_rows=baseline_rows,
        correction_window=args.correction_window,
    )
    exact_rankings = m200.top_window_rankings(
        candidates,
        baseline_rows,
        dense_rows,
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
    windows = m201.build_windows(
        candidates,
        baseline_rows,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
    )
    teachers = m201.teacher_rows(
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        windows=windows,
    )
    return DatasetBundle(
        name=dataset,
        dense_documents_raw=dense_documents_raw,
        dense_queries_raw=dense_queries_raw,
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        qrels=qrels,
        candidates=candidates,
        baseline_rows=baseline_rows,
        baseline_metrics=baseline_metrics,
        exact_metrics=exact_metrics,
        exact_delta=m200.metric_delta(exact_metrics, baseline_metrics),
        windows=windows,
        teachers=teachers,
    )


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def sample_indices(
    bundles: Sequence[DatasetBundle],
) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for bundle_index, bundle in enumerate(bundles):
        for row, window in enumerate(bundle.windows):
            if len(window) >= 8:
                samples.append((bundle_index, row))
    return samples


def train_shared_head(
    *,
    bundles: Sequence[DatasetBundle],
    init_basis: np.ndarray,
    dimension: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(seed + dimension)
    rng = random.Random(seed + dimension)
    doc_head = torch.nn.Parameter(
        torch.from_numpy(init_basis[:, :dimension].copy()).to(device),
    )
    query_head = torch.nn.Parameter(
        torch.from_numpy(init_basis[:, :dimension].copy()).to(device),
    )
    optimizer = torch.optim.AdamW(
        [doc_head, query_head],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    samples = sample_indices(bundles)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(samples)
        losses: list[float] = []
        mse_losses: list[float] = []
        kl_losses: list[float] = []
        for bundle_index, row in samples:
            bundle = bundles[bundle_index]
            docs = bundle.windows[row]
            doc_values = torch.from_numpy(
                bundle.dense_documents[docs],
            ).to(device) @ doc_head
            query_values = (
                torch.from_numpy(bundle.dense_queries[row]).to(device)
                @ query_head
            )
            scores = doc_values @ query_values
            target = normalize(torch.from_numpy(bundle.teachers[row]).to(device))
            prediction = normalize(scores)
            mse = F.mse_loss(prediction, target)
            target_distribution = torch.softmax(target / temperature, dim=0)
            kl = -torch.sum(
                target_distribution * F.log_softmax(
                    prediction / temperature,
                    dim=0,
                ),
            )
            loss = mse + 0.1 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([doc_head, query_head], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse.detach().cpu()))
            kl_losses.append(float(kl.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'mse': statistics.fmean(mse_losses),
                'kl': statistics.fmean(kl_losses),
            },
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            base.log(
                f'm202 dim={dimension} epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f}'
            )
    return (
        doc_head.detach().cpu().numpy().astype(np.float32),
        query_head.detach().cpu().numpy().astype(np.float32),
        history,
    )


def evaluate_bundle(
    *,
    bundle: DatasetBundle,
    doc_head: np.ndarray,
    query_head: np.ndarray,
    dimension: int,
    mode: str,
    args: argparse.Namespace,
    training_final: dict[str, float],
) -> dict[str, Any]:
    doc_values = (bundle.dense_documents @ doc_head).astype(
        np.float32,
        copy=False,
    )
    query_values = (bundle.dense_queries @ query_head).astype(
        np.float32,
        copy=False,
    )
    storage, bytes_per_document = m1971.storage_for_mode(doc_values, mode=mode)
    compact_rows, latency_ms, touched_docs = m200.score_compact_top256(
        storage=storage,
        query_values=query_values,
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        mode=mode,
        correction_window=args.correction_window,
    )
    rankings = m200.top_window_rankings(
        bundle.candidates,
        bundle.baseline_rows,
        compact_rows,
        weight=args.dense_weight,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        top_k=args.top_k,
    )
    metrics = m1971.retrieval_metrics(
        rankings,
        dense_queries=bundle.dense_queries_raw,
        dense_documents=bundle.dense_documents_raw,
        qrels=bundle.qrels,
    )
    delta = m200.metric_delta(metrics, bundle.baseline_metrics)
    return {
        'dimension': dimension,
        'mode': mode,
        'bytes_per_document_payload': bytes_per_document,
        'mean_payload_bytes_per_query': (
            statistics.fmean(touched_docs) * bytes_per_document
        ),
        'metrics': metrics,
        'delta_vs_fixed_hybrid': delta,
        'retention_vs_exact_top256_delta': {
            name: (
                float(delta[name]) / float(bundle.exact_delta[name])
                if bundle.exact_delta[name] != 0.0 else None
            )
            for name in base.METRIC_NAMES
        },
        'row_safe_at_0.002': min(delta.values()) >= -0.002,
        'score_latency_ms': {
            'p50': float(np.percentile(latency_ms, 50)),
            'p95': float(np.percentile(latency_ms, 95)),
        },
        'training_final': training_final,
    }


def macro_delta(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        name: statistics.fmean(
            row['delta_vs_fixed_hybrid'][name] for row in rows
        )
        for name in base.METRIC_NAMES
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    train_names = parse_csv(args.train_datasets)
    eval_names = parse_csv(args.eval_datasets)
    dimensions = base.parse_int_csv(args.dimensions)
    train_bundles = [
        load_bundle(dataset=name, args=args)
        for name in train_names
    ]
    bundle_cache = {bundle.name: bundle for bundle in train_bundles}
    eval_bundles: list[DatasetBundle] = []
    for name in eval_names:
        bundle = bundle_cache.get(name)
        if bundle is None:
            bundle = load_bundle(dataset=name, args=args)
            bundle_cache[name] = bundle
        eval_bundles.append(bundle)
    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )
    variants: list[dict[str, Any]] = []
    for dimension in dimensions:
        doc_head, query_head, history = train_shared_head(
            bundles=train_bundles,
            init_basis=basis,
            dimension=dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.softmax_temperature,
            seed=args.seed,
            device=device,
        )
        for mode in ('int4_row',):
            rows = [
                {
                    'dataset': bundle.name,
                    **evaluate_bundle(
                        bundle=bundle,
                        doc_head=doc_head,
                        query_head=query_head,
                        dimension=dimension,
                        mode=mode,
                        args=args,
                        training_final=history[-1],
                    ),
                }
                for bundle in eval_bundles
            ]
            variants.append(
                {
                    'dimension': dimension,
                    'mode': mode,
                    'training_final': history[-1],
                    'macro_delta_vs_fixed_hybrid': macro_delta(rows),
                    'unsafe_rows': [
                        row['dataset']
                        for row in rows
                        if not row['row_safe_at_0.002']
                    ],
                    'rows': rows,
                },
            )
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'shared projection head across train datasets',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'correction_window': args.correction_window,
            'protected_head': args.protected_head,
            'dense_weight': args.dense_weight,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'softmax_temperature': args.softmax_temperature,
            'device': device,
        },
        'dataset_baselines': {
            bundle.name: {
                'fixed_hybrid': bundle.baseline_metrics,
                'exact_dense_top256_correction': {
                    'metrics': bundle.exact_metrics,
                    'delta_vs_fixed_hybrid': bundle.exact_delta,
                },
                'counts': {
                    'documents': len(bundle.dense_documents_raw.ids),
                    'queries': len(bundle.dense_queries_raw.ids),
                },
            }
            for bundle in eval_bundles
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
    lines = [
        '# M202 shared top256 residual head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
        '',
        '## Variants',
        '',
        '| Dim | Mode | Macro dMAP | Macro dRecall | Unsafe rows |',
        '| ---: | --- | ---: | ---: | --- |',
    ]
    for variant in payload['variants']:
        macro = variant['macro_delta_vs_fixed_hybrid']
        unsafe = ', '.join(variant['unsafe_rows']) or 'none'
        lines.append(
            f"| {variant['dimension']} | {variant['mode']} | "
            f"{macro['map@100']:+.6f} | "
            f"{macro['recall@100']:+.6f} | {unsafe} |"
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
        print(f'[m202] failed: {exc}', file=sys.stderr)
        raise
