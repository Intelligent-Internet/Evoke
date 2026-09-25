#!/usr/bin/env python3
"""M201 top256-local residual projection distillation.

This is a learnability probe, not a deployable encoder. It trains small query
and document projection heads from frozen dense embeddings, using only exact
dense scores inside the frozen baseline top256 window as supervision.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m200_top256_tiny_int4_floor as m200


SCHEMA = 'm201_top256_distill_projection_v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train small top256-local residual projection heads.',
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
    parser.add_argument('--dimensions', default='32,64')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--softmax-temperature', type=float, default=0.35)
    parser.add_argument('--dense-weight', type=float, default=0.5)
    parser.add_argument('--protected-head', type=int, default=20)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def actual_device(name: str) -> str:
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return name


def build_windows(
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    *,
    protected_head: int,
    correction_window: int,
) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for candidates, baseline in zip(candidate_rows, baseline_rows):
        order = base.top_indices(baseline, top_k=len(candidates))
        window_size = min(correction_window, len(candidates))
        window = np.asarray(candidates, dtype=np.int64)[
            order[protected_head:window_size]
        ]
        windows.append(window.astype(np.int64, copy=False))
    return windows


def teacher_rows(
    *,
    dense_documents: np.ndarray,
    dense_queries: np.ndarray,
    windows: Sequence[np.ndarray],
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for row, documents in enumerate(windows):
        scores = dense_documents[documents] @ dense_queries[row]
        rows.append(scores.astype(np.float32, copy=False))
    return rows


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def train_projection(
    *,
    dense_documents: np.ndarray,
    dense_queries: np.ndarray,
    windows: Sequence[np.ndarray],
    teachers: Sequence[np.ndarray],
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
    doc_tensor = torch.from_numpy(dense_documents).to(device)
    query_tensor = torch.from_numpy(dense_queries).to(device)
    teacher_tensors = [
        torch.from_numpy(values).to(device)
        for values in teachers
    ]
    window_tensors = [
        torch.from_numpy(values.astype(np.int64, copy=False)).to(device)
        for values in windows
    ]
    trainable = [index for index, row in enumerate(windows) if len(row) >= 8]
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(trainable)
        losses: list[float] = []
        mse_losses: list[float] = []
        kl_losses: list[float] = []
        for row in trainable:
            docs = window_tensors[row]
            target = normalize(teacher_tensors[row])
            doc_values = doc_tensor.index_select(0, docs) @ doc_head
            query_values = query_tensor[row] @ query_head
            scores = doc_values @ query_values
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            base.log(
                f'm201 dim={dimension} epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f}'
            )
    return (
        doc_head.detach().cpu().numpy().astype(np.float32),
        query_head.detach().cpu().numpy().astype(np.float32),
        history,
    )


def score_projected_top256(
    *,
    storage: Any,
    query_values: np.ndarray,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    mode: str,
    correction_window: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    return m200.score_compact_top256(
        storage=storage,
        query_values=query_values,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        mode=mode,
        correction_window=correction_window,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    dimensions = base.parse_int_csv(args.dimensions)
    device = actual_device(args.device)
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
    candidate_dense_rows = m200.score_exact_top256(
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        candidate_rows=candidates,
        baseline_rows=baseline_score_rows,
        correction_window=args.correction_window,
    )
    exact_rankings = m200.top_window_rankings(
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
    windows = build_windows(
        candidates,
        baseline_score_rows,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
    )
    teachers = teacher_rows(
        dense_documents=dense_documents,
        dense_queries=dense_queries,
        windows=windows,
    )
    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )
    variants: list[dict[str, Any]] = []
    for dimension in dimensions:
        doc_head, query_head, history = train_projection(
            dense_documents=dense_documents,
            dense_queries=dense_queries,
            windows=windows,
            teachers=teachers,
            init_basis=basis,
            dimension=dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.softmax_temperature,
            seed=args.seed,
            device=device,
        )
        doc_values = (dense_documents @ doc_head).astype(np.float32, copy=False)
        query_values = (dense_queries @ query_head).astype(np.float32, copy=False)
        for mode in ('float32', 'int4_row'):
            storage, bytes_per_document = m1971.storage_for_mode(
                doc_values,
                mode=mode,
            )
            compact_rows, latency_ms, touched_docs = score_projected_top256(
                storage=storage,
                query_values=query_values,
                candidate_rows=candidates,
                baseline_rows=baseline_score_rows,
                mode=mode,
                correction_window=args.correction_window,
            )
            rankings = m200.top_window_rankings(
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
            delta = m200.metric_delta(metrics, baseline_metrics)
            exact_delta = m200.metric_delta(exact_metrics, baseline_metrics)
            variants.append(
                {
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
                            float(delta[name]) / float(exact_delta[name])
                            if exact_delta[name] != 0.0 else None
                        )
                        for name in base.METRIC_NAMES
                    },
                    'row_safe_at_0.002': min(delta.values()) >= -0.002,
                    'score_latency_ms': {
                        'p50': float(np.percentile(latency_ms, 50)),
                        'p95': float(np.percentile(latency_ms, 95)),
                    },
                    'training_final': history[-1],
                },
            )
    payload = {
        'schema': SCHEMA,
        'dataset': args.dataset,
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'same dataset top256-local learnability probe',
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
        'counts': {
            'documents': len(dense_documents_raw.ids),
            'queries': len(dense_queries_raw.ids),
            'mean_train_window': statistics.fmean(float(len(row)) for row in windows),
        },
        'fixed_hybrid': {
            'metrics': baseline_metrics,
        },
        'exact_dense_top256_correction': {
            'metrics': exact_metrics,
            'delta_vs_fixed_hybrid': m200.metric_delta(
                exact_metrics,
                baseline_metrics,
            ),
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
        '# M201 top256 distill projection',
        '',
        f"- Dataset: `{payload['dataset']}`",
        f"- Training scope: `{payload['training_scope']}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
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
        '## Learned Variants',
        '',
        '| Dim | Mode | Bytes/doc | dMAP | dRecall | p95 ms | Safe |',
        '| ---: | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for variant in payload['variants']:
        delta = variant['delta_vs_fixed_hybrid']
        lines.append(
            f"| {variant['dimension']} | {variant['mode']} | "
            f"{variant['bytes_per_document_payload']} | "
            f"{delta['map@100']:+.6f} | "
            f"{delta['recall@100']:+.6f} | "
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
        print(f'[m201] failed: {exc}', file=sys.stderr)
        raise
