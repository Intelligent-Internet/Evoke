#!/usr/bin/env python3
"""M205 compact mini-vector head with top-rank pairwise loss.

This continues the M204 route.  The target operator is the M203-selected
head0/window256/weight1.5 local rerank, but training adds a small top-rank
pairwise term so the projection head learns the high-rank ordering that became
unsafe when M204 removed top20 protection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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
import research_sae_m202_shared_top256_residual_head as m202


SCHEMA = 'm205_compact_pairwise_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


@dataclass(frozen=True)
class LowBitRows:
    values: np.ndarray
    scales: np.ndarray
    dimensions: int
    bits: int
    bytes_per_row: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate compact pairwise mini-vector heads.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimensions', default='16,32,64')
    parser.add_argument('--quantization', default='int2_row,int4_row')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--softmax-temperature', type=float, default=0.35)
    parser.add_argument('--pairwise-weight', type=float, default=0.2)
    parser.add_argument('--pair-top-k', type=int, default=16)
    parser.add_argument('--pair-negative-k', type=int, default=64)
    parser.add_argument('--dense-weight', type=float, default=1.5)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=205)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def quantize_lowbit_rows(matrix: np.ndarray, *, bits: int) -> LowBitRows:
    if bits not in (2, 4):
        raise ValueError('low-bit row quantization supports 2 or 4 bits')
    max_code = (1 << (bits - 1)) - 1
    levels_per_byte = 8 // bits
    scales = np.max(np.abs(matrix), axis=1).astype(np.float32) / max_code
    scales[scales <= 1.0e-12] = 1.0
    codes = np.clip(
        np.rint(matrix / scales[:, None]),
        -max_code,
        max_code,
    ).astype(np.int8)
    unsigned = (codes + max_code).astype(np.uint8)
    byte_count = int(math.ceil(matrix.shape[1] / levels_per_byte))
    values = np.zeros((matrix.shape[0], byte_count), dtype=np.uint8)
    for offset in range(levels_per_byte):
        part = unsigned[:, offset::levels_per_byte]
        values[:, :part.shape[1]] |= part << (bits * offset)
    return LowBitRows(
        values=values,
        scales=scales,
        dimensions=int(matrix.shape[1]),
        bits=bits,
        bytes_per_row=int(byte_count + 4),
    )


def lowbit_scores(
    documents: LowBitRows,
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    bits = documents.bits
    levels_per_byte = 8 // bits
    mask = (1 << bits) - 1
    max_code = (1 << (bits - 1)) - 1
    byte_count = documents.values.shape[1]
    padded = np.zeros(byte_count * levels_per_byte, dtype=np.float32)
    padded[:documents.dimensions] = query[:documents.dimensions]
    query_blocks = padded.reshape(byte_count, levels_per_byte)
    packed_codes = np.arange(256, dtype=np.uint16)
    table = np.zeros((byte_count, 256), dtype=np.float32)
    for offset in range(levels_per_byte):
        codes = (
            ((packed_codes >> (bits * offset)) & mask).astype(np.float32)
            - max_code
        )
        table += query_blocks[:, offset, None] * codes[None, :]
    selected = documents.values[np.asarray(candidates, dtype=np.int64)]
    positions = np.arange(byte_count, dtype=np.int64)[None, :]
    quantized_sum = table[positions, selected].sum(axis=1)
    return quantized_sum * documents.scales[candidates]


def storage_for_mode(documents: np.ndarray, *, mode: str) -> tuple[Any, int]:
    if mode == 'float32':
        return documents.astype(np.float32, copy=False), int(documents.shape[1] * 4)
    if mode == 'int2_row':
        rows = quantize_lowbit_rows(documents, bits=2)
        return rows, rows.bytes_per_row
    if mode == 'int4_row':
        rows = quantize_lowbit_rows(documents, bits=4)
        return rows, rows.bytes_per_row
    raise ValueError(f'unsupported mode: {mode}')


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
        if mode in ('int2_row', 'int4_row'):
            selected_scores = lowbit_scores(
                storage,
                query_values[row],
                selected_docs,
            )
        else:
            selected_scores = np.asarray(storage)[selected_docs] @ query_values[row]
        latency_ms.append((time.perf_counter_ns() - before) / 1_000_000.0)
        scores = np.zeros(len(candidates), dtype=np.float32)
        scores[selected_local] = selected_scores.astype(np.float32, copy=False)
        compact_rows.append(scores)
        touched_docs.append(float(window_size))
    return compact_rows, latency_ms, touched_docs


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def pairwise_top_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    top_k: int,
    negative_k: int,
) -> torch.Tensor:
    if prediction.numel() < 4 or top_k <= 0 or negative_k <= 0:
        return prediction.new_tensor(0.0)
    order = torch.argsort(target, descending=True)
    pos_count = min(top_k, max(1, prediction.numel() // 4))
    neg_start = pos_count
    neg_stop = min(prediction.numel(), neg_start + negative_k)
    if neg_stop <= neg_start:
        return prediction.new_tensor(0.0)
    positives = order[:pos_count]
    negatives = order[neg_start:neg_stop]
    margins = prediction[positives][:, None] - prediction[negatives][None, :]
    return F.softplus(-margins).mean()


def sample_indices(
    bundles: Sequence[m202.DatasetBundle],
) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for bundle_index, bundle in enumerate(bundles):
        for row, window in enumerate(bundle.windows):
            if len(window) >= 8:
                samples.append((bundle_index, row))
    return samples


def train_shared_head(
    *,
    bundles: Sequence[m202.DatasetBundle],
    init_basis: np.ndarray,
    dimension: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    pairwise_weight: float,
    pair_top_k: int,
    pair_negative_k: int,
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
        pair_losses: list[float] = []
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
            pair = pairwise_top_loss(
                prediction,
                target,
                top_k=pair_top_k,
                negative_k=pair_negative_k,
            )
            loss = mse + 0.1 * kl + pairwise_weight * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([doc_head, query_head], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse.detach().cpu()))
            kl_losses.append(float(kl.detach().cpu()))
            pair_losses.append(float(pair.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'mse': statistics.fmean(mse_losses),
                'kl': statistics.fmean(kl_losses),
                'pairwise': statistics.fmean(pair_losses),
            },
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            base.log(
                f'm205 dim={dimension} epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f} '
                f'pair={history[-1]["pairwise"]:.6f}'
            )
    return (
        doc_head.detach().cpu().numpy().astype(np.float32),
        query_head.detach().cpu().numpy().astype(np.float32),
        history,
    )


def metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in base.METRIC_NAMES
    }


def load_dense_reference(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['dataset']: row['pplx_dense']
        for row in payload.get('dataset_baselines', [])
        if 'pplx_dense' in row
    }


def evaluate_bundle(
    *,
    bundle: m202.DatasetBundle,
    doc_head: np.ndarray,
    query_head: np.ndarray,
    dimension: int,
    mode: str,
    dense_reference: dict[str, dict[str, float]],
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
    storage, bytes_per_document = storage_for_mode(doc_values, mode=mode)
    compact_rows, latency_ms, touched_docs = score_compact_top256(
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
    delta = metric_delta(metrics, bundle.baseline_metrics)
    dense_metrics = dense_reference.get(bundle.name)
    delta_vs_dense = (
        metric_delta(metrics, dense_metrics) if dense_metrics else None
    )
    return {
        'dimension': dimension,
        'mode': mode,
        'bytes_per_document_payload': bytes_per_document,
        'mean_payload_bytes_per_query': (
            statistics.fmean(touched_docs) * bytes_per_document
        ),
        'metrics': metrics,
        'delta_vs_fixed_hybrid': delta,
        'delta_vs_pplx_dense': delta_vs_dense,
        'retention_vs_exact_top256_delta': {
            name: (
                float(delta[name]) / float(bundle.exact_delta[name])
                if bundle.exact_delta[name] != 0.0 else None
            )
            for name in base.METRIC_NAMES
        },
        'row_safe_vs_fixed_at_0.002': min(delta[name] for name in PRIMARY_METRICS)
        >= -0.002,
        'row_safe_vs_dense_at_0.002': (
            min(delta_vs_dense[name] for name in PRIMARY_METRICS) >= -0.002
            if delta_vs_dense else None
        ),
        'score_latency_ms': {
            'p50': float(np.percentile(latency_ms, 50)),
            'p95': float(np.percentile(latency_ms, 95)),
        },
        'training_final': training_final,
    }


def macro_delta(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        name: statistics.fmean(row[key][name] for row in rows)
        for name in base.METRIC_NAMES
    }


def macro_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        name: statistics.fmean(row['metrics'][name] for row in rows)
        for name in base.METRIC_NAMES
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    dimensions = base.parse_int_csv(args.dimensions)
    modes = parse_str_csv(args.quantization)
    allowed_modes = {'float32', 'int2_row', 'int4_row'}
    if set(modes) - allowed_modes:
        raise ValueError(f'unsupported modes: {set(modes) - allowed_modes}')
    dense_reference = load_dense_reference(args.reference_summary)
    train_bundles = [
        m202.load_bundle(dataset=name, args=args)
        for name in train_names
    ]
    bundle_cache = {bundle.name: bundle for bundle in train_bundles}
    eval_bundles: list[m202.DatasetBundle] = []
    for name in eval_names:
        bundle = bundle_cache.get(name)
        if bundle is None:
            bundle = m202.load_bundle(dataset=name, args=args)
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
            pairwise_weight=args.pairwise_weight,
            pair_top_k=args.pair_top_k,
            pair_negative_k=args.pair_negative_k,
            seed=args.seed,
            device=device,
        )
        for mode in modes:
            rows = [
                {
                    'dataset': bundle.name,
                    **evaluate_bundle(
                        bundle=bundle,
                        doc_head=doc_head,
                        query_head=query_head,
                        dimension=dimension,
                        mode=mode,
                        dense_reference=dense_reference,
                        args=args,
                        training_final=history[-1],
                    ),
                }
                for bundle in eval_bundles
            ]
            dense_rows = [
                row for row in rows if row['delta_vs_pplx_dense'] is not None
            ]
            macro_vs_dense = (
                macro_delta(dense_rows, 'delta_vs_pplx_dense')
                if dense_rows else None
            )
            variants.append(
                {
                    'dimension': dimension,
                    'mode': mode,
                    'training_final': history[-1],
                    'macro_metrics': macro_metrics(rows),
                    'macro_delta_vs_fixed_hybrid': macro_delta(
                        rows,
                        'delta_vs_fixed_hybrid',
                    ),
                    'macro_delta_vs_pplx_dense': macro_vs_dense,
                    'beats_pplx_dense_primary': (
                        all(
                            macro_vs_dense[name] > 0.0
                            for name in PRIMARY_METRICS
                        )
                        if macro_vs_dense else None
                    ),
                    'unsafe_rows_vs_fixed': [
                        row['dataset']
                        for row in rows
                        if not row['row_safe_vs_fixed_at_0.002']
                    ],
                    'unsafe_rows_vs_dense': [
                        row['dataset']
                        for row in rows
                        if row['row_safe_vs_dense_at_0.002'] is False
                    ],
                    'rows': rows,
                },
            )
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'shared projection head with top-rank pairwise loss',
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
            'pairwise_weight': args.pairwise_weight,
            'pair_top_k': args.pair_top_k,
            'pair_negative_k': args.pair_negative_k,
            'device': device,
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
        '# M205 compact pairwise mini-vector head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
        '',
        '## Variants',
        '',
        '| Dim | Mode | Bytes/doc | Macro dMAP vs dense | '
        'Macro dRecall vs dense | Macro dMAP vs fixed | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for variant in payload['variants']:
        macro_dense = variant['macro_delta_vs_pplx_dense'] or {}
        macro_fixed = variant['macro_delta_vs_fixed_hybrid']
        first_row = variant['rows'][0]
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {variant['dimension']} | {variant['mode']} | "
            f"{first_row['bytes_per_document_payload']} | "
            f"{macro_dense.get('map@100', 0.0):+.6f} | "
            f"{macro_dense.get('recall@100', 0.0):+.6f} | "
            f"{macro_fixed['map@100']:+.6f} | {unsafe} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
