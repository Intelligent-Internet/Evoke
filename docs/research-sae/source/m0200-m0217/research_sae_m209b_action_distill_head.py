#!/usr/bin/env python3
"""M209B action-distilled mini-vector head.

M209A showed that a safer score operator recovers the M204/M205 quality band
but still misses dense.  This trainer targets the remaining boundary problem:
which top256 documents should be promoted into, or demoted out of, top100 under
the exact local score-blend teacher.  Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
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


SCHEMA = 'm209b_action_distill_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train compact heads on top100 action distillation.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimensions', default='32,64')
    parser.add_argument('--quantization', default='float32,int4_row')
    parser.add_argument('--eval-weights', default='1.0,1.25,1.5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--teacher-weight', type=float, default=1.25)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--boundary-negative-k', type=int, default=96)
    parser.add_argument('--shape-weight', type=float, default=0.05)
    parser.add_argument('--boundary-weight', type=float, default=0.2)
    parser.add_argument('--promotion-weight', type=float, default=2.0)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=219)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('float list must not be empty')
    return values


def torch_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = torch.min(values)
    maximum = torch.max(values)
    scale = torch.clamp(maximum - minimum, min=1.0e-6)
    return (values - minimum) / scale


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def baseline_window_values(bundle: m202.DatasetBundle) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for candidates, baseline, window in zip(
        bundle.candidates,
        bundle.baseline_rows,
        bundle.windows,
    ):
        positions = {int(doc): index for index, doc in enumerate(candidates)}
        values.append(
            np.asarray(
                [baseline[positions[int(doc)]] for doc in window],
                dtype=np.float32,
            ),
        )
    return values


def pair_loss(
    prediction: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    if positives.numel() == 0 or negatives.numel() == 0:
        return prediction.new_tensor(0.0)
    margins = prediction[positives][:, None] - prediction[negatives][None, :]
    return F.softplus(-margins).mean()


def action_loss(
    *,
    prediction_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    baseline: torch.Tensor,
    teacher_weight: float,
    top_k: int,
    boundary_negative_k: int,
    shape_weight: float,
    boundary_weight: float,
    promotion_weight: float,
) -> torch.Tensor:
    teacher_final = baseline + teacher_weight * torch_minmax(teacher_scores)
    predicted_final = baseline + teacher_weight * torch_minmax(
        prediction_scores,
    )
    teacher_order = torch.argsort(teacher_final, descending=True)
    baseline_order = torch.argsort(baseline, descending=True)
    top_count = min(top_k, teacher_order.numel())
    neg_stop = min(teacher_order.numel(), top_count + boundary_negative_k)
    positives = teacher_order[:top_count]
    negatives = teacher_order[top_count:neg_stop]
    boundary = pair_loss(predicted_final, positives, negatives)

    teacher_top = torch.zeros(
        teacher_order.numel(),
        device=teacher_order.device,
        dtype=torch.bool,
    )
    baseline_top = torch.zeros_like(teacher_top)
    teacher_top[teacher_order[:top_count]] = True
    baseline_top[baseline_order[:top_count]] = True
    promoted = torch.nonzero(teacher_top & ~baseline_top).flatten()
    demoted = torch.nonzero(baseline_top & ~teacher_top).flatten()
    promotion = pair_loss(predicted_final, promoted, demoted)

    labels = teacher_top.to(dtype=torch.float32)
    boundary_bce = F.binary_cross_entropy_with_logits(
        normalize(predicted_final),
        labels,
    )
    shape = F.mse_loss(normalize(prediction_scores), normalize(teacher_scores))
    return (
        boundary
        + promotion_weight * promotion
        + boundary_weight * boundary_bce
        + shape_weight * shape
    )


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
    teacher_weight: float,
    top_k: int,
    boundary_negative_k: int,
    shape_weight: float,
    boundary_weight: float,
    promotion_weight: float,
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
    baseline_windows = [baseline_window_values(bundle) for bundle in bundles]
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(samples)
        losses: list[float] = []
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
            prediction_scores = doc_values @ query_values
            teacher_scores = torch.from_numpy(bundle.teachers[row]).to(device)
            baseline = torch.from_numpy(
                baseline_windows[bundle_index][row],
            ).to(device)
            loss = action_loss(
                prediction_scores=prediction_scores,
                teacher_scores=teacher_scores,
                baseline=baseline,
                teacher_weight=teacher_weight,
                top_k=top_k,
                boundary_negative_k=boundary_negative_k,
                shape_weight=shape_weight,
                boundary_weight=boundary_weight,
                promotion_weight=promotion_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([doc_head, query_head], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
            },
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            base.log(
                f'm209b dim={dimension} epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f}'
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


def load_dense_reference(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['dataset']: row['pplx_dense']
        for row in payload.get('dataset_baselines', [])
    }


def evaluate_bundle(
    *,
    bundle: m202.DatasetBundle,
    doc_head: np.ndarray,
    query_head: np.ndarray,
    dimension: int,
    mode: str,
    weight: float,
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
        weight=weight,
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
    delta_vs_dense = metric_delta(metrics, dense_reference[bundle.name])
    return {
        'dimension': dimension,
        'mode': mode,
        'weight': weight,
        'bytes_per_document_payload': bytes_per_document,
        'mean_payload_bytes_per_query': (
            statistics.fmean(touched_docs) * bytes_per_document
        ),
        'metrics': metrics,
        'delta_vs_fixed_hybrid': delta,
        'delta_vs_pplx_dense': delta_vs_dense,
        'row_safe_vs_fixed_at_0.002': (
            min(delta[name] for name in PRIMARY_METRICS) >= -0.002
        ),
        'row_safe_vs_dense_at_0.002': (
            min(delta_vs_dense[name] for name in PRIMARY_METRICS) >= -0.002
        ),
        'score_latency_ms': {
            'p50': float(np.percentile(latency_ms, 50)),
            'p95': float(np.percentile(latency_ms, 95)),
        },
        'training_final': training_final,
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
    eval_weights = parse_float_csv(args.eval_weights)
    allowed_modes = {'float32', 'int4_row'}
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
            teacher_weight=args.teacher_weight,
            top_k=args.top_k,
            boundary_negative_k=args.boundary_negative_k,
            shape_weight=args.shape_weight,
            boundary_weight=args.boundary_weight,
            promotion_weight=args.promotion_weight,
            seed=args.seed,
            device=device,
        )
        for mode in modes:
            for weight in eval_weights:
                rows = [
                    {
                        'dataset': bundle.name,
                        **evaluate_bundle(
                            bundle=bundle,
                            doc_head=doc_head,
                            query_head=query_head,
                            dimension=dimension,
                            mode=mode,
                            weight=weight,
                            dense_reference=dense_reference,
                            args=args,
                            training_final=history[-1],
                        ),
                    }
                    for bundle in eval_bundles
                ]
                macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
                variants.append(
                    {
                        'dimension': dimension,
                        'mode': mode,
                        'weight': weight,
                        'training_final': history[-1],
                        'macro_metrics': macro_metrics(rows),
                        'macro_delta_vs_fixed_hybrid': macro_delta(
                            rows,
                            'delta_vs_fixed_hybrid',
                        ),
                        'macro_delta_vs_pplx_dense': macro_vs_dense,
                        'beats_pplx_dense_primary': all(
                            macro_vs_dense[name] > 0.0
                            for name in PRIMARY_METRICS
                        ),
                        'unsafe_rows_vs_fixed': [
                            row['dataset']
                            for row in rows
                            if not row['row_safe_vs_fixed_at_0.002']
                        ],
                        'unsafe_rows_vs_dense': [
                            row['dataset']
                            for row in rows
                            if not row['row_safe_vs_dense_at_0.002']
                        ],
                        'rows': rows,
                    },
                )
    variants.sort(
        key=lambda item: (
            item['dimension'] == 32 and item['mode'] == 'int4_row',
            item['beats_pplx_dense_primary'],
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'top100 promotion/demotion action distillation',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'correction_window': args.correction_window,
            'protected_head': args.protected_head,
            'teacher_weight': args.teacher_weight,
            'eval_weights': list(eval_weights),
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'boundary_negative_k': args.boundary_negative_k,
            'shape_weight': args.shape_weight,
            'boundary_weight': args.boundary_weight,
            'promotion_weight': args.promotion_weight,
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
        '# M209B action-distilled mini-vector head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
        f"- Teacher weight: `{payload['policy']['teacher_weight']}`",
        '',
        '## Variants',
        '',
        '| Dim | Mode | Weight | Bytes/doc | Macro dMAP vs dense | '
        'Macro dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for variant in payload['variants']:
        macro_dense = variant['macro_delta_vs_pplx_dense']
        first_row = variant['rows'][0]
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {variant['dimension']} | {variant['mode']} | "
            f"{variant['weight']:.3g} | "
            f"{first_row['bytes_per_document_payload']} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
