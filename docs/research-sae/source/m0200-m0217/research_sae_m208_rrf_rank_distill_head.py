#!/usr/bin/env python3
"""M208 RRF rank-distilled mini-vector head.

M207 found that an exact RRF last-mile operator beats PPLX dense while avoiding
score calibration as the core contract.  This trainer distills dense ordering
inside the fixed top256 window, then evaluates compact mini-vector scores with
the same RRF operator.  Qrels are used only for final metrics.
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


SCHEMA = 'm208_rrf_rank_distill_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train compact heads for RRF rank-distilled rerank.',
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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--softmax-temperature', type=float, default=0.2)
    parser.add_argument('--pairwise-weight', type=float, default=0.4)
    parser.add_argument('--pair-top-k', type=int, default=16)
    parser.add_argument('--pair-negative-k', type=int, default=96)
    parser.add_argument('--rrf-weight', type=float, default=2.0)
    parser.add_argument('--rrf-k', type=float, default=10.0)
    parser.add_argument('--dense-weight', type=float, default=2.0)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=208)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def rank_component_from_scores(
    scores: torch.Tensor,
    *,
    rrf_k: float,
) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty(scores.shape[0], device=scores.device)
    ranks[order] = torch.arange(
        scores.shape[0],
        device=scores.device,
        dtype=torch.float32,
    )
    return 1.0 / (float(rrf_k) + ranks + 1.0)


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
    rrf_k: float,
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
        list_losses: list[float] = []
        mse_losses: list[float] = []
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
            prediction_scores = doc_values @ query_values
            teacher_scores = torch.from_numpy(bundle.teachers[row]).to(device)
            target = rank_component_from_scores(
                teacher_scores,
                rrf_k=rrf_k,
            )
            target_norm = normalize(target)
            prediction_norm = normalize(prediction_scores)
            list_target = torch.softmax(target_norm / temperature, dim=0)
            list_loss = -torch.sum(
                list_target * F.log_softmax(
                    prediction_norm / temperature,
                    dim=0,
                ),
            )
            mse = F.mse_loss(prediction_norm, target_norm)
            pair = pairwise_top_loss(
                prediction_norm,
                target_norm,
                top_k=pair_top_k,
                negative_k=pair_negative_k,
            )
            loss = list_loss + 0.1 * mse + pairwise_weight * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([doc_head, query_head], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            list_losses.append(float(list_loss.detach().cpu()))
            mse_losses.append(float(mse.detach().cpu()))
            pair_losses.append(float(pair.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'list': statistics.fmean(list_losses),
                'mse': statistics.fmean(mse_losses),
                'pairwise': statistics.fmean(pair_losses),
            },
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            base.log(
                f'm208 dim={dimension} epoch={epoch + 1}/{epochs} '
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


def load_reference(path: Path) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float] | None,
]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    dense = {
        row['dataset']: row['pplx_dense']
        for row in payload.get('dataset_baselines', [])
    }
    exact_macro = None
    for variant in payload.get('variants', []):
        if (
            variant.get('family') == 'rrf'
            and variant.get('protected_head') == 0
            and variant.get('correction_window') == 256
            and float(variant.get('weight')) == 2.0
            and float(variant.get('rrf_k')) == 10.0
        ):
            exact_macro = variant.get('macro_metrics')
            break
    return dense, exact_macro


def rrf_rankings(
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    correction_rows: Sequence[np.ndarray],
    *,
    weight: float,
    protected_head: int,
    correction_window: int,
    rrf_k: float,
    top_k: int,
) -> list[np.ndarray]:
    rankings: list[np.ndarray] = []
    for candidates, baseline, correction in zip(
        candidate_rows,
        baseline_rows,
        correction_rows,
    ):
        baseline = np.asarray(baseline, dtype=np.float32)
        correction = np.asarray(correction, dtype=np.float32)
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        head_size = min(protected_head, top_k, len(candidates))
        window_size = min(correction_window, len(candidates))
        protected = baseline_order[:head_size]
        eligible = baseline_order[head_size:window_size]
        tail_budget = max(top_k - head_size, 0)
        if eligible.size == 0 or tail_budget == 0:
            local = protected
        else:
            dense_order = base.top_indices(
                correction[eligible],
                top_k=eligible.size,
            )
            dense_ranks = np.empty(eligible.size, dtype=np.float32)
            dense_ranks[dense_order] = np.arange(
                eligible.size,
                dtype=np.float32,
            )
            baseline_ranks = np.arange(eligible.size, dtype=np.float32)
            final = (
                1.0 / (rrf_k + baseline_ranks + 1.0)
                + weight / (rrf_k + dense_ranks + 1.0)
            )
            tail_order = base.top_indices(final, top_k=tail_budget)
            local = np.concatenate((protected, eligible[tail_order]))
        rankings.append(np.asarray(candidates, dtype=np.int64)[local])
    return rankings


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
    storage, bytes_per_document = m1971.storage_for_mode(doc_values, mode=mode)
    compact_rows, latency_ms, touched_docs = m200.score_compact_top256(
        storage=storage,
        query_values=query_values,
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        mode=mode,
        correction_window=args.correction_window,
    )
    rankings = rrf_rankings(
        bundle.candidates,
        bundle.baseline_rows,
        compact_rows,
        weight=args.rrf_weight,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        rrf_k=args.rrf_k,
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
    allowed_modes = {'float32', 'int4_row'}
    if set(modes) - allowed_modes:
        raise ValueError(f'unsupported modes: {set(modes) - allowed_modes}')
    dense_reference, exact_rrf_macro = load_reference(args.reference_summary)
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
            rrf_k=args.rrf_k,
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
            macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
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
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'shared projection head against RRF rank component',
        'exact_rrf_macro_reference': exact_rrf_macro,
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'correction_window': args.correction_window,
            'protected_head': args.protected_head,
            'rrf_weight': args.rrf_weight,
            'rrf_k': args.rrf_k,
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
        '# M208 RRF rank-distilled mini-vector head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
        f"- RRF weight: `{payload['policy']['rrf_weight']}`",
        f"- RRF k: `{payload['policy']['rrf_k']}`",
        '',
        '## Variants',
        '',
        '| Dim | Mode | Bytes/doc | Macro dMAP vs dense | '
        'Macro dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for variant in payload['variants']:
        macro_dense = variant['macro_delta_vs_pplx_dense']
        first_row = variant['rows'][0]
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {variant['dimension']} | {variant['mode']} | "
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
