#!/usr/bin/env python3
"""M210 non-linear compact mini-vector head.

M209B showed that a linear action-distilled head is not enough for the
last-mile top256 rerank problem.  M210A changes the model class while keeping
the deployment contract fixed: the exported payload is still a 32d INT4
mini-vector, but training can use a small residual MLP initialized from the
stable PCA/linear head.  Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
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


SCHEMA = 'm210_nonlinear_mini_vector_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


@dataclass(frozen=True)
class Operator:
    family: str
    protected_head: int
    correction_window: int
    weight: float
    gate_threshold: float | None = None
    gate_rank: int | None = None

    def key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.protected_head,
            self.correction_window,
            self.weight,
            self.gate_threshold,
            self.gate_rank,
        )

    def label(self) -> str:
        parts = [
            self.family,
            f'head={self.protected_head}',
            f'window={self.correction_window}',
            f'w={self.weight:g}',
        ]
        if self.gate_threshold is not None:
            parts.append(f'gate={self.gate_threshold:g}')
        if self.gate_rank is not None:
            parts.append(f'gate_rank={self.gate_rank}')
        return ', '.join(parts)


class ResidualMLPHead(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        init_basis: np.ndarray,
        hidden_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.base = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.residual_scale = float(residual_scale)
        with torch.no_grad():
            weight = torch.from_numpy(init_basis[:, :output_dim].T.copy())
            self.base.weight.copy_(weight)
            final = self.residual[-1]
            assert isinstance(final, torch.nn.Linear)
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.residual_scale * self.residual(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate residual MLP mini-vector heads.',
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
    parser.add_argument('--learning-rate', type=float, default=0.0015)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--residual-scale', type=float, default=0.25)
    parser.add_argument('--softmax-temperature', type=float, default=0.35)
    parser.add_argument('--pairwise-weight', type=float, default=0.05)
    parser.add_argument('--pair-top-k', type=int, default=16)
    parser.add_argument('--pair-negative-k', type=int, default=64)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--protected-heads', default='0')
    parser.add_argument('--correction-windows', default='256')
    parser.add_argument('--score-weights', default='0.75,1.0,1.25,1.5')
    parser.add_argument('--correction-margin-gates', default='0.1,0.2')
    parser.add_argument('--baseline-ambiguity-gates', default='0.1,0.2')
    parser.add_argument('--gate-rank', type=int, default=16)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=210)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
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


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('integer list must not be empty')
    return values


def normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - torch.mean(values)
    scale = torch.std(centered, unbiased=False)
    return centered / torch.clamp(scale, min=1.0e-6)


def torch_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = torch.min(values)
    maximum = torch.max(values)
    scale = torch.clamp(maximum - minimum, min=1.0e-6)
    return (values - minimum) / scale


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


def score_shape_loss(
    *,
    prediction_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
    pairwise_weight: float,
    pair_top_k: int,
    pair_negative_k: int,
) -> torch.Tensor:
    target = normalize(teacher_scores)
    prediction = normalize(prediction_scores)
    mse = F.mse_loss(prediction, target)
    target_distribution = torch.softmax(target / temperature, dim=0)
    kl = -torch.sum(
        target_distribution * F.log_softmax(prediction / temperature, dim=0),
    )
    pairwise = pairwise_top_loss(
        prediction,
        target,
        top_k=pair_top_k,
        negative_k=pair_negative_k,
    )
    return mse + 0.1 * kl + pairwise_weight * pairwise


def sample_indices(
    bundles: Sequence[m202.DatasetBundle],
) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for bundle_index, bundle in enumerate(bundles):
        for row, window in enumerate(bundle.windows):
            if len(window) >= 8:
                samples.append((bundle_index, row))
    return samples


def train_residual_mlp_head(
    *,
    bundles: Sequence[m202.DatasetBundle],
    init_basis: np.ndarray,
    dimension: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    residual_scale: float,
    temperature: float,
    pairwise_weight: float,
    pair_top_k: int,
    pair_negative_k: int,
    seed: int,
    device: str,
) -> tuple[ResidualMLPHead, ResidualMLPHead, list[dict[str, float]]]:
    torch.manual_seed(seed + dimension)
    rng = random.Random(seed + dimension)
    input_dim = int(init_basis.shape[0])
    doc_model = ResidualMLPHead(
        input_dim=input_dim,
        output_dim=dimension,
        init_basis=init_basis,
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
    ).to(device)
    query_model = ResidualMLPHead(
        input_dim=input_dim,
        output_dim=dimension,
        init_basis=init_basis,
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(doc_model.parameters()) + list(query_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    samples = sample_indices(bundles)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(samples)
        losses: list[float] = []
        for bundle_index, row in samples:
            bundle = bundles[bundle_index]
            docs = bundle.windows[row]
            doc_inputs = torch.from_numpy(bundle.dense_documents[docs]).to(device)
            query_inputs = torch.from_numpy(
                bundle.dense_queries[row:row + 1],
            ).to(device)
            doc_values = doc_model(doc_inputs)
            query_values = query_model(query_inputs)[0]
            prediction_scores = doc_values @ query_values
            teacher_scores = torch.from_numpy(bundle.teachers[row]).to(device)
            loss = score_shape_loss(
                prediction_scores=prediction_scores,
                teacher_scores=teacher_scores,
                temperature=temperature,
                pairwise_weight=pairwise_weight,
                pair_top_k=pair_top_k,
                pair_negative_k=pair_negative_k,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(doc_model.parameters()) + list(query_model.parameters()),
                5.0,
            )
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
                f'm210 residual_mlp dim={dimension} '
                f'epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f}'
            )
    return doc_model, query_model, history


def encode_matrix(
    *,
    model: torch.nn.Module,
    matrix: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(matrix.shape[0], start + batch_size)
            values = torch.from_numpy(matrix[start:stop]).to(device)
            encoded = model(values).detach().cpu().numpy()
            outputs.append(encoded.astype(np.float32, copy=False))
    return np.vstack(outputs).astype(np.float32, copy=False)


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


def row_safe(delta: dict[str, float], *, floor: float = -0.002) -> bool:
    return min(delta[name] for name in PRIMARY_METRICS) >= floor


def margin_at_rank(values: np.ndarray, *, rank: int) -> float:
    if values.size < 2:
        return 0.0
    normalized = m1971.minmax(values)
    order = base.top_indices(normalized, top_k=normalized.size)
    compare_at = min(max(rank, 2), order.size) - 1
    return float(normalized[order[0]] - normalized[order[compare_at]])


def final_scores(
    *,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    operator: Operator,
    gate_decisions: list[float],
) -> np.ndarray:
    corrected = (
        baseline_values
        + operator.weight * m1971.minmax(correction_values)
    )
    if operator.family == 'score_minmax':
        return corrected
    if operator.family == 'correction_margin_gate':
        assert operator.gate_threshold is not None
        assert operator.gate_rank is not None
        gate = (
            margin_at_rank(correction_values, rank=operator.gate_rank)
            >= operator.gate_threshold
        )
        gate_decisions.append(float(gate))
        return corrected if gate else baseline_values
    if operator.family == 'baseline_ambiguity_gate':
        assert operator.gate_threshold is not None
        assert operator.gate_rank is not None
        gate = (
            margin_at_rank(baseline_values, rank=operator.gate_rank)
            <= operator.gate_threshold
        )
        gate_decisions.append(float(gate))
        return corrected if gate else baseline_values
    raise ValueError(f'unsupported operator family: {operator.family}')


def operator_rankings(
    *,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    correction_rows: Sequence[np.ndarray],
    operator: Operator,
    top_k: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    rankings: list[np.ndarray] = []
    gate_decisions: list[float] = []
    for candidates, baseline, correction in zip(
        candidate_rows,
        baseline_rows,
        correction_rows,
    ):
        baseline = np.asarray(baseline, dtype=np.float32)
        correction = np.asarray(correction, dtype=np.float32)
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        head_size = min(operator.protected_head, top_k, len(candidates))
        window_size = min(operator.correction_window, len(candidates))
        protected = baseline_order[:head_size]
        eligible = baseline_order[head_size:window_size]
        tail_budget = max(top_k - head_size, 0)
        if eligible.size == 0 or tail_budget == 0:
            local = protected
        else:
            scores = final_scores(
                baseline_values=baseline[eligible],
                correction_values=correction[eligible],
                operator=operator,
                gate_decisions=gate_decisions,
            )
            tail_order = base.top_indices(scores, top_k=tail_budget)
            local = np.concatenate((protected, eligible[tail_order]))
        rankings.append(np.asarray(candidates, dtype=np.int64)[local])
    stats = {
        'gate_rate': (
            statistics.fmean(gate_decisions) if gate_decisions else 1.0
        ),
    }
    return rankings, stats


def build_operators(args: argparse.Namespace) -> list[Operator]:
    heads = parse_int_csv(args.protected_heads)
    windows = parse_int_csv(args.correction_windows)
    weights = parse_float_csv(args.score_weights)
    correction_gates = parse_float_csv(args.correction_margin_gates)
    baseline_gates = parse_float_csv(args.baseline_ambiguity_gates)
    operators: list[Operator] = []
    for head in heads:
        for window in windows:
            for weight in weights:
                operators.append(
                    Operator(
                        family='score_minmax',
                        protected_head=head,
                        correction_window=window,
                        weight=weight,
                    ),
                )
                for threshold in correction_gates:
                    operators.append(
                        Operator(
                            family='correction_margin_gate',
                            protected_head=head,
                            correction_window=window,
                            weight=weight,
                            gate_threshold=threshold,
                            gate_rank=args.gate_rank,
                        ),
                    )
                for threshold in baseline_gates:
                    operators.append(
                        Operator(
                            family='baseline_ambiguity_gate',
                            protected_head=head,
                            correction_window=window,
                            weight=weight,
                            gate_threshold=threshold,
                            gate_rank=args.gate_rank,
                        ),
                    )
    return operators


def load_dense_reference(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['dataset']: row['pplx_dense']
        for row in payload.get('dataset_baselines', [])
    }


def compact_correction_rows(
    *,
    bundle: m202.DatasetBundle,
    doc_values: np.ndarray,
    query_values: np.ndarray,
    mode: str,
    correction_window: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    storage, bytes_per_document = m1971.storage_for_mode(doc_values, mode=mode)
    compact_rows, latency_ms, touched_docs = m200.score_compact_top256(
        storage=storage,
        query_values=query_values,
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        mode=mode,
        correction_window=correction_window,
    )
    return compact_rows, {
        'bytes_per_document_payload': bytes_per_document,
        'mean_payload_bytes_per_query': (
            statistics.fmean(touched_docs) * bytes_per_document
        ),
        'score_latency_ms': {
            'p50': float(np.percentile(latency_ms, 50)),
            'p95': float(np.percentile(latency_ms, 95)),
        },
    }


def evaluate_operator(
    *,
    bundle: m202.DatasetBundle,
    correction_rows: Sequence[np.ndarray],
    operator: Operator,
    dense_reference: dict[str, dict[str, float]],
    top_k: int,
) -> dict[str, Any]:
    rankings, stats = operator_rankings(
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        correction_rows=correction_rows,
        operator=operator,
        top_k=top_k,
    )
    metrics = m1971.retrieval_metrics(
        rankings,
        dense_queries=bundle.dense_queries_raw,
        dense_documents=bundle.dense_documents_raw,
        qrels=bundle.qrels,
    )
    delta_vs_fixed = metric_delta(metrics, bundle.baseline_metrics)
    delta_vs_dense = metric_delta(metrics, dense_reference[bundle.name])
    return {
        'dataset': bundle.name,
        'metrics': metrics,
        'delta_vs_fixed_hybrid': delta_vs_fixed,
        'delta_vs_pplx_dense': delta_vs_dense,
        'safe_vs_fixed_at_0.002': row_safe(delta_vs_fixed),
        'safe_vs_dense_at_0.002': row_safe(delta_vs_dense),
        'gate_rate': stats['gate_rate'],
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
    operators = build_operators(args)
    max_window = max(operator.correction_window for operator in operators)
    dense_reference = load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    loader_args.correction_window = max_window
    loader_args.protected_head = 0
    train_bundles = [
        m202.load_bundle(dataset=name, args=loader_args)
        for name in train_names
    ]
    bundle_cache = {bundle.name: bundle for bundle in train_bundles}
    eval_bundles: list[m202.DatasetBundle] = []
    for name in eval_names:
        bundle = bundle_cache.get(name)
        if bundle is None:
            bundle = m202.load_bundle(dataset=name, args=loader_args)
            bundle_cache[name] = bundle
        eval_bundles.append(bundle)

    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )
    variants: list[dict[str, Any]] = []
    checkpoints: dict[int, tuple[ResidualMLPHead, ResidualMLPHead]] = {}
    histories: dict[int, list[dict[str, float]]] = {}
    for dimension in dimensions:
        doc_model, query_model, history = train_residual_mlp_head(
            bundles=train_bundles,
            init_basis=basis,
            dimension=dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            residual_scale=args.residual_scale,
            temperature=args.softmax_temperature,
            pairwise_weight=args.pairwise_weight,
            pair_top_k=args.pair_top_k,
            pair_negative_k=args.pair_negative_k,
            seed=args.seed,
            device=device,
        )
        checkpoints[dimension] = (doc_model, query_model)
        histories[dimension] = history
        encoded_by_dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for bundle in eval_bundles:
            doc_values = encode_matrix(
                model=doc_model,
                matrix=bundle.dense_documents,
                device=device,
                batch_size=args.encode_batch_size,
            )
            query_values = encode_matrix(
                model=query_model,
                matrix=bundle.dense_queries,
                device=device,
                batch_size=args.encode_batch_size,
            )
            encoded_by_dataset[bundle.name] = (doc_values, query_values)
        for mode in modes:
            correction_by_dataset: dict[str, list[np.ndarray]] = {}
            compact_stats_by_dataset: dict[str, dict[str, Any]] = {}
            for bundle in eval_bundles:
                doc_values, query_values = encoded_by_dataset[bundle.name]
                correction_rows, compact_stats = compact_correction_rows(
                    bundle=bundle,
                    doc_values=doc_values,
                    query_values=query_values,
                    mode=mode,
                    correction_window=max_window,
                )
                correction_by_dataset[bundle.name] = correction_rows
                compact_stats_by_dataset[bundle.name] = compact_stats
            for operator in operators:
                rows = [
                    evaluate_operator(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        operator=operator,
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in eval_bundles
                ]
                macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
                first_stats = compact_stats_by_dataset[eval_bundles[0].name]
                variants.append(
                    {
                        'architecture': 'residual_mlp',
                        'dimension': dimension,
                        'mode': mode,
                        'operator': operator.label(),
                        'operator_key': list(operator.key()),
                        'family': operator.family,
                        'protected_head': operator.protected_head,
                        'correction_window': operator.correction_window,
                        'weight': operator.weight,
                        'gate_threshold': operator.gate_threshold,
                        'gate_rank': operator.gate_rank,
                        'mean_gate_rate': statistics.fmean(
                            row['gate_rate'] for row in rows
                        ),
                        'training_final': history[-1],
                        'compact_stats': first_stats,
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
                            if not row['safe_vs_fixed_at_0.002']
                        ],
                        'unsafe_rows_vs_dense': [
                            row['dataset']
                            for row in rows
                            if not row['safe_vs_dense_at_0.002']
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
            -len(item['unsafe_rows_vs_dense']),
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses dense teacher only',
        'training_scope': 'residual MLP score-shape mini-vector head',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'operator_count': len(operators),
            'dimensions': list(dimensions),
            'quantization': list(modes),
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'hidden_dim': args.hidden_dim,
            'residual_scale': args.residual_scale,
            'softmax_temperature': args.softmax_temperature,
            'pairwise_weight': args.pairwise_weight,
            'pair_top_k': args.pair_top_k,
            'pair_negative_k': args.pair_negative_k,
            'protected_heads': parse_int_csv(args.protected_heads),
            'correction_windows': parse_int_csv(args.correction_windows),
            'score_weights': parse_float_csv(args.score_weights),
            'correction_margin_gates': parse_float_csv(
                args.correction_margin_gates,
            ),
            'baseline_ambiguity_gates': parse_float_csv(
                args.baseline_ambiguity_gates,
            ),
            'gate_rank': args.gate_rank,
            'device': device,
        },
        'training_history': {
            str(dimension): histories[dimension]
            for dimension in dimensions
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
    for dimension, (doc_model, query_model) in checkpoints.items():
        torch.save(
            {
                'schema': SCHEMA,
                'dimension': dimension,
                'doc_model': doc_model.cpu().state_dict(),
                'query_model': query_model.cpu().state_dict(),
                'policy': payload['policy'],
            },
            args.output_dir / f'residual_mlp_dim{dimension}.pt',
        )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M210 non-linear mini-vector head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Architecture: `{payload['training_scope']}`",
        f"- Epochs: `{payload['policy']['epochs']}`",
        '',
        '## Top 32d INT4 Variants',
        '',
        '| Rank | Operator | Bytes/doc | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | --- | --- |',
    ]
    rank = 0
    for variant in payload['variants']:
        if variant['dimension'] != 32 or variant['mode'] != 'int4_row':
            continue
        rank += 1
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        bytes_per_doc = variant['compact_stats']['bytes_per_document_payload']
        lines.append(
            f"| {rank} | {variant['operator']} | {bytes_per_doc} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
        if rank >= 30:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
