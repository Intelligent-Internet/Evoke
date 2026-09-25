#!/usr/bin/env python3
"""M211 query-gated mini-vector correction.

M210A showed that a residual MLP head improves recall but still misses dense
MAP under unconditional score-top256 correction.  M211 keeps the trained M210
mini-vector fixed and trains a small query-level gate from qrels-free features.
The gate learns whether the learned mini-vector rerank is closer to the exact
dense top256 teacher than the baseline ranking.  Qrels are used only for final
metrics.
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
import research_sae_m210_nonlinear_mini_vector_head as m210


SCHEMA = 'm211_query_gated_mini_vector_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


@dataclass(frozen=True)
class Operator:
    protected_head: int
    correction_window: int
    weight: float

    def label(self) -> str:
        return (
            f'score_minmax, head={self.protected_head}, '
            f'window={self.correction_window}, w={self.weight:g}'
        )


@dataclass
class GateState:
    mean: np.ndarray
    std: np.ndarray
    state_dict: dict[str, torch.Tensor] | None
    hidden_dim: int
    constant_probability: float | None
    feature_names: tuple[str, ...]
    history: list[dict[str, float]]
    positive_rate: float


class GateNet(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate query-gated M210 mini-vector correction.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimensions', default='32,64')
    parser.add_argument('--quantization', default='float32,int4_row')
    parser.add_argument('--score-weights', default='0.5,0.75,1.0,1.25')
    parser.add_argument('--gate-thresholds', default='0.4,0.5,0.6')
    parser.add_argument('--gate-hidden-dim', type=int, default=16)
    parser.add_argument('--gate-epochs', type=int, default=200)
    parser.add_argument('--gate-learning-rate', type=float, default=0.01)
    parser.add_argument('--label-margin', type=float, default=0.002)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=211)
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


def load_dense_reference(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['dataset']: row['pplx_dense']
        for row in payload.get('dataset_baselines', [])
    }


def load_m210_models(
    *,
    checkpoint_dir: Path,
    dimension: int,
    basis: np.ndarray,
    device: str,
) -> tuple[m210.ResidualMLPHead, m210.ResidualMLPHead, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_dir / f'residual_mlp_dim{dimension}.pt',
        map_location=device,
    )
    policy = checkpoint['policy']
    doc_model = m210.ResidualMLPHead(
        input_dim=basis.shape[0],
        output_dim=dimension,
        init_basis=basis,
        hidden_dim=int(policy['hidden_dim']),
        residual_scale=float(policy['residual_scale']),
    ).to(device)
    query_model = m210.ResidualMLPHead(
        input_dim=basis.shape[0],
        output_dim=dimension,
        init_basis=basis,
        hidden_dim=int(policy['hidden_dim']),
        residual_scale=float(policy['residual_scale']),
    ).to(device)
    doc_model.load_state_dict(checkpoint['doc_model'])
    query_model.load_state_dict(checkpoint['query_model'])
    doc_model.eval()
    query_model.eval()
    return doc_model, query_model, policy


def exact_dense_rows(
    bundle: m202.DatasetBundle,
    *,
    correction_window: int,
) -> list[np.ndarray]:
    return m200.score_exact_top256(
        dense_documents=bundle.dense_documents,
        dense_queries=bundle.dense_queries,
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        correction_window=correction_window,
    )


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


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1.0e-8 or right_std <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def top_overlap(left: np.ndarray, right: np.ndarray, *, top_k: int) -> float:
    count = min(top_k, left.size, right.size)
    if count <= 0:
        return 0.0
    return len(set(map(int, left[:count])) & set(map(int, right[:count]))) / count


def margin(values: np.ndarray, order: np.ndarray, *, rank: int) -> float:
    if values.size < 2 or order.size < 2:
        return 0.0
    normalized = m1971.minmax(values)
    compare_at = min(max(rank, 2), order.size) - 1
    return float(normalized[order[0]] - normalized[order[compare_at]])


def feature_names() -> tuple[str, ...]:
    return (
        'baseline_margin_16',
        'baseline_margin_100',
        'correction_margin_16',
        'correction_margin_100',
        'final_margin_16',
        'final_margin_100',
        'baseline_correction_corr',
        'overlap_baseline_correction_32',
        'overlap_baseline_correction_100',
        'overlap_baseline_final_32',
        'overlap_baseline_final_100',
        'correction_std',
        'final_std',
        'top1_changed',
        'top10_changed_fraction',
    )


def query_features(
    *,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    operator: Operator,
) -> np.ndarray:
    baseline = np.asarray(baseline_values, dtype=np.float32)
    correction = np.asarray(correction_values, dtype=np.float32)
    baseline_order = base.top_indices(baseline, top_k=baseline.size)
    head_size = min(operator.protected_head, baseline_order.size)
    window_size = min(operator.correction_window, baseline_order.size)
    eligible = baseline_order[head_size:window_size]
    if eligible.size == 0:
        return np.zeros(len(feature_names()), dtype=np.float32)
    local_baseline = baseline[eligible]
    local_correction = correction[eligible]
    correction_norm = m1971.minmax(local_correction)
    final = local_baseline + operator.weight * correction_norm
    b_order = base.top_indices(local_baseline, top_k=eligible.size)
    c_order = base.top_indices(local_correction, top_k=eligible.size)
    f_order = base.top_indices(final, top_k=eligible.size)
    top10 = min(10, eligible.size)
    top10_changed = 1.0 - top_overlap(b_order, f_order, top_k=top10)
    values = [
        margin(local_baseline, b_order, rank=16),
        margin(local_baseline, b_order, rank=100),
        margin(local_correction, c_order, rank=16),
        margin(local_correction, c_order, rank=100),
        margin(final, f_order, rank=16),
        margin(final, f_order, rank=100),
        safe_corr(m1971.minmax(local_baseline), correction_norm),
        top_overlap(b_order, c_order, top_k=32),
        top_overlap(b_order, c_order, top_k=100),
        top_overlap(b_order, f_order, top_k=32),
        top_overlap(b_order, f_order, top_k=100),
        float(np.std(correction_norm)),
        float(np.std(m1971.minmax(final))),
        float(int(b_order[0] != f_order[0])) if f_order.size else 0.0,
        top10_changed,
    ]
    return np.asarray(values, dtype=np.float32)


def agreement_score(
    predicted_order: np.ndarray,
    teacher_order: np.ndarray,
    *,
    top_k: int,
) -> float:
    k = min(top_k, predicted_order.size, teacher_order.size)
    if k <= 0:
        return 0.0
    gains = {
        int(doc): 1.0 / np.log2(rank + 2.0)
        for rank, doc in enumerate(teacher_order[:k])
    }
    dcg = 0.0
    for rank, doc in enumerate(predicted_order[:k]):
        dcg += gains.get(int(doc), 0.0) / np.log2(rank + 2.0)
    ideal_gains = sorted(gains.values(), reverse=True)
    ideal = sum(
        gain / np.log2(rank + 2.0)
        for rank, gain in enumerate(ideal_gains)
    )
    return float(dcg / ideal) if ideal > 0.0 else 0.0


def teacher_label(
    *,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    teacher_values: np.ndarray,
    operator: Operator,
    top_k: int,
    label_margin: float,
) -> tuple[float, float]:
    baseline = np.asarray(baseline_values, dtype=np.float32)
    correction = np.asarray(correction_values, dtype=np.float32)
    teacher = np.asarray(teacher_values, dtype=np.float32)
    baseline_order = base.top_indices(baseline, top_k=baseline.size)
    window_size = min(operator.correction_window, baseline_order.size)
    eligible = baseline_order[:window_size]
    if eligible.size == 0:
        return 0.0, 0.0
    local_baseline = baseline[eligible]
    learned_final = (
        local_baseline + operator.weight * m1971.minmax(correction[eligible])
    )
    teacher_final = (
        local_baseline + operator.weight * m1971.minmax(teacher[eligible])
    )
    baseline_local = base.top_indices(local_baseline, top_k=eligible.size)
    learned_local = base.top_indices(learned_final, top_k=eligible.size)
    teacher_local = base.top_indices(teacher_final, top_k=eligible.size)
    baseline_docs = eligible[baseline_local]
    learned_docs = eligible[learned_local]
    teacher_docs = eligible[teacher_local]
    learned_score = agreement_score(
        learned_docs,
        teacher_docs,
        top_k=top_k,
    )
    baseline_score = agreement_score(
        baseline_docs,
        teacher_docs,
        top_k=top_k,
    )
    benefit = learned_score - baseline_score
    return float(benefit > label_margin), float(benefit)


def collect_gate_training_data(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    exact_by_dataset: dict[str, list[np.ndarray]],
    operator: Operator,
    top_k: int,
    label_margin: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    benefits: list[float] = []
    for bundle in bundles:
        correction_rows = correction_by_dataset[bundle.name]
        exact_rows = exact_by_dataset[bundle.name]
        for baseline, correction, teacher in zip(
            bundle.baseline_rows,
            correction_rows,
            exact_rows,
        ):
            features.append(
                query_features(
                    baseline_values=baseline,
                    correction_values=correction,
                    operator=operator,
                ),
            )
            label, benefit = teacher_label(
                baseline_values=baseline,
                correction_values=correction,
                teacher_values=teacher,
                operator=operator,
                top_k=top_k,
                label_margin=label_margin,
            )
            labels.append(label)
            benefits.append(benefit)
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        benefits,
    )


def train_gate(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> GateState:
    names = feature_names()
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std <= 1.0e-6] = 1.0
    positive_rate = float(labels.mean()) if labels.size else 0.0
    if labels.size == 0 or positive_rate in (0.0, 1.0):
        return GateState(
            mean=mean,
            std=std,
            state_dict=None,
            hidden_dim=hidden_dim,
            constant_probability=positive_rate,
            feature_names=names,
            history=[],
            positive_rate=positive_rate,
        )
    torch.manual_seed(seed)
    random.seed(seed)
    model = GateNet(input_dim=features.shape[1], hidden_dim=hidden_dim).to(device)
    x = torch.from_numpy((features - mean) / std).to(device)
    y = torch.from_numpy(labels).to(device)
    pos = max(float(labels.sum()), 1.0)
    neg = max(float(labels.size - labels.sum()), 1.0)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                probs = torch.sigmoid(model(x))
                preds = (probs >= 0.5).to(dtype=torch.float32)
                accuracy = float((preds == y).to(dtype=torch.float32).mean())
            history.append(
                {
                    'epoch': float(epoch + 1),
                    'loss': float(loss.detach().cpu()),
                    'accuracy': accuracy,
                },
            )
    return GateState(
        mean=mean,
        std=std,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        hidden_dim=hidden_dim,
        constant_probability=None,
        feature_names=names,
        history=history,
        positive_rate=positive_rate,
    )


def gate_probabilities(
    *,
    state: GateState,
    features: np.ndarray,
    device: str,
) -> np.ndarray:
    if state.constant_probability is not None:
        return np.full(
            features.shape[0],
            state.constant_probability,
            dtype=np.float32,
        )
    model = GateNet(
        input_dim=features.shape[1],
        hidden_dim=state.hidden_dim,
    ).to(device)
    assert state.state_dict is not None
    model.load_state_dict(state.state_dict)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy((features - state.mean) / state.std).to(device)
        probs = torch.sigmoid(model(x)).detach().cpu().numpy()
    return probs.astype(np.float32, copy=False)


def collect_eval_features(
    *,
    bundle: m202.DatasetBundle,
    correction_rows: Sequence[np.ndarray],
    exact_rows: Sequence[np.ndarray],
    operator: Operator,
    top_k: int,
    label_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    oracle_labels: list[float] = []
    benefits: list[float] = []
    for baseline, correction, teacher in zip(
        bundle.baseline_rows,
        correction_rows,
        exact_rows,
    ):
        features.append(
            query_features(
                baseline_values=baseline,
                correction_values=correction,
                operator=operator,
            ),
        )
        label, benefit = teacher_label(
            baseline_values=baseline,
            correction_values=correction,
            teacher_values=teacher,
            operator=operator,
            top_k=top_k,
            label_margin=label_margin,
        )
        oracle_labels.append(label)
        benefits.append(benefit)
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(oracle_labels, dtype=np.float32),
        np.asarray(benefits, dtype=np.float32),
    )


def gated_rankings(
    *,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    correction_rows: Sequence[np.ndarray],
    operator: Operator,
    alphas: np.ndarray,
    top_k: int,
) -> list[np.ndarray]:
    rankings: list[np.ndarray] = []
    for candidates, baseline, correction, alpha in zip(
        candidate_rows,
        baseline_rows,
        correction_rows,
        alphas,
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
            scores = baseline[eligible] + (
                float(alpha)
                * operator.weight
                * m1971.minmax(correction[eligible])
            )
            tail_order = base.top_indices(scores, top_k=tail_budget)
            local = np.concatenate((protected, eligible[tail_order]))
        rankings.append(np.asarray(candidates, dtype=np.int64)[local])
    return rankings


def evaluate_variant(
    *,
    bundle: m202.DatasetBundle,
    correction_rows: Sequence[np.ndarray],
    operator: Operator,
    alphas: np.ndarray,
    dense_reference: dict[str, dict[str, float]],
    top_k: int,
) -> dict[str, Any]:
    rankings = gated_rankings(
        candidate_rows=bundle.candidates,
        baseline_rows=bundle.baseline_rows,
        correction_rows=correction_rows,
        operator=operator,
        alphas=alphas,
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
        'gate_rate': float(np.mean(alphas > 0.5)),
        'mean_alpha': float(np.mean(alphas)),
    }


def add_variant(
    *,
    variants: list[dict[str, Any]],
    dimension: int,
    mode: str,
    operator: Operator,
    gate_family: str,
    gate_threshold: float | None,
    deployable: bool,
    gate_state: GateState | None,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any],
) -> None:
    macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': dimension,
            'mode': mode,
            'operator': operator.label(),
            'gate_family': gate_family,
            'gate_threshold': gate_threshold,
            'deployable': deployable,
            'weight': operator.weight,
            'training_final': (
                gate_state.history[-1]
                if gate_state is not None and gate_state.history else None
            ),
            'gate_positive_rate': (
                gate_state.positive_rate if gate_state is not None else None
            ),
            'compact_stats': compact_stats,
            'macro_metrics': macro_metrics(rows),
            'macro_delta_vs_fixed_hybrid': macro_delta(
                rows,
                'delta_vs_fixed_hybrid',
            ),
            'macro_delta_vs_pplx_dense': macro_vs_dense,
            'beats_pplx_dense_primary': all(
                macro_vs_dense[name] > 0.0 for name in PRIMARY_METRICS
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    dimensions = base.parse_int_csv(args.dimensions)
    modes = parse_str_csv(args.quantization)
    weights = parse_float_csv(args.score_weights)
    thresholds = parse_float_csv(args.gate_thresholds)
    allowed_modes = {'float32', 'int4_row'}
    if set(modes) - allowed_modes:
        raise ValueError(f'unsupported modes: {set(modes) - allowed_modes}')
    dense_reference = load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
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
    all_bundles = list(bundle_cache.values())
    exact_by_dataset = {
        bundle.name: exact_dense_rows(
            bundle,
            correction_window=args.correction_window,
        )
        for bundle in all_bundles
    }
    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )

    variants: list[dict[str, Any]] = []
    gate_training: list[dict[str, Any]] = []
    for dimension in dimensions:
        doc_model, query_model, m210_policy = load_m210_models(
            checkpoint_dir=args.m210_checkpoint_dir,
            dimension=dimension,
            basis=basis,
            device=device,
        )
        encoded_by_dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for bundle in all_bundles:
            doc_values = m210.encode_matrix(
                model=doc_model,
                matrix=bundle.dense_documents,
                device=device,
                batch_size=args.encode_batch_size,
            )
            query_values = m210.encode_matrix(
                model=query_model,
                matrix=bundle.dense_queries,
                device=device,
                batch_size=args.encode_batch_size,
            )
            encoded_by_dataset[bundle.name] = (doc_values, query_values)
        for mode in modes:
            correction_by_dataset: dict[str, list[np.ndarray]] = {}
            compact_stats_by_dataset: dict[str, dict[str, Any]] = {}
            for bundle in all_bundles:
                doc_values, query_values = encoded_by_dataset[bundle.name]
                correction_rows, compact_stats = compact_correction_rows(
                    bundle=bundle,
                    doc_values=doc_values,
                    query_values=query_values,
                    mode=mode,
                    correction_window=args.correction_window,
                )
                correction_by_dataset[bundle.name] = correction_rows
                compact_stats_by_dataset[bundle.name] = compact_stats
            for weight in weights:
                operator = Operator(
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    weight=weight,
                )
                train_x, train_y, train_benefits = collect_gate_training_data(
                    bundles=train_bundles,
                    correction_by_dataset=correction_by_dataset,
                    exact_by_dataset=exact_by_dataset,
                    operator=operator,
                    top_k=args.top_k,
                    label_margin=args.label_margin,
                )
                gate_state = train_gate(
                    features=train_x,
                    labels=train_y,
                    hidden_dim=args.gate_hidden_dim,
                    epochs=args.gate_epochs,
                    learning_rate=args.gate_learning_rate,
                    seed=args.seed + dimension + int(weight * 1000),
                    device=device,
                )
                gate_training.append(
                    {
                        'dimension': dimension,
                        'mode': mode,
                        'weight': weight,
                        'positive_rate': gate_state.positive_rate,
                        'mean_teacher_benefit': (
                            statistics.fmean(train_benefits)
                            if train_benefits else 0.0
                        ),
                        'history': gate_state.history,
                    },
                )

                eval_feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                eval_benefit_cache: dict[str, np.ndarray] = {}
                prob_by_dataset: dict[str, np.ndarray] = {}
                for bundle in eval_bundles:
                    features, oracle_labels, benefits = collect_eval_features(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        exact_rows=exact_by_dataset[bundle.name],
                        operator=operator,
                        top_k=args.top_k,
                        label_margin=args.label_margin,
                    )
                    eval_feature_cache[bundle.name] = (features, oracle_labels)
                    eval_benefit_cache[bundle.name] = benefits
                    prob_by_dataset[bundle.name] = gate_probabilities(
                        state=gate_state,
                        features=features,
                        device=device,
                    )

                first_stats = compact_stats_by_dataset[eval_bundles[0].name]
                ungated_rows = [
                    evaluate_variant(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        operator=operator,
                        alphas=np.ones(
                            len(bundle.candidates),
                            dtype=np.float32,
                        ),
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in eval_bundles
                ]
                add_variant(
                    variants=variants,
                    dimension=dimension,
                    mode=mode,
                    operator=operator,
                    gate_family='ungated',
                    gate_threshold=None,
                    deployable=True,
                    gate_state=None,
                    rows=ungated_rows,
                    compact_stats=first_stats,
                )
                soft_rows = [
                    evaluate_variant(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        operator=operator,
                        alphas=prob_by_dataset[bundle.name],
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in eval_bundles
                ]
                add_variant(
                    variants=variants,
                    dimension=dimension,
                    mode=mode,
                    operator=operator,
                    gate_family='learned_soft',
                    gate_threshold=None,
                    deployable=True,
                    gate_state=gate_state,
                    rows=soft_rows,
                    compact_stats=first_stats,
                )
                for threshold in thresholds:
                    hard_rows = [
                        evaluate_variant(
                            bundle=bundle,
                            correction_rows=correction_by_dataset[bundle.name],
                            operator=operator,
                            alphas=(
                                prob_by_dataset[bundle.name] >= threshold
                            ).astype(np.float32),
                            dense_reference=dense_reference,
                            top_k=args.top_k,
                        )
                        for bundle in eval_bundles
                    ]
                    add_variant(
                        variants=variants,
                        dimension=dimension,
                        mode=mode,
                        operator=operator,
                        gate_family='learned_hard',
                        gate_threshold=threshold,
                        deployable=True,
                        gate_state=gate_state,
                        rows=hard_rows,
                        compact_stats=first_stats,
                    )
                oracle_rows = [
                    evaluate_variant(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        operator=operator,
                        alphas=eval_feature_cache[bundle.name][1],
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in eval_bundles
                ]
                add_variant(
                    variants=variants,
                    dimension=dimension,
                    mode=mode,
                    operator=operator,
                    gate_family='teacher_agreement_oracle',
                    gate_threshold=None,
                    deployable=False,
                    gate_state=None,
                    rows=oracle_rows,
                    compact_stats=first_stats,
                )
    variants.sort(
        key=lambda item: (
            item['deployable'],
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
        'qrels_usage': 'final metrics only; gate labels use dense teacher agreement',
        'training_scope': 'query gate over fixed M210 mini-vector correction',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimensions': list(dimensions),
            'quantization': list(modes),
            'score_weights': list(weights),
            'gate_thresholds': list(thresholds),
            'gate_hidden_dim': args.gate_hidden_dim,
            'gate_epochs': args.gate_epochs,
            'gate_learning_rate': args.gate_learning_rate,
            'label_margin': args.label_margin,
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
        },
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'gate_training': gate_training,
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
        '# M211 query-gated mini-vector correction',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Deployable 32d INT4 Variants',
        '',
        '| Rank | Gate | Operator | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | --- | --- | ---: | ---: | --- | --- |',
    ]
    rank = 0
    for variant in payload['variants']:
        if (
            not variant['deployable']
            or variant['dimension'] != 32
            or variant['mode'] != 'int4_row'
        ):
            continue
        rank += 1
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        gate = variant['gate_family']
        if variant['gate_threshold'] is not None:
            gate = f"{gate}@{variant['gate_threshold']:.2f}"
        lines.append(
            f"| {rank} | {gate} | {variant['operator']} | "
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
