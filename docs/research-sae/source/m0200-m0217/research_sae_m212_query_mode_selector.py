#!/usr/bin/env python3
"""M212 query-level mini-vector mode selector.

M211A showed that a binary query gate improves the M210 mini-vector path but
does not close the dense MAP gap.  M212A tests a more expressive selector:
for each query, choose a correction strength from a small discrete set,
including baseline-only.  The selector is trained from qrels-free agreement
with an exact dense top256 teacher.  Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211


SCHEMA = 'm212_query_mode_selector_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


class ModeNet(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate query-level correction mode selector.',
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
    parser.add_argument('--mode-weights', default='0.0,0.5,0.75,1.0,1.25,1.5')
    parser.add_argument('--confidence-thresholds', default='0.4,0.5,0.6')
    parser.add_argument('--teacher-weight', type=float, default=2.0)
    parser.add_argument('--selector-hidden-dim', type=int, default=32)
    parser.add_argument('--selector-epochs', type=int, default=250)
    parser.add_argument('--selector-learning-rate', type=float, default=0.01)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=212)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    return parser.parse_args()


def parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(',') if value.strip())
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


def feature_names(mode_weights: Sequence[float]) -> tuple[str, ...]:
    names: list[str] = []
    for weight in mode_weights:
        suffix = f'w{weight:g}'
        names.extend(
            f'{suffix}:{name}' for name in m211.feature_names()
        )
    return tuple(names)


def query_mode_features(
    *,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    mode_weights: Sequence[float],
    protected_head: int,
    correction_window: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for weight in mode_weights:
        operator = m211.Operator(
            protected_head=protected_head,
            correction_window=correction_window,
            weight=weight,
        )
        features.append(
            m211.query_features(
                baseline_values=baseline_values,
                correction_values=correction_values,
                operator=operator,
            ),
        )
    return np.concatenate(features).astype(np.float32, copy=False)


def local_docs_for_weight(
    *,
    candidates: np.ndarray,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    weight: float,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> np.ndarray:
    baseline = np.asarray(baseline_values, dtype=np.float32)
    correction = np.asarray(correction_values, dtype=np.float32)
    baseline_order = base.top_indices(baseline, top_k=len(candidates))
    head_size = min(protected_head, top_k, len(candidates))
    window_size = min(correction_window, len(candidates))
    protected = baseline_order[:head_size]
    eligible = baseline_order[head_size:window_size]
    tail_budget = max(top_k - head_size, 0)
    if eligible.size == 0 or tail_budget == 0:
        local = protected
    else:
        final = baseline[eligible] + weight * m1971.minmax(correction[eligible])
        tail_order = base.top_indices(final, top_k=tail_budget)
        local = np.concatenate((protected, eligible[tail_order]))
    return np.asarray(candidates, dtype=np.int64)[local]


def mode_label(
    *,
    candidates: np.ndarray,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    teacher_values: np.ndarray,
    mode_weights: Sequence[float],
    teacher_weight: float,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> tuple[int, float]:
    teacher_docs = local_docs_for_weight(
        candidates=candidates,
        baseline_values=baseline_values,
        correction_values=teacher_values,
        weight=teacher_weight,
        protected_head=protected_head,
        correction_window=correction_window,
        top_k=top_k,
    )
    scores: list[float] = []
    for weight in mode_weights:
        docs = local_docs_for_weight(
            candidates=candidates,
            baseline_values=baseline_values,
            correction_values=correction_values,
            weight=weight,
            protected_head=protected_head,
            correction_window=correction_window,
            top_k=top_k,
        )
        scores.append(m211.agreement_score(docs, teacher_docs, top_k=top_k))
    best = max(range(len(mode_weights)), key=lambda i: (scores[i], -mode_weights[i]))
    baseline_score = scores[0]
    return best, float(scores[best] - baseline_score)


def collect_selector_data(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    exact_by_dataset: dict[str, list[np.ndarray]],
    mode_weights: Sequence[float],
    teacher_weight: float,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    benefits: list[float] = []
    for bundle in bundles:
        correction_rows = correction_by_dataset[bundle.name]
        exact_rows = exact_by_dataset[bundle.name]
        for candidates, baseline, correction, teacher in zip(
            bundle.candidates,
            bundle.baseline_rows,
            correction_rows,
            exact_rows,
        ):
            features.append(
                query_mode_features(
                    baseline_values=baseline,
                    correction_values=correction,
                    mode_weights=mode_weights,
                    protected_head=protected_head,
                    correction_window=correction_window,
                ),
            )
            label, benefit = mode_label(
                candidates=candidates,
                baseline_values=baseline,
                correction_values=correction,
                teacher_values=teacher,
                mode_weights=mode_weights,
                teacher_weight=teacher_weight,
                protected_head=protected_head,
                correction_window=correction_window,
                top_k=top_k,
            )
            labels.append(label)
            benefits.append(benefit)
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        benefits,
    )


def train_selector(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(seed)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std <= 1.0e-6] = 1.0
    x = torch.from_numpy((features - mean) / std).to(device)
    y = torch.from_numpy(labels).to(device)
    model = ModeNet(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        classes=int(labels.max()) + 1,
    ).to(device)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    class_weights = torch.from_numpy(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                preds = torch.argmax(model(x), dim=1)
                accuracy = float((preds == y).to(dtype=torch.float32).mean())
            history.append(
                {
                    'epoch': float(epoch + 1),
                    'loss': float(loss.detach().cpu()),
                    'accuracy': accuracy,
                },
            )
    return (
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        mean,
        std,
        history,
    )


def selector_probabilities(
    *,
    state_dict: dict[str, torch.Tensor],
    mean: np.ndarray,
    std: np.ndarray,
    hidden_dim: int,
    features: np.ndarray,
    classes: int,
    device: str,
) -> np.ndarray:
    model = ModeNet(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        classes=classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy((features - mean) / std).to(device)
        probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()
    return probs.astype(np.float32, copy=False)


def alphas_from_modes(
    *,
    probabilities: np.ndarray,
    mode_weights: Sequence[float],
    family: str,
    confidence_threshold: float | None = None,
) -> np.ndarray:
    weights = np.asarray(mode_weights, dtype=np.float32)
    if family == 'mode_soft':
        return probabilities @ weights
    if family == 'mode_hard':
        return weights[np.argmax(probabilities, axis=1)]
    if family == 'mode_confident':
        assert confidence_threshold is not None
        best = np.argmax(probabilities, axis=1)
        confident = np.max(probabilities, axis=1) >= confidence_threshold
        return np.where(confident, weights[best], 0.0).astype(np.float32)
    if family == 'mode_oracle':
        return weights[np.argmax(probabilities, axis=1)]
    raise ValueError(f'unsupported selector family: {family}')


def evaluate_variant(
    *,
    bundle: m202.DatasetBundle,
    correction_rows: Sequence[np.ndarray],
    alphas: np.ndarray,
    dense_reference: dict[str, dict[str, float]],
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> dict[str, Any]:
    operator = m211.Operator(
        protected_head=protected_head,
        correction_window=correction_window,
        weight=1.0,
    )
    rankings = m211.gated_rankings(
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
        'mean_alpha': float(np.mean(alphas)),
        'baseline_rate': float(np.mean(alphas <= 1.0e-6)),
    }


def add_variant(
    *,
    variants: list[dict[str, Any]],
    dimension: int,
    mode: str,
    selector_family: str,
    confidence_threshold: float | None,
    deployable: bool,
    mode_weights: Sequence[float],
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any],
    training_final: dict[str, float] | None,
    label_distribution: dict[str, float],
    mean_teacher_benefit: float,
) -> None:
    macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': dimension,
            'mode': mode,
            'selector_family': selector_family,
            'confidence_threshold': confidence_threshold,
            'deployable': deployable,
            'mode_weights': list(mode_weights),
            'training_final': training_final,
            'label_distribution': label_distribution,
            'mean_teacher_benefit': mean_teacher_benefit,
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
    modes = m211.parse_str_csv(args.quantization)
    mode_weights = parse_float_csv(args.mode_weights)
    confidence_thresholds = parse_float_csv(args.confidence_thresholds)
    if abs(mode_weights[0]) > 1.0e-8:
        raise ValueError('--mode-weights must start with baseline weight 0.0')
    dense_reference = load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    loader_args.dense_weight = args.teacher_weight
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
        bundle.name: m211.exact_dense_rows(
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
    selector_training: list[dict[str, Any]] = []
    for dimension in dimensions:
        doc_model, query_model, _m210_policy = m211.load_m210_models(
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
                correction_rows, compact_stats = m211.compact_correction_rows(
                    bundle=bundle,
                    doc_values=doc_values,
                    query_values=query_values,
                    mode=mode,
                    correction_window=args.correction_window,
                )
                correction_by_dataset[bundle.name] = correction_rows
                compact_stats_by_dataset[bundle.name] = compact_stats
            train_x, train_y, train_benefits = collect_selector_data(
                bundles=train_bundles,
                correction_by_dataset=correction_by_dataset,
                exact_by_dataset=exact_by_dataset,
                mode_weights=mode_weights,
                teacher_weight=args.teacher_weight,
                protected_head=args.protected_head,
                correction_window=args.correction_window,
                top_k=args.top_k,
            )
            state_dict, mean, std, history = train_selector(
                features=train_x,
                labels=train_y,
                hidden_dim=args.selector_hidden_dim,
                epochs=args.selector_epochs,
                learning_rate=args.selector_learning_rate,
                seed=args.seed + dimension + len(mode),
                device=device,
            )
            counts = np.bincount(train_y, minlength=len(mode_weights))
            label_distribution = {
                f'w{mode_weights[index]:g}': float(count / train_y.size)
                for index, count in enumerate(counts)
            }
            training_final = history[-1] if history else None
            mean_teacher_benefit = (
                statistics.fmean(train_benefits) if train_benefits else 0.0
            )
            selector_training.append(
                {
                    'dimension': dimension,
                    'mode': mode,
                    'label_distribution': label_distribution,
                    'mean_teacher_benefit': mean_teacher_benefit,
                    'history': history,
                },
            )

            eval_features_by_dataset: dict[str, np.ndarray] = {}
            oracle_labels_by_dataset: dict[str, np.ndarray] = {}
            probabilities_by_dataset: dict[str, np.ndarray] = {}
            for bundle in eval_bundles:
                features, labels, _benefits = collect_selector_data(
                    bundles=[bundle],
                    correction_by_dataset=correction_by_dataset,
                    exact_by_dataset=exact_by_dataset,
                    mode_weights=mode_weights,
                    teacher_weight=args.teacher_weight,
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    top_k=args.top_k,
                )
                eval_features_by_dataset[bundle.name] = features
                oracle_labels_by_dataset[bundle.name] = labels
                probabilities_by_dataset[bundle.name] = selector_probabilities(
                    state_dict=state_dict,
                    mean=mean,
                    std=std,
                    hidden_dim=args.selector_hidden_dim,
                    features=features,
                    classes=len(mode_weights),
                    device=device,
                )
            first_stats = compact_stats_by_dataset[eval_bundles[0].name]
            for family, threshold, deployable in [
                ('mode_soft', None, True),
                ('mode_hard', None, True),
                ('mode_oracle', None, False),
            ]:
                rows = []
                for bundle in eval_bundles:
                    if family == 'mode_oracle':
                        probs = np.zeros(
                            (len(oracle_labels_by_dataset[bundle.name]), len(mode_weights)),
                            dtype=np.float32,
                        )
                        probs[
                            np.arange(probs.shape[0]),
                            oracle_labels_by_dataset[bundle.name],
                        ] = 1.0
                    else:
                        probs = probabilities_by_dataset[bundle.name]
                    alphas = alphas_from_modes(
                        probabilities=probs,
                        mode_weights=mode_weights,
                        family=family,
                    )
                    rows.append(
                        evaluate_variant(
                            bundle=bundle,
                            correction_rows=correction_by_dataset[bundle.name],
                            alphas=alphas,
                            dense_reference=dense_reference,
                            protected_head=args.protected_head,
                            correction_window=args.correction_window,
                            top_k=args.top_k,
                        ),
                    )
                add_variant(
                    variants=variants,
                    dimension=dimension,
                    mode=mode,
                    selector_family=family,
                    confidence_threshold=threshold,
                    deployable=deployable,
                    mode_weights=mode_weights,
                    rows=rows,
                    compact_stats=first_stats,
                    training_final=training_final,
                    label_distribution=label_distribution,
                    mean_teacher_benefit=mean_teacher_benefit,
                )
            for threshold in confidence_thresholds:
                rows = []
                for bundle in eval_bundles:
                    alphas = alphas_from_modes(
                        probabilities=probabilities_by_dataset[bundle.name],
                        mode_weights=mode_weights,
                        family='mode_confident',
                        confidence_threshold=threshold,
                    )
                    rows.append(
                        evaluate_variant(
                            bundle=bundle,
                            correction_rows=correction_by_dataset[bundle.name],
                            alphas=alphas,
                            dense_reference=dense_reference,
                            protected_head=args.protected_head,
                            correction_window=args.correction_window,
                            top_k=args.top_k,
                        ),
                    )
                add_variant(
                    variants=variants,
                    dimension=dimension,
                    mode=mode,
                    selector_family='mode_confident',
                    confidence_threshold=threshold,
                    deployable=True,
                    mode_weights=mode_weights,
                    rows=rows,
                    compact_stats=first_stats,
                    training_final=training_final,
                    label_distribution=label_distribution,
                    mean_teacher_benefit=mean_teacher_benefit,
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
        'qrels_usage': 'final metrics only; labels use exact dense teacher agreement',
        'training_scope': 'query-level discrete correction mode selector',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimensions': list(dimensions),
            'quantization': list(modes),
            'mode_weights': list(mode_weights),
            'confidence_thresholds': list(confidence_thresholds),
            'teacher_weight': args.teacher_weight,
            'selector_hidden_dim': args.selector_hidden_dim,
            'selector_epochs': args.selector_epochs,
            'selector_learning_rate': args.selector_learning_rate,
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
            'feature_names': list(feature_names(mode_weights)),
        },
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'selector_training': selector_training,
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
        '# M212 query-level mode selector',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Deployable 32d INT4 Variants',
        '',
        '| Rank | Selector | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | --- | --- |',
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
        selector = variant['selector_family']
        if variant['confidence_threshold'] is not None:
            selector = f"{selector}@{variant['confidence_threshold']:.2f}"
        lines.append(
            f"| {rank} | {selector} | "
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
