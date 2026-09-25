#!/usr/bin/env python3
"""M213 qrels-supervised safe mode selector diagnostic.

M212A showed that dense-teacher agreement is not the right selector target for
the fixed M210 mini-vector path.  M213A changes supervision: train a query-level
selector to choose the correction strength that maximizes train-qrels AP@100
with a small Recall tie-breaker.  This is a supervised diagnostic, not a
qrels-free benchmark.  It asks whether safe-MAP-oriented control has enough
headroom before spending compute on a true multi-expert head.
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

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m212_query_mode_selector as m212


SCHEMA = 'm213_qrels_safe_mode_selector_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate qrels-supervised safe mode selector.',
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
    parser.add_argument('--selector-hidden-dim', type=int, default=32)
    parser.add_argument('--selector-epochs', type=int, default=250)
    parser.add_argument('--selector-learning-rate', type=float, default=0.01)
    parser.add_argument('--recall-tie-weight', type=float, default=0.05)
    parser.add_argument('--min-objective-gain', type=float, default=0.0)
    parser.add_argument(
        '--dataset-feature-mode',
        choices=('none', 'train_onehot'),
        default='none',
    )
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=213)
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


def query_metrics(
    ranking: np.ndarray,
    *,
    query_id: str,
    doc_ids: Sequence[str],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    rels = qrels[query_id]
    rel_docs = set(rels)
    retrieved = [doc_ids[int(index)] for index in ranking[:100]]
    hits = len(set(retrieved) & rel_docs)
    recall = hits / len(rel_docs) if rel_docs else 0.0
    ap = base.average_precision(retrieved, rel_docs)
    return {
        'map@100': float(ap),
        'recall@100': float(recall),
    }


def qrels_mode_label(
    *,
    candidates: np.ndarray,
    baseline_values: np.ndarray,
    correction_values: np.ndarray,
    mode_weights: Sequence[float],
    query_id: str,
    doc_ids: Sequence[str],
    qrels: dict[str, dict[str, float]],
    protected_head: int,
    correction_window: int,
    top_k: int,
    recall_tie_weight: float,
    min_objective_gain: float,
) -> tuple[int, float, dict[str, float]]:
    scores: list[float] = []
    maps: list[float] = []
    recalls: list[float] = []
    for weight in mode_weights:
        docs = m212.local_docs_for_weight(
            candidates=candidates,
            baseline_values=baseline_values,
            correction_values=correction_values,
            weight=weight,
            protected_head=protected_head,
            correction_window=correction_window,
            top_k=top_k,
        )
        metrics = query_metrics(
            docs,
            query_id=query_id,
            doc_ids=doc_ids,
            qrels=qrels,
        )
        maps.append(metrics['map@100'])
        recalls.append(metrics['recall@100'])
        scores.append(
            metrics['map@100'] + recall_tie_weight * metrics['recall@100'],
        )
    baseline_score = scores[0]
    best = max(range(len(mode_weights)), key=lambda i: (scores[i], -mode_weights[i]))
    if scores[best] - baseline_score < min_objective_gain:
        best = 0
    diagnostics = {
        'best_objective_gain': float(scores[best] - baseline_score),
        'best_map_gain': float(maps[best] - maps[0]),
        'best_recall_gain': float(recalls[best] - recalls[0]),
    }
    return best, float(scores[best] - baseline_score), diagnostics


def collect_selector_data(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    mode_weights: Sequence[float],
    dataset_feature_names: Sequence[str],
    protected_head: int,
    correction_window: int,
    top_k: int,
    recall_tie_weight: float,
    min_objective_gain: float,
) -> tuple[np.ndarray, np.ndarray, list[float], list[dict[str, float]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    benefits: list[float] = []
    diagnostics: list[dict[str, float]] = []
    for bundle in bundles:
        correction_rows = correction_by_dataset[bundle.name]
        dataset_features = np.zeros(
            len(dataset_feature_names),
            dtype=np.float32,
        )
        if bundle.name in dataset_feature_names:
            dataset_features[list(dataset_feature_names).index(bundle.name)] = 1.0
        for row, (candidates, baseline, correction) in enumerate(
            zip(bundle.candidates, bundle.baseline_rows, correction_rows),
        ):
            base_features = m212.query_mode_features(
                baseline_values=baseline,
                correction_values=correction,
                mode_weights=mode_weights,
                protected_head=protected_head,
                correction_window=correction_window,
            )
            features.append(np.concatenate((base_features, dataset_features)))
            label, benefit, row_diagnostics = qrels_mode_label(
                candidates=candidates,
                baseline_values=baseline,
                correction_values=correction,
                mode_weights=mode_weights,
                query_id=bundle.dense_queries_raw.ids[row],
                doc_ids=bundle.dense_documents_raw.ids,
                qrels=bundle.qrels,
                protected_head=protected_head,
                correction_window=correction_window,
                top_k=top_k,
                recall_tie_weight=recall_tie_weight,
                min_objective_gain=min_objective_gain,
            )
            labels.append(label)
            benefits.append(benefit)
            diagnostics.append(row_diagnostics)
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        benefits,
        diagnostics,
    )


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
    mean_qrels_objective_gain: float,
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
            'mean_qrels_objective_gain': mean_qrels_objective_gain,
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
    dataset_feature_names: tuple[str, ...] = ()
    if args.dataset_feature_mode == 'train_onehot':
        dataset_feature_names = tuple(train_names)
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
            train_x, train_y, train_benefits, train_diag = collect_selector_data(
                bundles=train_bundles,
                correction_by_dataset=correction_by_dataset,
                mode_weights=mode_weights,
                dataset_feature_names=dataset_feature_names,
                protected_head=args.protected_head,
                correction_window=args.correction_window,
                top_k=args.top_k,
                recall_tie_weight=args.recall_tie_weight,
                min_objective_gain=args.min_objective_gain,
            )
            state_dict, mean, std, history = m212.train_selector(
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
            mean_qrels_gain = (
                statistics.fmean(train_benefits) if train_benefits else 0.0
            )
            selector_training.append(
                {
                    'dimension': dimension,
                    'mode': mode,
                    'label_distribution': label_distribution,
                    'mean_qrels_objective_gain': mean_qrels_gain,
                    'mean_map_gain': statistics.fmean(
                        row['best_map_gain'] for row in train_diag
                    ) if train_diag else 0.0,
                    'mean_recall_gain': statistics.fmean(
                        row['best_recall_gain'] for row in train_diag
                    ) if train_diag else 0.0,
                    'history': history,
                },
            )

            oracle_labels_by_dataset: dict[str, np.ndarray] = {}
            probabilities_by_dataset: dict[str, np.ndarray] = {}
            for bundle in eval_bundles:
                features, labels, _benefits, _diag = collect_selector_data(
                    bundles=[bundle],
                    correction_by_dataset=correction_by_dataset,
                    mode_weights=mode_weights,
                    dataset_feature_names=dataset_feature_names,
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    top_k=args.top_k,
                    recall_tie_weight=args.recall_tie_weight,
                    min_objective_gain=args.min_objective_gain,
                )
                oracle_labels_by_dataset[bundle.name] = labels
                probabilities_by_dataset[bundle.name] = m212.selector_probabilities(
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
                    alphas = m212.alphas_from_modes(
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
                    training_final=history[-1] if history else None,
                    label_distribution=label_distribution,
                    mean_qrels_objective_gain=mean_qrels_gain,
                )
            for threshold in confidence_thresholds:
                rows = []
                for bundle in eval_bundles:
                    alphas = m212.alphas_from_modes(
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
                    training_final=history[-1] if history else None,
                    label_distribution=label_distribution,
                    mean_qrels_objective_gain=mean_qrels_gain,
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
        'qrels_usage': (
            'train/eval qrels used for selector labels and oracle diagnostics; '
            'this is a supervised diagnostic, not a qrels-free benchmark'
        ),
        'training_scope': 'qrels-supervised safe MAP mode selector over M210 mini-vector',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimensions': list(dimensions),
            'quantization': list(modes),
            'mode_weights': list(mode_weights),
            'confidence_thresholds': list(confidence_thresholds),
            'selector_hidden_dim': args.selector_hidden_dim,
            'selector_epochs': args.selector_epochs,
            'selector_learning_rate': args.selector_learning_rate,
            'recall_tie_weight': args.recall_tie_weight,
            'min_objective_gain': args.min_objective_gain,
            'dataset_feature_mode': args.dataset_feature_mode,
            'dataset_feature_names': list(dataset_feature_names),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
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
        '# M213 qrels-supervised safe mode selector',
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
