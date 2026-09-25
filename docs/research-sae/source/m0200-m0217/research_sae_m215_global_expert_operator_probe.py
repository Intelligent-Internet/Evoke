#!/usr/bin/env python3
"""M215 global expert operator and selector probe.

M214B proved that qrels-pairwise global mini-vector heads can produce a stable
dense-safe win under a qrels-safe selector. M215 keeps those trained heads fixed
and tests cheaper control paths: fixed score weights and dense-teacher
agreement selectors. The representation checkpoint is supervised, but this
probe does not use qrels for selector labels; qrels are used only for final
metrics and oracle diagnostics.
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
import research_sae_m213_qrels_safe_mode_selector as m213


SCHEMA = 'm215_global_expert_operator_probe_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Probe fixed/qrels-free operators for M214 global experts.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--m214-checkpoint-dirs', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--fixed-weights', default='0.25,0.5,0.75,1.0,1.25,1.5,2.0')
    parser.add_argument('--mode-weights', default='0.0,0.5,0.75,1.0,1.25,1.5')
    parser.add_argument('--confidence-thresholds', default='0.4,0.5,0.6')
    parser.add_argument('--teacher-weight', type=float, default=2.0)
    parser.add_argument('--selector-hidden-dim', type=int, default=32)
    parser.add_argument('--selector-epochs', type=int, default=250)
    parser.add_argument('--selector-learning-rate', type=float, default=0.01)
    parser.add_argument(
        '--dataset-feature-mode',
        choices=('none', 'train_onehot'),
        default='train_onehot',
    )
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=215)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    return parser.parse_args()


def parse_path_csv(raw: str) -> tuple[Path, ...]:
    paths = tuple(Path(value.strip()) for value in raw.split(',') if value.strip())
    if not paths:
        raise ValueError('path list must not be empty')
    return paths


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
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


def load_global_model_pair(
    *,
    checkpoint_dir: Path,
    m210_checkpoint_dir: Path,
    dimension: int,
    basis: np.ndarray,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    doc_model, query_model, _policy = m211.load_m210_models(
        checkpoint_dir=m210_checkpoint_dir,
        dimension=dimension,
        basis=basis,
        device=device,
    )
    checkpoint = torch.load(
        checkpoint_dir / f'dim{dimension}_global.pt',
        map_location=device,
    )
    doc_model.load_state_dict(checkpoint['doc_model'])
    query_model.load_state_dict(checkpoint['query_model'])
    doc_model.eval()
    query_model.eval()
    return doc_model, query_model


def dataset_features(
    bundle_name: str,
    names: Sequence[str],
) -> np.ndarray:
    features = np.zeros(len(names), dtype=np.float32)
    if bundle_name in names:
        features[list(names).index(bundle_name)] = 1.0
    return features


def collect_dense_selector_data(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    exact_by_dataset: dict[str, list[np.ndarray]],
    mode_weights: Sequence[float],
    teacher_weight: float,
    dataset_feature_names: Sequence[str],
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    benefits: list[float] = []
    for bundle in bundles:
        dataset_extra = dataset_features(bundle.name, dataset_feature_names)
        correction_rows = correction_by_dataset[bundle.name]
        exact_rows = exact_by_dataset[bundle.name]
        for candidates, baseline, correction, teacher in zip(
            bundle.candidates,
            bundle.baseline_rows,
            correction_rows,
            exact_rows,
        ):
            base_features = m212.query_mode_features(
                baseline_values=baseline,
                correction_values=correction,
                mode_weights=mode_weights,
                protected_head=protected_head,
                correction_window=correction_window,
            )
            features.append(np.concatenate((base_features, dataset_extra)))
            label, benefit = m212.mode_label(
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


def add_variant(
    *,
    variants: list[dict[str, Any]],
    checkpoint: str,
    family: str,
    threshold: float | None,
    weight: float | None,
    deployable: bool,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any],
    training_final: dict[str, float] | None,
    label_distribution: dict[str, float] | None,
    mean_teacher_benefit: float | None,
) -> None:
    macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'checkpoint': checkpoint,
            'dimension': 32,
            'mode': 'int4_row',
            'selector_family': family,
            'confidence_threshold': threshold,
            'fixed_weight': weight,
            'deployable': deployable,
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
    checkpoint_dirs = parse_path_csv(args.m214_checkpoint_dirs)
    modes = parse_str_csv(args.quantization)
    if modes != ('int4_row',):
        raise ValueError('M215 currently expects --quantization int4_row')
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    mode_weights = m213.parse_float_csv(args.mode_weights)
    confidence_thresholds = m213.parse_float_csv(args.confidence_thresholds)
    dataset_feature_names: tuple[str, ...] = ()
    if args.dataset_feature_mode == 'train_onehot':
        dataset_feature_names = tuple(train_names)
    dense_reference = m213.load_dense_reference(args.reference_summary)

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
        bundle.name: m211.exact_dense_rows(
            bundle,
            correction_window=args.correction_window,
        )
        for bundle in all_bundles
    }
    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=args.dimension,
        report_dimensions=(args.dimension,),
    )

    variants: list[dict[str, Any]] = []
    selector_training: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_dir in enumerate(checkpoint_dirs):
        checkpoint_name = checkpoint_dir.name
        doc_model, query_model = load_global_model_pair(
            checkpoint_dir=checkpoint_dir,
            m210_checkpoint_dir=args.m210_checkpoint_dir,
            dimension=args.dimension,
            basis=basis,
            device=device,
        )
        correction_by_dataset: dict[str, list[np.ndarray]] = {}
        compact_stats_by_dataset: dict[str, dict[str, Any]] = {}
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
            correction_rows, compact_stats = m211.compact_correction_rows(
                bundle=bundle,
                doc_values=doc_values,
                query_values=query_values,
                mode='int4_row',
                correction_window=args.correction_window,
            )
            correction_by_dataset[bundle.name] = correction_rows
            compact_stats_by_dataset[bundle.name] = compact_stats

        first_stats = compact_stats_by_dataset[eval_bundles[0].name]
        for weight in fixed_weights:
            rows = [
                m213.evaluate_variant(
                    bundle=bundle,
                    correction_rows=correction_by_dataset[bundle.name],
                    alphas=np.full(
                        len(bundle.candidates),
                        weight,
                        dtype=np.float32,
                    ),
                    dense_reference=dense_reference,
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    top_k=args.top_k,
                )
                for bundle in eval_bundles
            ]
            add_variant(
                variants=variants,
                checkpoint=checkpoint_name,
                family='fixed_weight',
                threshold=None,
                weight=weight,
                deployable=True,
                rows=rows,
                compact_stats=first_stats,
                training_final=None,
                label_distribution=None,
                mean_teacher_benefit=None,
            )

        train_x, train_y, train_benefits = collect_dense_selector_data(
            bundles=train_bundles,
            correction_by_dataset=correction_by_dataset,
            exact_by_dataset=exact_by_dataset,
            mode_weights=mode_weights,
            teacher_weight=args.teacher_weight,
            dataset_feature_names=dataset_feature_names,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        )
        state_dict, mean, std, history = m212.train_selector(
            features=train_x,
            labels=train_y,
            hidden_dim=args.selector_hidden_dim,
            epochs=args.selector_epochs,
            learning_rate=args.selector_learning_rate,
            seed=args.seed + checkpoint_index,
            device=device,
        )
        counts = np.bincount(train_y, minlength=len(mode_weights))
        label_distribution = {
            f'w{mode_weights[index]:g}': float(count / train_y.size)
            for index, count in enumerate(counts)
        }
        selector_training.append(
            {
                'checkpoint': checkpoint_name,
                'label_distribution': label_distribution,
                'mean_teacher_benefit': (
                    statistics.fmean(train_benefits) if train_benefits else 0.0
                ),
                'history': history,
            },
        )

        probabilities_by_dataset: dict[str, np.ndarray] = {}
        oracle_labels_by_dataset: dict[str, np.ndarray] = {}
        for bundle in eval_bundles:
            features, labels, _benefits = collect_dense_selector_data(
                bundles=[bundle],
                correction_by_dataset=correction_by_dataset,
                exact_by_dataset=exact_by_dataset,
                mode_weights=mode_weights,
                teacher_weight=args.teacher_weight,
                dataset_feature_names=dataset_feature_names,
                protected_head=args.protected_head,
                correction_window=args.correction_window,
                top_k=args.top_k,
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

        for family, threshold, deployable in [
            ('dense_mode_soft', None, True),
            ('dense_mode_hard', None, True),
            ('dense_mode_oracle', None, False),
        ]:
            rows = []
            for bundle in eval_bundles:
                if family == 'dense_mode_oracle':
                    probs = np.zeros(
                        (len(oracle_labels_by_dataset[bundle.name]), len(mode_weights)),
                        dtype=np.float32,
                    )
                    probs[
                        np.arange(probs.shape[0]),
                        oracle_labels_by_dataset[bundle.name],
                    ] = 1.0
                    alpha_family = 'mode_oracle'
                else:
                    probs = probabilities_by_dataset[bundle.name]
                    alpha_family = family.removeprefix('dense_')
                alphas = m212.alphas_from_modes(
                    probabilities=probs,
                    mode_weights=mode_weights,
                    family=alpha_family,
                )
                rows.append(
                    m213.evaluate_variant(
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
                checkpoint=checkpoint_name,
                family=family,
                threshold=threshold,
                weight=None,
                deployable=deployable,
                rows=rows,
                compact_stats=first_stats,
                training_final=history[-1] if history else None,
                label_distribution=label_distribution,
                mean_teacher_benefit=(
                    statistics.fmean(train_benefits) if train_benefits else 0.0
                ),
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
                    m213.evaluate_variant(
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
                checkpoint=checkpoint_name,
                family='dense_mode_confident',
                threshold=threshold,
                weight=None,
                deployable=True,
                rows=rows,
                compact_stats=first_stats,
                training_final=history[-1] if history else None,
                label_distribution=label_distribution,
                mean_teacher_benefit=(
                    statistics.fmean(train_benefits) if train_benefits else 0.0
                ),
            )

    variants.sort(
        key=lambda item: (
            item['deployable'],
            item['beats_pplx_dense_primary'],
            -len(item['unsafe_rows_vs_dense']),
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': (
            'no qrels used for fixed operators or dense-teacher selector labels; '
            'qrels used only for final metrics and nondeployable diagnostics. '
            'The probed M214 checkpoints are qrels-supervised artifacts.'
        ),
        'training_scope': 'fixed and dense-teacher control over M214 global expert',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': list(modes),
            'fixed_weights': list(fixed_weights),
            'mode_weights': list(mode_weights),
            'confidence_thresholds': list(confidence_thresholds),
            'teacher_weight': args.teacher_weight,
            'selector_hidden_dim': args.selector_hidden_dim,
            'selector_epochs': args.selector_epochs,
            'selector_learning_rate': args.selector_learning_rate,
            'dataset_feature_mode': args.dataset_feature_mode,
            'dataset_feature_names': list(dataset_feature_names),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
        },
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'm214_checkpoint_dirs': [str(path) for path in checkpoint_dirs],
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
        '# M215 global expert operator probe',
        '',
        f"- Checkpoints: `{len(payload['m214_checkpoint_dirs'])}`",
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Deployable Variants',
        '',
        '| Rank | Checkpoint | Control | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | --- | --- | ---: | ---: | --- | --- |',
    ]
    rank = 0
    for variant in payload['variants']:
        if not variant['deployable']:
            continue
        rank += 1
        control = variant['selector_family']
        if variant['fixed_weight'] is not None:
            control = f"fixed@{variant['fixed_weight']:.2f}"
        if variant['confidence_threshold'] is not None:
            control = f"{control}@{variant['confidence_threshold']:.2f}"
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['checkpoint']} | {control} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
        if rank >= 40:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
