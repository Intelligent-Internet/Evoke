#!/usr/bin/env python3
"""M221 row-aware qrels-free router probe.

M220 showed that a qrels-free runtime confidence gate can turn the M218B
RankT5-safe mini-vector into a larger macro win, but the same dense-unsafe rows
remain. M221 keeps the compact heads fixed and evaluates structured router
policies built from qrels-free query/window features. Qrels are used only for
final metrics.
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
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m220_runtime_confidence_gate as m220


SCHEMA = 'm221_row_aware_router_probe_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


DEFAULT_POLICIES = (
    'm220c_joint',
    'joint_soft',
    'teacher_dense_strict',
    'student_dense_strict',
    'teacher_student_strict',
    'baseline_guard',
    'churn_guard',
    'dense_floor_guard',
    'family_dense_guard',
    'family_dense_half',
    'family_soft_risk',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate row-aware qrels-free router policies.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--teacher-score-cache-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-paths', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--fixed-weights', default='1.0,1.125,1.25,1.375')
    parser.add_argument('--router-policies', default=','.join(DEFAULT_POLICIES))
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    parser.add_argument('--overlap-k', type=int, default=24)
    parser.add_argument('--joint-threshold', type=float, default=0.30)
    return parser.parse_args()


def minmax_margin(values: np.ndarray, *, rank: int) -> float:
    if values.size < 2:
        return 0.0
    normalized = m1971.minmax(np.asarray(values, dtype=np.float32))
    order = base.top_indices(normalized, top_k=normalized.size)
    compare_at = min(max(rank, 2), order.size) - 1
    return float(normalized[order[0]] - normalized[order[compare_at]])


def feature_rows(
    *,
    bundle: m202.DatasetBundle,
    student_rows: Sequence[np.ndarray],
    teacher_rows: Sequence[np.ndarray],
    correction_window: int,
    overlap_k: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row, (student_row, teacher_row) in enumerate(zip(student_rows, teacher_rows)):
        student = m220.candidate_values_for_window(
            bundle=bundle,
            row=row,
            candidate_values=np.asarray(student_row, dtype=np.float32),
        )
        teacher = m220.candidate_values_for_window(
            bundle=bundle,
            row=row,
            candidate_values=np.asarray(teacher_row, dtype=np.float32),
        )
        dense = np.asarray(bundle.teachers[row], dtype=np.float32)
        baseline = m218.local_baseline_scores(bundle=bundle, row=row)
        size = min(correction_window, student.size, teacher.size, dense.size)
        if size <= 0:
            rows.append(
                {
                    'teacher_dense': 0.0,
                    'student_teacher': 0.0,
                    'student_dense': 0.0,
                    'baseline_dense': 0.0,
                    'baseline_teacher': 0.0,
                    'student_baseline': 0.0,
                    'joint': 0.0,
                    'churn': 1.0,
                    'baseline_margin_24': 0.0,
                    'teacher_margin_24': 0.0,
                    'student_margin_24': 0.0,
                    'dense_margin_24': 0.0,
                },
            )
            continue
        student = student[:size]
        teacher = teacher[:size]
        dense = dense[:size]
        baseline = baseline[:size]
        teacher_dense = m220.overlap(teacher, dense, k=overlap_k)
        student_teacher = m220.overlap(student, teacher, k=overlap_k)
        student_dense = m220.overlap(student, dense, k=overlap_k)
        baseline_dense = m220.overlap(baseline, dense, k=overlap_k)
        baseline_teacher = m220.overlap(baseline, teacher, k=overlap_k)
        student_baseline = m220.overlap(student, baseline, k=overlap_k)
        joint = min(teacher_dense, student_teacher)
        rows.append(
            {
                'teacher_dense': float(teacher_dense),
                'student_teacher': float(student_teacher),
                'student_dense': float(student_dense),
                'baseline_dense': float(baseline_dense),
                'baseline_teacher': float(baseline_teacher),
                'student_baseline': float(student_baseline),
                'joint': float(joint),
                'churn': float(1.0 - student_baseline),
                'baseline_margin_24': minmax_margin(baseline, rank=24),
                'teacher_margin_24': minmax_margin(teacher, rank=24),
                'student_margin_24': minmax_margin(student, rank=24),
                'dense_margin_24': minmax_margin(dense, rank=24),
            },
        )
    return rows


def family_stats(
    *,
    features_by_dataset: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    keys = (
        'teacher_dense',
        'student_teacher',
        'student_dense',
        'baseline_dense',
        'baseline_teacher',
        'student_baseline',
        'joint',
        'churn',
    )
    for dataset, rows in features_by_dataset.items():
        stats[dataset] = {
            key: statistics.fmean(row[key] for row in rows)
            for key in keys
        }
        stats[dataset]['dense_advantage_over_teacher'] = (
            stats[dataset]['baseline_dense'] - stats[dataset]['teacher_dense']
        )
        stats[dataset]['dense_advantage_over_student'] = (
            stats[dataset]['baseline_dense'] - stats[dataset]['student_dense']
        )
    return stats


def policy_alphas(
    *,
    policy: str,
    dataset: str,
    features: Sequence[dict[str, float]],
    family: dict[str, dict[str, float]],
    joint_threshold: float,
) -> np.ndarray:
    alphas: list[float] = []
    family_row = family[dataset]
    family_risk = max(
        0.0,
        family_row['dense_advantage_over_teacher'],
        family_row['dense_advantage_over_student'],
    )
    family_is_risky = family_risk >= 0.05
    for row in features:
        m220c = row['joint'] >= joint_threshold
        if policy == 'm220c_joint':
            alpha = 1.0 if m220c else 0.0
        elif policy == 'joint_soft':
            alpha = min(1.0, max(0.0, row['joint'] / 0.45))
        elif policy == 'teacher_dense_strict':
            alpha = 1.0 if m220c and row['teacher_dense'] >= 0.40 else 0.0
        elif policy == 'student_dense_strict':
            alpha = 1.0 if m220c and row['student_dense'] >= 0.35 else 0.0
        elif policy == 'teacher_student_strict':
            alpha = 1.0 if m220c and row['student_teacher'] >= 0.40 else 0.0
        elif policy == 'baseline_guard':
            guarded = row['baseline_dense'] >= 0.50 and row['baseline_teacher'] < 0.35
            alpha = 1.0 if m220c and not guarded else 0.0
        elif policy == 'churn_guard':
            guarded = row['student_baseline'] < 0.35 and row['baseline_dense'] >= 0.45
            alpha = 1.0 if m220c and not guarded else 0.0
        elif policy == 'dense_floor_guard':
            guarded = (
                row['baseline_dense'] > row['teacher_dense']
                and row['baseline_dense'] > row['student_dense']
            )
            alpha = 1.0 if m220c and not guarded else 0.0
        elif policy == 'family_dense_guard':
            alpha = 1.0 if m220c and not family_is_risky else 0.0
        elif policy == 'family_dense_half':
            alpha = 1.0 if m220c else 0.0
            if family_is_risky:
                alpha *= 0.5
        elif policy == 'family_soft_risk':
            alpha = 1.0 if m220c else 0.0
            alpha *= max(0.0, min(1.0, 1.0 - 2.0 * family_risk))
        else:
            raise ValueError(f'unknown router policy: {policy}')
        alphas.append(float(alpha))
    return np.asarray(alphas, dtype=np.float32)


def summarize_features(
    *,
    features_by_dataset: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    keys = tuple(next(iter(next(iter(features_by_dataset.values())))).keys())
    return {
        dataset: {
            key: statistics.fmean(row[key] for row in rows)
            for key in keys
        }
        for dataset, rows in features_by_dataset.items()
    }


def add_variant(
    *,
    variants: list[dict[str, Any]],
    checkpoint: str,
    teacher_policy: str,
    router_policy: str,
    weight: float,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any],
    mean_alpha: float,
) -> None:
    macro_vs_dense = m213.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'checkpoint': checkpoint,
            'teacher_policy': teacher_policy,
            'source': 'student_32d_int4',
            'router_policy': router_policy,
            'fixed_weight': weight,
            'deployable': True,
            'mean_alpha': mean_alpha,
            'compact_stats': compact_stats,
            'macro_metrics': m213.macro_metrics(rows),
            'macro_delta_vs_fixed_hybrid': m213.macro_delta(
                rows,
                'delta_vs_fixed_hybrid',
            ),
            'macro_delta_vs_pplx_dense': macro_vs_dense,
            'beats_pplx_dense_primary': all(
                macro_vs_dense[name] > 0.0 for name in PRIMARY_METRICS
            ),
            'unsafe_rows_vs_dense': [
                row['dataset']
                for row in rows
                if not row['safe_vs_dense_at_0.002']
            ],
            'unsafe_rows_vs_fixed': [
                row['dataset']
                for row in rows
                if not row['safe_vs_fixed_at_0.002']
            ],
            'rows': rows,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M221 expects a CUDA device')

    checkpoint_paths = m220.parse_paths(args.checkpoint_paths)
    eval_names = m202.parse_csv(args.eval_datasets)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    router_policies = m220.parse_str_csv(args.router_policies)
    dense_reference = m213.load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    loader_args.dense_weight = 0.5
    bundles = [
        m202.load_bundle(dataset=name, args=loader_args)
        for name in eval_names
    ]
    rankt5_by_dataset = {
        bundle.name: m218.load_cached_rankt5_scores(
            bundle=bundle,
            cache_dir=args.teacher_score_cache_dir,
        )
        for bundle in bundles
    }
    needed_teacher_policies = sorted(
        {
            str(
                torch.load(
                    path,
                    map_location='cpu',
                    weights_only=False,
                ).get('teacher_policy', path.stem),
            )
            for path in checkpoint_paths
        },
    )
    teacher_scores, _masks, diagnostics = m218.build_policy_scores(
        bundles=bundles,
        rankt5_by_dataset=rankt5_by_dataset,
        policies=needed_teacher_policies,
    )
    teacher_rows_by_policy = {
        policy: {
            bundle.name: m218.window_scores_to_candidate_rows(
                bundle=bundle,
                window_scores=teacher_scores[policy][bundle.name],
            )
            for bundle in bundles
        }
        for policy in needed_teacher_policies
    }

    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=args.dimension,
        report_dimensions=(args.dimension,),
    )

    variants: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    feature_summaries_by_checkpoint: dict[str, dict[str, dict[str, float]]] = {}
    family_stats_by_checkpoint: dict[str, dict[str, dict[str, float]]] = {}
    for path in checkpoint_paths:
        doc_model, query_model, teacher_policy = m220.load_checkpoint_models(
            checkpoint_path=path,
            m210_checkpoint_dir=args.m210_checkpoint_dir,
            dimension=args.dimension,
            basis=basis,
            device=device,
        )
        label = m220.checkpoint_label(path, teacher_policy)
        correction_by_dataset: dict[str, list[np.ndarray]] = {}
        compact_by_dataset: dict[str, dict[str, Any]] = {}
        features_by_dataset: dict[str, list[dict[str, float]]] = {}
        for bundle in bundles:
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
            compact_by_dataset[bundle.name] = compact_stats
            features_by_dataset[bundle.name] = feature_rows(
                bundle=bundle,
                student_rows=correction_rows,
                teacher_rows=teacher_rows_by_policy[teacher_policy][bundle.name],
                correction_window=args.correction_window,
                overlap_k=args.overlap_k,
            )
        family = family_stats(features_by_dataset=features_by_dataset)
        feature_summaries_by_checkpoint[label] = summarize_features(
            features_by_dataset=features_by_dataset,
        )
        family_stats_by_checkpoint[label] = family

        alpha_cache: dict[str, dict[str, np.ndarray]] = {}
        for policy in router_policies:
            alpha_cache[policy] = {
                bundle.name: policy_alphas(
                    policy=policy,
                    dataset=bundle.name,
                    features=features_by_dataset[bundle.name],
                    family=family,
                    joint_threshold=args.joint_threshold,
                )
                for bundle in bundles
            }

        for policy in router_policies:
            for weight in fixed_weights:
                operator = m211.Operator(
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    weight=weight,
                )
                rows = [
                    m211.evaluate_variant(
                        bundle=bundle,
                        correction_rows=correction_by_dataset[bundle.name],
                        operator=operator,
                        alphas=alpha_cache[policy][bundle.name],
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in bundles
                ]
                add_variant(
                    variants=variants,
                    checkpoint=label,
                    teacher_policy=teacher_policy,
                    router_policy=policy,
                    weight=weight,
                    rows=rows,
                    compact_stats=compact_by_dataset[bundles[0].name],
                    mean_alpha=statistics.fmean(
                        float(np.mean(alpha_cache[policy][bundle.name]))
                        for bundle in bundles
                    ),
                )
        checkpoints.append(
            {
                'path': str(path),
                'label': label,
                'teacher_policy': teacher_policy,
            },
        )
        del doc_model
        del query_model
        torch.cuda.empty_cache()

    variants.sort(
        key=lambda item: (
            item['beats_pplx_dense_primary'],
            -len(item['unsafe_rows_vs_dense']),
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'eval_datasets': list(eval_names),
        'qrels_usage': 'qrels used only for final metrics',
        'training_scope': 'row-aware qrels-free routers over fixed M218/M220 heads',
        'policy': {
            'dimension': args.dimension,
            'quantization': 'int4_row',
            'fixed_weights': list(fixed_weights),
            'router_policies': list(router_policies),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'overlap_k': args.overlap_k,
            'joint_threshold': args.joint_threshold,
            'top_k': args.top_k,
            'device': device,
        },
        'checkpoints': checkpoints,
        'teacher_score_cache_dir': str(args.teacher_score_cache_dir),
        'policy_diagnostics': diagnostics,
        'feature_summaries_by_checkpoint': feature_summaries_by_checkpoint,
        'family_stats_by_checkpoint': family_stats_by_checkpoint,
        'variants': variants,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m217.write_json(args.output_dir / 'summary.json', payload)
    (args.output_dir / 'summary.md').write_text(
        markdown_report(payload),
        encoding='utf-8',
    )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M221 row-aware router probe',
        '',
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Variants',
        '',
        '| Rank | Checkpoint | Router | Weight | dMAP vs dense | '
        'dRecall vs dense | Unsafe dense | Mean alpha |',
        '| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['checkpoint']} | "
            f"{variant['router_policy']} | {variant['fixed_weight']:.3f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | {unsafe} | "
            f"{variant['mean_alpha']:.3f} |"
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
