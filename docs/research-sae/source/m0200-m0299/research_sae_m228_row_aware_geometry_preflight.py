#!/usr/bin/env python3
"""M228 row-aware qrels-free score-geometry teacher preflight.

M227 showed that global qrels-free RankT5/dense score geometry has macro
headroom, but it is not dense-safe on FiQA and SciDocs. M228 keeps this as a
teacher-side preflight and tests whether qrels-free query/window features can
gate the geometry before any student training is attempted.

Teacher construction is qrels-free. Qrels are used only for final metrics. When
supervised teacher checkpoints are provided, they are used only as a diagnostic
reference for score-geometry similarity, not to construct any M228 policy.
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
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m223_supervised_teacher_bridge_head as m223
import research_sae_m224_teacher_ensemble_bridge_head as m224
import research_sae_m227_qrels_free_geometry_preflight as m227


SCHEMA = 'm228_row_aware_geometry_preflight_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')
DEFAULT_POLICIES = (
    'agreement_soft_mean_z',
    'agreement_hard_mean_z',
    'triad_consensus_mean_z',
    'dense_anchor_residual_soft',
    'dense_anchor_residual_hard',
    'rankt5_dense_rrf_guarded',
    'risk_dropout_mean_z',
    'family_damped_mean_z',
    'family_damped_residual',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate row-aware qrels-free score-geometry teachers.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--teacher-score-cache-dir', type=Path, required=True)
    parser.add_argument('--supervised-reference-checkpoint-paths', default='')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--teacher-policies', default=','.join(DEFAULT_POLICIES))
    parser.add_argument(
        '--fixed-weights',
        default='0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5',
    )
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--similarity-ks', default='24,64,100')
    parser.add_argument('--gate-k', type=int, default=24)
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def top_overlap(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    size = min(int(k), left.size, right.size)
    if size <= 0:
        return 0.0
    left_top = set(int(x) for x in base.top_indices(left, top_k=size))
    right_top = set(int(x) for x in base.top_indices(right, top_k=size))
    return float(len(left_top & right_top) / size)


def minmax_margin(values: np.ndarray, *, rank: int) -> float:
    if values.size < 2:
        return 0.0
    normalized = m1971.minmax(np.asarray(values, dtype=np.float32))
    order = base.top_indices(normalized, top_k=normalized.size)
    compare_at = min(max(rank, 2), order.size) - 1
    return float(normalized[order[0]] - normalized[order[compare_at]])


def row_features(
    *,
    rankt5_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
    gate_k: int,
) -> dict[str, float]:
    rankt5_dense = top_overlap(rankt5_scores, dense_scores, k=gate_k)
    rankt5_baseline = top_overlap(rankt5_scores, baseline_scores, k=gate_k)
    dense_baseline = top_overlap(dense_scores, baseline_scores, k=gate_k)
    rankt5_dense_64 = top_overlap(rankt5_scores, dense_scores, k=64)
    return {
        'rankt5_dense': rankt5_dense,
        'rankt5_baseline': rankt5_baseline,
        'dense_baseline': dense_baseline,
        'rankt5_dense_64': rankt5_dense_64,
        'triad': statistics.fmean(
            (rankt5_dense, rankt5_baseline, dense_baseline),
        ),
        'rankt5_margin': minmax_margin(rankt5_scores, rank=gate_k),
        'dense_margin': minmax_margin(dense_scores, rank=gate_k),
        'baseline_margin': minmax_margin(baseline_scores, rank=gate_k),
    }


def summarize_features(
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


def family_damping(dataset_summary: dict[str, float]) -> float:
    source_agreement = dataset_summary['rankt5_dense']
    triad = dataset_summary['triad']
    dense_baseline = dataset_summary['dense_baseline']
    risk = max(0.0, dense_baseline - source_agreement)
    return clamp(0.35 + 1.30 * source_agreement + 0.50 * triad - 0.80 * risk)


def soft_agreement_alpha(features: dict[str, float]) -> float:
    source = clamp((features['rankt5_dense'] - 0.20) / 0.30)
    broad = clamp((features['rankt5_dense_64'] - 0.30) / 0.30)
    support = clamp((features['triad'] - 0.28) / 0.25)
    return clamp(0.45 * source + 0.35 * broad + 0.20 * support)


def hard_agreement_alpha(features: dict[str, float]) -> float:
    if features['rankt5_dense'] >= 0.35 and features['rankt5_dense_64'] >= 0.45:
        return 1.0
    return 0.0


def risk_dropout_alpha(features: dict[str, float]) -> float:
    if (
        features['dense_baseline'] >= 0.45
        and features['rankt5_dense'] + 0.10 < features['dense_baseline']
    ):
        return 0.0
    if features['triad'] < 0.25 and features['rankt5_dense_64'] < 0.35:
        return 0.0
    return soft_agreement_alpha(features)


def policy_row(
    *,
    policy: str,
    rankt5_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
    features: dict[str, float],
    family_alpha: float,
    rrf_k: float,
) -> tuple[np.ndarray, float]:
    rankt5_z = m218.zscore(rankt5_scores)
    dense_z = m218.zscore(dense_scores)
    baseline_z = m218.zscore(baseline_scores)
    mean_z = ((rankt5_z + dense_z) / 2.0).astype(np.float32, copy=False)
    triad_z = ((rankt5_z + dense_z + baseline_z) / 3.0).astype(
        np.float32,
        copy=False,
    )
    residual = (dense_z + 0.50 * (rankt5_z - dense_z)).astype(
        np.float32,
        copy=False,
    )

    if policy == 'agreement_soft_mean_z':
        alpha = soft_agreement_alpha(features)
        return (alpha * mean_z).astype(np.float32, copy=False), alpha
    if policy == 'agreement_hard_mean_z':
        alpha = hard_agreement_alpha(features)
        return (alpha * mean_z).astype(np.float32, copy=False), alpha
    if policy == 'triad_consensus_mean_z':
        alpha = clamp((features['triad'] - 0.30) / 0.25)
        return (alpha * triad_z).astype(np.float32, copy=False), alpha
    if policy == 'dense_anchor_residual_soft':
        alpha = soft_agreement_alpha(features)
        return (alpha * residual).astype(np.float32, copy=False), alpha
    if policy == 'dense_anchor_residual_hard':
        alpha = hard_agreement_alpha(features)
        return (alpha * residual).astype(np.float32, copy=False), alpha
    if policy == 'rankt5_dense_rrf_guarded':
        alpha = risk_dropout_alpha(features)
        rrf = (
            m227.rrf_scores(rankt5_scores, rrf_k=rrf_k)
            + m227.rrf_scores(dense_scores, rrf_k=rrf_k)
        ) / 2.0
        return (alpha * rrf).astype(np.float32, copy=False), alpha
    if policy == 'risk_dropout_mean_z':
        alpha = risk_dropout_alpha(features)
        return (alpha * mean_z).astype(np.float32, copy=False), alpha
    if policy == 'family_damped_mean_z':
        alpha = family_alpha * risk_dropout_alpha(features)
        return (alpha * mean_z).astype(np.float32, copy=False), alpha
    if policy == 'family_damped_residual':
        alpha = family_alpha * risk_dropout_alpha(features)
        return (alpha * residual).astype(np.float32, copy=False), alpha
    raise ValueError(f'unknown teacher policy: {policy}')


def build_policy_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    rankt5_by_dataset: dict[str, list[np.ndarray]],
    policies: Sequence[str],
    gate_k: int,
    rrf_k: float,
) -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
]:
    features_by_dataset: dict[str, list[dict[str, float]]] = {}
    for bundle in bundles:
        rows: list[dict[str, float]] = []
        for row, rankt5_scores in enumerate(rankt5_by_dataset[bundle.name]):
            rows.append(
                row_features(
                    rankt5_scores=rankt5_scores,
                    dense_scores=np.asarray(bundle.teachers[row], dtype=np.float32),
                    baseline_scores=m218.local_baseline_scores(bundle=bundle, row=row),
                    gate_k=gate_k,
                ),
            )
        features_by_dataset[bundle.name] = rows

    feature_summary = summarize_features(features_by_dataset)
    scores_by_policy: dict[str, dict[str, list[np.ndarray]]] = {}
    alpha_summary: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        policy_scores: dict[str, list[np.ndarray]] = {}
        policy_alphas: dict[str, dict[str, float]] = {}
        for bundle in bundles:
            family_alpha = family_damping(feature_summary[bundle.name])
            rows = []
            alphas = []
            for row, rankt5_scores in enumerate(rankt5_by_dataset[bundle.name]):
                dense_scores = np.asarray(bundle.teachers[row], dtype=np.float32)
                baseline_scores = m218.local_baseline_scores(bundle=bundle, row=row)
                scores, alpha = policy_row(
                    policy=policy,
                    rankt5_scores=rankt5_scores,
                    dense_scores=dense_scores,
                    baseline_scores=baseline_scores,
                    features=features_by_dataset[bundle.name][row],
                    family_alpha=family_alpha,
                    rrf_k=rrf_k,
                )
                rows.append(scores)
                alphas.append(alpha)
            policy_scores[bundle.name] = rows
            policy_alphas[bundle.name] = {
                'mean_alpha': statistics.fmean(alphas),
                'zero_fraction': statistics.fmean(
                    1.0 if value <= 0.000001 else 0.0
                    for value in alphas
                ),
                'family_alpha': family_alpha,
            }
        scores_by_policy[policy] = policy_scores
        alpha_summary[policy] = policy_alphas
    return scores_by_policy, feature_summary, alpha_summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    policies = m227.parse_str_csv(args.teacher_policies)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    similarity_ks = m227.parse_int_csv(args.similarity_ks)
    eval_names = m202.parse_csv(args.eval_datasets)
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
    qrels_free_scores, feature_summary, alpha_summary = build_policy_scores(
        bundles=bundles,
        rankt5_by_dataset=rankt5_by_dataset,
        policies=policies,
        gate_k=args.gate_k,
        rrf_k=args.rrf_k,
    )

    supervised_paths = m227.parse_optional_paths(
        args.supervised_reference_checkpoint_paths,
    )
    reference_scores: dict[str, list[np.ndarray]] | None = None
    reference_metadata: list[dict[str, Any]] = []
    if supervised_paths:
        basis, _energy = base.load_principal_basis(
            args.projection_basis,
            dimensions=args.dimension,
            report_dimensions=(args.dimension,),
        )
        teacher_score_sets: list[dict[str, list[np.ndarray]]] = []
        for teacher_path in supervised_paths:
            teacher_doc, teacher_query, label, metadata = m223.load_teacher_pair(
                checkpoint_path=teacher_path,
                m210_checkpoint_dir=args.m210_checkpoint_dir,
                dimension=args.dimension,
                basis=basis,
                device=device,
            )
            base.log(f'm228 loading diagnostic reference {label}')
            reference_metadata.append(metadata)
            teacher_score_sets.append(
                m223.model_window_scores(
                    bundles=bundles,
                    doc_model=teacher_doc,
                    query_model=teacher_query,
                    device=device,
                    batch_size=args.encode_batch_size,
                ),
            )
            del teacher_doc
            del teacher_query
            if device == 'cuda':
                torch.cuda.empty_cache()
        reference_scores = m224.ensemble_scores(
            policy='mean_z',
            bundles=bundles,
            teacher_scores=teacher_score_sets,
            rrf_k=args.rrf_k,
        )

    variants: list[dict[str, Any]] = []
    similarities_by_policy: dict[str, Any] = {}
    for policy in policies:
        score_rows = qrels_free_scores[policy]
        similarity = None
        if reference_scores is not None:
            similarity = m227.similarity_to_reference(
                policy_scores=score_rows,
                reference_scores=reference_scores,
                ks=similarity_ks,
            )
            similarities_by_policy[policy] = similarity
        candidate_rows = m223.candidate_rows_from_window_scores(
            bundles=bundles,
            scores_by_dataset=score_rows,
        )
        for weight, rows in m218.evaluate_correction(
            bundles=bundles,
            correction_by_dataset=candidate_rows,
            dense_reference=dense_reference,
            fixed_weights=fixed_weights,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        ):
            m227.add_variant(
                variants=variants,
                teacher_policy=policy,
                weight=weight,
                rows=rows,
                similarity=similarity,
            )

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
        'qrels_usage': (
            'teacher construction uses qrels-free RankT5 cache, dense scores, '
            'baseline scores, and source-agreement features only; qrels are '
            'used only for final metrics. Optional supervised checkpoint '
            'references are diagnostic only.'
        ),
        'training_scope': 'row-aware qrels-free score-geometry teacher preflight',
        'policy': {
            'teacher_policies': list(policies),
            'fixed_weights': list(fixed_weights),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'top_k': args.top_k,
            'rrf_k': args.rrf_k,
            'gate_k': args.gate_k,
            'similarity_ks': list(similarity_ks),
            'device': device,
        },
        'teacher_score_cache_dir': str(args.teacher_score_cache_dir),
        'supervised_reference_checkpoints': reference_metadata,
        'feature_summary': feature_summary,
        'alpha_summary_by_policy': alpha_summary,
        'similarities_by_policy': similarities_by_policy,
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
        '# M228 row-aware qrels-free score-geometry preflight',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Policies: `{', '.join(payload['policy']['teacher_policies'])}`",
        '',
        '## Variants',
        '',
        '| Rank | Policy | Weight | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense | Similarity top24 | Pearson z |',
        '| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        similarity = variant.get('similarity_to_supervised_meanz') or {}
        top24 = similarity.get('mean_top24_overlap', float('nan'))
        pearson_z = similarity.get('mean_pearson_z', float('nan'))
        lines.append(
            f"| {rank} | {variant['teacher_policy']} | "
            f"{variant['fixed_weight']:.3f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} | "
            f"{top24:.6f} | {pearson_z:.6f} |"
        )
        if rank >= 100:
            break
    lines.extend(['', '## Alpha summary', ''])
    for policy, by_dataset in payload['alpha_summary_by_policy'].items():
        means = [
            row['mean_alpha']
            for row in by_dataset.values()
        ]
        zeros = [
            row['zero_fraction']
            for row in by_dataset.values()
        ]
        lines.append(
            f"- `{policy}`: mean alpha {statistics.fmean(means):.4f}, "
            f"mean zero fraction {statistics.fmean(zeros):.4f}"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
