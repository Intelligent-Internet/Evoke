#!/usr/bin/env python3
"""M227 qrels-free score-geometry teacher preflight.

M226 showed that the deployable 32d INT4 head can transfer a strong continuous
teacher score geometry. M227 checks whether existing qrels-free signals inside
the frozen top256 window have enough teacher-side headroom, row safety, and
geometric similarity to justify a later distillation run.

Teacher construction is qrels-free. Qrels are used only for final metrics. When
supervised teacher checkpoints are provided, they are used only as a diagnostic
reference for score-geometry similarity, not to construct any M227 policy.
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


SCHEMA = 'm227_qrels_free_geometry_preflight_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')
DEFAULT_POLICIES = (
    'rankt5_z',
    'dense_z',
    'baseline_z',
    'rankt5_dense_mean_z',
    'rankt5_dense_baseline_mean_z',
    'rankt5_dense_rrf',
    'dense_baseline_rrf',
    'rankt5_dense_intersection_128',
    'rankt5_dense_union_128',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate qrels-free top256 score-geometry teachers.',
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
    parser.add_argument('--fixed-weights', default='1.0,1.5,2.0,2.5,3.0,3.5')
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
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('integer list must not be empty')
    return values


def parse_optional_paths(raw: str) -> tuple[Path, ...]:
    return tuple(Path(value.strip()) for value in raw.split(',') if value.strip())


def rrf_scores(values: np.ndarray, *, rrf_k: float) -> np.ndarray:
    return m224.rrf_scores(np.asarray(values, dtype=np.float32), rrf_k=rrf_k)


def floor_invalid(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if bool(np.all(valid)):
        return scores.astype(np.float32, copy=False)
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    adjusted[~valid] = float(np.min(adjusted)) - 1.0
    return adjusted


def build_policy_row(
    *,
    policy: str,
    rankt5_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
    rrf_k: float,
) -> np.ndarray:
    rankt5_z = m218.zscore(rankt5_scores)
    dense_z = m218.zscore(dense_scores)
    baseline_z = m218.zscore(baseline_scores)
    rankt5_rank = m218.ranks_desc(rankt5_scores)
    dense_rank = m218.ranks_desc(dense_scores)
    baseline_rank = m218.ranks_desc(baseline_scores)

    if policy == 'rankt5_z':
        return rankt5_z.astype(np.float32, copy=False)
    if policy == 'dense_z':
        return dense_z.astype(np.float32, copy=False)
    if policy == 'baseline_z':
        return baseline_z.astype(np.float32, copy=False)
    if policy == 'rankt5_dense_mean_z':
        return ((rankt5_z + dense_z) / 2.0).astype(np.float32, copy=False)
    if policy == 'rankt5_dense_baseline_mean_z':
        return ((rankt5_z + dense_z + baseline_z) / 3.0).astype(
            np.float32,
            copy=False,
        )
    if policy == 'rankt5_dense_rrf':
        return (
            (rrf_scores(rankt5_scores, rrf_k=rrf_k)
             + rrf_scores(dense_scores, rrf_k=rrf_k))
            / 2.0
        ).astype(np.float32, copy=False)
    if policy == 'dense_baseline_rrf':
        return (
            (rrf_scores(dense_scores, rrf_k=rrf_k)
             + rrf_scores(baseline_scores, rrf_k=rrf_k))
            / 2.0
        ).astype(np.float32, copy=False)
    if policy == 'rankt5_dense_intersection_128':
        scores = (rankt5_z + dense_z) / 2.0
        valid = (rankt5_rank < 128) & (dense_rank < 128)
        if int(np.sum(valid)) < 16:
            valid = dense_rank < 128
        return floor_invalid(scores, valid)
    if policy == 'rankt5_dense_union_128':
        scores = (rankt5_z + dense_z + 0.25 * baseline_z) / 2.25
        valid = (rankt5_rank < 128) | (dense_rank < 128) | (baseline_rank < 128)
        return floor_invalid(scores, valid)
    raise ValueError(f'unknown teacher policy: {policy}')


def build_policy_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    rankt5_by_dataset: dict[str, list[np.ndarray]],
    policies: Sequence[str],
    rrf_k: float,
) -> dict[str, dict[str, list[np.ndarray]]]:
    scores_by_policy: dict[str, dict[str, list[np.ndarray]]] = {}
    for policy in policies:
        scores_by_dataset: dict[str, list[np.ndarray]] = {}
        for bundle in bundles:
            rows: list[np.ndarray] = []
            for row, rankt5_scores in enumerate(rankt5_by_dataset[bundle.name]):
                dense_scores = np.asarray(bundle.teachers[row], dtype=np.float32)
                baseline_scores = m218.local_baseline_scores(
                    bundle=bundle,
                    row=row,
                )
                rows.append(
                    build_policy_row(
                        policy=policy,
                        rankt5_scores=rankt5_scores,
                        dense_scores=dense_scores,
                        baseline_scores=baseline_scores,
                        rrf_k=rrf_k,
                    ),
                )
            scores_by_dataset[bundle.name] = rows
        scores_by_policy[policy] = scores_by_dataset
    return scores_by_policy


def top_overlap(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    size = min(int(k), left.size, right.size)
    if size <= 0:
        return 0.0
    left_top = set(int(x) for x in base.top_indices(left, top_k=size))
    right_top = set(int(x) for x in base.top_indices(right, top_k=size))
    return float(len(left_top & right_top) / size)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_z = m218.zscore(left)
    right_z = m218.zscore(right)
    if left_z.size == 0:
        return 0.0
    return float(np.mean(left_z * right_z))


def similarity_to_reference(
    *,
    policy_scores: dict[str, list[np.ndarray]],
    reference_scores: dict[str, list[np.ndarray]],
    ks: Sequence[int],
) -> dict[str, Any]:
    pearsons: list[float] = []
    mses: list[float] = []
    overlaps: dict[int, list[float]] = {int(k): [] for k in ks}
    by_dataset: dict[str, dict[str, float]] = {}
    for dataset, rows in policy_scores.items():
        dataset_pearsons: list[float] = []
        dataset_mses: list[float] = []
        dataset_overlaps: dict[int, list[float]] = {int(k): [] for k in ks}
        for row, scores in enumerate(rows):
            ref = np.asarray(reference_scores[dataset][row], dtype=np.float32)
            current = np.asarray(scores, dtype=np.float32)
            score_pearson = pearson(current, ref)
            score_mse = float(np.mean((m218.zscore(current) - m218.zscore(ref)) ** 2))
            pearsons.append(score_pearson)
            mses.append(score_mse)
            dataset_pearsons.append(score_pearson)
            dataset_mses.append(score_mse)
            for k in ks:
                overlap = top_overlap(current, ref, k=int(k))
                overlaps[int(k)].append(overlap)
                dataset_overlaps[int(k)].append(overlap)
        by_dataset[dataset] = {
            'mean_pearson_z': statistics.fmean(dataset_pearsons),
            'mean_z_mse': statistics.fmean(dataset_mses),
            **{
                f'mean_top{k}_overlap': statistics.fmean(values)
                for k, values in dataset_overlaps.items()
            },
        }
    return {
        'mean_pearson_z': statistics.fmean(pearsons),
        'mean_z_mse': statistics.fmean(mses),
        **{
            f'mean_top{k}_overlap': statistics.fmean(values)
            for k, values in overlaps.items()
        },
        'by_dataset': by_dataset,
    }


def add_variant(
    *,
    variants: list[dict[str, Any]],
    teacher_policy: str,
    weight: float,
    rows: list[dict[str, Any]],
    similarity: dict[str, Any] | None,
) -> None:
    macro_vs_dense = m213.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'source': 'qrels_free_teacher_preflight',
            'teacher_policy': teacher_policy,
            'fixed_weight': weight,
            'deployable': False,
            'macro_metrics': m213.macro_metrics(rows),
            'macro_delta_vs_fixed_hybrid': m213.macro_delta(
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
            'similarity_to_supervised_meanz': similarity,
            'rows': rows,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    policies = parse_str_csv(args.teacher_policies)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    similarity_ks = parse_int_csv(args.similarity_ks)
    eval_names = m202.parse_csv(args.eval_datasets)
    dense_reference = m213.load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    loader_args.dense_weight = 0.5
    bundles = [m202.load_bundle(dataset=name, args=loader_args)
               for name in eval_names]

    rankt5_by_dataset = {
        bundle.name: m218.load_cached_rankt5_scores(
            bundle=bundle,
            cache_dir=args.teacher_score_cache_dir,
        )
        for bundle in bundles
    }
    qrels_free_scores = build_policy_scores(
        bundles=bundles,
        rankt5_by_dataset=rankt5_by_dataset,
        policies=policies,
        rrf_k=args.rrf_k,
    )

    supervised_paths = parse_optional_paths(args.supervised_reference_checkpoint_paths)
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
            base.log(f'm227 loading diagnostic reference {label}')
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
            similarity = similarity_to_reference(
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
            add_variant(
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
            'and baseline scores only; qrels are used only for final metrics. '
            'Optional supervised checkpoint references are diagnostic only.'
        ),
        'training_scope': 'qrels-free score-geometry teacher preflight',
        'policy': {
            'teacher_policies': list(policies),
            'fixed_weights': list(fixed_weights),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'top_k': args.top_k,
            'rrf_k': args.rrf_k,
            'similarity_ks': list(similarity_ks),
            'device': device,
        },
        'teacher_score_cache_dir': str(args.teacher_score_cache_dir),
        'supervised_reference_checkpoints': reference_metadata,
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
        '# M227 qrels-free score-geometry preflight',
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
        if rank >= 80:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
