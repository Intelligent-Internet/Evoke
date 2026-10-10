#!/usr/bin/env python3
"""M222 row-risk pseudo-label mini-vector head.

M221 found a useful qrels-free family-risk runtime router, but it did not remove
the persistent dense-unsafe rows. M222 is a narrow weak-supervision probe: it
keeps the M218 training budget and product contract, but changes the pseudo
teacher construction on M221-identified risk families. The risk-family list is
an explicit weak label from prior diagnostics, not locked-test data.
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

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m216_dense_pairwise_global_head as m216
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m221_row_aware_router_probe as m221


SCHEMA = 'm222_row_risk_pseudolabel_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')

DEFAULT_POLICIES = (
    'row_prior_dense_anchor',
    'row_prior_consensus_anchor',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train row-risk pseudo-label global mini-vector heads.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--teacher-score-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--teacher-policies', default=','.join(DEFAULT_POLICIES))
    parser.add_argument(
        '--risk-datasets',
        default='arguana,fiqa,scidocs,trec-covid',
    )
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--distill-weight', type=float, default=0.2)
    parser.add_argument('--positive-k', type=int, default=8)
    parser.add_argument('--negative-start-rank', type=int, default=16)
    parser.add_argument('--negative-k', type=int, default=64)
    parser.add_argument('--min-positive-k', type=int, default=4)
    parser.add_argument('--min-negative-k', type=int, default=16)
    parser.add_argument('--fixed-weights', default='1.25,1.375,1.5')
    parser.add_argument(
        '--router-policies',
        default='fixed,family_dense_half,m220c_joint',
    )
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    parser.add_argument('--overlap-k', type=int, default=24)
    parser.add_argument('--joint-threshold', type=float, default=0.30)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def top_mask(ranks: np.ndarray, *, limit: int) -> np.ndarray:
    return np.asarray(ranks, dtype=np.int64) < int(limit)


def row_prior_scores(
    *,
    policy: str,
    dataset: str,
    risk_datasets: set[str],
    rankt5_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build weak row-risk pseudo-label scores.

    Non-risk families use the M218 `blend_d0p5` teacher. Risk families use a
    denser anchor so the student is less free to learn RankT5 corrections that
    repeatedly beat fixed hybrid but lose to PPLX dense.
    """
    if dataset not in risk_datasets:
        return m218.transform_scores(
            policy='blend_d0p5',
            rankt5_scores=rankt5_scores,
            dense_scores=dense_scores,
            baseline_scores=baseline_scores,
        )

    rankt5_z = m218.zscore(rankt5_scores)
    dense_z = m218.zscore(dense_scores)
    baseline_z = m218.zscore(baseline_scores)
    rankt5_rank = m218.ranks_desc(rankt5_scores)
    dense_rank = m218.ranks_desc(dense_scores)
    baseline_rank = m218.ranks_desc(baseline_scores)

    if policy == 'row_prior_dense_anchor':
        scores = 0.35 * rankt5_z + 1.00 * dense_z + 0.50 * baseline_z
        valid = top_mask(dense_rank, limit=160) | top_mask(baseline_rank, limit=128)
    elif policy == 'row_prior_consensus_anchor':
        scores = 0.50 * rankt5_z + 1.00 * dense_z + 0.75 * baseline_z
        valid = top_mask(dense_rank, limit=192) & (
            top_mask(rankt5_rank, limit=128)
            | top_mask(baseline_rank, limit=128)
        )
        if int(np.sum(valid)) < 8:
            valid = top_mask(dense_rank, limit=192) | top_mask(
                baseline_rank,
                limit=128,
            )
    else:
        raise ValueError(f'unknown M222 teacher policy: {policy}')

    if not bool(np.all(valid)):
        scores = scores.copy()
        scores[~valid] = float(np.min(scores)) - 1.0
    return scores.astype(np.float32, copy=False), valid


def build_policy_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    rankt5_by_dataset: dict[str, list[np.ndarray]],
    policies: Sequence[str],
    risk_datasets: set[str],
) -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, dict[str, list[np.ndarray]]],
    list[dict[str, Any]],
]:
    scores_by_policy: dict[str, dict[str, list[np.ndarray]]] = {}
    masks_by_policy: dict[str, dict[str, list[np.ndarray]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for policy in policies:
        score_by_dataset: dict[str, list[np.ndarray]] = {}
        mask_by_dataset: dict[str, list[np.ndarray]] = {}
        active_by_dataset: dict[str, float] = {}
        for bundle in bundles:
            score_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            active_rates: list[float] = []
            for row, rankt5_scores in enumerate(rankt5_by_dataset[bundle.name]):
                dense_scores = np.asarray(bundle.teachers[row], dtype=np.float32)
                baseline_scores = m218.local_baseline_scores(bundle=bundle, row=row)
                scores, valid = row_prior_scores(
                    policy=policy,
                    dataset=bundle.name,
                    risk_datasets=risk_datasets,
                    rankt5_scores=rankt5_scores,
                    dense_scores=dense_scores,
                    baseline_scores=baseline_scores,
                )
                score_rows.append(scores)
                mask_rows.append(valid)
                active_rates.append(float(np.mean(valid)))
            score_by_dataset[bundle.name] = score_rows
            mask_by_dataset[bundle.name] = mask_rows
            active_by_dataset[bundle.name] = statistics.fmean(active_rates)
        scores_by_policy[policy] = score_by_dataset
        masks_by_policy[policy] = mask_by_dataset
        diagnostics.append(
            {
                'policy': policy,
                'risk_datasets': sorted(risk_datasets),
                'mean_active_fraction': statistics.fmean(
                    active_by_dataset.values(),
                ),
                'active_fraction_by_dataset': active_by_dataset,
            },
        )
    return scores_by_policy, masks_by_policy, diagnostics


def add_variant(
    *,
    variants: list[dict[str, Any]],
    source: str,
    teacher_policy: str,
    router_policy: str,
    weight: float,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any] | None,
    training_final: dict[str, float] | None,
    deployable: bool,
    samples: int | None,
    mean_alpha: float | None,
) -> None:
    macro_vs_dense = m213.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': 32,
            'mode': 'int4_row',
            'source': source,
            'teacher_policy': teacher_policy,
            'router_policy': router_policy,
            'selector_family': 'fixed_weight_or_family_router',
            'fixed_weight': weight,
            'deployable': deployable,
            'mean_alpha': mean_alpha,
            'training_samples': samples,
            'training_final': training_final,
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


def evaluate_with_router(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    teacher_rows_by_dataset: dict[str, list[np.ndarray]],
    dense_reference: dict[str, dict[str, float]],
    router_policy: str,
    weight: float,
    protected_head: int,
    correction_window: int,
    overlap_k: int,
    joint_threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    operator = m211.Operator(
        protected_head=protected_head,
        correction_window=correction_window,
        weight=weight,
    )
    features_by_dataset = {
        bundle.name: m221.feature_rows(
            bundle=bundle,
            student_rows=correction_by_dataset[bundle.name],
            teacher_rows=teacher_rows_by_dataset[bundle.name],
            correction_window=correction_window,
            overlap_k=overlap_k,
        )
        for bundle in bundles
    }
    family = m221.family_stats(features_by_dataset=features_by_dataset)
    alpha_by_dataset: dict[str, np.ndarray] = {}
    for bundle in bundles:
        if router_policy == 'fixed':
            alphas = np.ones(len(bundle.candidates), dtype=np.float32)
        else:
            alphas = m221.policy_alphas(
                policy=router_policy,
                dataset=bundle.name,
                features=features_by_dataset[bundle.name],
                family=family,
                joint_threshold=joint_threshold,
            )
        alpha_by_dataset[bundle.name] = alphas
    rows = [
        m211.evaluate_variant(
            bundle=bundle,
            correction_rows=correction_by_dataset[bundle.name],
            operator=operator,
            alphas=alpha_by_dataset[bundle.name],
            dense_reference=dense_reference,
            top_k=top_k,
        )
        for bundle in bundles
    ]
    mean_alpha = statistics.fmean(
        float(np.mean(alpha_by_dataset[bundle.name]))
        for bundle in bundles
    )
    return rows, mean_alpha


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M222 training expects a CUDA device')
    if parse_str_csv(args.quantization) != ('int4_row',):
        raise ValueError('M222 currently expects --quantization int4_row')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    teacher_policies = parse_str_csv(args.teacher_policies)
    risk_datasets = set(m202.parse_csv(args.risk_datasets))
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    router_policies = parse_str_csv(args.router_policies)
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

    rankt5_by_dataset = {
        bundle.name: m218.load_cached_rankt5_scores(
            bundle=bundle,
            cache_dir=args.teacher_score_cache_dir,
        )
        for bundle in all_bundles
    }
    policy_scores, policy_masks, policy_diagnostics = build_policy_scores(
        bundles=all_bundles,
        rankt5_by_dataset=rankt5_by_dataset,
        policies=teacher_policies,
        risk_datasets=risk_datasets,
    )

    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=args.dimension,
        report_dimensions=(args.dimension,),
    )
    base_doc, base_query, m210_policy = m211.load_m210_models(
        checkpoint_dir=args.m210_checkpoint_dir,
        dimension=args.dimension,
        basis=basis,
        device=device,
    )
    base_doc.cpu()
    base_query.cpu()

    variants: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}

    for policy_index, teacher_policy in enumerate(teacher_policies):
        samples, sample_counts = m218.filtered_pairwise_samples(
            bundles=train_bundles,
            scores_by_dataset=policy_scores[teacher_policy],
            masks_by_dataset=policy_masks[teacher_policy],
            positive_k=args.positive_k,
            negative_start_rank=args.negative_start_rank,
            negative_k=args.negative_k,
            min_positive_k=args.min_positive_k,
            min_negative_k=args.min_negative_k,
        )
        base.log(
            f'm222 {teacher_policy} samples={len(samples)} '
            f'counts={sample_counts}'
        )
        doc_model, query_model, history = m218.train_policy_head(
            bundles=train_bundles,
            samples=samples,
            init_doc_model=base_doc,
            init_query_model=base_query,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            distill_weight=args.distill_weight,
            seed=args.seed + policy_index,
            device=device,
        )
        doc_model.to(device)
        query_model.to(device)

        correction_by_dataset: dict[str, list[np.ndarray]] = {}
        compact_stats_by_dataset: dict[str, dict[str, Any]] = {}
        for bundle in eval_bundles:
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

        teacher_rows_by_dataset = {
            bundle.name: m218.window_scores_to_candidate_rows(
                bundle=bundle,
                window_scores=policy_scores[teacher_policy][bundle.name],
            )
            for bundle in eval_bundles
        }

        first_stats = compact_stats_by_dataset[eval_bundles[0].name]
        for router_policy in router_policies:
            for weight in fixed_weights:
                rows, mean_alpha = evaluate_with_router(
                    bundles=eval_bundles,
                    correction_by_dataset=correction_by_dataset,
                    teacher_rows_by_dataset=teacher_rows_by_dataset,
                    dense_reference=dense_reference,
                    router_policy=router_policy,
                    weight=weight,
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    overlap_k=args.overlap_k,
                    joint_threshold=args.joint_threshold,
                    top_k=args.top_k,
                )
                add_variant(
                    variants=variants,
                    source='student_32d_int4',
                    teacher_policy=teacher_policy,
                    router_policy=router_policy,
                    weight=weight,
                    rows=rows,
                    compact_stats=first_stats,
                    training_final=history[-1],
                    deployable=True,
                    samples=len(samples),
                    mean_alpha=mean_alpha,
                )

        checkpoint_path = args.output_dir / f'{teacher_policy}_dim32_int4.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'schema': SCHEMA,
                'teacher_policy': teacher_policy,
                'dimension': args.dimension,
                'doc_model': doc_model.cpu().state_dict(),
                'query_model': query_model.cpu().state_dict(),
                'policy': vars(args),
                'training_history': history,
            },
            checkpoint_path,
        )
        checkpoint_paths[teacher_policy] = str(checkpoint_path)
        training[teacher_policy] = {
            'samples': len(samples),
            'sample_counts': sample_counts,
            'history': history,
            'checkpoint': str(checkpoint_path),
        }
        del doc_model
        del query_model
        torch.cuda.empty_cache()

    for teacher_policy in teacher_policies:
        teacher_rows_by_dataset = {
            bundle.name: m218.window_scores_to_candidate_rows(
                bundle=bundle,
                window_scores=policy_scores[teacher_policy][bundle.name],
            )
            for bundle in eval_bundles
        }
        for weight, rows in m218.evaluate_correction(
            bundles=eval_bundles,
            correction_by_dataset=teacher_rows_by_dataset,
            dense_reference=dense_reference,
            fixed_weights=fixed_weights,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        ):
            add_variant(
                variants=variants,
                source='teacher_ceiling',
                teacher_policy=teacher_policy,
                router_policy='fixed',
                weight=weight,
                rows=rows,
                compact_stats=None,
                training_final=None,
                deployable=False,
                samples=None,
                mean_alpha=1.0,
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
            'weak row-family risk tags come from prior non-locked diagnostic '
            'results; qrels are otherwise used only for final metrics'
        ),
        'training_scope': 'row-risk pseudo-label global 32d INT4 mini-vector head',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': ['int4_row'],
            'teacher_policies': list(teacher_policies),
            'risk_datasets': sorted(risk_datasets),
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'distill_weight': args.distill_weight,
            'positive_k': args.positive_k,
            'negative_start_rank': args.negative_start_rank,
            'negative_k': args.negative_k,
            'min_positive_k': args.min_positive_k,
            'min_negative_k': args.min_negative_k,
            'fixed_weights': list(fixed_weights),
            'router_policies': list(router_policies),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'overlap_k': args.overlap_k,
            'joint_threshold': args.joint_threshold,
            'device': device,
        },
        'teacher_score_cache_dir': str(args.teacher_score_cache_dir),
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'm210_policy': m210_policy,
        'policy_diagnostics': policy_diagnostics,
        'training': training,
        'checkpoint_paths': checkpoint_paths,
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
        '# M222 row-risk pseudo-label mini-vector head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Risk datasets: `{', '.join(payload['policy']['risk_datasets'])}`",
        f"- Teacher policies: `{', '.join(payload['policy']['teacher_policies'])}`",
        '',
        '## Variants',
        '',
        '| Rank | Source | Policy | Router | Weight | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['source']} | "
            f"{variant['teacher_policy']} | {variant['router_policy']} | "
            f"{variant['fixed_weight']:.3f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
        if rank >= 60:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
