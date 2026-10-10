#!/usr/bin/env python3
"""M225 action-distill supervised bridge mini-vector head.

M224 showed that a cross-seed M214B RRF teacher ensemble transfers better than
single teacher artifacts, but the student still leaves a large gap to the
teacher ceiling. M225 changes the supervision target from raw teacher score
order to the teacher's deployment action: which candidates should be promoted
or demoted after combining the teacher correction with the frozen baseline.

This is a supervised bridge diagnostic. The teacher artifacts were trained with
qrels; qrels are otherwise used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m222_row_risk_pseudolabel_head as m222
import research_sae_m223_supervised_teacher_bridge_head as m223
import research_sae_m224_teacher_ensemble_bridge_head as m224


SCHEMA = 'm225_action_distill_bridge_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')
DEFAULT_ACTION_POLICIES = ('promote_demote', 'frontier')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Distill teacher rerank actions into a 32d INT4 student.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--teacher-checkpoint-paths', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--ensemble-policy', default='rrf')
    parser.add_argument(
        '--action-policies',
        default=','.join(DEFAULT_ACTION_POLICIES),
    )
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--distill-weight', type=float, default=0.2)
    parser.add_argument('--positive-k', type=int, default=16)
    parser.add_argument('--negative-start-rank', type=int, default=96)
    parser.add_argument('--negative-k', type=int, default=96)
    parser.add_argument('--min-positive-k', type=int, default=4)
    parser.add_argument('--min-negative-k', type=int, default=16)
    parser.add_argument('--fixed-weights', default='2.0,2.25,2.5,2.75,3.0')
    parser.add_argument(
        '--router-policies',
        default='fixed,family_dense_half,m220c_joint',
    )
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--teacher-action-weight', type=float, default=2.5)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=225)
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


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 1.0e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / span).astype(np.float32, copy=False)


def unique_ordered(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def teacher_action_order(
    *,
    bundle: m202.DatasetBundle,
    row: int,
    teacher_candidate_scores: np.ndarray,
    weight: float,
    protected_head: int,
    correction_window: int,
) -> np.ndarray:
    baseline = np.asarray(bundle.baseline_rows[row], dtype=np.float32)
    baseline_order = base.top_indices(baseline, top_k=baseline.size)
    window_stop = min(int(correction_window), int(baseline_order.size))
    protected = baseline_order[:protected_head]
    active = baseline_order[protected_head:window_stop]
    tail = baseline_order[window_stop:]
    if active.size == 0:
        return np.concatenate([protected, tail]).astype(np.int64, copy=False)
    active_scores = (
        baseline[active]
        + float(weight) * minmax(teacher_candidate_scores[active])
    )
    active_order = active[base.top_indices(active_scores, top_k=active.size)]
    return np.concatenate([protected, active_order, tail]).astype(
        np.int64,
        copy=False,
    )


def candidate_indices_to_window_positions(
    *,
    bundle: m202.DatasetBundle,
    row: int,
    indices: Sequence[int],
) -> list[int]:
    candidates = np.asarray(bundle.candidates[row], dtype=np.int64)
    window = np.asarray(bundle.windows[row], dtype=np.int64)
    positions = {int(doc): pos for pos, doc in enumerate(window)}
    converted: list[int] = []
    for index in indices:
        doc = int(candidates[int(index)])
        if doc in positions:
            converted.append(int(positions[doc]))
    return unique_ordered(converted)


def action_pairs_for_row(
    *,
    policy: str,
    bundle: m202.DatasetBundle,
    row: int,
    teacher_candidate_scores: np.ndarray,
    teacher_action_weight: float,
    positive_k: int,
    negative_start_rank: int,
    negative_k: int,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    baseline = np.asarray(bundle.baseline_rows[row], dtype=np.float32)
    baseline_order = base.top_indices(baseline, top_k=baseline.size)
    action_order = teacher_action_order(
        bundle=bundle,
        row=row,
        teacher_candidate_scores=teacher_candidate_scores,
        weight=teacher_action_weight,
        protected_head=protected_head,
        correction_window=correction_window,
    )
    baseline_top = set(int(index) for index in baseline_order[:top_k])
    action_top = set(int(index) for index in action_order[:top_k])
    promoted = [int(index) for index in action_order[:top_k]
                if int(index) not in baseline_top]
    demoted = [int(index) for index in baseline_order[:top_k]
               if int(index) not in action_top]

    if policy == 'promote_demote':
        positive_candidates = unique_ordered(
            list(action_order[:positive_k]) + promoted,
        )
        negative_candidates = unique_ordered(
            demoted + list(action_order[negative_start_rank:]),
        )
    elif policy == 'frontier':
        positive_candidates = unique_ordered(action_order[:positive_k])
        negative_candidates = unique_ordered(
            action_order[negative_start_rank:],
        )
    else:
        raise ValueError(f'unknown action policy: {policy}')

    positive_positions = candidate_indices_to_window_positions(
        bundle=bundle,
        row=row,
        indices=positive_candidates,
    )[:positive_k]
    positive_set = set(positive_positions)
    negative_positions = [
        index
        for index in candidate_indices_to_window_positions(
            bundle=bundle,
            row=row,
            indices=negative_candidates,
        )
        if index not in positive_set
    ][:negative_k]
    diagnostics = {
        'promoted': len(promoted),
        'demoted': len(demoted),
        'positive_positions': len(positive_positions),
        'negative_positions': len(negative_positions),
    }
    return positive_positions, negative_positions, diagnostics


def action_pairwise_samples(
    *,
    policy: str,
    bundles: Sequence[m202.DatasetBundle],
    teacher_rows_by_dataset: dict[str, list[np.ndarray]],
    positive_k: int,
    negative_start_rank: int,
    negative_k: int,
    min_positive_k: int,
    min_negative_k: int,
    teacher_action_weight: float,
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], dict[str, Any]]:
    samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    counts: dict[str, int] = {}
    row_stats: dict[str, dict[str, float]] = {}
    for bundle_index, bundle in enumerate(bundles):
        count = 0
        promoted_counts: list[float] = []
        demoted_counts: list[float] = []
        positive_counts: list[float] = []
        negative_counts: list[float] = []
        for row, teacher_scores in enumerate(teacher_rows_by_dataset[bundle.name]):
            positives, negatives, diagnostics = action_pairs_for_row(
                policy=policy,
                bundle=bundle,
                row=row,
                teacher_candidate_scores=teacher_scores,
                teacher_action_weight=teacher_action_weight,
                positive_k=positive_k,
                negative_start_rank=negative_start_rank,
                negative_k=negative_k,
                protected_head=protected_head,
                correction_window=correction_window,
                top_k=top_k,
            )
            promoted_counts.append(float(diagnostics['promoted']))
            demoted_counts.append(float(diagnostics['demoted']))
            positive_counts.append(float(diagnostics['positive_positions']))
            negative_counts.append(float(diagnostics['negative_positions']))
            if len(positives) < min_positive_k:
                continue
            if len(negatives) < min_negative_k:
                continue
            samples.append(
                (
                    bundle_index,
                    row,
                    np.asarray(positives, dtype=np.int64),
                    np.asarray(negatives, dtype=np.int64),
                ),
            )
            count += 1
        counts[bundle.name] = count
        row_stats[bundle.name] = {
            'mean_promoted_top100': statistics.fmean(promoted_counts),
            'mean_demoted_top100': statistics.fmean(demoted_counts),
            'mean_positive_positions': statistics.fmean(positive_counts),
            'mean_negative_positions': statistics.fmean(negative_counts),
        }
    return samples, {'sample_counts': counts, 'row_stats': row_stats}


def add_variant(
    *,
    variants: list[dict[str, Any]],
    source: str,
    ensemble_policy: str,
    action_policy: str,
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
            'ensemble_policy': ensemble_policy,
            'action_policy': action_policy,
            'router_policy': router_policy,
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


def evaluate_student(
    *,
    variants: list[dict[str, Any]],
    ensemble_policy: str,
    action_policy: str,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    teacher_rows_by_dataset: dict[str, list[np.ndarray]],
    dense_reference: dict[str, dict[str, float]],
    router_policies: Sequence[str],
    fixed_weights: Sequence[float],
    compact_stats: dict[str, Any],
    training_final: dict[str, float],
    samples: int,
    protected_head: int,
    correction_window: int,
    overlap_k: int,
    joint_threshold: float,
    top_k: int,
) -> None:
    for router_policy in router_policies:
        for weight in fixed_weights:
            rows, mean_alpha = m222.evaluate_with_router(
                bundles=bundles,
                correction_by_dataset=correction_by_dataset,
                teacher_rows_by_dataset=teacher_rows_by_dataset,
                dense_reference=dense_reference,
                router_policy=router_policy,
                weight=weight,
                protected_head=protected_head,
                correction_window=correction_window,
                overlap_k=overlap_k,
                joint_threshold=joint_threshold,
                top_k=top_k,
            )
            add_variant(
                variants=variants,
                source='student_32d_int4',
                ensemble_policy=ensemble_policy,
                action_policy=action_policy,
                router_policy=router_policy,
                weight=weight,
                rows=rows,
                compact_stats=compact_stats,
                training_final=training_final,
                deployable=True,
                samples=samples,
                mean_alpha=mean_alpha,
            )


def load_bundles(
    *,
    args: argparse.Namespace,
    train_names: Sequence[str],
    eval_names: Sequence[str],
) -> tuple[list[m202.DatasetBundle], list[m202.DatasetBundle], list[m202.DatasetBundle]]:
    loader_args = argparse.Namespace(**vars(args))
    loader_args.dense_weight = 0.5
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
    return train_bundles, eval_bundles, list(bundle_cache.values())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M225 training expects a CUDA device')
    if parse_str_csv(args.quantization) != ('int4_row',):
        raise ValueError('M225 currently expects --quantization int4_row')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    action_policies = parse_str_csv(args.action_policies)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    router_policies = parse_str_csv(args.router_policies)
    teacher_paths = m223.parse_paths(args.teacher_checkpoint_paths)
    dense_reference = m213.load_dense_reference(args.reference_summary)
    train_bundles, eval_bundles, all_bundles = load_bundles(
        args=args,
        train_names=train_names,
        eval_names=eval_names,
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

    teacher_metadata: list[dict[str, Any]] = []
    teacher_score_sets: list[dict[str, list[np.ndarray]]] = []
    for teacher_path in teacher_paths:
        teacher_doc, teacher_query, teacher_label, metadata = m223.load_teacher_pair(
            checkpoint_path=teacher_path,
            m210_checkpoint_dir=args.m210_checkpoint_dir,
            dimension=args.dimension,
            basis=basis,
            device=device,
        )
        base.log(f'm225 loading teacher {teacher_label}')
        teacher_metadata.append(metadata)
        teacher_score_sets.append(
            m223.model_window_scores(
                bundles=all_bundles,
                doc_model=teacher_doc,
                query_model=teacher_query,
                device=device,
                batch_size=args.encode_batch_size,
            ),
        )
        del teacher_doc
        del teacher_query
        torch.cuda.empty_cache()

    ensemble_scores = m224.ensemble_scores(
        policy=args.ensemble_policy,
        bundles=all_bundles,
        teacher_scores=teacher_score_sets,
        rrf_k=args.rrf_k,
    )
    teacher_rows_train = m223.candidate_rows_from_window_scores(
        bundles=train_bundles,
        scores_by_dataset=ensemble_scores,
    )
    teacher_rows_eval = m223.candidate_rows_from_window_scores(
        bundles=eval_bundles,
        scores_by_dataset=ensemble_scores,
    )

    variants: list[dict[str, Any]] = []
    for weight, rows in m218.evaluate_correction(
        bundles=eval_bundles,
        correction_by_dataset=teacher_rows_eval,
        dense_reference=dense_reference,
        fixed_weights=fixed_weights,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        top_k=args.top_k,
    ):
        add_variant(
            variants=variants,
            source='teacher_action_source_ceiling',
            ensemble_policy=args.ensemble_policy,
            action_policy='teacher_source',
            router_policy='fixed',
            weight=weight,
            rows=rows,
            compact_stats=None,
            training_final=None,
            deployable=False,
            samples=None,
            mean_alpha=1.0,
        )

    training: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    for policy_index, action_policy in enumerate(action_policies):
        samples, sample_diagnostics = action_pairwise_samples(
            policy=action_policy,
            bundles=train_bundles,
            teacher_rows_by_dataset=teacher_rows_train,
            positive_k=args.positive_k,
            negative_start_rank=args.negative_start_rank,
            negative_k=args.negative_k,
            min_positive_k=args.min_positive_k,
            min_negative_k=args.min_negative_k,
            teacher_action_weight=args.teacher_action_weight,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        )
        base.log(
            f'm225 {action_policy} samples={len(samples)} '
            f'counts={sample_diagnostics["sample_counts"]}'
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
        correction_by_dataset, compact_stats_by_dataset = (
            m223.compact_rows_from_models(
                bundles=eval_bundles,
                doc_model=doc_model,
                query_model=query_model,
                device=device,
                batch_size=args.encode_batch_size,
                correction_window=args.correction_window,
            )
        )
        evaluate_student(
            variants=variants,
            ensemble_policy=args.ensemble_policy,
            action_policy=action_policy,
            bundles=eval_bundles,
            correction_by_dataset=correction_by_dataset,
            teacher_rows_by_dataset=teacher_rows_eval,
            dense_reference=dense_reference,
            router_policies=router_policies,
            fixed_weights=fixed_weights,
            compact_stats=compact_stats_by_dataset[eval_bundles[0].name],
            training_final=history[-1],
            samples=len(samples),
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            overlap_k=args.overlap_k,
            joint_threshold=args.joint_threshold,
            top_k=args.top_k,
        )
        checkpoint_path = args.output_dir / f'{action_policy}_student.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'schema': SCHEMA,
                'ensemble_policy': args.ensemble_policy,
                'action_policy': action_policy,
                'teacher_checkpoints': teacher_metadata,
                'dimension': args.dimension,
                'doc_model': doc_model.cpu().state_dict(),
                'query_model': query_model.cpu().state_dict(),
                'policy': vars(args),
                'training_history': history,
            },
            checkpoint_path,
        )
        checkpoint_paths[action_policy] = str(checkpoint_path)
        training[action_policy] = {
            'samples': len(samples),
            **sample_diagnostics,
            'history': history,
            'checkpoint': str(checkpoint_path),
        }
        del doc_model
        del query_model
        torch.cuda.empty_cache()

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
            'training uses M214 qrels-supervised checkpoint artifacts through '
            'an M224-style teacher ensemble, without reading qrels directly; '
            'qrels are otherwise used only for final metrics'
        ),
        'training_scope': (
            'action-distill supervised bridge into a 32d INT4 mini-vector head'
        ),
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': ['int4_row'],
            'ensemble_policy': args.ensemble_policy,
            'action_policies': list(action_policies),
            'teacher_action_weight': args.teacher_action_weight,
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
            'rrf_k': args.rrf_k,
            'device': device,
        },
        'teacher_checkpoints': teacher_metadata,
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'm210_policy': m210_policy,
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
        '# M225 action-distill supervised bridge head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Ensemble policy: `{payload['policy']['ensemble_policy']}`",
        f"- Action policies: `{', '.join(payload['policy']['action_policies'])}`",
        '',
        '## Variants',
        '',
        '| Rank | Source | Action | Router | Weight | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['source']} | "
            f"{variant['action_policy']} | {variant['router_policy']} | "
            f"{variant['fixed_weight']:.3f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
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
