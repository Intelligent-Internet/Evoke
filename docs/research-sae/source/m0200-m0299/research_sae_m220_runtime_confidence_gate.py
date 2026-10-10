#!/usr/bin/env python3
"""M220 runtime confidence gates for RankT5-safe mini-vector heads.

M218B showed a stable but thin qrels-free macro win. M219A showed that simple
student-transfer changes do not improve MAP. M220A keeps the trained M218B
student checkpoints fixed and asks whether qrels-free query-level confidence
gates can preserve the macro win while reducing unsafe row damage.

Qrels are used only for final metrics.
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
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218


SCHEMA = 'm220_runtime_confidence_gate_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate runtime gates over fixed M218B mini-vector heads.',
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
    parser.add_argument('--fixed-weights', default='0.75,1.0,1.25,1.5')
    parser.add_argument(
        '--gate-policies',
        default=(
            'fixed,teacher_dense_soft,student_teacher_soft,'
            'joint_soft,joint_hard_0p25,joint_hard_0p40,joint_hard_0p55,'
            'teacher_dense_hard_0p40'
        ),
    )
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    parser.add_argument('--overlap-k', type=int, default=32)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def parse_paths(raw: str) -> tuple[Path, ...]:
    values = tuple(Path(value.strip()) for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('path list must not be empty')
    return values


def top_set(values: np.ndarray, *, k: int) -> set[int]:
    size = min(int(k), int(values.size))
    if size <= 0:
        return set()
    return set(int(index) for index in np.argsort(-values)[:size])


def overlap(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    left_set = top_set(left, k=k)
    right_set = top_set(right, k=k)
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set) / min(len(left_set), len(right_set)))


def checkpoint_label(path: Path, teacher_policy: str) -> str:
    parent = path.parent.name
    stem = path.stem.replace('_dim32_int4', '')
    return f'{parent}:{stem}:{teacher_policy}'


def load_checkpoint_models(
    *,
    checkpoint_path: Path,
    m210_checkpoint_dir: Path,
    dimension: int,
    basis: np.ndarray,
    device: str,
) -> tuple[m210.ResidualMLPHead, m210.ResidualMLPHead, str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    teacher_policy = str(checkpoint.get('teacher_policy', checkpoint_path.stem))
    doc_model, query_model, _policy = m211.load_m210_models(
        checkpoint_dir=m210_checkpoint_dir,
        dimension=dimension,
        basis=basis,
        device=device,
    )
    doc_model.load_state_dict(checkpoint['doc_model'])
    query_model.load_state_dict(checkpoint['query_model'])
    doc_model.eval()
    query_model.eval()
    return doc_model, query_model, teacher_policy


def candidate_values_for_window(
    *,
    bundle: m202.DatasetBundle,
    row: int,
    candidate_values: np.ndarray,
) -> np.ndarray:
    positions = {
        int(doc): index
        for index, doc in enumerate(np.asarray(bundle.candidates[row]))
    }
    return np.asarray(
        [candidate_values[positions[int(doc)]] for doc in bundle.windows[row]],
        dtype=np.float32,
    )


def query_alphas(
    *,
    bundle: m202.DatasetBundle,
    student_rows: Sequence[np.ndarray],
    teacher_rows: Sequence[np.ndarray],
    gate_policy: str,
    correction_window: int,
    overlap_k: int,
) -> np.ndarray:
    alphas: list[float] = []
    for row, (student_row, teacher_row) in enumerate(zip(student_rows, teacher_rows)):
        student = candidate_values_for_window(
            bundle=bundle,
            row=row,
            candidate_values=np.asarray(student_row, dtype=np.float32),
        )
        teacher = candidate_values_for_window(
            bundle=bundle,
            row=row,
            candidate_values=np.asarray(teacher_row, dtype=np.float32),
        )
        dense = np.asarray(bundle.teachers[row], dtype=np.float32)
        window_size = min(correction_window, student.size, teacher.size, dense.size)
        if window_size == 0:
            alphas.append(0.0)
            continue
        student = student[:window_size]
        teacher = teacher[:window_size]
        dense = dense[:window_size]

        teacher_dense = overlap(teacher, dense, k=overlap_k)
        student_teacher = overlap(student, teacher, k=overlap_k)
        joint = min(teacher_dense, student_teacher)
        if gate_policy == 'fixed':
            alpha = 1.0
        elif gate_policy == 'teacher_dense_soft':
            alpha = teacher_dense
        elif gate_policy == 'student_teacher_soft':
            alpha = student_teacher
        elif gate_policy == 'joint_soft':
            alpha = joint
        elif gate_policy.startswith('joint_hard_'):
            threshold = float(gate_policy.rsplit('_', 1)[1].replace('p', '.'))
            alpha = 1.0 if joint >= threshold else 0.0
        elif gate_policy.startswith('teacher_dense_hard_'):
            threshold = float(gate_policy.rsplit('_', 1)[1].replace('p', '.'))
            alpha = 1.0 if teacher_dense >= threshold else 0.0
        else:
            raise ValueError(f'unknown gate policy: {gate_policy}')
        alphas.append(float(alpha))
    return np.asarray(alphas, dtype=np.float32)


def add_variant(
    *,
    variants: list[dict[str, Any]],
    checkpoint: str,
    teacher_policy: str,
    gate_policy: str,
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
            'gate_policy': gate_policy,
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
        raise ValueError('M220 expects a CUDA device')

    checkpoint_paths = parse_paths(args.checkpoint_paths)
    eval_names = m202.parse_csv(args.eval_datasets)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    gate_policies = parse_str_csv(args.gate_policies)
    dense_reference = m213.load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    # m202.load_bundle computes an exact-dense diagnostic with this field.
    # M220 does not use that diagnostic for decisions, so keep the historical
    # m202 default instead of changing the M220 protocol.
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
    for path in checkpoint_paths:
        doc_model, query_model, teacher_policy = load_checkpoint_models(
            checkpoint_path=path,
            m210_checkpoint_dir=args.m210_checkpoint_dir,
            dimension=args.dimension,
            basis=basis,
            device=device,
        )
        label = checkpoint_label(path, teacher_policy)
        correction_by_dataset: dict[str, list[np.ndarray]] = {}
        compact_by_dataset: dict[str, dict[str, Any]] = {}
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

        alpha_cache: dict[str, dict[str, np.ndarray]] = {}
        for gate_policy in gate_policies:
            alpha_cache[gate_policy] = {
                bundle.name: query_alphas(
                    bundle=bundle,
                    student_rows=correction_by_dataset[bundle.name],
                    teacher_rows=teacher_rows_by_policy[teacher_policy][bundle.name],
                    gate_policy=gate_policy,
                    correction_window=args.correction_window,
                    overlap_k=args.overlap_k,
                )
                for bundle in bundles
            }

        for gate_policy in gate_policies:
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
                        alphas=alpha_cache[gate_policy][bundle.name],
                        dense_reference=dense_reference,
                        top_k=args.top_k,
                    )
                    for bundle in bundles
                ]
                add_variant(
                    variants=variants,
                    checkpoint=label,
                    teacher_policy=teacher_policy,
                    gate_policy=gate_policy,
                    weight=weight,
                    rows=rows,
                    compact_stats=compact_by_dataset[bundles[0].name],
                    mean_alpha=statistics.fmean(
                        float(np.mean(alpha_cache[gate_policy][bundle.name]))
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
        'training_scope': 'runtime confidence gates over fixed M218B heads',
        'policy': {
            'dimension': args.dimension,
            'quantization': 'int4_row',
            'fixed_weights': list(fixed_weights),
            'gate_policies': list(gate_policies),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'overlap_k': args.overlap_k,
            'top_k': args.top_k,
            'device': device,
        },
        'checkpoints': checkpoints,
        'teacher_score_cache_dir': str(args.teacher_score_cache_dir),
        'policy_diagnostics': diagnostics,
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
        '# M220 runtime confidence gate',
        '',
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Variants',
        '',
        '| Rank | Checkpoint | Gate | Weight | dMAP vs dense | '
        'dRecall vs dense | Unsafe dense | Mean alpha |',
        '| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['checkpoint']} | "
            f"{variant['gate_policy']} | {variant['fixed_weight']:.2f} | "
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
