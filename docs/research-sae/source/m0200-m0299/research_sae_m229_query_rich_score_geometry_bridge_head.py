#!/usr/bin/env python3
"""M229 query-rich score-geometry supervised bridge.

M226 showed that a 32d INT4 product head can carry a strong supervised
score-geometry target. M229 keeps the document-side product contract fixed
while allowing the online query tower to be richer. The goal is to find the
quality ceiling before a later query-cost-control pass.

This is a supervised bridge diagnostic. The teacher artifacts were trained with
qrels; qrels are otherwise used only for final metrics.
"""

from __future__ import annotations

import argparse
import copy
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
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m216_dense_pairwise_global_head as m216
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m223_supervised_teacher_bridge_head as m223
import research_sae_m226_score_geometry_bridge_head as m226


SCHEMA = 'm229_query_rich_score_geometry_bridge_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')
DEFAULT_ARCHITECTURES = ('query_rich_512x2', 'query_rich_768x3')


@dataclass(frozen=True)
class QueryArchitecture:
    label: str
    hidden_dim: int
    depth: int
    residual_scale: float


class QueryRichHead(torch.nn.Module):
    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.base = copy.deepcopy(base_model)
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _index in range(depth):
            layers.extend(
                [
                    torch.nn.Linear(current_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(hidden_dim),
                ],
            )
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.adapter = torch.nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)
        with torch.no_grad():
            final = self.adapter[-1]
            assert isinstance(final, torch.nn.Linear)
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.residual_scale * self.adapter(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate query-rich score-geometry bridge heads.',
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
    parser.add_argument('--query-architectures', default=','.join(DEFAULT_ARCHITECTURES))
    parser.add_argument('--objective-policy', default='meanz_pair_meanz_score')
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.0004)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--distill-weight', type=float, default=0.2)
    parser.add_argument('--score-geometry-weight', type=float, default=0.2)
    parser.add_argument('--positive-k', type=int, default=16)
    parser.add_argument('--negative-start-rank', type=int, default=16)
    parser.add_argument('--negative-k', type=int, default=128)
    parser.add_argument('--min-positive-k', type=int, default=8)
    parser.add_argument('--min-negative-k', type=int, default=32)
    parser.add_argument('--fixed-weights', default='2.5,3.0,3.25,3.5,3.75,4.0')
    parser.add_argument(
        '--router-policies',
        default='fixed,family_dense_half,m220c_joint',
    )
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=229)
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


def parse_architecture(label: str) -> QueryArchitecture:
    if not label.startswith('query_rich_') or 'x' not in label:
        raise ValueError(f'unknown query architecture: {label}')
    spec = label.removeprefix('query_rich_')
    hidden_raw, depth_raw = spec.split('x', 1)
    hidden_dim = int(hidden_raw)
    depth = int(depth_raw)
    if hidden_dim <= 0 or depth <= 0:
        raise ValueError(f'invalid query architecture: {label}')
    return QueryArchitecture(
        label=label,
        hidden_dim=hidden_dim,
        depth=depth,
        residual_scale=0.35,
    )


def architecture_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def train_query_rich_head(
    *,
    bundles: Sequence[m202.DatasetBundle],
    samples: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    score_targets_by_dataset: dict[str, list[np.ndarray]],
    init_doc_model: torch.nn.Module,
    init_query_model: torch.nn.Module,
    architecture: QueryArchitecture,
    input_dim: int,
    output_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    distill_weight: float,
    score_geometry_weight: float,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, float]]]:
    if not samples:
        raise ValueError('no training samples after filtering')
    doc_model = copy.deepcopy(init_doc_model).to(device)
    query_model = QueryRichHead(
        base_model=init_query_model,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=architecture.hidden_dim,
        depth=architecture.depth,
        residual_scale=architecture.residual_scale,
    ).to(device)
    base_doc, base_query = m216.clone_pair(
        init_doc_model,
        init_query_model,
        device=device,
    )
    m216.freeze(base_doc)
    m216.freeze(base_query)
    optimizer = torch.optim.AdamW(
        list(doc_model.parameters()) + list(query_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    mutable_samples = list(samples)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(mutable_samples)
        losses: list[float] = []
        pair_losses: list[float] = []
        distill_losses: list[float] = []
        geometry_losses: list[float] = []
        for bundle_index, row, positives, negatives in mutable_samples:
            bundle = bundles[bundle_index]
            docs = bundle.windows[row]
            doc_inputs = torch.from_numpy(bundle.dense_documents[docs]).to(device)
            query_inputs = torch.from_numpy(
                bundle.dense_queries[row:row + 1],
            ).to(device)
            doc_values = doc_model(doc_inputs)
            query_values = query_model(query_inputs)[0]
            scores = doc_values @ query_values

            pos_index = torch.from_numpy(positives).to(device)
            neg_index = torch.from_numpy(negatives).to(device)
            pair_loss = F.softplus(
                -(scores[pos_index][:, None] - scores[neg_index][None, :]),
            ).mean()

            target_scores = torch.from_numpy(
                np.asarray(
                    score_targets_by_dataset[bundle.name][row],
                    dtype=np.float32,
                ),
            ).to(device)
            geometry_loss = m216.normalized_mse(scores, target_scores)

            with torch.no_grad():
                base_doc_values = base_doc(doc_inputs)
                base_query_values = base_query(query_inputs)[0]
                base_scores = base_doc_values @ base_query_values
            distill_loss = m216.normalized_mse(scores, base_scores)
            magnitude_loss = 0.001 * (
                doc_values.pow(2).mean() + query_values.pow(2).mean()
            )
            loss = (
                pair_loss
                + distill_weight * distill_loss
                + score_geometry_weight * geometry_loss
                + magnitude_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(doc_model.parameters()) + list(query_model.parameters()),
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pair_losses.append(float(pair_loss.detach().cpu()))
            distill_losses.append(float(distill_loss.detach().cpu()))
            geometry_losses.append(float(geometry_loss.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'pair_loss': statistics.fmean(pair_losses),
                'distill_loss': statistics.fmean(distill_losses),
                'score_geometry_loss': statistics.fmean(geometry_losses),
            },
        )
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 2 == 0:
            base.log(
                f'm229 {architecture.label} epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f} '
                f'pair={history[-1]["pair_loss"]:.6f} '
                f'geom={history[-1]["score_geometry_loss"]:.6f}'
            )
    return doc_model.cpu(), query_model.cpu(), history


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M229 training expects a CUDA device')
    if parse_str_csv(args.quantization) != ('int4_row',):
        raise ValueError('M229 currently expects --quantization int4_row')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    teacher_paths = m223.parse_paths(args.teacher_checkpoint_paths)
    architectures = [
        parse_architecture(label)
        for label in parse_str_csv(args.query_architectures)
    ]
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    router_policies = parse_str_csv(args.router_policies)
    dense_reference = m213.load_dense_reference(args.reference_summary)

    pair_policy, score_policy = m226.objective_components(args.objective_policy)
    if pair_policy != 'mean_z' or score_policy != 'mean_z':
        raise ValueError('M229A expects meanz_pair_meanz_score')

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
    all_bundles = list(bundle_cache.values())

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
        base.log(f'm229 loading teacher {teacher_label}')
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

    score_scores = m226.ensemble_scores(
        policy='mean_z',
        bundles=all_bundles,
        teacher_scores=teacher_score_sets,
        rrf_k=args.rrf_k,
    )
    pair_masks = m223.all_valid_masks(score_scores)
    teacher_rows_by_dataset = m223.candidate_rows_from_window_scores(
        bundles=eval_bundles,
        scores_by_dataset=score_scores,
    )

    variants: list[dict[str, Any]] = []
    for weight, rows in m218.evaluate_correction(
        bundles=eval_bundles,
        correction_by_dataset=teacher_rows_by_dataset,
        dense_reference=dense_reference,
        fixed_weights=fixed_weights,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        top_k=args.top_k,
    ):
        m226.add_variant(
            variants=variants,
            source='teacher_score_ceiling',
            objective_policy=args.objective_policy,
            pair_policy=pair_policy,
            score_policy=score_policy,
            router_policy='fixed',
            weight=weight,
            rows=rows,
            compact_stats=None,
            training_final=None,
            deployable=False,
            samples=None,
            mean_alpha=1.0,
        )

    samples, sample_counts = m218.filtered_pairwise_samples(
        bundles=train_bundles,
        scores_by_dataset=score_scores,
        masks_by_dataset=pair_masks,
        positive_k=args.positive_k,
        negative_start_rank=args.negative_start_rank,
        negative_k=args.negative_k,
        min_positive_k=args.min_positive_k,
        min_negative_k=args.min_negative_k,
    )
    base.log(f'm229 samples={len(samples)} counts={sample_counts}')

    training: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    for arch_index, architecture in enumerate(architectures):
        doc_model, query_model, history = train_query_rich_head(
            bundles=train_bundles,
            samples=samples,
            score_targets_by_dataset=score_scores,
            init_doc_model=base_doc,
            init_query_model=base_query,
            architecture=architecture,
            input_dim=int(basis.shape[0]),
            output_dim=args.dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            distill_weight=args.distill_weight,
            score_geometry_weight=args.score_geometry_weight,
            seed=args.seed + arch_index,
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
        before_count = len(variants)
        m226.evaluate_student(
            variants=variants,
            objective_policy=args.objective_policy,
            pair_policy=pair_policy,
            score_policy=score_policy,
            bundles=eval_bundles,
            correction_by_dataset=correction_by_dataset,
            teacher_rows_by_dataset=teacher_rows_by_dataset,
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
        for variant in variants[before_count:]:
            variant['architecture'] = architecture.label
            variant['query_model_stats'] = architecture_summary(query_model)
            variant['doc_model_stats'] = architecture_summary(doc_model)

        checkpoint_path = args.output_dir / f'{architecture.label}_student.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'schema': SCHEMA,
                'architecture': architecture.__dict__,
                'objective_policy': args.objective_policy,
                'pair_policy': pair_policy,
                'score_policy': score_policy,
                'teacher_checkpoints': teacher_metadata,
                'dimension': args.dimension,
                'doc_model': doc_model.cpu().state_dict(),
                'query_model': query_model.cpu().state_dict(),
                'policy': vars(args),
                'training_history': history,
            },
            checkpoint_path,
        )
        checkpoint_paths[architecture.label] = str(checkpoint_path)
        training[architecture.label] = {
            'architecture': architecture.__dict__,
            'samples': len(samples),
            'sample_counts': sample_counts,
            'history': history,
            'checkpoint': str(checkpoint_path),
            'query_model_stats': architecture_summary(query_model),
            'doc_model_stats': architecture_summary(doc_model),
        }
        del doc_model
        del query_model
        torch.cuda.empty_cache()

    for variant in variants:
        variant.setdefault('architecture', None)

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
            'training uses M214 qrels-supervised checkpoint artifacts as an '
            'ensemble teacher, without reading qrels directly; qrels are '
            'otherwise used only for final metrics'
        ),
        'training_scope': (
            'query-rich score-geometry supervised-teacher bridge distillation '
            'with a 32d INT4 document-side product contract'
        ),
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': ['int4_row'],
            'objective_policy': args.objective_policy,
            'query_architectures': [arch.__dict__ for arch in architectures],
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'distill_weight': args.distill_weight,
            'score_geometry_weight': args.score_geometry_weight,
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
        '# M229 query-rich score-geometry bridge head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Objective: `{payload['policy']['objective_policy']}`",
        '',
        '## Variants',
        '',
        '| Rank | Source | Architecture | Router | Weight | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['source']} | "
            f"{variant.get('architecture') or '-'} | "
            f"{variant['router_policy']} | "
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
