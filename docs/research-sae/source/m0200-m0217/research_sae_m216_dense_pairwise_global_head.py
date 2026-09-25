#!/usr/bin/env python3
"""M216 qrels-free dense-pairwise global mini-vector head.

M215A showed that a qrels-supervised M214 global head does not need a learned
selector: a fixed weight can produce a stable dense-safe win. M216 tests whether
the same product-side route can be trained without qrels by replacing the M214
qrels-pairwise loss with exact-dense-teacher pairwise ordering inside top256.

Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
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
import research_sae_m213_qrels_safe_mode_selector as m213


SCHEMA = 'm216_dense_pairwise_global_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate dense-teacher pairwise global head.',
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
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--distill-weight', type=float, default=0.2)
    parser.add_argument('--positive-k', type=int, default=8)
    parser.add_argument('--negative-start-rank', type=int, default=16)
    parser.add_argument('--negative-k', type=int, default=64)
    parser.add_argument('--fixed-weights', default='0.75,1.0,1.25,1.5,2.0')
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=216)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def clone_pair(
    doc_model: torch.nn.Module,
    query_model: torch.nn.Module,
    *,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    return copy.deepcopy(doc_model).to(device), copy.deepcopy(query_model).to(device)


def freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def dense_pairwise_samples(
    bundles: Sequence[m202.DatasetBundle],
    *,
    positive_k: int,
    negative_start_rank: int,
    negative_k: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for bundle_index, bundle in enumerate(bundles):
        for row, teacher in enumerate(bundle.teachers):
            if teacher.size < positive_k + 2:
                continue
            order = np.argsort(-teacher)
            pos = order[:min(positive_k, order.size)]
            start = min(max(negative_start_rank, pos.size), order.size)
            stop = min(order.size, start + negative_k)
            neg = order[start:stop]
            if pos.size and neg.size:
                samples.append(
                    (
                        bundle_index,
                        row,
                        pos.astype(np.int64, copy=False),
                        neg.astype(np.int64, copy=False),
                    ),
                )
    return samples


def normalized_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(m210.normalize(left), m210.normalize(right))


def train_dense_pairwise_head(
    *,
    bundles: Sequence[m202.DatasetBundle],
    init_doc_model: torch.nn.Module,
    init_query_model: torch.nn.Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    distill_weight: float,
    positive_k: int,
    negative_start_rank: int,
    negative_k: int,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, float]], int]:
    doc_model, query_model = clone_pair(
        init_doc_model,
        init_query_model,
        device=device,
    )
    base_doc, base_query = clone_pair(
        init_doc_model,
        init_query_model,
        device=device,
    )
    freeze(base_doc)
    freeze(base_query)
    optimizer = torch.optim.AdamW(
        list(doc_model.parameters()) + list(query_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    samples = dense_pairwise_samples(
        bundles,
        positive_k=positive_k,
        negative_start_rank=negative_start_rank,
        negative_k=negative_k,
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        rng.shuffle(samples)
        losses: list[float] = []
        pair_losses: list[float] = []
        distill_losses: list[float] = []
        for bundle_index, row, positives, negatives in samples:
            bundle = bundles[bundle_index]
            docs = bundle.windows[row]
            doc_inputs = torch.from_numpy(bundle.dense_documents[docs]).to(device)
            query_inputs = torch.from_numpy(bundle.dense_queries[row:row + 1]).to(device)
            doc_values = doc_model(doc_inputs)
            query_values = query_model(query_inputs)[0]
            scores = doc_values @ query_values

            pos_index = torch.from_numpy(positives).to(device)
            neg_index = torch.from_numpy(negatives).to(device)
            pair_loss = F.softplus(
                -(scores[pos_index][:, None] - scores[neg_index][None, :]),
            ).mean()

            with torch.no_grad():
                base_doc_values = base_doc(doc_inputs)
                base_query_values = base_query(query_inputs)[0]
                base_scores = base_doc_values @ base_query_values
            distill_loss = normalized_mse(scores, base_scores)
            magnitude_loss = 0.001 * (
                doc_values.pow(2).mean() + query_values.pow(2).mean()
            )
            loss = pair_loss + distill_weight * distill_loss + magnitude_loss

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
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'pair_loss': statistics.fmean(pair_losses),
                'distill_loss': statistics.fmean(distill_losses),
            },
        )
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 2 == 0:
            base.log(
                f'm216 dense_pairwise epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f} '
                f'pair={history[-1]["pair_loss"]:.6f}'
            )
    return doc_model.cpu(), query_model.cpu(), history, len(samples)


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


def add_variant(
    *,
    variants: list[dict[str, Any]],
    weight: float,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any],
    training_final: dict[str, float],
) -> None:
    macro_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': 32,
            'mode': 'int4_row',
            'selector_family': 'fixed_weight',
            'fixed_weight': weight,
            'deployable': True,
            'training_final': training_final,
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
    modes = parse_str_csv(args.quantization)
    if modes != ('int4_row',):
        raise ValueError('M216 currently expects --quantization int4_row')
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
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
    doc_model, query_model, history, samples = train_dense_pairwise_head(
        bundles=train_bundles,
        init_doc_model=base_doc,
        init_query_model=base_query,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        distill_weight=args.distill_weight,
        positive_k=args.positive_k,
        negative_start_rank=args.negative_start_rank,
        negative_k=args.negative_k,
        seed=args.seed,
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

    first_stats = compact_stats_by_dataset[eval_bundles[0].name]
    variants: list[dict[str, Any]] = []
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
            weight=weight,
            rows=rows,
            compact_stats=first_stats,
            training_final=history[-1],
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
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': 'final metrics only; training uses exact dense teacher only',
        'training_scope': 'dense-teacher pairwise global 32d mini-vector head',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': list(modes),
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'distill_weight': args.distill_weight,
            'positive_k': args.positive_k,
            'negative_start_rank': args.negative_start_rank,
            'negative_k': args.negative_k,
            'fixed_weights': list(fixed_weights),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
        },
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'm210_policy': m210_policy,
        'training_samples': samples,
        'training_history': history,
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
    torch.save(
        {
            'schema': SCHEMA,
            'dimension': args.dimension,
            'doc_model': doc_model.cpu().state_dict(),
            'query_model': query_model.cpu().state_dict(),
            'policy': payload['policy'],
        },
        args.output_dir / f'dense_pairwise_dim{args.dimension}.pt',
    )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M216 dense-pairwise global mini-vector head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Training samples: `{payload['training_samples']}`",
        '',
        '## Fixed Weight Variants',
        '',
        '| Rank | Weight | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['fixed_weight']:.2f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
