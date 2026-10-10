#!/usr/bin/env python3
"""M218 safe RankT5-pseudo global mini-vector head.

M217 showed that ordinary RankT5 pseudo-labels have macro headroom but are not
row-safe, and the compact student loses most of that headroom. M218 keeps the
same product contract and reuses the completed M217 RankT5 cache, but trains
against conservative qrels-free teacher variants that blend or gate RankT5 with
the exact dense top256 signal.

Qrels are used only for final metrics and teacher diagnostics.
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
import torch.nn.functional as F

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m201_top256_distill_projection as m201
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m210_nonlinear_mini_vector_head as m210
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m216_dense_pairwise_global_head as m216
import research_sae_m217_rankt5_pairwise_global_head as m217


SCHEMA = 'm218_safe_rankt5_global_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train safe RankT5-pseudo global mini-vector heads.',
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
    parser.add_argument(
        '--teacher-policies',
        default='blend_d1p0,blend_d0p5,dense_top128_gate,rank_dense_consensus',
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
    parser.add_argument('--fixed-weights', default='0.75,1.0,1.25,1.5,2.0')
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=218)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.mean(values))
    scale = max(float(np.std(centered)), 1.0e-6)
    return centered / scale


def ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=np.float32))
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def local_baseline_scores(
    *,
    bundle: m202.DatasetBundle,
    row: int,
) -> np.ndarray:
    candidates = bundle.candidates[row]
    window = bundle.windows[row]
    baseline = np.asarray(bundle.baseline_rows[row], dtype=np.float32)
    positions = {int(doc): index for index, doc in enumerate(candidates)}
    return np.asarray(
        [baseline[positions[int(doc)]] for doc in window],
        dtype=np.float32,
    )


def load_cached_rankt5_scores(
    *,
    bundle: m202.DatasetBundle,
    cache_dir: Path,
) -> list[np.ndarray]:
    score_path = cache_dir / f'{bundle.name}.rankt5.top256.npz'
    manifest_path = cache_dir / f'{bundle.name}.rankt5.top256.json'
    if not score_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f'missing RankT5 cache for {bundle.name}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('fingerprint') != m217.fingerprint_bundle(bundle):
        raise ValueError(f'RankT5 cache fingerprint mismatch: {bundle.name}')
    packed = np.load(score_path)
    return m217.unpack_rows(packed['offsets'], packed['scores'])


def transform_scores(
    *,
    policy: str,
    rankt5_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rankt5_z = zscore(rankt5_scores)
    dense_z = zscore(dense_scores)
    baseline_z = zscore(baseline_scores)
    rankt5_rank = ranks_desc(rankt5_scores)
    dense_rank = ranks_desc(dense_scores)
    baseline_rank = ranks_desc(baseline_scores)
    valid = np.ones(rankt5_z.size, dtype=bool)

    if policy == 'raw':
        scores = rankt5_z
    elif policy == 'blend_d0p5':
        scores = rankt5_z + 0.5 * dense_z
    elif policy == 'blend_d1p0':
        scores = rankt5_z + dense_z
    elif policy == 'dense_top128_gate':
        scores = rankt5_z.copy()
        valid = dense_rank < 128
    elif policy == 'dense_top64_gate':
        scores = rankt5_z.copy()
        valid = dense_rank < 64
    elif policy == 'rank_dense_consensus':
        scores = rankt5_z + 0.5 * dense_z
        valid = (rankt5_rank < 64) & (dense_rank < 128)
        if int(np.sum(valid)) < 8:
            valid = dense_rank < 128
    elif policy == 'baseline_dense_consensus':
        scores = rankt5_z + 0.5 * dense_z + 0.25 * baseline_z
        valid = (dense_rank < 128) | (baseline_rank < 128)
    else:
        raise ValueError(f'unknown teacher policy: {policy}')

    if not bool(np.all(valid)):
        floor = float(np.min(scores)) - 1.0
        scores = scores.copy()
        scores[~valid] = floor
    return scores.astype(np.float32, copy=False), valid


def build_policy_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    rankt5_by_dataset: dict[str, list[np.ndarray]],
    policies: Sequence[str],
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
        active_rates: list[float] = []
        for bundle in bundles:
            score_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            for row, rankt5_scores in enumerate(rankt5_by_dataset[bundle.name]):
                dense_scores = np.asarray(bundle.teachers[row], dtype=np.float32)
                baseline_scores = local_baseline_scores(bundle=bundle, row=row)
                scores, valid = transform_scores(
                    policy=policy,
                    rankt5_scores=rankt5_scores,
                    dense_scores=dense_scores,
                    baseline_scores=baseline_scores,
                )
                score_rows.append(scores)
                mask_rows.append(valid)
                active_rates.append(float(np.mean(valid)))
            score_by_dataset[bundle.name] = score_rows
            mask_by_dataset[bundle.name] = mask_rows
        scores_by_policy[policy] = score_by_dataset
        masks_by_policy[policy] = mask_by_dataset
        diagnostics.append(
            {
                'policy': policy,
                'mean_active_fraction': statistics.fmean(active_rates),
            },
        )
    return scores_by_policy, masks_by_policy, diagnostics


def filtered_pairwise_samples(
    *,
    bundles: Sequence[m202.DatasetBundle],
    scores_by_dataset: dict[str, list[np.ndarray]],
    masks_by_dataset: dict[str, list[np.ndarray]],
    positive_k: int,
    negative_start_rank: int,
    negative_k: int,
    min_positive_k: int,
    min_negative_k: int,
) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], dict[str, int]]:
    samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    counts: dict[str, int] = {}
    for bundle_index, bundle in enumerate(bundles):
        count = 0
        for row, scores in enumerate(scores_by_dataset[bundle.name]):
            scores = np.asarray(scores, dtype=np.float32)
            if scores.size < min_positive_k + min_negative_k:
                continue
            valid = np.asarray(masks_by_dataset[bundle.name][row], dtype=bool)
            order = np.argsort(-scores)
            positives = [int(index) for index in order if valid[int(index)]]
            positives = positives[:positive_k]
            if len(positives) < min_positive_k:
                continue
            positive_set = set(positives)
            start = min(max(negative_start_rank, len(positives)), order.size)
            negatives = [
                int(index)
                for index in order[start:]
                if int(index) not in positive_set
            ]
            negatives = negatives[:negative_k]
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
    return samples, counts


def train_policy_head(
    *,
    bundles: Sequence[m202.DatasetBundle],
    samples: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    init_doc_model: torch.nn.Module,
    init_query_model: torch.nn.Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    distill_weight: float,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, float]]]:
    if not samples:
        raise ValueError('no training samples after filtering')
    doc_model, query_model = m216.clone_pair(
        init_doc_model,
        init_query_model,
        device=device,
    )
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
        for bundle_index, row, positives, negatives in mutable_samples:
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
            distill_loss = m216.normalized_mse(scores, base_scores)
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
                f'm218 policy epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f} '
                f'pair={history[-1]["pair_loss"]:.6f}'
            )
    return doc_model.cpu(), query_model.cpu(), history


def window_scores_to_candidate_rows(
    *,
    bundle: m202.DatasetBundle,
    window_scores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    return m217.window_scores_to_candidate_rows(bundle, window_scores)


def add_variant(
    *,
    variants: list[dict[str, Any]],
    source: str,
    teacher_policy: str,
    weight: float,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any] | None,
    training_final: dict[str, float] | None,
    deployable: bool,
    samples: int | None,
) -> None:
    macro_vs_dense = m213.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': 32,
            'mode': 'int4_row',
            'source': source,
            'teacher_policy': teacher_policy,
            'selector_family': 'fixed_weight',
            'fixed_weight': weight,
            'deployable': deployable,
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


def evaluate_correction(
    *,
    bundles: Sequence[m202.DatasetBundle],
    correction_by_dataset: dict[str, list[np.ndarray]],
    dense_reference: dict[str, dict[str, float]],
    fixed_weights: Sequence[float],
    protected_head: int,
    correction_window: int,
    top_k: int,
) -> list[tuple[float, list[dict[str, Any]]]]:
    results = []
    for weight in fixed_weights:
        rows = [
            m213.evaluate_variant(
                bundle=bundle,
                correction_rows=correction_by_dataset[bundle.name],
                alphas=np.full(len(bundle.candidates), weight, dtype=np.float32),
                dense_reference=dense_reference,
                protected_head=protected_head,
                correction_window=correction_window,
                top_k=top_k,
            )
            for bundle in bundles
        ]
        results.append((weight, rows))
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M218 training expects a CUDA device')
    if parse_str_csv(args.quantization) != ('int4_row',):
        raise ValueError('M218 currently expects --quantization int4_row')

    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    teacher_policies = parse_str_csv(args.teacher_policies)
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
    all_bundles = list(bundle_cache.values())

    rankt5_by_dataset = {
        bundle.name: load_cached_rankt5_scores(
            bundle=bundle,
            cache_dir=args.teacher_score_cache_dir,
        )
        for bundle in all_bundles
    }
    policy_scores, policy_masks, policy_diagnostics = build_policy_scores(
        bundles=all_bundles,
        rankt5_by_dataset=rankt5_by_dataset,
        policies=teacher_policies,
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
        samples, sample_counts = filtered_pairwise_samples(
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
            f'm218 {teacher_policy} samples={len(samples)} '
            f'counts={sample_counts}'
        )
        doc_model, query_model, history = train_policy_head(
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

        first_stats = compact_stats_by_dataset[eval_bundles[0].name]
        for weight, rows in evaluate_correction(
            bundles=eval_bundles,
            correction_by_dataset=correction_by_dataset,
            dense_reference=dense_reference,
            fixed_weights=fixed_weights,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
        ):
            add_variant(
                variants=variants,
                source='student_32d_int4',
                teacher_policy=teacher_policy,
                weight=weight,
                rows=rows,
                compact_stats=first_stats,
                training_final=history[-1],
                deployable=True,
                samples=len(samples),
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
        teacher_correction_by_dataset = {
            bundle.name: window_scores_to_candidate_rows(
                bundle=bundle,
                window_scores=policy_scores[teacher_policy][bundle.name],
            )
            for bundle in eval_bundles
        }
        for weight, rows in evaluate_correction(
            bundles=eval_bundles,
            correction_by_dataset=teacher_correction_by_dataset,
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
                weight=weight,
                rows=rows,
                compact_stats=None,
                training_final=None,
                deployable=False,
                samples=None,
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
            'qrels used only for final metrics and teacher diagnostics; '
            'training uses cached RankT5 plus dense/baseline safety filters'
        ),
        'training_scope': 'safe RankT5-pseudo global 32d INT4 mini-vector head',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimension': args.dimension,
            'quantization': ['int4_row'],
            'teacher_policies': list(teacher_policies),
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
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
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
        '# M218 safe RankT5-pseudo global mini-vector head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Teacher policies: `{', '.join(payload['policy']['teacher_policies'])}`",
        '',
        '## Variants',
        '',
        '| Rank | Source | Policy | Weight | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['source']} | "
            f"{variant['teacher_policy']} | "
            f"{variant['fixed_weight']:.2f} | "
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
