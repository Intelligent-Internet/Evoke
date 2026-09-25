#!/usr/bin/env python3
"""M214 row-family expert mini-vector head diagnostic.

M213C proved the fixed M210 32d INT4 mini-vector can beat PPLX dense when a
qrels-supervised row-conditioned selector decides how strongly to apply it. This
run moves the row-family signal one layer earlier: train qrels-supervised
global and per-row-family mini-vector experts initialized from M210A, then reuse
the M213 safe mode selector over the resulting 32d INT4 scores.

This is a supervised diagnostic, not a qrels-free product benchmark.
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
import research_sae_m212_query_mode_selector as m212
import research_sae_m213_qrels_safe_mode_selector as m213


SCHEMA = 'm214_row_expert_mini_vector_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate qrels-supervised row expert mini-vectors.',
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
    parser.add_argument('--dimensions', default='32')
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--routes', default='base,global,expert_or_global,expert_or_base')
    parser.add_argument('--expert-epochs', type=int, default=8)
    parser.add_argument('--expert-learning-rate', type=float, default=0.0005)
    parser.add_argument('--expert-weight-decay', type=float, default=0.0001)
    parser.add_argument('--distill-weight', type=float, default=0.2)
    parser.add_argument('--pair-positive-k', type=int, default=8)
    parser.add_argument('--pair-negative-k', type=int, default=64)
    parser.add_argument('--mode-weights', default='0.0,0.5,0.75,1.0,1.25,1.5')
    parser.add_argument('--confidence-thresholds', default='0.4,0.5,0.6')
    parser.add_argument('--selector-hidden-dim', type=int, default=32)
    parser.add_argument('--selector-epochs', type=int, default=250)
    parser.add_argument('--selector-learning-rate', type=float, default=0.01)
    parser.add_argument('--recall-tie-weight', type=float, default=0.05)
    parser.add_argument('--min-objective-gain', type=float, default=0.0)
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
    parser.add_argument('--seed', type=int, default=214)
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


def relevant_doc_ids(bundle: m202.DatasetBundle, row: int) -> set[str]:
    query_id = bundle.dense_queries_raw.ids[row]
    return {
        doc_id
        for doc_id, relevance in bundle.qrels.get(query_id, {}).items()
        if float(relevance) > 0.0
    }


def qrels_training_samples(
    bundles: Sequence[m202.DatasetBundle],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for bundle_index, bundle in enumerate(bundles):
        doc_ids = bundle.dense_documents_raw.ids
        for row, docs in enumerate(bundle.windows):
            rels = relevant_doc_ids(bundle, row)
            if not rels:
                continue
            positives: list[int] = []
            negatives: list[int] = []
            for local, doc_index in enumerate(docs):
                if doc_ids[int(doc_index)] in rels:
                    positives.append(local)
                else:
                    negatives.append(local)
            if positives and negatives:
                samples.append(
                    (
                        bundle_index,
                        row,
                        np.asarray(positives, dtype=np.int64),
                        np.asarray(negatives, dtype=np.int64),
                    ),
                )
    return samples


def normalized_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(m210.normalize(left), m210.normalize(right))


def train_qrels_expert(
    *,
    bundles: Sequence[m202.DatasetBundle],
    init_doc_model: torch.nn.Module,
    init_query_model: torch.nn.Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    distill_weight: float,
    pair_positive_k: int,
    pair_negative_k: int,
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
    samples = qrels_training_samples(bundles)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    history: list[dict[str, float]] = []
    if not samples:
        return doc_model.cpu(), query_model.cpu(), history, 0

    for epoch in range(epochs):
        rng.shuffle(samples)
        losses: list[float] = []
        pair_losses: list[float] = []
        distill_losses: list[float] = []
        for bundle_index, row, positives, negatives in samples:
            bundle = bundles[bundle_index]
            pos = positives.copy()
            neg = negatives.copy()
            if pos.size > pair_positive_k:
                pos = np.asarray(rng.sample(list(pos), pair_positive_k), dtype=np.int64)
            if neg.size > pair_negative_k:
                neg = np.asarray(list(neg[:pair_negative_k]), dtype=np.int64)

            docs = bundle.windows[row]
            doc_inputs = torch.from_numpy(bundle.dense_documents[docs]).to(device)
            query_inputs = torch.from_numpy(bundle.dense_queries[row:row + 1]).to(device)
            doc_values = doc_model(doc_inputs)
            query_values = query_model(query_inputs)[0]
            scores = doc_values @ query_values

            pos_index = torch.from_numpy(pos).to(device)
            neg_index = torch.from_numpy(neg).to(device)
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
                f'm214 expert seed={seed} epoch={epoch + 1}/{epochs} '
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


def route_pair(
    *,
    route: str,
    dataset: str,
    base_pair: tuple[torch.nn.Module, torch.nn.Module],
    global_pair: tuple[torch.nn.Module, torch.nn.Module],
    experts: dict[str, tuple[torch.nn.Module, torch.nn.Module]],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if route == 'base':
        return base_pair
    if route == 'global':
        return global_pair
    if route == 'expert_or_global':
        return experts.get(dataset, global_pair)
    if route == 'expert_or_base':
        return experts.get(dataset, base_pair)
    raise ValueError(f'unsupported route: {route}')


def encode_bundle(
    *,
    bundle: m202.DatasetBundle,
    model_pair: tuple[torch.nn.Module, torch.nn.Module],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    doc_model, query_model = model_pair
    doc_model.to(device)
    query_model.to(device)
    doc_values = m210.encode_matrix(
        model=doc_model,
        matrix=bundle.dense_documents,
        device=device,
        batch_size=batch_size,
    )
    query_values = m210.encode_matrix(
        model=query_model,
        matrix=bundle.dense_queries,
        device=device,
        batch_size=batch_size,
    )
    doc_model.cpu()
    query_model.cpu()
    return doc_values, query_values


def evaluate_route(
    *,
    route: str,
    dimension: int,
    mode: str,
    train_bundles: Sequence[m202.DatasetBundle],
    eval_bundles: Sequence[m202.DatasetBundle],
    base_pair: tuple[torch.nn.Module, torch.nn.Module],
    global_pair: tuple[torch.nn.Module, torch.nn.Module],
    experts: dict[str, tuple[torch.nn.Module, torch.nn.Module]],
    dense_reference: dict[str, dict[str, float]],
    mode_weights: Sequence[float],
    confidence_thresholds: Sequence[float],
    dataset_feature_names: Sequence[str],
    args: argparse.Namespace,
    device: str,
    selector_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    correction_by_dataset: dict[str, list[np.ndarray]] = {}
    compact_stats_by_dataset: dict[str, dict[str, Any]] = {}
    for bundle in eval_bundles:
        pair = route_pair(
            route=route,
            dataset=bundle.name,
            base_pair=base_pair,
            global_pair=global_pair,
            experts=experts,
        )
        doc_values, query_values = encode_bundle(
            bundle=bundle,
            model_pair=pair,
            device=device,
            batch_size=args.encode_batch_size,
        )
        correction_rows, compact_stats = m211.compact_correction_rows(
            bundle=bundle,
            doc_values=doc_values,
            query_values=query_values,
            mode=mode,
            correction_window=args.correction_window,
        )
        correction_by_dataset[bundle.name] = correction_rows
        compact_stats_by_dataset[bundle.name] = compact_stats

    train_x, train_y, train_benefits, train_diag = m213.collect_selector_data(
        bundles=train_bundles,
        correction_by_dataset=correction_by_dataset,
        mode_weights=mode_weights,
        dataset_feature_names=dataset_feature_names,
        protected_head=args.protected_head,
        correction_window=args.correction_window,
        top_k=args.top_k,
        recall_tie_weight=args.recall_tie_weight,
        min_objective_gain=args.min_objective_gain,
    )
    state_dict, mean, std, history = m212.train_selector(
        features=train_x,
        labels=train_y,
        hidden_dim=args.selector_hidden_dim,
        epochs=args.selector_epochs,
        learning_rate=args.selector_learning_rate,
        seed=selector_seed,
        device=device,
    )
    counts = np.bincount(train_y, minlength=len(mode_weights))
    label_distribution = {
        f'w{mode_weights[index]:g}': float(count / train_y.size)
        for index, count in enumerate(counts)
    }
    selector_summary = {
        'route': route,
        'dimension': dimension,
        'mode': mode,
        'label_distribution': label_distribution,
        'mean_qrels_objective_gain': (
            statistics.fmean(train_benefits) if train_benefits else 0.0
        ),
        'mean_map_gain': statistics.fmean(
            row['best_map_gain'] for row in train_diag
        ) if train_diag else 0.0,
        'mean_recall_gain': statistics.fmean(
            row['best_recall_gain'] for row in train_diag
        ) if train_diag else 0.0,
        'history': history,
    }

    oracle_labels_by_dataset: dict[str, np.ndarray] = {}
    probabilities_by_dataset: dict[str, np.ndarray] = {}
    for bundle in eval_bundles:
        features, labels, _benefits, _diag = m213.collect_selector_data(
            bundles=[bundle],
            correction_by_dataset=correction_by_dataset,
            mode_weights=mode_weights,
            dataset_feature_names=dataset_feature_names,
            protected_head=args.protected_head,
            correction_window=args.correction_window,
            top_k=args.top_k,
            recall_tie_weight=args.recall_tie_weight,
            min_objective_gain=args.min_objective_gain,
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

    variants: list[dict[str, Any]] = []
    first_stats = compact_stats_by_dataset[eval_bundles[0].name]
    for family, threshold, deployable in [
        ('mode_soft', None, True),
        ('mode_hard', None, True),
        ('mode_oracle', None, False),
    ]:
        rows = []
        for bundle in eval_bundles:
            if family == 'mode_oracle':
                probs = np.zeros(
                    (len(oracle_labels_by_dataset[bundle.name]), len(mode_weights)),
                    dtype=np.float32,
                )
                probs[
                    np.arange(probs.shape[0]),
                    oracle_labels_by_dataset[bundle.name],
                ] = 1.0
            else:
                probs = probabilities_by_dataset[bundle.name]
            alphas = m212.alphas_from_modes(
                probabilities=probs,
                mode_weights=mode_weights,
                family=family,
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
        m213.add_variant(
            variants=variants,
            dimension=dimension,
            mode=mode,
            selector_family=family,
            confidence_threshold=threshold,
            deployable=deployable,
            mode_weights=mode_weights,
            rows=rows,
            compact_stats=first_stats,
            training_final=history[-1] if history else None,
            label_distribution=label_distribution,
            mean_qrels_objective_gain=selector_summary[
                'mean_qrels_objective_gain'
            ],
        )
        variants[-1]['route'] = route
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
        m213.add_variant(
            variants=variants,
            dimension=dimension,
            mode=mode,
            selector_family='mode_confident',
            confidence_threshold=threshold,
            deployable=True,
            mode_weights=mode_weights,
            rows=rows,
            compact_stats=first_stats,
            training_final=history[-1] if history else None,
            label_distribution=label_distribution,
            mean_qrels_objective_gain=selector_summary[
                'mean_qrels_objective_gain'
            ],
        )
        variants[-1]['route'] = route
    return variants, selector_summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    dimensions = base.parse_int_csv(args.dimensions)
    modes = parse_str_csv(args.quantization)
    routes = parse_str_csv(args.routes)
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

    basis, _energy = base.load_principal_basis(
        args.projection_basis,
        dimensions=max(dimensions),
        report_dimensions=dimensions,
    )

    variants: list[dict[str, Any]] = []
    selector_training: list[dict[str, Any]] = []
    expert_training: dict[str, Any] = {}
    checkpoint_pairs: dict[str, tuple[torch.nn.Module, torch.nn.Module]] = {}

    for dimension in dimensions:
        base_doc, base_query, m210_policy = m211.load_m210_models(
            checkpoint_dir=args.m210_checkpoint_dir,
            dimension=dimension,
            basis=basis,
            device=device,
        )
        base_doc.cpu()
        base_query.cpu()
        base_pair = (base_doc, base_query)
        global_doc, global_query, global_history, global_samples = train_qrels_expert(
            bundles=train_bundles,
            init_doc_model=base_doc,
            init_query_model=base_query,
            epochs=args.expert_epochs,
            learning_rate=args.expert_learning_rate,
            weight_decay=args.expert_weight_decay,
            distill_weight=args.distill_weight,
            pair_positive_k=args.pair_positive_k,
            pair_negative_k=args.pair_negative_k,
            seed=args.seed + dimension * 1000,
            device=device,
        )
        global_pair = (global_doc, global_query)
        experts: dict[str, tuple[torch.nn.Module, torch.nn.Module]] = {}
        expert_training[f'{dimension}:global'] = {
            'samples': global_samples,
            'history': global_history,
        }
        checkpoint_pairs[f'dim{dimension}_global'] = global_pair
        for index, bundle in enumerate(train_bundles):
            doc_model, query_model, history, samples = train_qrels_expert(
                bundles=[bundle],
                init_doc_model=base_doc,
                init_query_model=base_query,
                epochs=args.expert_epochs,
                learning_rate=args.expert_learning_rate,
                weight_decay=args.expert_weight_decay,
                distill_weight=args.distill_weight,
                pair_positive_k=args.pair_positive_k,
                pair_negative_k=args.pair_negative_k,
                seed=args.seed + dimension * 1000 + index + 1,
                device=device,
            )
            experts[bundle.name] = (doc_model, query_model)
            expert_training[f'{dimension}:{bundle.name}'] = {
                'samples': samples,
                'history': history,
            }
            checkpoint_pairs[f'dim{dimension}_{bundle.name}'] = (
                doc_model,
                query_model,
            )

        for mode_index, mode in enumerate(modes):
            for route_index, route in enumerate(routes):
                route_variants, selector_summary = evaluate_route(
                    route=route,
                    dimension=dimension,
                    mode=mode,
                    train_bundles=train_bundles,
                    eval_bundles=eval_bundles,
                    base_pair=base_pair,
                    global_pair=global_pair,
                    experts=experts,
                    dense_reference=dense_reference,
                    mode_weights=mode_weights,
                    confidence_thresholds=confidence_thresholds,
                    dataset_feature_names=dataset_feature_names,
                    args=args,
                    device=device,
                    selector_seed=(
                        args.seed
                        + dimension * 100
                        + mode_index * 10
                        + route_index
                    ),
                )
                variants.extend(route_variants)
                selector_training.append(selector_summary)
        checkpoint_pairs[f'dim{dimension}_base'] = base_pair
        expert_training[f'{dimension}:m210_policy'] = m210_policy

    variants.sort(
        key=lambda item: (
            item['deployable'],
            item['dimension'] == 32 and item['mode'] == 'int4_row',
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
            'train qrels used for expert pairwise training and selector labels; '
            'eval qrels used for metrics and oracle diagnostics; supervised '
            'diagnostic, not qrels-free benchmark'
        ),
        'training_scope': 'qrels-supervised row-family expert 32d mini-vector head',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'dimensions': list(dimensions),
            'quantization': list(modes),
            'routes': list(routes),
            'expert_epochs': args.expert_epochs,
            'expert_learning_rate': args.expert_learning_rate,
            'expert_weight_decay': args.expert_weight_decay,
            'distill_weight': args.distill_weight,
            'pair_positive_k': args.pair_positive_k,
            'pair_negative_k': args.pair_negative_k,
            'mode_weights': list(mode_weights),
            'confidence_thresholds': list(confidence_thresholds),
            'selector_hidden_dim': args.selector_hidden_dim,
            'selector_epochs': args.selector_epochs,
            'selector_learning_rate': args.selector_learning_rate,
            'recall_tie_weight': args.recall_tie_weight,
            'min_objective_gain': args.min_objective_gain,
            'dataset_feature_mode': args.dataset_feature_mode,
            'dataset_feature_names': list(dataset_feature_names),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'device': device,
        },
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'expert_training': expert_training,
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
    for name, (doc_model, query_model) in checkpoint_pairs.items():
        torch.save(
            {
                'schema': SCHEMA,
                'name': name,
                'doc_model': doc_model.cpu().state_dict(),
                'query_model': query_model.cpu().state_dict(),
                'policy': payload['policy'],
            },
            args.output_dir / f'{name}.pt',
        )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M214 row-family expert mini-vector head',
        '',
        f"- Train datasets: `{', '.join(payload['train_datasets'])}`",
        f"- Eval datasets: `{', '.join(payload['eval_datasets'])}`",
        f"- Scope: `{payload['training_scope']}`",
        '',
        '## Top Deployable 32d INT4 Variants',
        '',
        '| Rank | Route | Selector | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | --- | --- | ---: | ---: | --- | --- |',
    ]
    rank = 0
    for variant in payload['variants']:
        if (
            not variant['deployable']
            or variant['dimension'] != 32
            or variant['mode'] != 'int4_row'
        ):
            continue
        rank += 1
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        selector = variant['selector_family']
        if variant['confidence_threshold'] is not None:
            selector = f"{selector}@{variant['confidence_threshold']:.2f}"
        lines.append(
            f"| {rank} | {variant['route']} | {selector} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
        if rank >= 30:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
