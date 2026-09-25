#!/usr/bin/env python3
"""M207 last-mile rank operator scout.

M203 proved that exact dense scores inside the BM25+SSR top256 window can beat
PPLX dense, but M204-M206 failed to transfer that score-blend operator into a
compact mini-vector.  This script keeps training out of the loop and tests
whether rank-based or gated local operators give a better exact ceiling for the
mini-vector's last-mile sorting role.  Qrels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research_sae_m1970_compact_dense_cascade_oracle as base
import research_sae_m1971_compact_dense_fusion_oracle as m1971
import research_sae_m200_top256_tiny_int4_floor as m200
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m203_dual_head_operator_ceiling as m203


SCHEMA = 'm207_last_mile_rank_operator_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


@dataclass(frozen=True)
class Operator:
    family: str
    protected_head: int
    correction_window: int
    weight: float
    rrf_k: float | None = None
    gate_threshold: float | None = None
    gate_rank: int | None = None

    def key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.protected_head,
            self.correction_window,
            self.weight,
            self.rrf_k,
            self.gate_threshold,
            self.gate_rank,
        )

    def label(self) -> str:
        parts = [
            self.family,
            f'head={self.protected_head}',
            f'window={self.correction_window}',
            f'w={self.weight:g}',
        ]
        if self.rrf_k is not None:
            parts.append(f'k={self.rrf_k:g}')
        if self.gate_threshold is not None:
            parts.append(f'gate={self.gate_threshold:g}')
        if self.gate_rank is not None:
            parts.append(f'gate_rank={self.gate_rank}')
        return ', '.join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Scout exact rank/gated operators for local mini-vector.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--datasets',
        default='arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,'
        'webis-touche2020',
    )
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--protected-heads', default='0,5')
    parser.add_argument('--correction-windows', default='100,256')
    parser.add_argument('--score-weights', default='0.75,1.0,1.25,1.5,2.0')
    parser.add_argument('--rank-weights', default='0.25,0.5,0.75,1.0,1.5,2.0')
    parser.add_argument('--rrf-weights', default='0.5,1.0,1.5,2.0')
    parser.add_argument('--rrf-ks', default='10,30,60')
    parser.add_argument('--gate-thresholds', default='0.05,0.1,0.15,0.2,0.3')
    parser.add_argument('--gate-rank', type=int, default=16)
    parser.add_argument('--block-size', type=int, default=64)
    return parser.parse_args()


def parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('float list must not be empty')
    return values


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(',') if value}))
    if not values:
        raise ValueError('integer list must not be empty')
    return values


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


def row_safe(delta: dict[str, float], *, floor: float = -0.002) -> bool:
    return min(delta[name] for name in PRIMARY_METRICS) >= floor


def rank_scores(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    return np.linspace(1.0, 0.0, num=length, dtype=np.float32)


def inverse_ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(order.shape[0], dtype=np.float32)
    ranks[order] = np.arange(order.shape[0], dtype=np.float32)
    return ranks


def dense_margin_gate(
    dense_values: np.ndarray,
    *,
    threshold: float,
    gate_rank: int,
) -> bool:
    if dense_values.size < 2:
        return False
    normalized = m1971.minmax(dense_values)
    order = base.top_indices(normalized, top_k=normalized.size)
    compare_at = min(max(gate_rank, 2), order.size) - 1
    margin = float(normalized[order[0]] - normalized[order[compare_at]])
    return margin >= threshold


def operator_rankings(
    *,
    candidate_rows: Sequence[np.ndarray],
    baseline_rows: Sequence[np.ndarray],
    dense_rows: Sequence[np.ndarray],
    operator: Operator,
    top_k: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    rankings: list[np.ndarray] = []
    gate_decisions: list[float] = []
    for candidates, baseline, dense in zip(
        candidate_rows,
        baseline_rows,
        dense_rows,
    ):
        baseline = np.asarray(baseline, dtype=np.float32)
        dense = np.asarray(dense, dtype=np.float32)
        baseline_order = base.top_indices(baseline, top_k=len(candidates))
        head_size = min(operator.protected_head, top_k, len(candidates))
        window_size = min(operator.correction_window, len(candidates))
        protected = baseline_order[:head_size]
        eligible = baseline_order[head_size:window_size]
        tail_budget = max(top_k - head_size, 0)
        if eligible.size == 0 or tail_budget == 0:
            local = protected
        else:
            final = operator_scores(
                baseline=baseline,
                dense=dense,
                eligible=eligible,
                operator=operator,
                gate_decisions=gate_decisions,
            )
            tail_order = base.top_indices(final, top_k=tail_budget)
            local = np.concatenate((protected, eligible[tail_order]))
        rankings.append(np.asarray(candidates, dtype=np.int64)[local])
    stats = {
        'gate_rate': (
            statistics.fmean(gate_decisions) if gate_decisions else 1.0
        ),
    }
    return rankings, stats


def operator_scores(
    *,
    baseline: np.ndarray,
    dense: np.ndarray,
    eligible: np.ndarray,
    operator: Operator,
    gate_decisions: list[float],
) -> np.ndarray:
    baseline_values = baseline[eligible]
    dense_values = dense[eligible]
    if operator.family == 'score_minmax':
        return baseline_values + operator.weight * m1971.minmax(dense_values)
    if operator.family == 'score_margin_gate':
        assert operator.gate_threshold is not None
        assert operator.gate_rank is not None
        gate = dense_margin_gate(
            dense_values,
            threshold=operator.gate_threshold,
            gate_rank=operator.gate_rank,
        )
        gate_decisions.append(float(gate))
        if not gate:
            return baseline_values
        return baseline_values + operator.weight * m1971.minmax(dense_values)
    dense_order = base.top_indices(dense_values, top_k=eligible.size)
    dense_ranks = inverse_ranks(dense_order)
    baseline_rank_scores = rank_scores(eligible.size)
    dense_rank_scores = rank_scores(eligible.size)[dense_ranks.astype(np.int64)]
    if operator.family == 'rank_linear':
        return baseline_rank_scores + operator.weight * dense_rank_scores
    if operator.family == 'rrf':
        assert operator.rrf_k is not None
        baseline_ranks = np.arange(eligible.size, dtype=np.float32)
        return (
            1.0 / (operator.rrf_k + baseline_ranks + 1.0)
            + operator.weight / (operator.rrf_k + dense_ranks + 1.0)
        )
    raise ValueError(f'unsupported operator family: {operator.family}')


def build_operators(args: argparse.Namespace) -> list[Operator]:
    heads = parse_int_csv(args.protected_heads)
    windows = parse_int_csv(args.correction_windows)
    score_weights = parse_float_csv(args.score_weights)
    rank_weights = parse_float_csv(args.rank_weights)
    rrf_weights = parse_float_csv(args.rrf_weights)
    rrf_ks = parse_float_csv(args.rrf_ks)
    gate_thresholds = parse_float_csv(args.gate_thresholds)
    operators: list[Operator] = []
    for head in heads:
        for window in windows:
            for weight in score_weights:
                operators.append(
                    Operator(
                        family='score_minmax',
                        protected_head=head,
                        correction_window=window,
                        weight=weight,
                    ),
                )
                for threshold in gate_thresholds:
                    operators.append(
                        Operator(
                            family='score_margin_gate',
                            protected_head=head,
                            correction_window=window,
                            weight=weight,
                            gate_threshold=threshold,
                            gate_rank=args.gate_rank,
                        ),
                    )
            for weight in rank_weights:
                operators.append(
                    Operator(
                        family='rank_linear',
                        protected_head=head,
                        correction_window=window,
                        weight=weight,
                    ),
                )
            for weight in rrf_weights:
                for rrf_k in rrf_ks:
                    operators.append(
                        Operator(
                            family='rrf',
                            protected_head=head,
                            correction_window=window,
                            weight=weight,
                            rrf_k=rrf_k,
                        ),
                    )
    return operators


def evaluate_dataset(
    *,
    bundle: m202.DatasetBundle,
    dense_metrics: dict[str, float],
    dense_rows: Sequence[np.ndarray],
    operators: Sequence[Operator],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operator in operators:
        rankings, stats = operator_rankings(
            candidate_rows=bundle.candidates,
            baseline_rows=bundle.baseline_rows,
            dense_rows=dense_rows,
            operator=operator,
            top_k=top_k,
        )
        metrics = m1971.retrieval_metrics(
            rankings,
            dense_queries=bundle.dense_queries_raw,
            dense_documents=bundle.dense_documents_raw,
            qrels=bundle.qrels,
        )
        delta_vs_dense = metric_delta(metrics, dense_metrics)
        delta_vs_fixed = metric_delta(metrics, bundle.baseline_metrics)
        rows.append(
            {
                'dataset': bundle.name,
                'operator': operator.label(),
                'operator_key': list(operator.key()),
                'family': operator.family,
                'protected_head': operator.protected_head,
                'correction_window': operator.correction_window,
                'weight': operator.weight,
                'rrf_k': operator.rrf_k,
                'gate_threshold': operator.gate_threshold,
                'gate_rank': operator.gate_rank,
                'gate_rate': stats['gate_rate'],
                'metrics': metrics,
                'delta_vs_pplx_dense': delta_vs_dense,
                'delta_vs_fixed_hybrid': delta_vs_fixed,
                'safe_vs_pplx_dense_at_0.002': row_safe(delta_vs_dense),
                'safe_vs_fixed_hybrid_at_0.002': row_safe(delta_vs_fixed),
            },
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.source_depth <= 0 or args.top_k <= 0:
        raise ValueError('--source-depth and --top-k must be positive')
    operators = build_operators(args)
    max_window = max(operator.correction_window for operator in operators)
    min_head = min(operator.protected_head for operator in operators)
    datasets = m202.parse_csv(args.datasets)
    started = time.perf_counter()

    loader_args = argparse.Namespace(**vars(args))
    loader_args.correction_window = max_window
    loader_args.protected_head = min_head
    loader_args.dense_weight = 1.5

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {
        operator.key(): [] for operator in operators
    }
    dataset_payloads: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = m202.load_bundle(dataset=dataset, args=loader_args)
        dense_rankings = m203.dense_full_rankings(
            dense_documents=bundle.dense_documents,
            dense_queries=bundle.dense_queries,
            top_k=args.top_k,
            block_size=args.block_size,
        )
        dense_metrics = m1971.retrieval_metrics(
            dense_rankings,
            dense_queries=bundle.dense_queries_raw,
            dense_documents=bundle.dense_documents_raw,
            qrels=bundle.qrels,
        )
        dense_rows = m203.candidate_dense_rows(bundle)
        rows = evaluate_dataset(
            bundle=bundle,
            dense_metrics=dense_metrics,
            dense_rows=dense_rows,
            operators=operators,
            top_k=args.top_k,
        )
        for row in rows:
            grouped[tuple(row['operator_key'])].append(row)
        dataset_payloads.append(
            {
                'dataset': dataset,
                'pplx_dense': dense_metrics,
                'fixed_hybrid': bundle.baseline_metrics,
                'fixed_hybrid_delta_vs_pplx_dense': metric_delta(
                    bundle.baseline_metrics,
                    dense_metrics,
                ),
            },
        )

    operator_by_key = {operator.key(): operator for operator in operators}
    variants: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        operator = operator_by_key[key]
        delta_vs_dense = macro_delta(rows, 'delta_vs_pplx_dense')
        variants.append(
            {
                'operator': operator.label(),
                'operator_key': list(key),
                'family': operator.family,
                'protected_head': operator.protected_head,
                'correction_window': operator.correction_window,
                'weight': operator.weight,
                'rrf_k': operator.rrf_k,
                'gate_threshold': operator.gate_threshold,
                'gate_rank': operator.gate_rank,
                'mean_gate_rate': statistics.fmean(
                    row['gate_rate'] for row in rows
                ),
                'macro_metrics': macro_metrics(rows),
                'macro_delta_vs_pplx_dense': delta_vs_dense,
                'macro_delta_vs_fixed_hybrid': macro_delta(
                    rows,
                    'delta_vs_fixed_hybrid',
                ),
                'beats_pplx_dense_primary': all(
                    delta_vs_dense[name] > 0.0 for name in PRIMARY_METRICS
                ),
                'unsafe_rows_vs_pplx_dense': [
                    row['dataset']
                    for row in rows
                    if not row['safe_vs_pplx_dense_at_0.002']
                ],
                'unsafe_rows_vs_fixed_hybrid': [
                    row['dataset']
                    for row in rows
                    if not row['safe_vs_fixed_hybrid_at_0.002']
                ],
                'rows': rows,
            },
        )
    variants.sort(
        key=lambda item: (
            item['beats_pplx_dense_primary'],
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
            -len(item['unsafe_rows_vs_pplx_dense']),
        ),
        reverse=True,
    )
    payload = {
        'schema': SCHEMA,
        'datasets': list(datasets),
        'qrels_usage': 'final metrics only; no training or qrel selection',
        'candidate_source': 'M190/SSR latent postings plus BM25',
        'target_route': 'last-mile local rank operator for mini-vector rerank',
        'policy': {
            'source_depth': args.source_depth,
            'top_k': args.top_k,
            'operator_count': len(operators),
            'protected_heads': parse_int_csv(args.protected_heads),
            'correction_windows': parse_int_csv(args.correction_windows),
            'score_weights': parse_float_csv(args.score_weights),
            'rank_weights': parse_float_csv(args.rank_weights),
            'rrf_weights': parse_float_csv(args.rrf_weights),
            'rrf_ks': parse_float_csv(args.rrf_ks),
            'gate_thresholds': parse_float_csv(args.gate_thresholds),
            'gate_rank': args.gate_rank,
        },
        'dataset_baselines': dataset_payloads,
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
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M207 last-mile rank operator scout',
        '',
        f"- Candidate source: {payload['candidate_source']}",
        f"- Target route: {payload['target_route']}",
        f"- Source depth: `{payload['policy']['source_depth']}`",
        f"- Operator count: `{payload['policy']['operator_count']}`",
        '',
        '## Top Variants',
        '',
        '| Rank | Family | Head | Window | Weight | Extra | '
        'dMAP vs dense | dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'][:30], start=1):
        delta = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_pplx_dense']) or 'none'
        extra_parts = []
        if variant['rrf_k'] is not None:
            extra_parts.append(f"k={variant['rrf_k']:g}")
        if variant['gate_threshold'] is not None:
            extra_parts.append(f"gate={variant['gate_threshold']:g}")
            extra_parts.append(f"rate={variant['mean_gate_rate']:.3f}")
        extra = ', '.join(extra_parts) or '-'
        lines.append(
            f"| {rank} | {variant['family']} | "
            f"{variant['protected_head']} | "
            f"{variant['correction_window']} | "
            f"{variant['weight']:.3g} | {extra} | "
            f"{delta['map@100']:+.6f} | "
            f"{delta['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
