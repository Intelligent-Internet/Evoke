#!/usr/bin/env python3
"""M217 RankT5-pseudo pairwise global mini-vector head.

M216 showed that exact dense-teacher pairwise ordering is not enough to learn
the last-mile gains found by qrels-supervised M214/M215. M217 keeps the same
deployable product contract, but replaces the dense-teacher ordering with a
RankT5 cross-encoder pseudo-relevance teacher inside the frozen top256 window.

Qrels are used only for final retrieval metrics and teacher diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from dataclasses import replace
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


SCHEMA = 'm217_rankt5_pairwise_global_head_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate RankT5-pseudo pairwise global head.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--rankt5-model', type=Path, required=True)
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
    parser.add_argument('--seed', type=int, default=217)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=8192)
    parser.add_argument('--rankt5-batch-size', type=int, default=8)
    parser.add_argument('--rankt5-max-length', type=int, default=512)
    parser.add_argument('--teacher-score-cache-dir', type=Path)
    parser.add_argument('--teacher-progress-interval', type=int, default=32)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def fingerprint_bundle(bundle: m202.DatasetBundle) -> str:
    digest = hashlib.sha256()
    digest.update(bundle.name.encode())
    digest.update(str(len(bundle.windows)).encode())
    for query_id, window in zip(bundle.dense_queries_raw.ids, bundle.windows):
        digest.update(b'\0')
        digest.update(query_id.encode())
        digest.update(np.asarray(window, dtype=np.int64).tobytes())
    return digest.hexdigest()


def pack_rows(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + int(len(row)))
    if offsets[-1]:
        values = np.concatenate([
            np.asarray(row, dtype=np.float32)
            for row in rows
        ]).astype(np.float32, copy=False)
    else:
        values = np.empty(0, dtype=np.float32)
    return np.asarray(offsets, dtype=np.int64), values


def unpack_rows(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[offsets[index]:offsets[index + 1]], dtype=np.float32)
        for index in range(len(offsets) - 1)
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def load_rankt5(model_path: Path, device: str) -> tuple[Any, Any, int]:
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
    token_id = tokenizer.convert_tokens_to_ids('<extra_id_10>')
    if token_id == tokenizer.unk_token_id:
        raise ValueError('RankT5 relevance token is missing')
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
        )
    except TypeError:
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
    model.to(device).eval()
    return tokenizer, model, int(token_id)


def score_text_pairs(
    *,
    tokenizer: Any,
    model: Any,
    token_id: int,
    texts: Sequence[str],
    device: str,
    max_length: int,
) -> np.ndarray:
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    ).to(device)
    decoder = torch.full(
        (len(texts), 1),
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )
    logits = model(**inputs, decoder_input_ids=decoder).logits[:, 0, token_id]
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite RankT5 scores')
    return logits.detach().cpu().float().numpy()


def score_rankt5_bundle(
    *,
    bundle: m202.DatasetBundle,
    tokenizer: Any,
    model: Any,
    token_id: int,
    cache_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    progress_interval: int,
) -> list[np.ndarray]:
    fingerprint = fingerprint_bundle(bundle)
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_path = cache_dir / f'{bundle.name}.rankt5.top256.npz'
    manifest_path = cache_dir / f'{bundle.name}.rankt5.top256.json'
    if score_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (
            manifest.get('schema') == SCHEMA
            and manifest.get('dataset') == bundle.name
            and manifest.get('fingerprint') == fingerprint
        ):
            packed = np.load(score_path)
            return unpack_rows(packed['offsets'], packed['scores'])

    started = time.perf_counter()
    rows: list[np.ndarray] = []
    completed_pairs = 0
    with torch.inference_mode():
        for row, documents in enumerate(bundle.windows):
            query = bundle.dense_queries_raw.texts[row]
            prompts = [
                'Query: '
                + query
                + ' Document: '
                + bundle.dense_documents_raw.texts[int(doc)]
                for doc in documents
            ]
            lengths = [
                len(tokenizer(prompt, truncation=False)['input_ids'])
                for prompt in prompts
            ]
            order = sorted(range(len(prompts)), key=lambda index: lengths[index])
            scores = np.empty(len(prompts), dtype=np.float32)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                batch_scores = score_text_pairs(
                    tokenizer=tokenizer,
                    model=model,
                    token_id=token_id,
                    texts=[prompts[index] for index in indices],
                    device=device,
                    max_length=max_length,
                )
                scores[np.asarray(indices, dtype=np.int64)] = batch_scores
            rows.append(scores)
            completed_pairs += len(scores)
            if (row + 1) % progress_interval == 0 or row + 1 == len(bundle.windows):
                elapsed = time.perf_counter() - started
                write_json(
                    output_dir / 'rankt5-progress.json',
                    {
                        'dataset': bundle.name,
                        'completed_queries': row + 1,
                        'total_queries': len(bundle.windows),
                        'completed_pairs': completed_pairs,
                        'elapsed_seconds': elapsed,
                        'estimated_dataset_seconds': (
                            elapsed * len(bundle.windows) / (row + 1)
                        ),
                    },
                )
                base.log(
                    f'm217 rankt5 {bundle.name} '
                    f'{row + 1}/{len(bundle.windows)} '
                    f'pairs={completed_pairs}'
                )

    offsets, values = pack_rows(rows)
    np.savez_compressed(score_path, offsets=offsets, scores=values)
    write_json(
        manifest_path,
        {
            'schema': SCHEMA,
            'dataset': bundle.name,
            'fingerprint': fingerprint,
            'queries': len(rows),
            'pairs': int(values.size),
            'batch_size': batch_size,
            'max_length': max_length,
            'seconds': time.perf_counter() - started,
        },
    )
    return rows


def window_scores_to_candidate_rows(
    bundle: m202.DatasetBundle,
    window_scores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for candidates, window, scores in zip(
        bundle.candidates,
        bundle.windows,
        window_scores,
    ):
        positions = {int(doc): index for index, doc in enumerate(candidates)}
        correction = np.zeros(len(candidates), dtype=np.float32)
        for doc, score in zip(window, scores):
            correction[positions[int(doc)]] = float(score)
        rows.append(correction)
    return rows


def add_variant(
    *,
    variants: list[dict[str, Any]],
    source: str,
    weight: float,
    rows: list[dict[str, Any]],
    compact_stats: dict[str, Any] | None,
    training_final: dict[str, float] | None,
    deployable: bool,
) -> None:
    macro_vs_dense = m216.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'dimension': 32,
            'mode': 'int4_row',
            'source': source,
            'selector_family': 'fixed_weight',
            'fixed_weight': weight,
            'deployable': deployable,
            'training_final': training_final,
            'compact_stats': compact_stats,
            'macro_metrics': m216.macro_metrics(rows),
            'macro_delta_vs_fixed_hybrid': m216.macro_delta(
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
    started = time.perf_counter()
    if args.output_dir.exists() and (args.output_dir / 'summary.json').exists():
        raise FileExistsError(args.output_dir / 'summary.json')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M217 RankT5 scoring requires a CUDA device')
    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    modes = parse_str_csv(args.quantization)
    if modes != ('int4_row',):
        raise ValueError('M217 currently expects --quantization int4_row')
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

    tokenizer, rankt5, token_id = load_rankt5(args.rankt5_model, device)
    cache_dir = args.teacher_score_cache_dir or args.output_dir / 'teacher_scores'
    teacher_window_by_dataset = {
        bundle.name: score_rankt5_bundle(
            bundle=bundle,
            tokenizer=tokenizer,
            model=rankt5,
            token_id=token_id,
            cache_dir=cache_dir,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.rankt5_batch_size,
            max_length=args.rankt5_max_length,
            progress_interval=args.teacher_progress_interval,
        )
        for bundle in bundle_cache.values()
    }
    del rankt5
    torch.cuda.empty_cache()

    rankt5_train_bundles = [
        replace(bundle, teachers=teacher_window_by_dataset[bundle.name])
        for bundle in train_bundles
    ]
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
    doc_model, query_model, history, samples = m216.train_dense_pairwise_head(
        bundles=rankt5_train_bundles,
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

    variants: list[dict[str, Any]] = []
    first_stats = compact_stats_by_dataset[eval_bundles[0].name]
    for weight in fixed_weights:
        rows = [
            m213.evaluate_variant(
                bundle=bundle,
                correction_rows=correction_by_dataset[bundle.name],
                alphas=np.full(len(bundle.candidates), weight, dtype=np.float32),
                dense_reference=dense_reference,
                protected_head=args.protected_head,
                correction_window=args.correction_window,
                top_k=args.top_k,
            )
            for bundle in eval_bundles
        ]
        add_variant(
            variants=variants,
            source='student_32d_int4',
            weight=weight,
            rows=rows,
            compact_stats=first_stats,
            training_final=history[-1],
            deployable=True,
        )

    for weight in fixed_weights:
        rows = []
        for bundle in eval_bundles:
            teacher_rows = window_scores_to_candidate_rows(
                bundle,
                teacher_window_by_dataset[bundle.name],
            )
            rows.append(
                m213.evaluate_variant(
                    bundle=bundle,
                    correction_rows=teacher_rows,
                    alphas=np.full(len(bundle.candidates), weight, dtype=np.float32),
                    dense_reference=dense_reference,
                    protected_head=args.protected_head,
                    correction_window=args.correction_window,
                    top_k=args.top_k,
                ),
            )
        add_variant(
            variants=variants,
            source='rankt5_teacher_ceiling',
            weight=weight,
            rows=rows,
            compact_stats=None,
            training_final=None,
            deployable=False,
        )

    variants.sort(
        key=lambda item: (
            item['source'] == 'student_32d_int4',
            item['beats_pplx_dense_primary'],
            -len(item['unsafe_rows_vs_dense']),
            item['macro_delta_vs_pplx_dense']['map@100'],
            item['macro_delta_vs_pplx_dense']['recall@100'],
        ),
        reverse=True,
    )
    teacher_cache_manifests = {
        name: str((cache_dir / f'{name}.rankt5.top256.json').resolve())
        for name in sorted(teacher_window_by_dataset)
    }
    payload = {
        'schema': SCHEMA,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': (
            'qrels used only for final metrics and teacher diagnostics; '
            'training uses RankT5 pseudo-labels inside top256'
        ),
        'training_scope': 'RankT5-pseudo pairwise global 32d INT4 mini-vector head',
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
            'rankt5_batch_size': args.rankt5_batch_size,
            'rankt5_max_length': args.rankt5_max_length,
            'device': device,
        },
        'rankt5_model': str(args.rankt5_model),
        'teacher_cache_dir': str(cache_dir),
        'teacher_cache_manifests': teacher_cache_manifests,
        'm210_checkpoint_dir': str(args.m210_checkpoint_dir),
        'm210_policy': m210_policy,
        'training_samples': samples,
        'training_history': history,
        'variants': variants,
        'elapsed_seconds': time.perf_counter() - started,
    }
    write_json(args.output_dir / 'summary.json', payload)
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
        args.output_dir / f'rankt5_pairwise_dim{args.dimension}.pt',
    )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        '# M217 RankT5-pseudo pairwise global mini-vector head',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Training samples: `{payload['training_samples']}`",
        '',
        '## Fixed Weight Variants',
        '',
        '| Rank | Source | Weight | dMAP vs dense | dRecall vs dense | '
        'Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['source']} | "
            f"{variant['fixed_weight']:.2f} | "
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
