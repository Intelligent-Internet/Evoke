#!/usr/bin/env python3
"""M233 EmbeddingGemma teacher preflight.

This experiment treats Google's EmbeddingGemma 2 as a qrels-free external
teacher source inside the existing II-42 top256 correction window. It measures
teacher-side headroom before any 32d INT4 student distillation is attempted.

Teacher construction uses only input text, EmbeddingGemma 2 embeddings, PPLX
dense scores, and baseline scores. Qrels are used only for final metrics.
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
import research_sae_m202_shared_top256_residual_head as m202
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m223_supervised_teacher_bridge_head as m223
import research_sae_m224_teacher_ensemble_bridge_head as m224


SCHEMA = 'm233_embeddinggemma_teacher_preflight_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')
QUERY_PREFIX = 'task: search result | query: '
DOCUMENT_PREFIX = 'title: none | text: '
DEFAULT_DIMS = '128,256,512,768'
DEFAULT_WEIGHTS = '0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0'
DEFAULT_POLICIES = (
    'eg2_{dim}_z',
    'eg2_{dim}_dense_mean_z',
    'eg2_{dim}_dense_baseline_mean_z',
    'eg2_{dim}_dense_rrf',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate EmbeddingGemma 2 as a teacher source.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--model-name', default='google/embeddinggemma-2')
    parser.add_argument('--embedding-dimensions', default=DEFAULT_DIMS)
    parser.add_argument('--teacher-policy-templates', default='')
    parser.add_argument('--fixed-weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--protected-head', type=int, default=0)
    parser.add_argument('--correction-window', type=int, default=256)
    parser.add_argument('--dense-weight', type=float, default=1.25)
    parser.add_argument('--source-depth', type=int, default=1000)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--encode-batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--rrf-k', type=float, default=60.0)
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


def actual_device(raw: str) -> str:
    if raw == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if raw == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return raw


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1.0e-6, None)


def cache_path(
    *,
    cache_dir: Path,
    dataset: str,
    role: str,
    model_name: str,
    max_length: int,
) -> Path:
    safe_model = model_name.replace('/', '--')
    return cache_dir / f'{dataset}.{role}.{safe_model}.len{max_length}.npz'


def load_npz(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def save_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_text_encoder(
    *,
    model_name: str,
    device: str,
    max_length: int,
) -> Any:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    dtype = torch.float32
    if device == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, 'vision_config'):
        config.vision_config = None
    if hasattr(config, 'audio_config'):
        config.audio_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return {
        'tokenizer': tokenizer,
        'model': model,
        'device': device,
        'max_length': int(max_length),
    }


def mean_pool(
    *,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = torch.sum(hidden * mask, dim=1)
    counts = torch.clamp(torch.sum(mask, dim=1), min=1.0e-6)
    return summed / counts


def encode_texts(
    *,
    encoder: dict[str, Any],
    texts: Sequence[str],
    prefix: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer = encoder['tokenizer']
    model = encoder['model']
    device = str(encoder['device'])
    max_length = int(encoder['max_length'])
    chunks: list[np.ndarray] = []
    total = len(texts)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_texts = [
                prefix + str(text)
                for text in texts[start:start + batch_size]
            ]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }
            outputs = model(**inputs)
            pooled = None
            for name in (
                'sentence_embedding',
                'pooler_output',
                'text_embeds',
                'embedding',
                'embeddings',
            ):
                value = getattr(outputs, name, None)
                if value is not None:
                    pooled = value
                    break
            if pooled is None:
                hidden = getattr(outputs, 'last_hidden_state', None)
                if hidden is None and isinstance(outputs, (tuple, list)):
                    hidden = outputs[0]
                if hidden is None:
                    raise ValueError('model output has no embedding tensor')
                pooled = mean_pool(
                    hidden=hidden,
                    attention_mask=inputs['attention_mask'],
                )
            chunks.append(pooled.float().cpu().numpy())
            base.log(f'm233 encoded texts={min(start + batch_size, total)}/{total}')
    matrix = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError(f'invalid embedding matrix shape: {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise ValueError('EmbeddingGemma produced non-finite embeddings')
    return l2_rows(matrix.astype(np.float32, copy=False))


def cached_query_vectors(
    *,
    encoder: dict[str, Any],
    bundle: m202.DatasetBundle,
    cache_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    path = cache_path(
        cache_dir=cache_dir,
        dataset=bundle.name,
        role='queries',
        model_name=model_name,
        max_length=max_length,
    )
    cached = load_npz(path)
    if cached is not None:
        ids = tuple(str(value) for value in cached['ids'].tolist())
        if ids == bundle.dense_queries_raw.ids:
            base.log(f'm233 query cache hit: {path}')
            return np.asarray(cached['vectors'], dtype=np.float32)
        base.log(f'm233 ignoring stale query cache: {path}')
    base.log(f'm233 encoding {bundle.name} queries with EmbeddingGemma 2')
    vectors = encode_texts(
        encoder=encoder,
        texts=bundle.dense_queries_raw.texts,
        prefix=QUERY_PREFIX,
        batch_size=batch_size,
    )
    save_npz(path, ids=np.asarray(bundle.dense_queries_raw.ids), vectors=vectors)
    return vectors


def cached_window_doc_vectors(
    *,
    encoder: dict[str, Any],
    bundle: m202.DatasetBundle,
    cache_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, int]]:
    path = cache_path(
        cache_dir=cache_dir,
        dataset=bundle.name,
        role='window_docs',
        model_name=model_name,
        max_length=max_length,
    )
    unique_doc_indices = sorted(
        {int(index) for window in bundle.windows for index in window},
    )
    cached = load_npz(path)
    if cached is not None:
        cached_indices = [int(value) for value in cached['doc_indices']]
        if cached_indices == unique_doc_indices:
            base.log(f'm233 document cache hit: {path}')
            positions = {doc: pos for pos, doc in enumerate(cached_indices)}
            return np.asarray(cached['vectors'], dtype=np.float32), positions
        base.log(f'm233 ignoring stale document cache: {path}')
    base.log(
        f'm233 encoding {bundle.name} window docs with EmbeddingGemma 2: '
        f'{len(unique_doc_indices)} unique docs',
    )
    texts = [bundle.dense_documents_raw.texts[index]
             for index in unique_doc_indices]
    vectors = encode_texts(
        encoder=encoder,
        texts=texts,
        prefix=DOCUMENT_PREFIX,
        batch_size=batch_size,
    )
    save_npz(
        path,
        doc_indices=np.asarray(unique_doc_indices, dtype=np.int64),
        vectors=vectors,
    )
    positions = {doc: pos for pos, doc in enumerate(unique_doc_indices)}
    return vectors, positions


def build_eg2_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    encoder: dict[str, Any],
    cache_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
) -> dict[str, dict[str, list[np.ndarray]]]:
    scores_by_dim: dict[str, dict[str, list[np.ndarray]]] = {}
    for bundle in bundles:
        query_vectors = cached_query_vectors(
            encoder=encoder,
            bundle=bundle,
            cache_dir=cache_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
        )
        doc_vectors, doc_positions = cached_window_doc_vectors(
            encoder=encoder,
            bundle=bundle,
            cache_dir=cache_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
        )
        for dim in (128, 256, 512, 768):
            if dim > query_vectors.shape[1]:
                continue
            dim_key = str(dim)
            scores_by_dim.setdefault(dim_key, {})
            query_dim = l2_rows(query_vectors[:, :dim])
            doc_dim = l2_rows(doc_vectors[:, :dim])
            rows: list[np.ndarray] = []
            for row, window in enumerate(bundle.windows):
                local_doc_rows = [doc_positions[int(index)] for index in window]
                scores = doc_dim[local_doc_rows] @ query_dim[row]
                rows.append(scores.astype(np.float32, copy=False))
            scores_by_dim[dim_key][bundle.name] = rows
    return scores_by_dim


def rrf_scores(values: np.ndarray, *, rrf_k: float) -> np.ndarray:
    return m224.rrf_scores(np.asarray(values, dtype=np.float32), rrf_k=rrf_k)


def build_policy_row(
    *,
    policy: str,
    eg2_scores: np.ndarray,
    dense_scores: np.ndarray,
    baseline_scores: np.ndarray,
    rrf_k: float,
) -> np.ndarray:
    eg2_z = m218.zscore(eg2_scores)
    dense_z = m218.zscore(dense_scores)
    baseline_z = m218.zscore(baseline_scores)
    if policy.endswith('_z'):
        return eg2_z.astype(np.float32, copy=False)
    if policy.endswith('_dense_mean_z'):
        return ((eg2_z + dense_z) / 2.0).astype(np.float32, copy=False)
    if policy.endswith('_dense_baseline_mean_z'):
        return ((eg2_z + dense_z + baseline_z) / 3.0).astype(
            np.float32,
            copy=False,
        )
    if policy.endswith('_dense_rrf'):
        return (
            (rrf_scores(eg2_scores, rrf_k=rrf_k)
             + rrf_scores(dense_scores, rrf_k=rrf_k))
            / 2.0
        ).astype(np.float32, copy=False)
    raise ValueError(f'unknown teacher policy: {policy}')


def expand_policy_templates(
    *,
    templates: Sequence[str],
    dimensions: Sequence[int],
) -> tuple[str, ...]:
    return tuple(
        template.format(dim=int(dim))
        for dim in dimensions
        for template in templates
    )


def build_policy_scores(
    *,
    bundles: Sequence[m202.DatasetBundle],
    scores_by_dim: dict[str, dict[str, list[np.ndarray]]],
    policies: Sequence[str],
    rrf_k: float,
) -> dict[str, dict[str, list[np.ndarray]]]:
    by_policy: dict[str, dict[str, list[np.ndarray]]] = {}
    for policy in policies:
        parts = policy.split('_', 2)
        if len(parts) < 3 or parts[0] != 'eg2':
            raise ValueError(f'invalid policy name: {policy}')
        dim_key = parts[1]
        rows_by_dataset: dict[str, list[np.ndarray]] = {}
        for bundle in bundles:
            rows: list[np.ndarray] = []
            for row, eg2_scores in enumerate(scores_by_dim[dim_key][bundle.name]):
                dense_scores = np.asarray(bundle.teachers[row], dtype=np.float32)
                baseline_scores = m218.local_baseline_scores(
                    bundle=bundle,
                    row=row,
                )
                rows.append(
                    build_policy_row(
                        policy=policy,
                        eg2_scores=eg2_scores,
                        dense_scores=dense_scores,
                        baseline_scores=baseline_scores,
                        rrf_k=rrf_k,
                    ),
                )
            rows_by_dataset[bundle.name] = rows
        by_policy[policy] = rows_by_dataset
    return by_policy


def add_variant(
    *,
    variants: list[dict[str, Any]],
    policy: str,
    weight: float,
    rows: list[dict[str, Any]],
) -> None:
    macro_vs_dense = m213.macro_delta(rows, 'delta_vs_pplx_dense')
    variants.append(
        {
            'source': 'embeddinggemma2_teacher_preflight',
            'teacher_policy': policy,
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
            'rows': rows,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = actual_device(args.device)
    eval_names = m202.parse_csv(args.eval_datasets)
    dimensions = parse_int_csv(args.embedding_dimensions)
    templates = (
        parse_str_csv(args.teacher_policy_templates)
        if args.teacher_policy_templates
        else DEFAULT_POLICIES
    )
    policies = expand_policy_templates(templates=templates, dimensions=dimensions)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    dense_reference = m213.load_dense_reference(args.reference_summary)

    loader_args = argparse.Namespace(**vars(args))
    bundles = [m202.load_bundle(dataset=name, args=loader_args)
               for name in eval_names]
    encoder = load_text_encoder(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
    )
    scores_by_dim = build_eg2_scores(
        bundles=bundles,
        encoder=encoder,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.encode_batch_size,
    )
    del encoder
    if device == 'cuda':
        torch.cuda.empty_cache()

    available_dimensions = tuple(int(value) for value in scores_by_dim)
    policy_scores = build_policy_scores(
        bundles=bundles,
        scores_by_dim=scores_by_dim,
        policies=policies,
        rrf_k=args.rrf_k,
    )
    variants: list[dict[str, Any]] = []
    for policy in policies:
        candidate_rows = m223.candidate_rows_from_window_scores(
            bundles=bundles,
            scores_by_dataset=policy_scores[policy],
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
                policy=policy,
                weight=weight,
                rows=rows,
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
            'teacher construction uses EmbeddingGemma 2, dense scores, and '
            'baseline scores only; qrels are used only for final metrics.'
        ),
        'training_scope': 'EmbeddingGemma 2 qrels-free teacher preflight',
        'model': {
            'name': args.model_name,
            'query_prefix': QUERY_PREFIX,
            'document_prefix': DOCUMENT_PREFIX,
            'max_length': args.max_length,
            'available_dimensions': list(available_dimensions),
        },
        'policy': {
            'teacher_policies': list(policies),
            'fixed_weights': list(fixed_weights),
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'top_k': args.top_k,
            'rrf_k': args.rrf_k,
            'device': device,
        },
        'cache_dir': str(args.cache_dir),
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
        '# M233 EmbeddingGemma 2 teacher preflight',
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- Model: `{payload['model']['name']}`",
        f"- Query prefix: `{payload['model']['query_prefix']}`",
        f"- Document prefix: `{payload['model']['document_prefix']}`",
        f"- Dimensions: `{payload['model']['available_dimensions']}`",
        '',
        '## Variants',
        '',
        '| Rank | Policy | Weight | MAP | Recall | dMAP vs dense | '
        'dRecall vs dense | Beats dense | Unsafe dense |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for rank, variant in enumerate(payload['variants'], 1):
        macro = variant['macro_metrics']
        macro_dense = variant['macro_delta_vs_pplx_dense']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | {variant['teacher_policy']} | "
            f"{variant['fixed_weight']:.3f} | "
            f"{macro['map@100']:.6f} | {macro['recall@100']:.6f} | "
            f"{macro_dense['map@100']:+.6f} | "
            f"{macro_dense['recall@100']:+.6f} | "
            f"{variant['beats_pplx_dense_primary']} | {unsafe} |"
        )
        if rank >= 100:
            break
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
