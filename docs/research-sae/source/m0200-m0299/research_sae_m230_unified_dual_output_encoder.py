#!/usr/bin/env python3
"""M230 unified dual-output encoder probe.

M229 proved that the 32d INT4 mini-vector sidecar can carry a strong supervised
score-geometry signal. M230A asks whether that signal can be read from the same
SAE encoder trunk that emits unified sparse postings.

This first probe freezes the SAE encoder and trains only two small mini-dense
readout heads from frozen hidden pooled states. The SAE posting output is not
mutated; qrels-supervised M214 artifacts are used only as fixed teacher
artifacts, and qrels are otherwise used only for final metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
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
import research_sae_m211_query_gated_mini_vector as m211
import research_sae_m213_qrels_safe_mode_selector as m213
import research_sae_m216_dense_pairwise_global_head as m216
import research_sae_m217_rankt5_pairwise_global_head as m217
import research_sae_m218_safe_rankt5_global_head as m218
import research_sae_m223_supervised_teacher_bridge_head as m223
import research_sae_m226_score_geometry_bridge_head as m226
import research_sae_m229_query_rich_score_geometry_bridge_head as m229


SCHEMA = 'm230_unified_dual_output_encoder_v1'
PRIMARY_METRICS = ('map@100', 'recall@100')


@dataclass
class FeatureBundle:
    bundle: m202.DatasetBundle
    doc_ids: np.ndarray
    doc_features: np.ndarray
    query_features: np.ndarray
    window_feature_indices: list[np.ndarray]


@dataclass(frozen=True)
class FeatureLayout:
    trunk_hidden_dim: int
    token_count: int
    token_start: int
    token_stop: int
    global_indices: tuple[int, ...]


class MiniDenseReadout(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0 or depth <= 0:
            raise ValueError('invalid mini-dense readout dimensions')
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _index in range(depth):
            layers.extend(
                (
                    torch.nn.Linear(current_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(hidden_dim),
                ),
            )
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TokenSummaryAdapterReadout(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        layout: FeatureLayout,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError('token adapter depth must be positive')
        self.input_dim = int(input_dim)
        self.trunk_hidden_dim = int(layout.trunk_hidden_dim)
        self.token_count = int(layout.token_count)
        self.token_start = int(layout.token_start)
        self.token_stop = int(layout.token_stop)
        global_indices = torch.tensor(layout.global_indices, dtype=torch.long)
        self.register_buffer('global_indices', global_indices)
        self.token_projection = torch.nn.Linear(self.trunk_hidden_dim, hidden_dim)
        self.global_projection = torch.nn.Linear(
            int(global_indices.numel()),
            hidden_dim,
        )
        self.attention = torch.nn.Linear(hidden_dim, 1)
        layers: list[torch.nn.Module] = [torch.nn.LayerNorm(hidden_dim)]
        for _index in range(depth):
            layers.extend(
                (
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(hidden_dim),
                ),
            )
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.output = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = values[:, self.token_start:self.token_stop].reshape(
            values.shape[0],
            self.token_count,
            self.trunk_hidden_dim,
        )
        global_values = values.index_select(1, self.global_indices)
        token_hidden = self.token_projection(tokens)
        global_hidden = self.global_projection(global_values)
        attention_logits = self.attention(
            torch.tanh(token_hidden + global_hidden[:, None, :]),
        ).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1)
        pooled = (attention[..., None] * token_hidden).sum(dim=1)
        return self.output(pooled + global_hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train/evaluate a frozen-SAE dual-output mini-dense probe.',
    )
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dense-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--projection-basis', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--m210-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--teacher-checkpoint-paths', default='')
    parser.add_argument('--teacher-source', default='m214_meanz')
    parser.add_argument('--m229-checkpoint-paths', default='')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-label', default='M230A')
    parser.add_argument('--feature-cache-dir', type=Path, required=True)
    parser.add_argument('--ng80-research-root', type=Path, required=True)
    parser.add_argument('--ng80-source-dir', type=Path, required=True)
    parser.add_argument('--ng80-config', type=Path, required=True)
    parser.add_argument('--sae-checkpoint', type=Path, required=True)
    parser.add_argument('--train-datasets', required=True)
    parser.add_argument('--eval-datasets', required=True)
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--quantization', default='int4_row')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--score-geometry-weight', type=float, default=0.2)
    parser.add_argument('--positive-k', type=int, default=16)
    parser.add_argument('--negative-start-rank', type=int, default=16)
    parser.add_argument('--negative-k', type=int, default=128)
    parser.add_argument('--min-positive-k', type=int, default=8)
    parser.add_argument('--min-negative-k', type=int, default=32)
    parser.add_argument('--fixed-weights', default='2.5,3.0,3.5,4.0,4.5')
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
    parser.add_argument('--seed', type=int, default=230)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--feature-batch-size', type=int, default=24)
    parser.add_argument('--head-batch-size', type=int, default=8192)
    parser.add_argument('--head-hidden-dim', type=int, default=512)
    parser.add_argument('--head-depth', type=int, default=2)
    parser.add_argument(
        '--head-kind',
        choices=('mlp', 'token_adapter'),
        default='mlp',
    )
    parser.add_argument('--feature-taps', default='mean')
    parser.add_argument('--sparse-summary-top-k', type=int, default=32)
    parser.add_argument('--token-summary-top-k', type=int, default=8)
    parser.add_argument('--overlap-k', type=int, default=24)
    parser.add_argument('--joint-threshold', type=float, default=0.30)
    return parser.parse_args()


def parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not values:
        raise ValueError('string list must not be empty')
    return values


def parse_paths_optional(raw: str) -> tuple[Path, ...]:
    return tuple(Path(value.strip()) for value in raw.split(',') if value.strip())


def infer_feature_layout(
    *,
    input_dim: int,
    taps: Sequence[str],
    sparse_summary_top_k: int,
    token_summary_top_k: int,
) -> FeatureLayout:
    hidden_coeff = 0
    fixed_dim = 0
    for tap in taps:
        if tap in ('mean', 'max', 'first'):
            hidden_coeff += 1
        elif tap == 'token_topk_hidden':
            hidden_coeff += token_summary_top_k
        elif tap == 'sparse_summary':
            fixed_dim += 5 + sparse_summary_top_k
        else:
            raise ValueError(f'unknown feature tap: {tap}')
    if hidden_coeff <= 0 or input_dim <= fixed_dim:
        raise ValueError('cannot infer hidden dimension from feature taps')
    remainder = input_dim - fixed_dim
    if remainder % hidden_coeff != 0:
        raise ValueError(
            'feature dimension is incompatible with taps: '
            f'input_dim={input_dim} fixed_dim={fixed_dim} '
            f'hidden_coeff={hidden_coeff}'
        )
    trunk_hidden_dim = remainder // hidden_coeff
    offset = 0
    token_start = -1
    token_stop = -1
    global_indices: list[int] = []
    for tap in taps:
        if tap in ('mean', 'max', 'first'):
            stop = offset + trunk_hidden_dim
            global_indices.extend(range(offset, stop))
            offset = stop
        elif tap == 'token_topk_hidden':
            token_start = offset
            token_stop = offset + token_summary_top_k * trunk_hidden_dim
            offset = token_stop
        elif tap == 'sparse_summary':
            stop = offset + 5 + sparse_summary_top_k
            global_indices.extend(range(offset, stop))
            offset = stop
    if token_start < 0 or token_stop <= token_start:
        raise ValueError('token_adapter requires token_topk_hidden feature tap')
    if offset != input_dim:
        raise ValueError(f'feature layout ended at {offset}, expected {input_dim}')
    return FeatureLayout(
        trunk_hidden_dim=trunk_hidden_dim,
        token_count=token_summary_top_k,
        token_start=token_start,
        token_stop=token_stop,
        global_indices=tuple(global_indices),
    )


def stable_json_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_ng80_encoder(args: argparse.Namespace, device: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(args.ng80_source_dir))
    module_path = args.ng80_source_dir / 'ng71_training.py'
    spec = importlib.util.spec_from_file_location('m230_ng71_training', module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot load ng71_training from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(args.ng80_config.read_text(encoding='utf-8'))
    encoder = module.Encoder(
        args.ng80_research_root,
        config,
        device,
        checkpoint=args.sae_checkpoint,
    )
    encoder.model.eval()
    for parameter in encoder.model.parameters():
        parameter.requires_grad_(False)
    return encoder, module


def load_m229_pair(
    *,
    checkpoint_path: Path,
    m210_checkpoint_dir: Path,
    dimension: int,
    basis: np.ndarray,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, str, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    architecture = checkpoint.get('architecture')
    if not isinstance(architecture, dict):
        raise ValueError(f'{checkpoint_path}: missing M229 architecture')
    doc_model, base_query, _policy = m211.load_m210_models(
        checkpoint_dir=m210_checkpoint_dir,
        dimension=dimension,
        basis=basis,
        device=device,
    )
    query_model = m229.QueryRichHead(
        base_model=base_query,
        input_dim=int(basis.shape[0]),
        output_dim=dimension,
        hidden_dim=int(architecture['hidden_dim']),
        depth=int(architecture['depth']),
        residual_scale=float(architecture['residual_scale']),
    ).to(device)
    doc_model.load_state_dict(checkpoint['doc_model'])
    query_model.load_state_dict(checkpoint['query_model'])
    doc_model.eval()
    query_model.eval()
    label = f'{checkpoint_path.parent.name}:{architecture.get("label", "m229")}'
    metadata = {
        'path': str(checkpoint_path),
        'label': label,
        'schema': checkpoint.get('schema'),
        'architecture': architecture,
        'objective_policy': checkpoint.get('objective_policy'),
    }
    return doc_model, query_model, label, metadata


def load_teacher_scores(
    *,
    args: argparse.Namespace,
    bundles: Sequence[m202.DatasetBundle],
    basis: np.ndarray,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, list[np.ndarray]]]]:
    if args.teacher_source == 'm214_meanz':
        paths = m223.parse_paths(args.teacher_checkpoint_paths)
        loader = m223.load_teacher_pair
    elif args.teacher_source == 'm229_meanz':
        paths = parse_paths_optional(args.m229_checkpoint_paths)
        if not paths:
            raise ValueError('--m229-checkpoint-paths required for m229_meanz')
        loader = load_m229_pair
    else:
        raise ValueError(f'unknown teacher source: {args.teacher_source}')

    metadata_rows: list[dict[str, Any]] = []
    score_sets: list[dict[str, list[np.ndarray]]] = []
    for teacher_path in paths:
        teacher_doc, teacher_query, teacher_label, metadata = loader(
            checkpoint_path=teacher_path,
            m210_checkpoint_dir=args.m210_checkpoint_dir,
            dimension=args.dimension,
            basis=basis,
            device=device,
        )
        base.log(f'm230 loading teacher {teacher_label}')
        metadata_rows.append(metadata)
        score_sets.append(
            m223.model_window_scores(
                bundles=bundles,
                doc_model=teacher_doc,
                query_model=teacher_query,
                device=device,
                batch_size=args.head_batch_size,
            ),
        )
        del teacher_doc
        del teacher_query
        torch.cuda.empty_cache()
    return metadata_rows, score_sets


def pooled_features(
    *,
    encoder: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    taps: Sequence[str],
    sparse_summary_top_k: int,
    token_summary_top_k: int,
) -> torch.Tensor:
    mask_bool = attention_mask.bool()
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    outputs: list[torch.Tensor] = []
    for tap in taps:
        if tap == 'mean':
            outputs.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
        elif tap == 'max':
            masked = hidden.masked_fill(~mask_bool[..., None], -1.0e9)
            outputs.append(masked.amax(1))
        elif tap == 'first':
            outputs.append(hidden[:, 0])
        elif tap == 'sparse_summary':
            if sparse_summary_top_k <= 0:
                raise ValueError('sparse summary top-k must be positive')
            values = encoder.model.lm_head(hidden).relu().log1p()
            values = values.masked_fill(~mask_bool[..., None], 0).amax(1)
            active = values > 0
            active_count = active.sum(1, keepdim=True).to(values.dtype)
            total = values.sum(1, keepdim=True)
            maximum = values.amax(1, keepdim=True)
            active_mean = total / active_count.clamp_min(1)
            norm = values.norm(dim=1, keepdim=True)
            top_values = torch.topk(
                values,
                k=min(sparse_summary_top_k, values.shape[1]),
                dim=1,
            ).values
            if top_values.shape[1] < sparse_summary_top_k:
                padding = top_values.new_zeros(
                    (top_values.shape[0], sparse_summary_top_k - top_values.shape[1]),
                )
                top_values = torch.cat((top_values, padding), dim=1)
            outputs.append(
                torch.cat(
                    (active_count, total, maximum, active_mean, norm, top_values),
                    dim=1,
                ),
            )
        elif tap == 'token_topk_hidden':
            if token_summary_top_k <= 0:
                raise ValueError('token summary top-k must be positive')
            scores = hidden.norm(dim=-1).masked_fill(~mask_bool, -1.0e9)
            top_k = min(token_summary_top_k, hidden.shape[1])
            top_indices = torch.topk(scores, k=top_k, dim=1).indices
            gather_index = top_indices[..., None].expand(
                -1,
                -1,
                hidden.shape[-1],
            )
            top_hidden = torch.gather(hidden, dim=1, index=gather_index)
            if top_k < token_summary_top_k:
                padding = top_hidden.new_zeros(
                    (
                        top_hidden.shape[0],
                        token_summary_top_k - top_k,
                        top_hidden.shape[2],
                    ),
                )
                top_hidden = torch.cat((top_hidden, padding), dim=1)
            outputs.append(top_hidden.flatten(1))
        else:
            raise ValueError(f'unknown feature tap: {tap}')
    return torch.cat(outputs, dim=1)


def encode_features(
    encoder: Any,
    *,
    texts: Sequence[str],
    role: str,
    batch_size: int,
    taps: Sequence[str],
    sparse_summary_top_k: int,
    token_summary_top_k: int,
) -> np.ndarray:
    if role not in ('query', 'document'):
        raise ValueError(f'invalid role: {role}')
    if not texts:
        raise ValueError('cannot encode empty text list')
    limit = int(encoder.config['base'][role + '_input_tokens'])
    outputs: list[np.ndarray] = []
    device = encoder.device
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            tokens = encoder.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=limit,
                return_tensors='pt',
            ).to(device)
            hidden = encoder.model.base_model(**tokens).last_hidden_state
            features = pooled_features(
                encoder=encoder,
                hidden=hidden,
                attention_mask=tokens['attention_mask'],
                taps=taps,
                sparse_summary_top_k=sparse_summary_top_k,
                token_summary_top_k=token_summary_top_k,
            )
            values = features.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError('non-finite hidden feature')
            outputs.append(values)
            if start == 0 or (start // batch_size) % 64 == 0:
                base.log(
                    f'm230 encoded {role} {min(start + batch_size, len(texts))}/'
                    f'{len(texts)}'
                )
    return np.vstack(outputs).astype(np.float32, copy=False)


def correction_doc_ids(
    bundle: m202.DatasetBundle,
    *,
    correction_window: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for candidates, baseline, window in zip(
        bundle.candidates,
        bundle.baseline_rows,
        bundle.windows,
    ):
        order = base.top_indices(baseline, top_k=len(candidates))
        window_size = min(correction_window, len(candidates))
        selected = np.asarray(candidates, dtype=np.int64)[order[:window_size]]
        chunks.append(selected)
        chunks.append(np.asarray(window, dtype=np.int64))
    merged = np.unique(np.concatenate(chunks))
    if merged.size == 0:
        raise ValueError(f'{bundle.name}: no correction documents selected')
    return merged.astype(np.int64, copy=False)


def feature_cache_path(
    *,
    cache_dir: Path,
    dataset: str,
    fingerprint: dict[str, Any],
) -> Path:
    digest = stable_json_sha(fingerprint)
    return cache_dir / f'{dataset}-{digest}-ng80-hidden.npz'


def build_feature_bundle(
    *,
    bundle: m202.DatasetBundle,
    encoder: Any,
    args: argparse.Namespace,
    sae_hash: str,
) -> FeatureBundle:
    doc_ids = correction_doc_ids(
        bundle,
        correction_window=args.correction_window,
    )
    fingerprint = {
        'schema': SCHEMA,
        'dataset': bundle.name,
        'sae_checkpoint': str(args.sae_checkpoint),
        'sae_hash': sae_hash,
        'ng80_config': str(args.ng80_config),
        'feature_taps': list(parse_str_csv(args.feature_taps)),
        'sparse_summary_top_k': args.sparse_summary_top_k,
        'token_summary_top_k': args.token_summary_top_k,
        'doc_count': len(bundle.dense_documents_raw.ids),
        'query_count': len(bundle.dense_queries_raw.ids),
        'doc_ids_sha256': hashlib.sha256(doc_ids.tobytes()).hexdigest(),
        'correction_window': args.correction_window,
    }
    cache_path = feature_cache_path(
        cache_dir=args.feature_cache_dir,
        dataset=bundle.name,
        fingerprint=fingerprint,
    )
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(data['metadata'].item()))
        if metadata != fingerprint:
            raise ValueError(f'{cache_path}: feature cache metadata mismatch')
        doc_features = np.asarray(data['doc_features'], dtype=np.float32)
        query_features = np.asarray(data['query_features'], dtype=np.float32)
        cached_doc_ids = np.asarray(data['doc_ids'], dtype=np.int64)
        if not np.array_equal(doc_ids, cached_doc_ids):
            raise ValueError(f'{cache_path}: cached doc ids changed')
        base.log(f'm230 feature cache hit {bundle.name}: {cache_path}')
    else:
        args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        doc_texts = [bundle.dense_documents_raw.texts[int(doc)] for doc in doc_ids]
        query_texts = list(bundle.dense_queries_raw.texts)
        base.log(
            f'm230 feature cache miss {bundle.name}: docs={len(doc_texts)} '
            f'queries={len(query_texts)}'
        )
        taps = parse_str_csv(args.feature_taps)
        doc_features = encode_features(
            encoder,
            texts=doc_texts,
            role='document',
            batch_size=args.feature_batch_size,
            taps=taps,
            sparse_summary_top_k=args.sparse_summary_top_k,
            token_summary_top_k=args.token_summary_top_k,
        )
        query_features = encode_features(
            encoder,
            texts=query_texts,
            role='query',
            batch_size=max(1, min(args.feature_batch_size, 16)),
            taps=taps,
            sparse_summary_top_k=args.sparse_summary_top_k,
            token_summary_top_k=args.token_summary_top_k,
        )
        np.savez_compressed(
            cache_path,
            metadata=np.asarray(json.dumps(fingerprint, sort_keys=True)),
            doc_ids=doc_ids,
            doc_features=doc_features,
            query_features=query_features,
        )
        base.log(f'm230 wrote feature cache {bundle.name}: {cache_path}')
    doc_lookup = {int(doc): index for index, doc in enumerate(doc_ids)}
    window_indices: list[np.ndarray] = []
    for window in bundle.windows:
        window_indices.append(
            np.asarray([doc_lookup[int(doc)] for doc in window], dtype=np.int64),
        )
    return FeatureBundle(
        bundle=bundle,
        doc_ids=doc_ids,
        doc_features=doc_features,
        query_features=query_features,
        window_feature_indices=window_indices,
    )


def encode_head_matrix(
    *,
    model: torch.nn.Module,
    matrix: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(matrix.shape[0], start + batch_size)
            values = torch.from_numpy(matrix[start:stop]).to(device)
            encoded = model(values).detach().cpu().numpy()
            outputs.append(encoded.astype(np.float32, copy=False))
    return np.vstack(outputs).astype(np.float32, copy=False)


def train_dual_readout(
    *,
    feature_bundles: Sequence[FeatureBundle],
    samples: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    score_targets_by_dataset: dict[str, list[np.ndarray]],
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    head_kind: str,
    feature_taps: Sequence[str],
    sparse_summary_top_k: int,
    token_summary_top_k: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    score_geometry_weight: float,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, float]]]:
    if not samples:
        raise ValueError('no training samples after filtering')
    layout: FeatureLayout | None = None
    if head_kind == 'token_adapter':
        layout = infer_feature_layout(
            input_dim=input_dim,
            taps=feature_taps,
            sparse_summary_top_k=sparse_summary_top_k,
            token_summary_top_k=token_summary_top_k,
        )
        doc_head = TokenSummaryAdapterReadout(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            layout=layout,
        ).to(device)
        query_head = TokenSummaryAdapterReadout(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            layout=layout,
        ).to(device)
    elif head_kind == 'mlp':
        doc_head = MiniDenseReadout(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(device)
        query_head = MiniDenseReadout(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(device)
    else:
        raise ValueError(f'unknown head kind: {head_kind}')
    optimizer = torch.optim.AdamW(
        list(doc_head.parameters()) + list(query_head.parameters()),
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
        geometry_losses: list[float] = []
        magnitude_losses: list[float] = []
        for bundle_index, row, positives, negatives in mutable_samples:
            features = feature_bundles[bundle_index]
            doc_inputs = torch.from_numpy(
                features.doc_features[features.window_feature_indices[row]],
            ).to(device)
            query_inputs = torch.from_numpy(
                features.query_features[row:row + 1],
            ).to(device)
            doc_values = doc_head(doc_inputs)
            query_values = query_head(query_inputs)[0]
            scores = doc_values @ query_values

            pos_index = torch.from_numpy(positives).to(device)
            neg_index = torch.from_numpy(negatives).to(device)
            pair_loss = F.softplus(
                -(scores[pos_index][:, None] - scores[neg_index][None, :]),
            ).mean()

            target_scores = torch.from_numpy(
                np.asarray(
                    score_targets_by_dataset[features.bundle.name][row],
                    dtype=np.float32,
                ),
            ).to(device)
            geometry_loss = m216.normalized_mse(scores, target_scores)
            magnitude_loss = 0.001 * (
                doc_values.pow(2).mean() + query_values.pow(2).mean()
            )
            loss = pair_loss + score_geometry_weight * geometry_loss + magnitude_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(doc_head.parameters()) + list(query_head.parameters()),
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pair_losses.append(float(pair_loss.detach().cpu()))
            geometry_losses.append(float(geometry_loss.detach().cpu()))
            magnitude_losses.append(float(magnitude_loss.detach().cpu()))
        history.append(
            {
                'epoch': float(epoch + 1),
                'loss': statistics.fmean(losses),
                'pair_loss': statistics.fmean(pair_losses),
                'score_geometry_loss': statistics.fmean(geometry_losses),
                'magnitude_loss': statistics.fmean(magnitude_losses),
            },
        )
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 2 == 0:
            base.log(
                f'm230 dual_readout epoch={epoch + 1}/{epochs} '
                f'loss={history[-1]["loss"]:.6f} '
                f'pair={history[-1]["pair_loss"]:.6f} '
                f'geom={history[-1]["score_geometry_loss"]:.6f}'
            )
    return doc_head.cpu(), query_head.cpu(), history


def compact_rows_from_feature_heads(
    *,
    feature_bundles: Sequence[FeatureBundle],
    doc_head: torch.nn.Module,
    query_head: torch.nn.Module,
    device: str,
    batch_size: int,
    correction_window: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, Any]]]:
    rows_by_dataset: dict[str, list[np.ndarray]] = {}
    stats_by_dataset: dict[str, dict[str, Any]] = {}
    for features in feature_bundles:
        bundle = features.bundle
        selected_doc_values = encode_head_matrix(
            model=doc_head,
            matrix=features.doc_features,
            device=device,
            batch_size=batch_size,
        )
        query_values = encode_head_matrix(
            model=query_head,
            matrix=features.query_features,
            device=device,
            batch_size=batch_size,
        )
        doc_values = np.zeros(
            (len(bundle.dense_documents_raw.ids), selected_doc_values.shape[1]),
            dtype=np.float32,
        )
        doc_values[features.doc_ids] = selected_doc_values
        rows, stats = m211.compact_correction_rows(
            bundle=bundle,
            doc_values=doc_values,
            query_values=query_values,
            mode='int4_row',
            correction_window=correction_window,
        )
        rows_by_dataset[bundle.name] = rows
        stats_by_dataset[bundle.name] = {
            **stats,
            'selected_documents_encoded': int(len(features.doc_ids)),
            'full_document_rows': int(len(bundle.dense_documents_raw.ids)),
        }
    return rows_by_dataset, stats_by_dataset


def model_stats(model: torch.nn.Module) -> dict[str, int]:
    return {
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def feature_summary(features: Sequence[FeatureBundle]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for item in features:
        rows[item.bundle.name] = {
            'selected_documents': int(len(item.doc_ids)),
            'queries': int(item.query_features.shape[0]),
            'hidden_dim': int(item.doc_features.shape[1]),
            'doc_feature_mb': (
                item.doc_features.nbytes / 1024.0 / 1024.0
            ),
            'query_feature_mb': (
                item.query_features.nbytes / 1024.0 / 1024.0
            ),
        }
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    device = m201.actual_device(args.device)
    if device != 'cuda':
        raise ValueError('M230A expects a CUDA device')
    if parse_str_csv(args.quantization) != ('int4_row',):
        raise ValueError('M230A currently expects --quantization int4_row')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_names = m202.parse_csv(args.train_datasets)
    eval_names = m202.parse_csv(args.eval_datasets)
    fixed_weights = m213.parse_float_csv(args.fixed_weights)
    router_policies = parse_str_csv(args.router_policies)
    dense_reference = m213.load_dense_reference(args.reference_summary)

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
    teacher_metadata, teacher_score_sets = load_teacher_scores(
        args=args,
        bundles=all_bundles,
        basis=basis,
        device=device,
    )

    pair_scores = m226.ensemble_scores(
        policy='mean_z',
        bundles=all_bundles,
        teacher_scores=teacher_score_sets,
        rrf_k=args.rrf_k,
    )
    score_scores = pair_scores
    pair_masks = m223.all_valid_masks(pair_scores)
    teacher_rows_by_dataset = m223.candidate_rows_from_window_scores(
        bundles=eval_bundles,
        scores_by_dataset=score_scores,
    )

    encoder, ng71 = load_ng80_encoder(args, device)
    sae_hash_before = ng71.parameter_hash(encoder.model)
    feature_by_dataset = {
        bundle.name: build_feature_bundle(
            bundle=bundle,
            encoder=encoder,
            args=args,
            sae_hash=sae_hash_before,
        )
        for bundle in all_bundles
    }
    sae_hash_after_features = ng71.parameter_hash(encoder.model)
    if sae_hash_before != sae_hash_after_features:
        raise ValueError('SAE encoder mutated during hidden feature extraction')

    train_features = [feature_by_dataset[bundle.name] for bundle in train_bundles]
    eval_features = [feature_by_dataset[bundle.name] for bundle in eval_bundles]
    input_dim = int(next(iter(feature_by_dataset.values())).doc_features.shape[1])
    samples, sample_counts = m218.filtered_pairwise_samples(
        bundles=train_bundles,
        scores_by_dataset=pair_scores,
        masks_by_dataset=pair_masks,
        positive_k=args.positive_k,
        negative_start_rank=args.negative_start_rank,
        negative_k=args.negative_k,
        min_positive_k=args.min_positive_k,
        min_negative_k=args.min_negative_k,
    )
    base.log(f'm230 samples={len(samples)} counts={sample_counts}')

    doc_head, query_head, history = train_dual_readout(
        feature_bundles=train_features,
        samples=samples,
        score_targets_by_dataset=score_scores,
        input_dim=input_dim,
        output_dim=args.dimension,
        hidden_dim=args.head_hidden_dim,
        depth=args.head_depth,
        head_kind=args.head_kind,
        feature_taps=parse_str_csv(args.feature_taps),
        sparse_summary_top_k=args.sparse_summary_top_k,
        token_summary_top_k=args.token_summary_top_k,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        score_geometry_weight=args.score_geometry_weight,
        seed=args.seed,
        device=device,
    )
    sae_hash_after_training = ng71.parameter_hash(encoder.model)
    if sae_hash_before != sae_hash_after_training:
        raise ValueError('SAE encoder mutated during mini-dense training')

    doc_head.to(device)
    query_head.to(device)
    correction_by_dataset, compact_stats_by_dataset = compact_rows_from_feature_heads(
        feature_bundles=eval_features,
        doc_head=doc_head,
        query_head=query_head,
        device=device,
        batch_size=args.head_batch_size,
        correction_window=args.correction_window,
    )

    variants: list[dict[str, Any]] = []
    m226.evaluate_student(
        variants=variants,
        objective_policy='frozen_sae_trunk_meanz_score',
        pair_policy='mean_z',
        score_policy='mean_z',
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
    for variant in variants:
        variant['source'] = (
            f'{args.run_label.lower()}_frozen_sae_trunk_student_32d_int4'
        )
        variant['doc_head_stats'] = model_stats(doc_head)
        variant['query_head_stats'] = model_stats(query_head)
        variant['sae_trunk_frozen'] = True

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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.output_dir / f'{args.run_label.lower()}_frozen_sae_dual_head.pt'
    )
    torch.save(
        {
            'schema': SCHEMA,
            'dimension': args.dimension,
            'sae_checkpoint': str(args.sae_checkpoint),
            'sae_parameter_hash': sae_hash_before,
            'input_dim': input_dim,
            'head_kind': args.head_kind,
            'head_hidden_dim': args.head_hidden_dim,
            'head_depth': args.head_depth,
            'doc_head': doc_head.cpu().state_dict(),
            'query_head': query_head.cpu().state_dict(),
            'teacher_checkpoints': teacher_metadata,
            'policy': vars(args),
            'training_history': history,
        },
        checkpoint_path,
    )

    payload = {
        'schema': SCHEMA,
        'run_label': args.run_label,
        'train_datasets': list(train_names),
        'eval_datasets': list(eval_names),
        'qrels_usage': (
            'training uses fixed teacher checkpoint artifacts as score geometry; '
            'qrels are otherwise used only for final metrics'
        ),
        'training_scope': (
            'frozen SAE encoder dual-output probe; SAE posting head is frozen '
            f'and only 32d mini-dense {args.head_kind} heads are trained'
        ),
        'policy': {
            'dimension': args.dimension,
            'quantization': ['int4_row'],
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'score_geometry_weight': args.score_geometry_weight,
            'positive_k': args.positive_k,
            'negative_start_rank': args.negative_start_rank,
            'negative_k': args.negative_k,
            'fixed_weights': list(fixed_weights),
            'router_policies': list(router_policies),
            'head_kind': args.head_kind,
            'head_hidden_dim': args.head_hidden_dim,
            'head_depth': args.head_depth,
            'protected_head': args.protected_head,
            'correction_window': args.correction_window,
            'overlap_k': args.overlap_k,
            'joint_threshold': args.joint_threshold,
            'device': device,
            'teacher_source': args.teacher_source,
            'feature_taps': list(parse_str_csv(args.feature_taps)),
            'sparse_summary_top_k': args.sparse_summary_top_k,
            'token_summary_top_k': args.token_summary_top_k,
        },
        'sae_encoder': {
            'research_root': str(args.ng80_research_root),
            'source_dir': str(args.ng80_source_dir),
            'config': str(args.ng80_config),
            'checkpoint': str(args.sae_checkpoint),
            'parameter_hash_before': sae_hash_before,
            'parameter_hash_after_features': sae_hash_after_features,
            'parameter_hash_after_training': sae_hash_after_training,
            'posting_invariance_verified': (
                sae_hash_before
                == sae_hash_after_features
                == sae_hash_after_training
            ),
        },
        'teacher_checkpoints': teacher_metadata,
        'feature_summary': feature_summary(list(feature_by_dataset.values())),
        'sample_counts': sample_counts,
        'training': {
            'samples': len(samples),
            'history': history,
            'checkpoint': str(checkpoint_path),
        },
        'variants': variants,
        'elapsed_seconds': time.perf_counter() - started,
    }
    m217.write_json(args.output_dir / 'summary.json', payload)
    (args.output_dir / 'summary.md').write_text(
        markdown_report(payload),
        encoding='utf-8',
    )
    return payload


def markdown_report(payload: dict[str, Any]) -> str:
    best = payload['variants'][0] if payload['variants'] else None
    lines = [
        f"# {payload.get('run_label', 'M230')} frozen SAE dual-output encoder probe",
        '',
        f"- Scope: `{payload['training_scope']}`",
        f"- SAE checkpoint: `{payload['sae_encoder']['checkpoint']}`",
        f"- Posting invariance verified: "
        f"`{payload['sae_encoder']['posting_invariance_verified']}`",
        f"- Training samples: `{payload['training']['samples']}`",
        '',
    ]
    if best is not None:
        macro = best['macro_delta_vs_pplx_dense']
        metrics = best['macro_metrics']
        unsafe = ', '.join(best['unsafe_rows_vs_dense']) or 'none'
        lines.extend(
            [
                '## Best Variant',
                '',
                f"- Router: `{best['router_policy']}`",
                f"- Weight: `{best['fixed_weight']}`",
                f"- MAP: {metrics['map@100']:.6f}",
                f"- Recall: {metrics['recall@100']:.6f}",
                f"- dMAP vs dense: {macro['map@100']:+.6f}",
                f"- dRecall vs dense: {macro['recall@100']:+.6f}",
                f"- Dense-unsafe rows: {unsafe}",
                '',
            ],
        )
    lines.extend(
        [
            '## Variants',
            '',
            '| Rank | Router | Weight | MAP | Recall | dMAP vs dense | '
            'dRecall vs dense | Unsafe dense |',
            '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |',
        ],
    )
    for rank, variant in enumerate(payload['variants'], 1):
        macro = variant['macro_delta_vs_pplx_dense']
        metrics = variant['macro_metrics']
        unsafe = ', '.join(variant['unsafe_rows_vs_dense']) or 'none'
        lines.append(
            f"| {rank} | `{variant['router_policy']}` | "
            f"{variant['fixed_weight']:.3g} | "
            f"{metrics['map@100']:.6f} | {metrics['recall@100']:.6f} | "
            f"{macro['map@100']:+.6f} | {macro['recall@100']:+.6f} | "
            f"{unsafe} |"
        )
    lines.extend(
        [
            '',
            '## Interpretation',
            '',
            'M230A is a readout feasibility test. A positive result means the '
            'SAE encoder already carries enough information to host the '
            'mini-dense head without mutating sparse postings. A weak result '
            'means the next route must be shared-trunk multitask training, '
            'with explicit posting drift guards.',
            '',
        ],
    )
    return '\n'.join(lines)


def main() -> None:
    payload = run(parse_args())
    best = payload['variants'][0] if payload['variants'] else None
    if best is None:
        base.log('m230 complete without variants')
        return
    macro = best['macro_delta_vs_pplx_dense']
    base.log(
        'm230 complete best '
        f'router={best["router_policy"]} weight={best["fixed_weight"]} '
        f'dMAP={macro["map@100"]:+.6f} '
        f'dRecall={macro["recall@100"]:+.6f} '
        f'unsafe={best["unsafe_rows_vs_dense"]}'
    )


if __name__ == '__main__':
    main()
