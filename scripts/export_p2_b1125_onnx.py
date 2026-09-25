#!/usr/bin/env python3
"""Export frozen P2 b1.125 query and document compilers to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


MODEL_REVISION = 'ad82b1fd09541c998c8d45045d601c51fdb8a9b7'
QUERY_ACTIVE_DIMS = 50
QUERY_GAMMA = 1.8518644571304321
DOCUMENT_ACTIVE_DIMS = 192
DOCUMENT_GAMMA = 0.5627960562705994
DOCUMENT_SCORE_SCALE = 1.0
QUERY_SCORE_SCALE = 0.6963680386543274
MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--validation-text',
        action='append',
        default=[],
        help='Text used for PyTorch/ONNX parity; may be repeated.',
    )
    parser.add_argument('--skip-validation', action='store_true')
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f'{path} already exists; pass --overwrite to replace it',
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True)


def validate_model_path(path: Path) -> None:
    required = (
        'config.json',
        'model.safetensors',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{path} is missing required artifacts: {missing}',
        )


def build_wrapper(
    model: Any,
    *,
    active_dims: int,
    gamma: float,
    score_scale: float,
) -> Any:
    import torch

    class P2SemanticCompiler(torch.nn.Module):
        def __init__(self, encoder: Any) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
        ) -> tuple[Any, Any]:
            logits = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0].float()
            mask = attention_mask.unsqueeze(-1).to(logits.dtype)
            pooled = torch.log1p(torch.relu(logits * mask)).amax(dim=1)
            weights, ids = torch.topk(
                pooled,
                k=active_dims,
                dim=1,
                largest=True,
                sorted=True,
            )
            weights = torch.pow(weights, gamma) * score_scale
            return ids.to(torch.int64), weights.to(torch.float32)

    return P2SemanticCompiler(model).eval()


def load_model(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def export_model(
    wrapper: Any,
    output: Path,
    *,
    opset: int,
) -> None:
    import torch

    input_ids = torch.tensor([[0, 31414, 232, 2]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        output,
        input_names=['input_ids', 'attention_mask'],
        output_names=['semantic_ids', 'semantic_weights'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'semantic_ids': {0: 'batch'},
            'semantic_weights': {0: 'batch'},
        },
        do_constant_folding=True,
        opset_version=opset,
        dynamo=False,
    )


def validate_export(
    tokenizer: Any,
    wrapper: Any,
    model_path: Path,
    texts: list[str],
    *,
    compiler: str,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            'onnxruntime Python package is required for export validation',
        ) from exc
    import torch

    session = ort.InferenceSession(
        str(model_path),
        providers=['CPUExecutionProvider'],
    )
    cases: list[dict[str, Any]] = []
    exact_positive_ids = True
    max_abs_delta = 0.0
    max_relative_delta = 0.0
    for text in texts:
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(torch.int64)
        attention_mask = encoded['attention_mask'].to(torch.int64)
        with torch.inference_mode():
            expected_ids, expected_weights = wrapper(
                input_ids,
                attention_mask,
            )
        actual_ids, actual_weights = session.run(
            ['semantic_ids', 'semantic_weights'],
            {
                'input_ids': input_ids.numpy(),
                'attention_mask': attention_mask.numpy(),
            },
        )
        expected_ids_np = expected_ids.numpy()
        expected_weights_np = expected_weights.numpy()
        expected_positive = expected_weights_np > 0.0
        actual_positive = actual_weights > 0.0
        expected_positive_ids = expected_ids_np[expected_positive]
        actual_positive_ids = actual_ids[actual_positive]
        ids_equal = bool(
            np.array_equal(expected_positive_ids, actual_positive_ids),
        )
        expected_positive_weights = expected_weights_np[expected_positive]
        actual_positive_weights = actual_weights[actual_positive]
        if expected_positive_weights.shape != actual_positive_weights.shape:
            absolute_delta = float('inf')
            relative_delta = float('inf')
        else:
            absolute_delta = float(
                np.max(
                    np.abs(
                        expected_positive_weights - actual_positive_weights,
                    ),
                    initial=0.0,
                ),
            )
            denominator = np.maximum(
                np.abs(expected_positive_weights),
                1e-8,
            )
            relative_delta = float(
                np.max(
                    np.abs(
                        expected_positive_weights - actual_positive_weights
                    ) / denominator,
                    initial=0.0,
                ),
            )
        exact_positive_ids = exact_positive_ids and ids_equal
        max_abs_delta = max(max_abs_delta, absolute_delta)
        max_relative_delta = max(max_relative_delta, relative_delta)
        cases.append({
            'compiler': compiler,
            'exact_positive_ids': ids_equal,
            'max_absolute_weight_delta': absolute_delta,
            'max_relative_weight_delta': relative_delta,
            'text': text,
            'positive_atom_count': int(expected_positive.sum()),
            'token_count': int(attention_mask.sum().item()),
        })
    return {
        'cases': cases,
        'exact_positive_ids': exact_positive_ids,
        'max_absolute_weight_delta': max_abs_delta,
        'max_relative_weight_delta': max_relative_delta,
        'provider': 'CPUExecutionProvider',
    }


def copy_tokenizer_artifacts(model_path: Path, output: Path) -> None:
    for name in (
        'merges.txt',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
    ):
        shutil.copy2(model_path / name, output / name)


def main() -> int:
    args = parse_args()
    validate_model_path(args.model_path)
    if args.opset < 17:
        raise ValueError('P2 semantic compiler requires ONNX opset >= 17')
    prepare_output(args.output, args.overwrite)
    tokenizer, model = load_model(args.model_path)
    query_wrapper = build_wrapper(
        model,
        active_dims=QUERY_ACTIVE_DIMS,
        gamma=QUERY_GAMMA,
        score_scale=QUERY_SCORE_SCALE,
    )
    document_wrapper = build_wrapper(
        model,
        active_dims=DOCUMENT_ACTIVE_DIMS,
        gamma=DOCUMENT_GAMMA,
        score_scale=DOCUMENT_SCORE_SCALE,
    )
    query_encoder_path = args.output / 'semantic_query_compiler.onnx'
    document_encoder_path = args.output / 'semantic_document_compiler.onnx'
    export_model(query_wrapper, query_encoder_path, opset=args.opset)
    export_model(document_wrapper, document_encoder_path, opset=args.opset)
    copy_tokenizer_artifacts(args.model_path, args.output)

    texts = args.validation_text or [
        'What causes oxidative stress in human cells?',
        'retrieval systems with exact lexical and semantic evidence',
        'COVID-19 vaccine efficacy in older adults',
    ]
    query_validation: dict[str, Any] | None = None
    document_validation: dict[str, Any] | None = None
    if not args.skip_validation:
        query_validation = validate_export(
            tokenizer,
            query_wrapper,
            query_encoder_path,
            texts,
            compiler='query',
        )
        document_validation = validate_export(
            tokenizer,
            document_wrapper,
            document_encoder_path,
            texts,
            compiler='document',
        )
        if not query_validation['exact_positive_ids']:
            raise RuntimeError(
                'ONNX export changed positive P2 semantic Top-50 IDs',
            )
        if not document_validation['exact_positive_ids']:
            raise RuntimeError(
                'ONNX export changed positive P2 semantic Top-192 IDs',
            )
        max_delta = max(
            query_validation['max_absolute_weight_delta'],
            document_validation['max_absolute_weight_delta'],
        )
        if max_delta > 1e-4:
            raise RuntimeError(
                'ONNX export exceeds the semantic weight parity tolerance: '
                f'{max_delta}',
            )

    manifest = {
        'abi': 'evoke_p2_semantic_onnx_v2',
        'artifacts': {
            name: {
                'path': name,
                'sha256': sha256_file(args.output / name),
                'size': (args.output / name).stat().st_size,
            }
            for name in (
                'semantic_query_compiler.onnx',
                'semantic_document_compiler.onnx',
                'merges.txt',
                'special_tokens_map.json',
                'tokenizer.json',
                'tokenizer_config.json',
                'vocab.json',
            )
        },
        'inputs': {
            'attention_mask': 'int64[batch,sequence]',
            'input_ids': 'int64[batch,sequence]',
        },
        'max_length': MAX_LENGTH,
        'model': {
            'family': 'ibm-granite/granite-embedding-30m-sparse',
            'revision': MODEL_REVISION,
        },
        'opset': args.opset,
        'outputs': {
            'semantic_ids': 'int64[batch,active_dims]',
            'semantic_weights': 'float32[batch,active_dims]',
        },
        'document_transform': {
            'active_dims': DOCUMENT_ACTIVE_DIMS,
            'gamma': DOCUMENT_GAMMA,
            'pooling': 'relu_log1p_attention_masked_max',
            'runtime_filter': 'semantic_weight > 0',
            'score_scale': DOCUMENT_SCORE_SCALE,
        },
        'query_transform': {
            'active_dims': QUERY_ACTIVE_DIMS,
            'gamma': QUERY_GAMMA,
            'pooling': 'relu_log1p_attention_masked_max',
            'runtime_filter': 'semantic_weight > 0',
            'score_scale': QUERY_SCORE_SCALE,
        },
        'validation': {
            'document': document_validation,
            'provider': 'CPUExecutionProvider',
            'query': query_validation,
        },
        'version': 2,
    }
    manifest_path = args.output / 'semantic_runtime.json'
    write_json(manifest_path, manifest)
    print(json.dumps({
        'document_encoder': str(document_encoder_path),
        'document_encoder_sha256': sha256_file(document_encoder_path),
        'document_encoder_size': document_encoder_path.stat().st_size,
        'manifest': str(manifest_path),
        'query_encoder': str(query_encoder_path),
        'query_encoder_sha256': sha256_file(query_encoder_path),
        'query_encoder_size': query_encoder_path.stat().st_size,
        'validation': {
            'document': document_validation,
            'query': query_validation,
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
