#!/usr/bin/env python3
"""Compile natural-language text into frozen P2 b1.125 query postings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evoke_plain_query import plain_text_to_raw_terms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text')
    inputs.add_argument('--input-jsonl', type=Path)
    encoder = parser.add_mutually_exclusive_group(required=True)
    encoder.add_argument('--model-path', type=Path)
    encoder.add_argument('--raw-semantic-json', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summary-output', type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def read_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.text is not None:
        return [{'id': '', 'text': args.text}]
    assert args.input_jsonl is not None
    rows: list[dict[str, str]] = []
    with args.input_jsonl.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    f'{args.input_jsonl}:{line_no}: expected object',
                )
            row_id = value.get('id')
            text = value.get('text')
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(
                    f'{args.input_jsonl}:{line_no}: invalid id',
                )
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f'{args.input_jsonl}:{line_no}: invalid text',
                )
            rows.append({'id': row_id, 'text': text})
    if not rows:
        raise ValueError(f'{args.input_jsonl}: no query rows')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'{args.input_jsonl}: duplicate query ids')
    return rows


def load_contract(checkout: Path) -> dict[str, Any]:
    manifest = read_json(checkout / 'manifest.json')
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
    ):
        raise ValueError('checkout is not a current Evoke model contract')
    supported_checkout = (
        manifest.get('model_format') == 'offline_p2_m1934_compiler'
        or (
            manifest.get('model_format') == 'onnx'
            and manifest.get('runtime_abi')
            == 'evoke_p2_unified_text_atoms_v2'
        )
    )
    if not supported_checkout:
        raise ValueError('checkout is not a P2 M1934 compiler')
    scoring = read_json(checkout / 'calibration/scoring_profile.json')
    atom_space = read_json(checkout / 'sae/atom_space.json')
    vocabulary = json.loads(
        (checkout / 'lexical/vocabulary.json').read_text(encoding='utf-8'),
    )
    if not isinstance(vocabulary, list):
        raise ValueError('invalid P2 lexical vocabulary')
    stats_config = scoring.get('query_compiler', {}).get(
        'corpus_statistics',
    )
    if not isinstance(stats_config, dict):
        raise ValueError('checkout has no P2 query calibration statistics')
    stats_path = checkout / stats_config['artifact']
    with np.load(stats_path) as stats:
        rms = stats['rms'].astype(np.float32, copy=True)
        document_count = int(stats['document_count'])
        total_dims = int(stats['total_dims'])
    if total_dims != int(atom_space['latent_dims']) or rms.shape != (
        total_dims,
    ):
        raise ValueError('P2 query calibration dimensions do not match')
    if document_count != int(stats_config['document_count']):
        raise ValueError('P2 query calibration document count mismatch')
    if not np.isfinite(rms).all() or np.any(rms < 0.0):
        raise ValueError('P2 query calibration contains invalid RMS values')
    return {
        'atom_space': atom_space,
        'rms': rms,
        'scoring': scoring,
        'vocabulary': vocabulary,
    }


def validate_semantic_atoms(
    atom_ids: Sequence[int],
    impacts: Sequence[float],
    *,
    semantic_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(atom_ids, dtype=np.int64)
    values = np.asarray(impacts, dtype=np.float32)
    if ids.ndim != 1 or values.ndim != 1 or ids.shape != values.shape:
        raise ValueError('semantic atom payload shape mismatch')
    if ids.size == 0 or ids.size > 50:
        raise ValueError('P2 semantic query must contain 1..50 atoms')
    if np.any(ids < 0) or np.any(ids >= semantic_dims):
        raise ValueError('semantic atom id outside Granite vocabulary')
    if len(set(int(value) for value in ids)) != ids.size:
        raise ValueError('semantic atom ids must be unique')
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError('semantic impacts must be finite and positive')
    order = np.argsort(ids)
    return ids[order], values[order]


def read_raw_semantic(
    path: Path,
    *,
    semantic_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = read_json(path)
    return validate_semantic_atoms(
        payload['atom_ids'],
        payload['atom_impacts'],
        semantic_dims=semantic_dims,
    )


def load_model(model_path: Path, device_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, device


def encode_semantic(
    text: str,
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    active_dims: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode():
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**inputs).logits
        mask = inputs['attention_mask'].unsqueeze(-1).to(logits.dtype)
        pooled = logits.mul(mask).relu_().log1p_().amax(dim=1)
        values, ids = torch.topk(pooled, k=active_dims, dim=1)
        values = values[0].float().cpu().numpy()
        ids = ids[0].cpu().numpy()
    keep = values > 0.0
    return validate_semantic_atoms(
        ids[keep],
        values[keep],
        semantic_dims=pooled.shape[1],
    )


def compile_query(
    text: str,
    *,
    contract: dict[str, Any],
    semantic_ids: np.ndarray,
    semantic_impacts: np.ndarray,
) -> dict[str, Any]:
    scoring = contract['scoring']
    query_config = scoring['query_compiler']
    semantic_config = query_config['semantic_encoder']
    atom_space = contract['atom_space']
    lexical_dims = int(atom_space['lexical']['end_exclusive'])
    semantic_dims = int(atom_space['semantic']['end_exclusive']) - lexical_dims
    semantic_ids, semantic_impacts = validate_semantic_atoms(
        semantic_ids,
        semantic_impacts,
        semantic_dims=semantic_dims,
    )

    vocabulary = contract['vocabulary']
    token_to_id = {
        token: index
        for index, token in enumerate(vocabulary)
    }
    term_counts = Counter(plain_text_to_raw_terms(text).split())
    lexical_pairs = sorted(
        (token_to_id[token], float(frequency))
        for token, frequency in term_counts.items()
        if token in token_to_id
    )

    calibrated_semantic = (
        np.power(
            semantic_impacts,
            float(semantic_config['gamma']),
            dtype=np.float32,
        )
        * np.float32(semantic_config['score_scale'])
    ).astype(np.float32, copy=False)
    rms = contract['rms']
    lexical_proxy = sum(
        impact * float(rms[atom_id])
        for atom_id, impact in lexical_pairs
    )
    semantic_proxy = float(
        np.dot(
            calibrated_semantic.astype(np.float64),
            rms[lexical_dims + semantic_ids].astype(np.float64),
        ),
    )
    global_scale = float(query_config['global_scale'])
    if lexical_proxy > 0.0 and semantic_proxy > 0.0:
        raw_scale = lexical_proxy / semantic_proxy
    else:
        raw_scale = global_scale
    query_scale = float(
        np.clip(
            raw_scale * float(query_config['multiplier']),
            global_scale * float(query_config['clip_min']),
            global_scale * float(query_config['clip_max']),
        ),
    )
    calibrated_semantic *= np.float32(query_scale)

    atom_ids = [atom_id for atom_id, _impact in lexical_pairs]
    impacts = [impact for _atom_id, impact in lexical_pairs]
    atom_ids.extend(int(lexical_dims + atom_id) for atom_id in semantic_ids)
    impacts.extend(float(value) for value in calibrated_semantic)
    order = np.argsort(np.asarray(atom_ids, dtype=np.int64))
    return {
        'atom_ids': [atom_ids[int(index)] for index in order],
        'atom_impacts': [impacts[int(index)] for index in order],
        'compiler': {
            'lexical_atoms': len(lexical_pairs),
            'lexical_proxy': lexical_proxy,
            'query_scale': query_scale,
            'semantic_atoms': int(semantic_ids.size),
            'semantic_proxy': semantic_proxy,
        },
        'product': 'P2',
        'route': 'M1934-b1.125-rms_m4',
        'text': text,
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError('warmup must be non-negative and repeat positive')
    contract = load_contract(args.checkout)
    semantic_config = contract['scoring']['query_compiler'][
        'semantic_encoder'
    ]
    semantic_dims = int(semantic_config['vocabulary_size'])
    inputs = read_inputs(args)
    if len(inputs) > 1 and args.raw_semantic_json is not None:
        raise ValueError('raw semantic input only supports one query')
    if len(inputs) > 1 and args.output is None:
        raise ValueError('--output is required for batch query compilation')
    latencies: list[float] = []
    compiled_rows: list[dict[str, Any]] = []
    if args.raw_semantic_json is not None:
        semantic_ids, semantic_impacts = read_raw_semantic(
            args.raw_semantic_json,
            semantic_dims=semantic_dims,
        )
        compiled_rows.append(
            compile_query(
                inputs[0]['text'],
                contract=contract,
                semantic_ids=semantic_ids,
                semantic_impacts=semantic_impacts,
            ),
        )
    else:
        assert args.model_path is not None
        tokenizer, model, device = load_model(args.model_path, args.device)
        for _iteration in range(args.warmup):
            encode_semantic(
                inputs[0]['text'],
                tokenizer=tokenizer,
                model=model,
                device=device,
                active_dims=int(semantic_config['active_dims']),
                max_length=int(semantic_config['max_length']),
            )
        for query in inputs:
            semantic_ids = np.empty(0, dtype=np.int64)
            semantic_impacts = np.empty(0, dtype=np.float32)
            query_latencies: list[float] = []
            for _iteration in range(args.repeat):
                started = time.perf_counter()
                semantic_ids, semantic_impacts = encode_semantic(
                    query['text'],
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    active_dims=int(semantic_config['active_dims']),
                    max_length=int(semantic_config['max_length']),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(elapsed_ms)
                query_latencies.append(elapsed_ms)
            compiled = compile_query(
                query['text'],
                contract=contract,
                semantic_ids=semantic_ids,
                semantic_impacts=semantic_impacts,
            )
            compiled['encoder_latency_ms'] = {
                'mean': statistics.fmean(query_latencies),
                'samples': len(query_latencies),
            }
            compiled_rows.append(compiled)

    for query, compiled in zip(inputs, compiled_rows, strict=True):
        if query['id']:
            compiled['id'] = query['id']
    summary = {
        'product': 'P2',
        'query_count': len(compiled_rows),
        'route': 'M1934-b1.125-rms_m4',
    }
    if latencies:
        summary['encoder_latency_ms'] = {
            'maximum': max(latencies),
            'mean': statistics.fmean(latencies),
            'minimum': min(latencies),
            'p95': float(np.quantile(latencies, 0.95)),
            'samples': len(latencies),
        }
    if len(compiled_rows) == 1 and args.input_jsonl is None:
        payload = compiled_rows[0]
        if 'encoder_latency_ms' in summary:
            payload['encoder_latency_ms'] = summary['encoder_latency_ms']
        serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized, end='')
    else:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            ''.join(
                json.dumps(row, sort_keys=True) + '\n'
                for row in compiled_rows
            ),
            encoding='utf-8',
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
