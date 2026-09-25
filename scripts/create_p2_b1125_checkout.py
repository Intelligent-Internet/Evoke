#!/usr/bin/env python3
"""Package the frozen M1934 b1.125 route as an Evoke P2 checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_REVISION = 'ad82b1fd09541c998c8d45045d601c51fdb8a9b7'
DEFAULT_SEMANTIC_DIMS = 50_265
DEFAULT_BUDGET_RATIO = 1.125
DEFAULT_GLOBAL_SCALE = 6.281606583836263
DEFAULT_SCALE_MULTIPLIER = 4.0
DEFAULT_CLIP_MIN = 0.5
DEFAULT_CLIP_MAX = 4.0
DEFAULT_SCORING_PROFILE = 'p2_unified_sparse_dot_v1'
M1914_DOCUMENT_ACTIVE_DIMS = 192
M1914_DOCUMENT_SCORE_SCALE = 1.0
M1914_GAMMA_DOCUMENT = 0.5627960562705994
M1914_GAMMA_QUERY = 1.8518644571304321
M1914_MAX_LENGTH = 512
M1914_QUERY_ACTIVE_DIMS = 50
M1914_SCORE_SCALE = 0.6963680386543274


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--lexical-vocabulary', required=True, type=Path)
    parser.add_argument('--query-atoms', required=True, type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--semantic-runtime-dir', type=Path)
    parser.add_argument('--query-calibration-runtime', type=Path)
    parser.add_argument('--query-calibration-statistics', type=Path)
    parser.add_argument('--model-id', default='')
    parser.add_argument('--atom-space', default='')
    parser.add_argument(
        '--scoring-profile',
        default=DEFAULT_SCORING_PROFILE,
    )
    parser.add_argument('--semantic-dims', type=int, default=DEFAULT_SEMANTIC_DIMS)
    parser.add_argument('--budget-ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--global-scale', type=float, default=DEFAULT_GLOBAL_SCALE)
    parser.add_argument(
        '--scale-multiplier',
        type=float,
        default=DEFAULT_SCALE_MULTIPLIER,
    )
    parser.add_argument('--clip-min', type=float, default=DEFAULT_CLIP_MIN)
    parser.add_argument('--clip-max', type=float, default=DEFAULT_CLIP_MAX)
    parser.add_argument('--model-revision', default=DEFAULT_MODEL_REVISION)
    parser.add_argument('--impact-head-size', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> dict[str, Any]:
    return {
        'path': str(path.resolve()),
        'sha256': sha256_file(path),
        'size': path.stat().st_size,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_no}: expected object row')
            yield line_no, value


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


def validate_vocabulary(path: Path) -> list[str]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError('lexical vocabulary must be a non-empty JSON list')
    if not all(isinstance(token, str) and token for token in value):
        raise ValueError('lexical vocabulary contains a non-string token')
    if len(set(value)) != len(value):
        raise ValueError('lexical vocabulary contains duplicate tokens')
    return value


def validate_query_atoms(path: Path, total_dims: int) -> dict[str, int]:
    rows = 0
    postings = 0
    maximum_atom = -1
    for line_no, row in read_jsonl(path):
        atom_ids = row.get('atom_ids')
        impacts = row.get('atom_impacts')
        if not isinstance(atom_ids, list) or not atom_ids:
            raise ValueError(f'{path}:{line_no}: atom_ids must be non-empty')
        if not isinstance(impacts, list) or len(impacts) != len(atom_ids):
            raise ValueError(f'{path}:{line_no}: atom payload length mismatch')
        normalized_ids = [int(atom_id) for atom_id in atom_ids]
        if normalized_ids != sorted(set(normalized_ids)):
            raise ValueError(f'{path}:{line_no}: atom_ids must be sorted/unique')
        if normalized_ids[0] < 0 or normalized_ids[-1] >= total_dims:
            raise ValueError(f'{path}:{line_no}: atom id outside P2 atom space')
        if any(
            not math.isfinite(float(impact)) or float(impact) < 0.0
            for impact in impacts
        ):
            raise ValueError(f'{path}:{line_no}: invalid atom impact')
        rows += 1
        postings += len(normalized_ids)
        maximum_atom = max(maximum_atom, normalized_ids[-1])
    if rows == 0:
        raise ValueError(f'{path}: no query rows')
    return {
        'maximum_atom_id': maximum_atom,
        'postings': postings,
        'rows': rows,
    }


def validate_query_calibration_runtime(
    path: Path,
    total_dims: int,
) -> dict[str, int | float]:
    contents = path.read_bytes()
    header_size = struct.calcsize('<8sIIQ')
    expected_size = header_size + total_dims * 4
    if len(contents) != expected_size:
        raise ValueError(
            'P2 query calibration size does not match the atom space: '
            f'expected={expected_size} actual={len(contents)}',
        )
    magic, version, artifact_dims, document_count = struct.unpack_from(
        '<8sIIQ',
        contents,
    )
    if magic != b'II42P2R1' or version != 1:
        raise ValueError('invalid P2 query calibration header')
    if artifact_dims != total_dims:
        raise ValueError(
            'P2 query calibration dimensions do not match: '
            f'expected={total_dims} actual={artifact_dims}',
        )
    if document_count <= 0:
        raise ValueError('P2 query calibration document count must be positive')
    rms_values = tuple(
        value[0]
        for value in struct.iter_unpack('<f', contents[header_size:])
    )
    if any(not math.isfinite(value) or value < 0.0 for value in rms_values):
        raise ValueError('P2 query calibration contains invalid RMS values')
    return {
        'active_dimensions': sum(value > 0.0 for value in rms_values),
        'document_count': document_count,
        'maximum_rms': max(rms_values, default=0.0),
        'total_dims': artifact_dims,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.strip():
        raise ValueError('--dataset must not be empty')
    for path in (args.lexical_vocabulary, args.query_atoms):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if args.source_manifest is not None:
        if not args.source_manifest.is_file():
            raise FileNotFoundError(args.source_manifest)
    if args.semantic_runtime_dir is not None:
        required = (
            'semantic_document_compiler.onnx',
            'semantic_query_compiler.onnx',
            'semantic_runtime.json',
            'merges.txt',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json',
        )
        missing = [
            name
            for name in required
            if not (args.semantic_runtime_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f'{args.semantic_runtime_dir} is missing {missing}',
            )
        if args.query_calibration_runtime is None:
            raise ValueError(
                '--query-calibration-runtime is required with '
                '--semantic-runtime-dir',
            )
    if (
        args.query_calibration_statistics is not None
        and args.query_calibration_runtime is None
    ):
        raise ValueError(
            '--query-calibration-statistics requires '
            '--query-calibration-runtime',
        )
    for path in (
        args.query_calibration_runtime,
        args.query_calibration_statistics,
    ):
        if path is not None and (not path.is_file() or path.stat().st_size <= 0):
            raise FileNotFoundError(path)
    if args.semantic_dims <= 0:
        raise ValueError('--semantic-dims must be positive')
    if args.budget_ratio <= 0.0:
        raise ValueError('--budget-ratio must be positive')
    if args.global_scale <= 0.0 or args.scale_multiplier <= 0.0:
        raise ValueError('P2 calibration scales must be positive')
    if args.clip_min <= 0.0 or args.clip_max < args.clip_min:
        raise ValueError('invalid P2 calibration clip interval')
    if args.impact_head_size <= 0:
        raise ValueError('--impact-head-size must be positive')

def main() -> int:
    args = parse_args()
    validate_args(args)
    vocabulary = validate_vocabulary(args.lexical_vocabulary)
    lexical_dims = len(vocabulary)
    total_dims = lexical_dims + args.semantic_dims
    query_stats = validate_query_atoms(args.query_atoms, total_dims)
    calibration_stats = None
    if args.query_calibration_runtime is not None:
        calibration_stats = validate_query_calibration_runtime(
            args.query_calibration_runtime,
            total_dims,
        )
    vocabulary_fingerprint = sha256_file(args.lexical_vocabulary)
    dataset_key = safe_name(args.dataset)
    model_id = args.model_id or f'ii42_p2_m1934_b1125_{dataset_key}'
    atom_space = args.atom_space or (
        f'p2_m1934_b1125_{dataset_key}_{vocabulary_fingerprint[:12]}'
    )

    prepare_output(args.output, args.overwrite)
    lexical_dir = args.output / 'lexical'
    encoder_dir = args.output / 'encoder'
    sae_dir = args.output / 'sae'
    calibration_dir = args.output / 'calibration'
    provenance_dir = args.output / 'provenance'
    for path in (
        lexical_dir,
        encoder_dir,
        sae_dir,
        calibration_dir,
        provenance_dir,
    ):
        path.mkdir()

    vocabulary_path = lexical_dir / 'vocabulary.json'
    shutil.copy2(args.lexical_vocabulary, vocabulary_path)
    atom_space_path = sae_dir / 'atom_space.json'
    scoring_path = calibration_dir / 'scoring_profile.json'
    contract_path = provenance_dir / 'p2_contract.json'
    calibration_runtime_path = calibration_dir / 'query_calibration_rms.f32'
    calibration_statistics_path = (
        calibration_dir / 'query_calibration_stats.npz'
    )

    atom_space_payload = {
        'atom_space': atom_space,
        'latent_dims': total_dims,
        'lexical': {
            'end_exclusive': lexical_dims,
            'source': 0,
            'start': 0,
            'vocabulary_sha256': vocabulary_fingerprint,
        },
        'namespace': 'm1934_exact_lexical_plus_granite_sparse_v1',
        'semantic': {
            'end_exclusive': total_dims,
            'model_revision': args.model_revision,
            'source': 1,
            'start': lexical_dims,
        },
        'source_types': {
            '0': 'exact_lexical_bm25_impact',
            '1': 'granite_sparse_semantic_impact',
        },
    }
    scoring_payload = {
        'candidate_k': 1000,
        'dense_runtime_dependency': False,
        'document_compiler': {
            'semantic_encoder': {
                'active_dims': M1914_DOCUMENT_ACTIVE_DIMS,
                'gamma': M1914_GAMMA_DOCUMENT,
                'max_length': M1914_MAX_LENGTH,
                'pooling': 'relu_log1p_masked_max',
                'score_scale': M1914_DOCUMENT_SCORE_SCALE,
            },
            'semantic_budget_ratio_to_lexical': args.budget_ratio,
            'semantic_pruning': 'm1933_balanced_prune_rows',
        },
        'impact_head_size': args.impact_head_size,
        'profile': args.scoring_profile,
        'query_compiler': {
            'clip_max': args.clip_max,
            'clip_min': args.clip_min,
            'global_scale': args.global_scale,
            'lexical_encoder': {
                'impact': 'query_term_frequency',
                'tokenizer': 'ii42_plain_query_v1',
                'unknown_terms': 'drop',
            },
            'mode': 'rms',
            'multiplier': args.scale_multiplier,
            'scale_constraint_order': 'multiply_then_clip',
            'semantic_encoder': {
                'active_dims': M1914_QUERY_ACTIVE_DIMS,
                'gamma': M1914_GAMMA_QUERY,
                'max_length': M1914_MAX_LENGTH,
                'model_family': (
                    'ibm-granite/granite-embedding-30m-sparse'
                ),
                'model_revision': args.model_revision,
                'pooling': 'relu_log1p_masked_max',
                'score_scale': M1914_SCORE_SCALE,
                'term_log_scale': 'identity',
                'vocabulary_size': args.semantic_dims,
            },
        },
        'score': 'sum(query_impact * document_impact)',
        'score_inputs': ['lexical_postings', 'semantic_postings'],
        'tie_break': ['score desc', 'document_id asc'],
        'version': 1,
    }
    contract_payload: dict[str, Any] = {
        'budget_ratio': args.budget_ratio,
        'dataset': args.dataset,
        'model': {
            'family': 'ibm-granite/granite-embedding-30m-sparse',
            'revision': args.model_revision,
        },
        'm1914_transform': {
            'document_active_dims': M1914_DOCUMENT_ACTIVE_DIMS,
            'gamma_document': M1914_GAMMA_DOCUMENT,
            'gamma_query': M1914_GAMMA_QUERY,
            'max_length': M1914_MAX_LENGTH,
            'query_active_dims': M1914_QUERY_ACTIVE_DIMS,
            'score_scale': M1914_SCORE_SCALE,
            'selected_branch': 'global_power',
            'selected_step': 1000,
            'term_log_scale': 'identity',
        },
        'product_name': 'P2',
        'qrels_usage': 'evaluation_only_after_frozen_transform',
        'query_atoms': signature(args.query_atoms),
        'query_stats': query_stats,
        'route': 'M1934-b1.125-rms_m4',
        'selection_surface': 'M1933 selection plus M1934 unseen transfer',
    }
    if args.source_manifest is not None:
        contract_payload['source_manifest'] = signature(args.source_manifest)
    if calibration_stats is not None:
        assert args.query_calibration_runtime is not None
        shutil.copy2(
            args.query_calibration_runtime,
            calibration_runtime_path,
        )
        scoring_payload['query_compiler']['corpus_statistics'] = {
            'active_dimensions': calibration_stats['active_dimensions'],
            'document_count': calibration_stats['document_count'],
            'method': 'column_rms_over_all_documents',
            'runtime_artifact': 'calibration/query_calibration_rms.f32',
        }
        contract_payload['query_calibration_runtime'] = {
            **signature(args.query_calibration_runtime),
            **calibration_stats,
        }
        if args.query_calibration_statistics is not None:
            shutil.copy2(
                args.query_calibration_statistics,
                calibration_statistics_path,
            )
            scoring_payload['query_compiler']['corpus_statistics'][
                'artifact'
            ] = 'calibration/query_calibration_stats.npz'
            contract_payload['query_calibration_statistics'] = signature(
                args.query_calibration_statistics,
            )

    write_json(atom_space_path, atom_space_payload)
    write_json(scoring_path, scoring_payload)
    write_json(contract_path, contract_payload)

    artifacts = {
        'atom_space': {
            'path': 'sae/atom_space.json',
            'sha256': sha256_file(atom_space_path),
        },
        'lexical_vocabulary': {
            'path': 'lexical/vocabulary.json',
            'sha256': sha256_file(vocabulary_path),
        },
        'p2_contract': {
            'path': 'provenance/p2_contract.json',
            'sha256': sha256_file(contract_path),
        },
        'scoring_profile': {
            'path': 'calibration/scoring_profile.json',
            'sha256': sha256_file(scoring_path),
        },
    }
    if calibration_stats is not None:
        artifacts['query_calibration_runtime'] = {
            'path': 'calibration/query_calibration_rms.f32',
            'sha256': sha256_file(calibration_runtime_path),
        }
        if args.query_calibration_statistics is not None:
            artifacts['query_calibration_statistics'] = {
                'path': 'calibration/query_calibration_stats.npz',
                'sha256': sha256_file(calibration_statistics_path),
            }
    if args.semantic_runtime_dir is not None:
        runtime_artifacts = {
            'document_encoder': 'semantic_document_compiler.onnx',
            'query_encoder': 'semantic_query_compiler.onnx',
            'semantic_runtime': 'semantic_runtime.json',
            'tokenizer': 'tokenizer.json',
            'tokenizer_config': 'tokenizer_config.json',
            'tokenizer_merges': 'merges.txt',
            'tokenizer_vocabulary': 'vocab.json',
        }
        for artifact_name, filename in runtime_artifacts.items():
            target = encoder_dir / filename
            shutil.copy2(args.semantic_runtime_dir / filename, target)
            artifacts[artifact_name] = {
                'path': f'encoder/{filename}',
                'sha256': sha256_file(target),
            }
        semantic_runtime = read_json(
            encoder_dir / 'semantic_runtime.json',
        )
        if semantic_runtime.get('abi') != 'evoke_p2_semantic_onnx_v2':
            raise ValueError('unsupported P2 semantic runtime artifact')
        validated_provider = semantic_runtime.get('validation', {}).get(
            'provider',
            'CPUExecutionProvider',
        )
        supported_validation_providers = {
            'CPUExecutionProvider',
            'CUDAExecutionProvider',
            'CoreMLExecutionProvider',
        }
        if validated_provider not in supported_validation_providers:
            raise ValueError(
                'unsupported P2 semantic runtime validation provider: '
                f'{validated_provider}',
            )
    has_runtime = args.semantic_runtime_dir is not None
    manifest = {
        'api_version': 'evoke_model_v1',
        'artifacts': artifacts,
        'doc_active': 0,
        'encoder_type': 'granite_sparse_exact_lexical_unified_postings',
        'index_compatibility': {
            'atom_space': atom_space,
            'scoring': args.scoring_profile,
        },
        'latent_dims': total_dims,
        'model_format': 'onnx' if has_runtime else (
            'offline_p2_m1934_compiler'
        ),
        'model_id': model_id,
        'query_active': 0,
        'runtime': 'onnxruntime' if has_runtime else (
            'offline_precomputed_atoms'
        ),
        'schema_version': 1,
    }
    if has_runtime:
        manifest.update({
            'runtime_abi': 'evoke_p2_unified_text_atoms_v2',
            'runtime_io': {
                'attention_mask': 'attention_mask',
                'input_ids': 'input_ids',
                'semantic_ids': 'semantic_ids',
                'semantic_weights': 'semantic_weights',
            },
            'runtime_output': {
                'atom_id_dtype': 'int64',
                'atom_weight_dtype': 'float32',
                'document_semantic_budget_ratio': args.budget_ratio,
                'max_atoms': 512,
                'document_semantic_max_atoms': M1914_DOCUMENT_ACTIVE_DIMS,
                'lexical_dims': lexical_dims,
                'query_semantic_max_atoms': M1914_QUERY_ACTIVE_DIMS,
                'semantic_dims': args.semantic_dims,
            },
        })
    manifest_path = args.output / 'manifest.json'
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                'atom_space': atom_space,
                'dataset': args.dataset,
                'lexical_dims': lexical_dims,
                'manifest': str(manifest_path),
                'model_id': model_id,
                'path': str(args.output),
                'query_rows': query_stats['rows'],
                'semantic_dims': args.semantic_dims,
                'total_dims': total_dims,
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
