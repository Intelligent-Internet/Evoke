#!/usr/bin/env python3
"""Create one-variable P2 runtime checkouts for isolated native experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


MODEL_API_VERSION = 'evoke_model_v1'
MODEL_SCHEMA_VERSION = 1
RUNTIME_ABI = 'evoke_p2_unified_text_atoms_v2'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--document-semantic-budget-ratio',
        type=float,
    )
    group.add_argument(
        '--document-semantic-max-atoms',
        type=int,
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(
    root: Path,
    manifest: dict[str, Any],
    artifact_name: str,
) -> Path:
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('source manifest has no artifact map')
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, dict):
        raise ValueError(f'source manifest has no {artifact_name} artifact')
    relative_path = artifact.get('path')
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f'{artifact_name} artifact path is invalid')
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() == output.resolve():
        raise ValueError('source and output must be different directories')
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f'{output} exists; pass --overwrite to replace it',
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    shutil.copytree(source, output, copy_function=shutil.copy2)


def validate_source(source: Path) -> dict[str, Any]:
    manifest_path = source / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if (
        manifest.get('schema_version') != MODEL_SCHEMA_VERSION
        or manifest.get('api_version') != MODEL_API_VERSION
        or manifest.get('runtime_abi') != RUNTIME_ABI
    ):
        raise ValueError(
            'source checkout must use the current Evoke model contract',
        )
    runtime_output = manifest.get('runtime_output')
    if not isinstance(runtime_output, dict):
        raise ValueError('source manifest has no runtime_output contract')
    for field in (
        'document_semantic_budget_ratio',
        'document_semantic_max_atoms',
    ):
        if field not in runtime_output:
            raise ValueError(f'source manifest is missing {field}')
    artifact_path(source, manifest, 'scoring_profile')
    artifact_path(source, manifest, 'p2_contract')
    return manifest


def variant_definition(
    *,
    budget_ratio: float | None,
    max_atoms: int | None,
) -> tuple[str, float | int]:
    if budget_ratio is not None:
        if not math.isfinite(budget_ratio) or budget_ratio <= 0.0:
            raise ValueError('semantic budget ratio must be finite and positive')
        return 'document_semantic_budget_ratio', budget_ratio
    if max_atoms is not None:
        if max_atoms <= 0:
            raise ValueError('document semantic max atoms must be positive')
        return 'document_semantic_max_atoms', max_atoms
    raise ValueError('exactly one runtime variant must be selected')


def create_variant(
    *,
    source: Path,
    output: Path,
    model_id: str,
    budget_ratio: float | None = None,
    max_atoms: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not model_id.strip():
        raise ValueError('model id must not be empty')
    source_manifest = validate_source(source)
    field, value = variant_definition(
        budget_ratio=budget_ratio,
        max_atoms=max_atoms,
    )
    prepare_output(source, output, overwrite)

    manifest_path = output / 'manifest.json'
    manifest = read_json(manifest_path)
    runtime_output = manifest['runtime_output']
    source_value = runtime_output[field]
    if float(source_value) == float(value):
        raise ValueError(f'{field} does not change the source checkout')
    manifest['model_id'] = model_id
    runtime_output[field] = value

    scoring_path = artifact_path(output, manifest, 'scoring_profile')
    scoring = read_json(scoring_path)
    document_compiler = scoring.get('document_compiler')
    if not isinstance(document_compiler, dict):
        raise ValueError('scoring profile has no document compiler')
    if field == 'document_semantic_budget_ratio':
        document_compiler['semantic_budget_ratio_to_lexical'] = value
    else:
        semantic_encoder = document_compiler.get('semantic_encoder')
        if not isinstance(semantic_encoder, dict):
            raise ValueError('scoring profile has no semantic encoder')
        semantic_encoder['active_dims'] = value
    write_json(scoring_path, scoring)

    contract_path = artifact_path(output, manifest, 'p2_contract')
    contract = read_json(contract_path)
    contract['lossy_variant'] = {
        'field': field,
        'source_manifest_sha256': sha256_file(source / 'manifest.json'),
        'source_model_id': source_manifest.get('model_id'),
        'source_value': source_value,
        'value': value,
    }
    if field == 'document_semantic_budget_ratio':
        contract['budget_ratio'] = value
    write_json(contract_path, contract)

    artifacts = manifest['artifacts']
    artifacts['scoring_profile']['sha256'] = sha256_file(scoring_path)
    artifacts['p2_contract']['sha256'] = sha256_file(contract_path)
    write_json(manifest_path, manifest)

    return {
        'field': field,
        'model_id': model_id,
        'output': str(output),
        'source_model_id': source_manifest.get('model_id'),
        'source_value': source_value,
        'value': value,
    }


def main() -> int:
    args = parse_args()
    summary = create_variant(
        source=args.source,
        output=args.output,
        model_id=args.model_id,
        budget_ratio=args.document_semantic_budget_ratio,
        max_atoms=args.document_semantic_max_atoms,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
