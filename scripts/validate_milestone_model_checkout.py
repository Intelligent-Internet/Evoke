#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / 'packaging/milestone-model.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


def safe_artifact_path(checkout: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        raise ValueError(f'unsafe model artifact path: {relative}')
    path = checkout / relative_path
    if path.is_symlink():
        raise ValueError(f'model artifacts must not be symlinks: {relative}')
    return path


def validate_checkout(
    checkout: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    checkout = checkout.expanduser().resolve()
    lock = load_json(lock_path)
    if lock.get('schema_version') != 1:
        raise ValueError('unsupported milestone-model lock schema')
    if not checkout.is_dir():
        raise FileNotFoundError(f'model checkout not found: {checkout}')
    checkout_entries = list(checkout.rglob('*'))
    symlinks = sorted(
        str(path.relative_to(checkout))
        for path in checkout_entries
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(
            f'milestone checkout must not contain symlinks: {symlinks}',
        )

    manifest_path = checkout / 'manifest.json'
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f'model manifest not found: {manifest_path}')
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != lock.get('manifest_sha256'):
        raise ValueError(
            'milestone manifest digest mismatch: '
            f'{manifest_sha256} != {lock.get("manifest_sha256")}',
        )
    manifest = load_json(manifest_path)

    contract = lock.get('manifest_contract')
    if not isinstance(contract, dict):
        raise ValueError('milestone lock has no manifest contract')
    actual_contract = {
        'api_version': manifest.get('api_version'),
        'model_id': manifest.get('model_id'),
        'runtime': manifest.get('runtime'),
        'runtime_abi': manifest.get('runtime_abi'),
        'model_format': manifest.get('model_format'),
        'encoder_type': manifest.get('encoder_type'),
        'latent_dims': manifest.get('latent_dims'),
    }
    if actual_contract != contract:
        raise ValueError(
            'milestone manifest contract mismatch: '
            f'{actual_contract!r} != {contract!r}',
        )

    locked_artifacts = lock.get('artifacts')
    manifest_artifacts = manifest.get('artifacts')
    if not isinstance(locked_artifacts, dict):
        raise ValueError('milestone lock has no artifact map')
    if manifest_artifacts != locked_artifacts:
        raise ValueError('milestone manifest artifact map does not match lock')

    checkout_bytes = manifest_path.stat().st_size
    expected_files = {'manifest.json'}
    for name, entry in locked_artifacts.items():
        if not isinstance(entry, dict):
            raise ValueError(f'invalid lock entry for {name}')
        relative = entry.get('path')
        expected_sha256 = entry.get('sha256')
        if not isinstance(relative, str) or not isinstance(
            expected_sha256,
            str,
        ):
            raise ValueError(f'invalid lock entry for {name}')
        artifact_path = safe_artifact_path(checkout, relative)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f'milestone artifact not found: {artifact_path}',
            )
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f'milestone artifact digest mismatch for {name}: '
                f'{actual_sha256} != {expected_sha256}',
            )
        checkout_bytes += artifact_path.stat().st_size
        expected_files.add(relative)

    actual_files = {
        str(path.relative_to(checkout))
        for path in checkout_entries
        if path.is_file() and path.name != '.DS_Store'
    }
    unexpected = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if unexpected or missing:
        raise ValueError(
            'milestone checkout file inventory mismatch: '
            f'unexpected={unexpected}, missing={missing}',
        )

    return {
        'artifact_count': len(locked_artifacts),
        'bundle_name': lock.get('bundle_name'),
        'checkout': str(checkout),
        'checkout_bytes': checkout_bytes,
        'manifest_sha256': manifest_sha256,
        'model_id': manifest.get('model_id'),
        'runtime_abi': manifest.get('runtime_abi'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate the frozen Evoke milestone model checkout.',
    )
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_checkout(args.checkout, args.lock)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
