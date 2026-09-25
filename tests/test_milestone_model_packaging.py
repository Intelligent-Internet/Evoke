from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_milestone_model_archive import archive_checkout  # noqa: E402
from validate_milestone_model_checkout import validate_checkout  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkout(root: Path) -> tuple[Path, Path]:
    checkout = root / 'checkout'
    (checkout / 'encoder').mkdir(parents=True)
    (checkout / 'lexical').mkdir()
    query_path = checkout / 'encoder/query.onnx'
    vocabulary_path = checkout / 'lexical/vocabulary.json'
    query_path.write_bytes(b'test-onnx')
    vocabulary_path.write_text('["search"]\n', encoding='utf-8')
    artifacts = {
        'query_encoder': {
            'path': 'encoder/query.onnx',
            'sha256': sha256_file(query_path),
        },
        'lexical_vocabulary': {
            'path': 'lexical/vocabulary.json',
            'sha256': sha256_file(vocabulary_path),
        },
    }
    manifest = {
        'api_version': 'evoke_model_v1',
        'artifacts': artifacts,
        'model_id': 'test_milestone',
        'runtime': 'onnxruntime',
        'runtime_abi': 'evoke_p2_unified_text_atoms_v2',
        'model_format': 'onnx',
        'encoder_type': 'test_unified_postings',
        'latent_dims': 8,
    }
    manifest_path = checkout / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    lock = {
        'schema_version': 1,
        'bundle_name': 'test-bundle',
        'manifest_sha256': sha256_file(manifest_path),
        'manifest_contract': {
            'api_version': 'evoke_model_v1',
            'model_id': 'test_milestone',
            'runtime': 'onnxruntime',
            'runtime_abi': 'evoke_p2_unified_text_atoms_v2',
            'model_format': 'onnx',
            'encoder_type': 'test_unified_postings',
            'latent_dims': 8,
        },
        'artifacts': artifacts,
    }
    lock_path = root / 'lock.json'
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return checkout, lock_path


def test_validator_accepts_locked_checkout(tmp_path: Path) -> None:
    checkout, lock_path = write_checkout(tmp_path)

    result = validate_checkout(checkout, lock_path)

    assert result['model_id'] == 'test_milestone'
    assert result['artifact_count'] == 2


def test_validator_rejects_tampered_or_extra_files(tmp_path: Path) -> None:
    checkout, lock_path = write_checkout(tmp_path)
    query_path = checkout / 'encoder/query.onnx'
    query_path.write_bytes(b'tampered')

    with pytest.raises(ValueError, match='digest mismatch'):
        validate_checkout(checkout, lock_path)

    query_path.write_bytes(b'test-onnx')
    (checkout / 'unexpected.txt').write_text('extra', encoding='utf-8')
    with pytest.raises(ValueError, match='file inventory mismatch'):
        validate_checkout(checkout, lock_path)


def test_validator_rejects_checkout_symlinks(tmp_path: Path) -> None:
    checkout, lock_path = write_checkout(tmp_path)
    link_path = checkout / 'encoder/query-link.onnx'
    try:
        link_path.symlink_to('query.onnx')
    except OSError as error:
        pytest.skip(f'symlinks are unavailable: {error}')

    with pytest.raises(ValueError, match='must not contain symlinks'):
        validate_checkout(checkout, lock_path)


def test_fetcher_safely_extracts_and_validates_archive(
    tmp_path: Path,
) -> None:
    checkout, lock_path = write_checkout(tmp_path)
    archive_path = tmp_path / 'model.zip'
    with zipfile.ZipFile(archive_path, mode='w') as archive:
        for path in checkout.rglob('*'):
            if path.is_file():
                archive.write(
                    path,
                    Path('milestone') / path.relative_to(checkout),
                )
    output = tmp_path / 'fetched'

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'scripts/fetch_milestone_model.py'),
            '--url',
            archive_path.as_uri(),
            '--output',
            str(output),
            '--lock',
            str(lock_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert validate_checkout(output, lock_path)['model_id'] == 'test_milestone'


def test_archive_builder_is_deterministic_and_fetchable(
    tmp_path: Path,
) -> None:
    checkout, lock_path = write_checkout(tmp_path)
    first = tmp_path / 'first.zip'
    second = tmp_path / 'second.zip'

    first_result = archive_checkout(checkout, first, lock_path)
    second_result = archive_checkout(checkout, second, lock_path)

    assert first_result['archive_sha256'] == second_result['archive_sha256']
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name('first.zip.sha256').read_text(
        encoding='utf-8',
    ).endswith('  first.zip\n')

    output = tmp_path / 'fetched'
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'scripts/fetch_milestone_model.py'),
            '--url',
            first.as_uri(),
            '--archive-sha256',
            str(first_result['archive_sha256']),
            '--output',
            str(output),
            '--lock',
            str(lock_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert validate_checkout(output, lock_path)['model_id'] == 'test_milestone'
