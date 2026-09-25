#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from validate_milestone_model_checkout import DEFAULT_LOCK
from validate_milestone_model_checkout import validate_checkout


def safe_member_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsafe archive member: {name}')


def extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                safe_member_path(member.filename)
                file_type = (member.external_attr >> 16) & 0o170000
                if file_type == 0o120000:
                    raise ValueError(
                        f'archive symlink is not allowed: {member.filename}',
                    )
            source.extractall(destination)
        return

    try:
        source = tarfile.open(archive, mode='r:*')
    except tarfile.TarError as error:
        raise ValueError(f'unsupported model archive: {archive}') from error
    with source:
        for member in source.getmembers():
            safe_member_path(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f'archive link or special file is not allowed: '
                    f'{member.name}',
                )
        source.extractall(destination, filter='data')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkout(extracted: Path) -> Path:
    manifests = sorted(extracted.rglob('manifest.json'))
    if len(manifests) != 1:
        raise ValueError(
            'model archive must contain exactly one checkout manifest; '
            f'found {len(manifests)}',
        )
    return manifests[0].parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and validate the frozen Evoke milestone model.',
    )
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--archive-sha256')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'output already exists: {output}')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.ii42-model-',
        dir=output.parent,
    ) as temp_name:
        temp = Path(temp_name)
        archive = temp / 'model.archive'
        with urllib.request.urlopen(args.url) as response:
            with archive.open('wb') as handle:
                shutil.copyfileobj(response, handle)
        if args.archive_sha256 is not None:
            actual_sha256 = sha256_file(archive)
            if actual_sha256 != args.archive_sha256:
                raise ValueError(
                    'model archive digest mismatch: '
                    f'{actual_sha256} != {args.archive_sha256}',
                )

        extracted = temp / 'extracted'
        extracted.mkdir()
        extract_archive(archive, extracted)
        checkout = find_checkout(extracted)
        validate_checkout(checkout, args.lock)

        staged = temp / 'checkout'
        shutil.copytree(checkout, staged)
        validate_checkout(staged, args.lock)
        if output.exists():
            shutil.rmtree(output)
        staged.rename(output)

    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
