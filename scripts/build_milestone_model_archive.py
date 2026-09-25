#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from validate_milestone_model_checkout import DEFAULT_LOCK
from validate_milestone_model_checkout import validate_checkout


ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def archive_checkout(
    checkout: Path,
    output: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, object]:
    checkout = checkout.expanduser().resolve()
    output = output.expanduser().resolve()
    validation = validate_checkout(checkout, lock_path)
    bundle_name = validation.get('bundle_name')
    if not isinstance(bundle_name, str) or not bundle_name:
        raise ValueError('milestone-model lock has no bundle_name')
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f'.{output.name}.',
        suffix='.tmp',
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode='w',
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in sorted(
                path for path in checkout.rglob('*') if path.is_file()
            ):
                relative = source.relative_to(checkout)
                member = zipfile.ZipInfo(
                    f'{bundle_name}/{relative.as_posix()}',
                    date_time=ARCHIVE_TIMESTAMP,
                )
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                with source.open('rb') as source_handle:
                    with archive.open(member, mode='w') as target_handle:
                        shutil.copyfileobj(
                            source_handle,
                            target_handle,
                            length=1024 * 1024,
                        )
        temporary_path.replace(output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    digest = sha256_file(output)
    checksum_path = output.with_name(f'{output.name}.sha256')
    checksum_path.write_text(
        f'{digest}  {output.name}\n',
        encoding='utf-8',
    )
    return {
        'archive': str(output),
        'archive_bytes': output.stat().st_size,
        'archive_sha256': digest,
        'bundle_name': bundle_name,
        'checksum': str(checksum_path),
        'model_id': validation['model_id'],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the deterministic Evoke milestone model asset.',
    )
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = archive_checkout(args.checkout, args.output, args.lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
