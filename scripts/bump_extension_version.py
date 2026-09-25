#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Version the single current Evoke install catalog.',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--version')
    group.add_argument('--series')
    parser.add_argument(
        '--rename-current-sql',
        action='store_true',
        help=(
            'Confirm that the current install SQL remains the reviewed '
            'catalog for the new version.'
        ),
    )
    return parser.parse_args()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def read_current_version(control_path: pathlib.Path) -> str:
    text = control_path.read_text(encoding='utf-8')
    match = re.search(r"default_version = '([0-9]+\.[0-9]+\.[0-9]+)'", text)
    if not match:
        raise RuntimeError('could not find default_version in control file')
    return match.group(1)


def next_patch_for_series(
    root: pathlib.Path, series: str, current_version: str
) -> str:
    output = subprocess.check_output(
        ['git', 'tag', '--list', f'v{series}.*'],
        cwd=root,
        text=True,
    )
    max_patch = 0
    pattern = re.compile(rf'^v{re.escape(series)}\.(\d+)$')
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            max_patch = max(max_patch, int(match.group(1)))
    current_prefix, _, current_patch_text = current_version.rpartition('.')
    if current_prefix == series and current_patch_text.isdigit():
        max_patch = max(max_patch, int(current_patch_text))
    return f'{series}.{max_patch + 1}'


def replace_default_version(control_path: pathlib.Path, version: str) -> None:
    text = control_path.read_text(encoding='utf-8')
    updated = re.sub(
        r"default_version = '[0-9]+\.[0-9]+\.[0-9]+'",
        f"default_version = '{version}'",
        text,
        count=1,
    )
    control_path.write_text(updated, encoding='utf-8')


def main() -> int:
    args = parse_args()
    if not args.rename_current_sql:
        raise RuntimeError(
            'refusing to rename the current extension SQL without '
            '--rename-current-sql; update and review the install SQL first '
            'when the SQL surface changes'
        )

    root = repo_root()
    control_path = root / 'evoke.control'
    sql_dir = root / 'sql'

    current_version = read_current_version(control_path)
    next_version = args.version or next_patch_for_series(
        root, args.series, current_version
    )

    if next_version == current_version:
        raise RuntimeError(
            f'next version {next_version} matches current version'
        )

    current_sql = sql_dir / f'evoke--{current_version}.sql'
    next_sql = sql_dir / f'evoke--{next_version}.sql'
    if not current_sql.exists():
        raise RuntimeError(f'missing install script: {current_sql.name}')
    if next_sql.exists():
        raise RuntimeError('target version file already exists')
    versioned_sql_files = sorted(sql_dir.glob('evoke--*.sql'))
    if versioned_sql_files != [current_sql]:
        raise RuntimeError(
            'expected exactly one current install SQL before versioning; '
            f'found {[path.name for path in versioned_sql_files]}'
        )

    current_sql.rename(next_sql)
    replace_default_version(control_path, next_version)

    print(next_version)
    return 0


if __name__ == '__main__':
    sys.exit(main())
