#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
DEFAULT_OUTPUT = (
    Path(tempfile.gettempdir()) / 'evoke_product_maturity_suite.json'
)
PINNED_ONNXRUNTIME_VERSION = (
    REPO_ROOT / 'packaging/onnxruntime.version'
).read_text(encoding='utf-8').strip()
MILESTONE_MODEL_LOCK = REPO_ROOT / 'packaging/milestone-model.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the Evoke unified-index product maturity suite.',
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--package-root',
        type=Path,
        required=True,
        help=(
            'DESTDIR-style staged release root. Its load-bearing files must '
            'match the PostgreSQL installation selected by --pg-bin.'
        ),
    )
    parser.add_argument(
        '--source-package-root',
        type=Path,
        required=True,
        help=(
            'DESTDIR-style staged psql_bm25s package used to qualify the '
            'only supported historical source-table migration.'
        ),
    )
    parser.add_argument('--lifecycle-docs', type=int, default=50_000)
    parser.add_argument('--benchmark-docs', type=int, default=75_000)
    parser.add_argument('--benchmark-repeats', type=int, default=4)
    parser.add_argument('--benchmark-concurrency', type=int, default=96)
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Run smokes only, skipping the product path benchmark.',
    )
    parser.add_argument(
        '--skip-restart-smoke',
        action='store_true',
        help='Skip the runtime-worker crash-restart smoke.',
    )
    parser.add_argument(
        '--restart-wait-seconds',
        type=float,
        default=15.0,
        help='Maximum wait for the five-second runtime-worker restart.',
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pg_config_value(pg_bin: Path, option: str) -> str:
    pg_config = pg_bin / 'pg_config'
    result = subprocess.run(
        [str(pg_config), option],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f'{pg_config} {option} failed: {detail or "unknown error"}'
        )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f'{pg_config} {option} returned an empty value')
    return value


def staged_install_path(package_root: Path, install_path: Path) -> Path:
    if not install_path.is_absolute():
        raise ValueError(f'install path is not absolute: {install_path}')
    return package_root / install_path.relative_to(install_path.anchor)


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition(':')
        if separator:
            values[key.strip()] = value.strip()
    return values


def current_source_identity() -> dict[str, str] | None:
    inside = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', '--is-inside-work-tree'],
        text=True,
        capture_output=True,
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != 'true':
        return None

    commit = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            'git',
            '-C',
            str(REPO_ROOT),
            'status',
            '--porcelain',
            '--untracked-files=normal',
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return {
        'commit': commit,
        'tree': 'clean' if not status.strip() else 'dirty',
    }


def validate_build_info(
    build_info: dict[str, str],
    source_identity: dict[str, str] | None,
) -> list[str]:
    errors: list[str] = []
    if build_info.get('Git tree') != 'clean':
        errors.append('BUILD-INFO Git tree must be clean')
    if source_identity is not None:
        if source_identity['tree'] != 'clean':
            errors.append('current source tree is not clean')
        if build_info.get('Git commit') != source_identity['commit']:
            errors.append(
                'BUILD-INFO Git commit does not match current source'
            )
    if build_info.get('ONNX Runtime') != PINNED_ONNXRUNTIME_VERSION:
        errors.append(
            'BUILD-INFO ONNX Runtime does not match the pinned version'
        )
    if (
        build_info.get('ONNX Runtime linkage')
        != 'bundled in PostgreSQL pkglibdir'
    ):
        errors.append('BUILD-INFO ONNX Runtime linkage is not bundled')
    return errors


def inspect_package_metadata(
    package_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = {
        'README.md': package_root / 'README.md',
        'EVOKE-LICENSE': package_root / 'LICENSES/EVOKE-LICENSE',
        'ONNXRUNTIME-LICENSE': (
            package_root / 'LICENSES/ONNXRUNTIME-LICENSE'
        ),
    }
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, path in paths.items():
        artifact: dict[str, Any] = {
            'name': name,
            'path': str(path),
        }
        if not path.is_file() or path.stat().st_size == 0:
            artifact['error'] = 'missing or empty'
            errors.append(f'package metadata is missing or empty: {name}')
        else:
            artifact['identity'] = artifact_identity(path)
        artifacts.append(artifact)

    expected = {
        'README.md': REPO_ROOT / 'README.md',
        'EVOKE-LICENSE': REPO_ROOT / 'LICENSE',
    }
    for artifact in artifacts:
        expected_path = expected.get(str(artifact['name']))
        if expected_path is None or 'identity' not in artifact:
            continue
        expected_identity = artifact_identity(expected_path)
        artifact['expected_identity'] = expected_identity
        if artifact['identity'] != expected_identity:
            errors.append(
                f"package metadata does not match source: "
                f"{artifact['name']}"
            )
    return artifacts, errors


def control_version(path: Path) -> str:
    match = re.search(
        r"^default_version\s*=\s*'([^']+)'\s*$",
        path.read_text(encoding='utf-8'),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f'default_version is missing from {path}')
    return match.group(1)


def artifact_identity(path: Path) -> dict[str, str]:
    if path.is_symlink():
        return {
            'kind': 'symlink',
            'target': os.readlink(path),
        }
    return {
        'kind': 'file',
        'sha256': sha256_file(path),
    }


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


def inspect_model_binding(
    package_root: Path,
    sharedir: Path,
    build_info: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        'passed': False,
        'errors': [],
        'artifacts': [],
    }
    errors: list[str] = report['errors']
    try:
        expected_location = sharedir / 'evoke/models/default'
        location_text = build_info.get('Milestone model location')
        if location_text != str(expected_location):
            raise ValueError(
                'BUILD-INFO milestone model location does not match '
                'PostgreSQL sharedir'
            )
        staged_model = staged_install_path(package_root, expected_location)
        installed_model = expected_location
        report['staged_model'] = str(staged_model)
        report['installed_model'] = str(installed_model)

        lock = load_json_object(MILESTONE_MODEL_LOCK)
        manifest_path = staged_model / 'manifest.json'
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError('staged milestone model manifest is missing')
        manifest = load_json_object(manifest_path)
        manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 != lock.get('manifest_sha256'):
            errors.append('staged model manifest does not match source lock')
        if manifest_sha256 != build_info.get(
            'Milestone model manifest SHA-256'
        ):
            errors.append('staged model manifest does not match BUILD-INFO')
        if manifest.get('model_id') != build_info.get('Milestone model'):
            errors.append('staged model id does not match BUILD-INFO')
        if manifest.get('artifacts') != lock.get('artifacts'):
            errors.append('staged model artifact map does not match source lock')

        expected_files = {'manifest.json'}
        artifact_specs: list[tuple[str, str]] = [
            ('manifest.json', manifest_sha256),
        ]
        locked_artifacts = lock.get('artifacts')
        if not isinstance(locked_artifacts, dict):
            raise ValueError('milestone model lock has no artifact map')
        for name, entry in sorted(locked_artifacts.items()):
            if not isinstance(entry, dict):
                raise ValueError(f'invalid milestone model lock entry: {name}')
            relative = entry.get('path')
            expected_sha256 = entry.get('sha256')
            if not isinstance(relative, str) or not isinstance(
                expected_sha256,
                str,
            ):
                raise ValueError(
                    f'invalid milestone model artifact identity: {name}'
                )
            relative_path = Path(relative)
            if relative_path.is_absolute() or '..' in relative_path.parts:
                raise ValueError(
                    f'unsafe milestone model artifact path: {relative}'
                )
            expected_files.add(relative)
            artifact_specs.append((relative, expected_sha256))

        for label, root in (
            ('staged', staged_model),
            ('installed', installed_model),
        ):
            actual_files = {
                str(path.relative_to(root))
                for path in root.rglob('*')
                if path.is_file() and not path.is_symlink()
            } if root.is_dir() else set()
            symlinks = sorted(
                str(path.relative_to(root))
                for path in root.rglob('*')
                if path.is_symlink()
            ) if root.is_dir() else []
            if symlinks:
                errors.append(
                    f'{label} milestone model contains symlinks: {symlinks}'
                )
            if actual_files != expected_files:
                errors.append(
                    f'{label} milestone model inventory mismatch: '
                    f'unexpected={sorted(actual_files - expected_files)}, '
                    f'missing={sorted(expected_files - actual_files)}'
                )

        artifacts: list[dict[str, Any]] = report['artifacts']
        for relative, expected_sha256 in artifact_specs:
            staged_path = staged_model / relative
            installed_path = installed_model / relative
            artifact = compare_package_artifact(staged_path, installed_path)
            artifact['name'] = relative
            artifact['expected_sha256'] = expected_sha256
            artifacts.append(artifact)
            if not staged_path.is_file() or staged_path.is_symlink():
                errors.append(f'staged model artifact is missing: {relative}')
                continue
            if sha256_file(staged_path) != expected_sha256:
                errors.append(
                    f'staged model artifact digest mismatch: {relative}'
                )
            if not artifact['matched']:
                errors.append(
                    f'installed model artifact mismatch: {relative}'
                )
        report['model_id'] = manifest.get('model_id')
        report['manifest_sha256'] = manifest_sha256
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    report['passed'] = not errors
    return report


def compare_package_artifact(
    staged_path: Path,
    installed_path: Path,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        'staged_path': str(staged_path),
        'installed_path': str(installed_path),
        'matched': False,
    }
    if not os.path.lexists(staged_path):
        artifact['error'] = 'staged artifact is missing'
        return artifact
    if not os.path.lexists(installed_path):
        artifact['error'] = 'installed artifact is missing'
        return artifact

    staged_identity = artifact_identity(staged_path)
    installed_identity = artifact_identity(installed_path)
    artifact['staged'] = staged_identity
    artifact['installed'] = installed_identity
    artifact['matched'] = staged_identity == installed_identity
    if not artifact['matched']:
        artifact['error'] = 'staged and installed identities differ'
    return artifact


def extension_sql_artifacts(
    extension_dir: Path,
    version: str,
) -> list[Path]:
    install_sql = extension_dir / f'evoke--{version}.sql'
    actual_sql = sorted(extension_dir.glob('evoke--*.sql'))
    expected_sql = [install_sql]
    if actual_sql != expected_sql:
        raise ValueError(
            'staged extension directory must contain exactly one current '
            'install SQL and no Evoke beta upgrade scripts; expected '
            f'{[path.name for path in expected_sql]}, found '
            f'{[path.name for path in actual_sql]}'
        )
    return expected_sql


def inspect_package_binding(
    package_root: Path,
    pg_bin: Path,
) -> dict[str, Any]:
    package_root = package_root.expanduser().resolve()
    report: dict[str, Any] = {
        'package_root': str(package_root),
        'pg_bin': str(pg_bin),
        'passed': False,
        'errors': [],
        'artifacts': [],
    }
    errors: list[str] = report['errors']
    try:
        if not package_root.is_dir():
            raise ValueError(
                f'staged package root is not a directory: {package_root}'
            )

        build_info_path = package_root / 'BUILD-INFO.txt'
        if not build_info_path.is_file():
            raise ValueError(f'BUILD-INFO.txt is missing: {package_root}')
        build_info = parse_key_value_file(build_info_path)
        report['build_info'] = build_info
        report['build_info_sha256'] = sha256_file(build_info_path)
        source_identity = current_source_identity()
        report['source_identity'] = source_identity
        errors.extend(validate_build_info(build_info, source_identity))

        metadata_artifacts, metadata_errors = inspect_package_metadata(
            package_root
        )
        report['metadata_artifacts'] = metadata_artifacts
        errors.extend(metadata_errors)

        pkglibdir = Path(pg_config_value(pg_bin, '--pkglibdir'))
        sharedir = Path(pg_config_value(pg_bin, '--sharedir'))
        pg_version = pg_config_value(pg_bin, '--version')
        report['pg_config'] = {
            'pkglibdir': str(pkglibdir),
            'sharedir': str(sharedir),
            'version': pg_version,
        }
        if build_info.get('PostgreSQL pkglibdir') != str(pkglibdir):
            errors.append('BUILD-INFO pkglibdir does not match --pg-bin')
        if build_info.get('PostgreSQL sharedir') != str(sharedir):
            errors.append('BUILD-INFO sharedir does not match --pg-bin')

        major_match = re.search(r'PostgreSQL\s+(\d+)', pg_version)
        if major_match is None:
            errors.append(f'cannot parse PostgreSQL major from {pg_version!r}')
        elif build_info.get('PostgreSQL major') != major_match.group(1):
            errors.append(
                'BUILD-INFO PostgreSQL major does not match --pg-bin'
            )

        staged_pkglibdir = staged_install_path(package_root, pkglibdir)
        staged_extension_dir = staged_install_path(
            package_root,
            sharedir / 'extension',
        )
        control_path = staged_extension_dir / 'evoke.control'
        if not control_path.is_file():
            raise ValueError(
                f'staged extension control file is missing: {control_path}'
            )
        version = control_version(control_path)
        report['extension_version'] = version
        if build_info.get('Version') != version:
            errors.append(
                'BUILD-INFO version does not match staged evoke.control'
            )

        extension_candidates = sorted(
            path for path in staged_pkglibdir.glob('evoke.*')
            if path.suffix in {'.dylib', '.so'}
        )
        if len(extension_candidates) != 1:
            raise ValueError(
                'staged package must contain exactly one evoke shared library '
                f'in {staged_pkglibdir}'
            )

        staged_artifacts = [
            extension_candidates[0],
            control_path,
            *extension_sql_artifacts(staged_extension_dir, version),
        ]
        runtime_artifacts = sorted(
            path for path in staged_pkglibdir.iterdir()
            if path.name.startswith('libonnxruntime')
        )
        if not runtime_artifacts:
            errors.append(
                f'no bundled ONNX Runtime library in {staged_pkglibdir}'
            )
        staged_artifacts.extend(runtime_artifacts)

        artifacts: list[dict[str, Any]] = report['artifacts']
        for staged_path in staged_artifacts:
            if staged_path.parent == staged_pkglibdir:
                installed_path = pkglibdir / staged_path.name
            else:
                installed_path = sharedir / 'extension' / staged_path.name
            artifact = compare_package_artifact(
                staged_path,
                installed_path,
            )
            artifact['name'] = staged_path.name
            artifacts.append(artifact)
            if not artifact['matched']:
                errors.append(
                    f'package binding mismatch: {staged_path.name}'
                )

        model_binding = inspect_model_binding(
            package_root,
            sharedir,
            build_info,
        )
        report['model_binding'] = model_binding
        errors.extend(model_binding['errors'])

        fingerprint_payload = {
            'build_info_sha256': report['build_info_sha256'],
            'artifacts': [
                {
                    'name': artifact['name'],
                    'staged': artifact.get('staged'),
                }
                for artifact in artifacts
            ],
            'metadata_artifacts': [
                {
                    'name': artifact['name'],
                    'identity': artifact.get('identity'),
                }
                for artifact in metadata_artifacts
            ],
            'model_artifacts': [
                {
                    'name': artifact['name'],
                    'staged': artifact.get('staged'),
                }
                for artifact in model_binding['artifacts']
            ],
        }
        fingerprint_json = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        report['fingerprint'] = hashlib.sha256(
            fingerprint_json
        ).hexdigest()
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))

    report['passed'] = not errors
    return report


def run_step(name: str, cmd: list[str | Path]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        [str(item) for item in cmd],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary: dict[str, Any] = {
        'name': name,
        'command': [str(item) for item in cmd],
        'returncode': result.returncode,
        'elapsed_ms': round(elapsed_ms, 3),
        'stdout_tail': result.stdout[-4000:],
        'stderr_tail': result.stderr[-4000:],
    }
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return summary


def benchmark_output_path(output: Path) -> Path:
    stem = output.stem
    return output.with_name(f'{stem}.benchmark.json')


def main() -> int:
    args = parse_args()
    args.pg_bin = args.pg_bin.expanduser().resolve()
    args.package_root = args.package_root.expanduser().resolve()
    args.source_package_root = (
        args.source_package_root.expanduser().resolve()
    )
    args.output = args.output.expanduser().resolve()
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output = benchmark_output_path(args.output)
    package_preflight = inspect_package_binding(
        args.package_root,
        args.pg_bin,
    )
    system_pkglibdir = Path(pg_config_value(args.pg_bin, '--pkglibdir'))
    system_sharedir = Path(pg_config_value(args.pg_bin, '--sharedir'))
    staged_pkglibdir = staged_install_path(
        args.package_root,
        system_pkglibdir,
    )
    staged_sharedir = staged_install_path(
        args.package_root,
        system_sharedir,
    )
    staged_extension_args: list[str | Path] = [
        '--extension-libdir',
        staged_pkglibdir,
        '--extension-control-dir',
        staged_sharedir,
    ]
    empty_lifecycle_cmd: list[str | Path] = [
        sys.executable,
        'scripts/test_empty_unlogged_lifecycle_smoke.py',
        '--pg-bin',
        args.pg_bin,
        *staged_extension_args,
    ]
    if args.model_path is not None:
        empty_lifecycle_cmd.extend(['--model-path', args.model_path])
    provider_matrix_cmd: list[str | Path] = [
        sys.executable,
        'scripts/test_onnxruntime_provider_matrix.py',
        '--pg-bin',
        args.pg_bin,
        '--providers',
        'auto',
        '--iterations',
        '2',
        '--strict',
    ]
    if args.model_path is not None:
        provider_matrix_cmd.extend(['--model-path', args.model_path])
    source_migration_cmd: list[str | Path] = [
        sys.executable,
        'scripts/test_psql_bm25s_source_migration_smoke.py',
        '--pg-bin',
        args.pg_bin,
        '--source-package-root',
        args.source_package_root,
        '--output',
        args.output.with_name(
            f'{args.output.stem}.source-migration.json'
        ),
    ]
    steps: list[tuple[str, list[str | Path]]] = [
        (
            'product convergence inventory',
            [sys.executable, 'scripts/test_product_convergence_inventory.py'],
        ),
        (
            'runtime server build',
            [
                'make',
                'runtime-server',
                f'PG_CONFIG={Path(args.pg_bin) / "pg_config"}',
            ],
        ),
        (
            'product Python unit tests',
            [sys.executable, '-m', 'pytest', '-q'],
        ),
        (
            'product Python compileall',
            [
                sys.executable,
                '-m',
                'compileall',
                '-q',
                'scripts',
                'tests',
            ],
        ),
        (
            'isolated product regression',
            [
                sys.executable,
                'scripts/test_extension_regression_temp_pg.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
            ],
        ),
        (
            'isolated extension schema placement smoke',
            [
                sys.executable,
                'scripts/test_extension_schema_smoke.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
            ],
        ),
        (
            'independent page-native golden',
            [
                sys.executable,
                'scripts/test_page_native_golden.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.page-native-golden.json'
                ),
            ],
        ),
        (
            'convergent v3 lifecycle smoke',
            [
                sys.executable,
                'scripts/test_convergent_segment_read_smoke.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.convergent-v3.json'
                ),
            ],
        ),
        (
            'convergent SSR lexical-first lifecycle smoke',
            [
                sys.executable,
                'scripts/test_convergent_sae_lifecycle_smoke.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.convergent-sae.json'
                ),
            ],
        ),
        (
            'compact maintenance builder smoke',
            [
                sys.executable,
                'scripts/test_compact_maintenance_builder.py',
                '--bindir',
                args.pg_bin,
                '--builder',
                'compact',
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
            ],
        ),
        (
            'spill maintenance builder smoke',
            [
                sys.executable,
                'scripts/test_compact_maintenance_builder.py',
                '--bindir',
                args.pg_bin,
                '--builder',
                'spill',
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
            ],
        ),
        (
            'convergent VACUUM hard-frontier smoke',
            [
                sys.executable,
                'scripts/test_convergent_vacuum_frontier_smoke.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.vacuum-frontier.json'
                ),
            ],
        ),
        (
            'retired storage fail-closed and REINDEX recovery smoke',
            [
                sys.executable,
                'scripts/test_storage_layout_boundary.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--json-output',
                args.output.with_name(
                    f'{args.output.stem}.v3-storage-boundary.json'
                ),
            ],
        ),
        (
            'checked metapage read boundary smoke',
            [
                sys.executable,
                'scripts/test_metapage_read_boundary.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--json-output',
                args.output.with_name(
                    f'{args.output.stem}.metapage-read-boundary.json'
                ),
            ],
        ),
        (
            'source-table BM25S migration smoke',
            source_migration_cmd,
        ),
        (
            'isolated ONNX Runtime ABI smoke',
            [
                sys.executable,
                'scripts/test_onnxruntime_smoke.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
            ],
        ),
        (
            'ONNX Runtime provider matrix',
            provider_matrix_cmd,
        ),
        (
            'empty and UNLOGGED lifecycle smoke',
            empty_lifecycle_cmd,
        ),
        (
            'mutable BM25 crash recovery smoke',
            [
                sys.executable,
                'scripts/test_crash_recovery_smoke.py',
                '--pg-bin',
                args.pg_bin,
            ],
        ),
        (
            'same-index writer concurrency smoke',
            [
                sys.executable,
                'scripts/test_same_index_writer_concurrency_temp_pg.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.writer-concurrency.json'
                ),
            ],
        ),
        (
            'SSR two-phase transaction lifecycle smoke',
            [
                sys.executable,
                'scripts/test_transactional_delta_lifecycle.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.sae-two-phase.json'
                ),
            ],
        ),
        (
            'SSR page-native transaction memory and RSS smoke',
            [
                sys.executable,
                'scripts/test_sae_transaction_many_index_budget.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.sae-transaction-memory.json'
                ),
            ],
        ),
        (
            'backend memory ownership smoke',
            [
                sys.executable,
                'scripts/test_backend_memory_ownership.py',
                '--bindir',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.backend-memory.json'
                ),
            ],
        ),
        (
            'eventual semantic quarantine smoke',
            [
                sys.executable,
                'scripts/test_eventual_semantic_quarantine.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.semantic-quarantine.json'
                ),
            ],
        ),
        (
            'eventual semantic maintenance fairness smoke',
            [
                sys.executable,
                'scripts/test_eventual_semantic_maintenance_fairness.py',
                '--pg-bin',
                args.pg_bin,
                '--model-path',
                args.model_path,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.semantic-fairness.json'
                ),
            ],
        ),
        (
            'cache publication and allocation failure-safety smoke',
            [
                sys.executable,
                'scripts/test_cache_failure_safety_temp_pg.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
            ],
        ),
        (
            'maintenance discovery scale benchmark',
            [
                sys.executable,
                'scripts/benchmark_maintenance_discovery_scale.py',
                '--pg-bin',
                args.pg_bin,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.maintenance-discovery.json'
                ),
            ],
        ),
        (
            'runtime service required smoke',
            [
                sys.executable,
                'scripts/test_runtime_service_required_smoke.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
            ],
        ),
        (
            'runtime service privilege smoke',
            [
                sys.executable,
                'scripts/test_runtime_service_privilege_smoke.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
            ],
        ),
        (
            'runtime service temp pg smoke',
            [
                sys.executable,
                'scripts/test_runtime_service_temp_pg.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--runtime-server-binary',
                REPO_ROOT / 'build/evoke-runtime-server',
            ],
        ),
        (
            'runtime failover backpressure smoke',
            [
                sys.executable,
                'scripts/test_runtime_service_temp_pg.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
                '--runtime-server-binary',
                REPO_ROOT / 'build/evoke-runtime-server',
                '--runtime-liveness-timeout-ms',
                '3000',
                '--failover-backpressure-only',
            ],
        ),
        (
            'shared preload lifecycle closure',
            [
                sys.executable,
                'scripts/test_shared_preload_lifecycle_closure.py',
                '--bindir',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
            ],
        ),
        (
            'shared preload automatic warmup smoke',
            [
                sys.executable,
                'scripts/test_shared_preload_auto_preload.py',
                '--bindir',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
            ],
        ),
        (
            'standby exact-root automatic preload smoke',
            [
                sys.executable,
                'scripts/test_shared_preload_standby_auto_preload.py',
                '--pg-bin',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.standby-preload.json'
                ),
            ],
        ),
        (
            'physical replication lifecycle smoke',
            [
                sys.executable,
                'scripts/test_replication_lifecycle_smoke.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--model-path',
                args.model_path,
            ],
        ),
        (
            'payload health corruption smoke',
            [
                sys.executable,
                'scripts/test_payload_health_corruption_smoke.py',
                '--bindir',
                args.pg_bin,
                '--extension-libdir',
                staged_pkglibdir,
                '--extension-control-dir',
                staged_sharedir,
                '--json-output',
                args.output.with_name(
                    f'{args.output.stem}.payload-health.json'
                ),
            ],
        ),
        (
            'ONNX Runtime resource soak',
            [
                sys.executable,
                'scripts/test_onnxruntime_resource_soak.py',
                '--pg-bin',
                args.pg_bin,
                *staged_extension_args,
                '--iterations',
                '1000',
                '--sample-every',
                '50',
                '--model-path',
                args.model_path,
                '--output',
                args.output.with_name(
                    f'{args.output.stem}.onnx-soak.json'
                ),
            ],
        ),
    ]
    if args.model_path is not None:
        steps.append(
            (
                'production model unified lifecycle smoke',
                [
                    sys.executable,
                    'scripts/test_unified_index_lifecycle_smoke.py',
                    '--pg-bin',
                    args.pg_bin,
                    *staged_extension_args,
                    '--model-path',
                    args.model_path,
                    '--soak-queries',
                    '300',
                    '--mixed-soak-cycles',
                    '8',
                    '--concurrent-crud-cycles',
                    '6',
                    '--concurrent-readers',
                    '4',
                    '--concurrent-writers',
                    '2',
                    '--output',
                    args.output.with_name(
                        f'{args.output.stem}.model-lifecycle.json'
                    ),
                ],
            )
        )
        steps.append(
            (
                'concurrent DDL and relation rewrite lifecycle smoke',
                [
                    sys.executable,
                    'scripts/test_concurrent_ddl_lifecycle_smoke.py',
                    '--pg-bin',
                    args.pg_bin,
                    '--model-path',
                    args.model_path,
                    '--output',
                    args.output.with_name(
                        f'{args.output.stem}.concurrent-ddl.json'
                    ),
                ],
            )
        )
    if not args.skip_restart_smoke:
        steps.append(
            (
                'runtime service restart smoke',
                [
                    sys.executable,
                    'scripts/test_runtime_service_restart_smoke.py',
                    '--pg-bin',
                    args.pg_bin,
                    '--model-path',
                    args.model_path,
                    '--restart-wait-seconds',
                    str(args.restart_wait_seconds),
                ],
            )
        )
    medium_lifecycle_cmd: list[str | Path] = [
        sys.executable,
        'scripts/test_model_lifecycle_medium_perf.py',
        '--pg-bin',
        args.pg_bin,
        '--docs',
        str(args.lifecycle_docs),
        '--output',
        args.output.with_name(
            f'{args.output.stem}.medium-lifecycle.json'
        ),
    ]
    if args.model_path is not None:
        medium_lifecycle_cmd.extend(['--model-path', args.model_path])
    steps.append(
        ('mutable lifecycle medium smoke', medium_lifecycle_cmd)
    )
    if not args.skip_benchmark:
        benchmark_cmd: list[str | Path] = [
            sys.executable,
            'scripts/benchmark_evoke_product_path.py',
            '--pg-bin',
            args.pg_bin,
            '--docs',
            str(args.benchmark_docs),
            '--repeats',
            str(args.benchmark_repeats),
            '--concurrency',
            str(args.benchmark_concurrency),
            '--output',
            benchmark_output,
        ]
        if args.model_path is not None:
            benchmark_cmd.extend(['--model-path', args.model_path])
        steps.append(('product path benchmark', benchmark_cmd))

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        if package_preflight['passed']:
            for name, cmd in steps:
                print(f'running: {name}', flush=True)
                step = run_step(name, cmd)
                results.append(step)
                if step['returncode'] != 0:
                    break
    finally:
        package_postflight = inspect_package_binding(
            args.package_root,
            args.pg_bin,
        )
        package_stable = (
            package_preflight.get('fingerprint') is not None
            and package_preflight.get('fingerprint')
            == package_postflight.get('fingerprint')
        )
        failed_step: str | None = None
        if not package_preflight['passed']:
            failed_step = 'staged package binding preflight'
        else:
            failed_result = next(
                (
                    step for step in results
                    if step['returncode'] != 0
                ),
                None,
            )
            if failed_result is not None:
                failed_step = str(failed_result['name'])
            elif not package_postflight['passed'] or not package_stable:
                failed_step = 'staged package binding postflight'

        steps_passed = (
            len(results) == len(steps)
            and all(step['returncode'] == 0 for step in results)
        )
        suite = {
            'api_version': 'evoke_index_v1',
            'suite': 'evoke_product_maturity',
            'elapsed_ms': round((time.perf_counter() - started) * 1000.0, 3),
            'pg_bin': str(args.pg_bin),
            'output': str(args.output),
            'source_package_root': str(args.source_package_root),
            'benchmark_output': None
                if args.skip_benchmark
                else str(benchmark_output),
            'package_binding': {
                'preflight': package_preflight,
                'postflight': package_postflight,
                'stable': package_stable,
            },
            'steps': results,
            'failed_step': failed_step,
            'passed': package_preflight['passed']
                and package_postflight['passed']
                and package_stable
                and steps_passed,
        }
        args.output.write_text(
            json.dumps(suite, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    if not suite['passed']:
        return 1
    print(f'product maturity suite passed -> {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
