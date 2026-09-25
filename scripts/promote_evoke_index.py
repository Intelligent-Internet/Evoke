#!/usr/bin/env python3
"""Publish one qualified Evoke candidate under its stable product name."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_$]*$')
REPO_ROOT = Path(__file__).resolve().parents[1]
ONNXRUNTIME_VERSION_PATH = REPO_ROOT / 'packaging/onnxruntime.version'
COMMONS_MATRIX_CONTRACT_VERSION = 10
MAX_RESTART_EVIDENCE_AGE_SECONDS = 600.0
REQUIRED_CONCURRENCIES = {1, 4, 8}
REQUIRED_COMMONS_CASES = {
    'arxiv_broad_date',
    'arxiv_date_category',
    'arxiv_organization_broad',
    'arxiv_organization_selective',
    'arxiv_unfiltered',
    'ca_chunk_scope_ordinary',
    'ca_chunk_scope_stress',
    'ca_policy_union',
    'pubmed_broad_date',
    'pubmed_date_category',
    'pubmed_date_journal',
    'pubmed_partial_date',
    'pubmed_unfiltered',
    'system_chunk_scope',
    'tx_chunk_scope_ordinary',
    'tx_policy_union',
    'wa_chunk_scope_ordinary',
    'wa_policy_union',
}
REQUIRED_SERIAL_RANK_ORACLE_CASES = {
    'arxiv_date_category',
    'ca_chunk_scope_stress',
    'pubmed_date_category',
    'pubmed_date_journal',
    'system_chunk_scope',
}


def pinned_onnxruntime_version() -> str:
    try:
        version = ONNXRUNTIME_VERSION_PATH.read_text(
            encoding='utf-8',
        ).strip()
    except OSError as exc:
        raise ValueError(
            'promotion requires the pinned ONNX Runtime version; '
            f'could not read {ONNXRUNTIME_VERSION_PATH}: {exc}'
        ) from exc
    if not version:
        raise ValueError(
            'promotion requires a non-empty pinned ONNX Runtime version'
        )
    return version


REQUIRED_COMMONS_FIXED_INDEXES = {
    'commons.data_arxiv__title_abstract__field_aware_bm25_idx',
    'commons.data_policy_ca__title_description__field_aware_bm25_idx',
    'commons.data_policy_ca_chunks__content__bm25_idx',
    'commons.data_policy_tx__title_description__field_aware_bm25_idx',
    'commons.data_policy_tx_chunks__content__bm25_idx',
    'commons.data_policy_wa__title_description__field_aware_bm25_idx',
    'commons.data_policy_wa_chunks__content__bm25_idx',
    'commons.sys_chunks__content__bm25_idx',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Atomically promote, roll back, or finalize one Evoke candidate. '
            'Promotion keeps the previous stable index under a rollback name; '
            'an explicit bootstrap publishes the first stable index.'
        ),
    )
    parser.add_argument('action', choices=('promote', 'rollback', 'finalize'))
    parser.add_argument('--dsn', default=os.environ.get('DATABASE_URL', ''))
    parser.add_argument('--psql', default=shutil.which('psql') or 'psql')
    parser.add_argument('--stable', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--rollback', required=True)
    parser.add_argument(
        '--qualification-evidence',
        action='append',
        type=Path,
        default=[],
        help='CQ-7 matrix JSON; promotion requires concurrency 1, 4, and 8',
    )
    parser.add_argument(
        '--release-evidence',
        type=Path,
        help=(
            'passing evoke_product_maturity JSON for the installed staged '
            'package; required for promotion and finalization'
        ),
    )
    parser.add_argument(
        '--post-swap-evidence',
        type=Path,
        help='passing stable-name matrix required before finalize',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help=(
            'publish or finalize the first stable index when no rollback '
            'index exists; invalid for rollback and ordinary replacement'
        ),
    )
    return parser.parse_args()


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_relation(value: str) -> tuple[str, str]:
    parts = value.split('.')
    if len(parts) != 2 or not all(IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise ValueError(
            f'expected an unquoted schema-qualified identifier, got {value!r}'
        )
    return parts[0], parts[1]


def qualified_name(relation: tuple[str, str]) -> str:
    return f'{quote_ident(relation[0])}.{quote_ident(relation[1])}'


def run_psql(args: argparse.Namespace, sql: str) -> str:
    command = [
        args.psql,
        '-X',
        '-qAt',
        '-v',
        'ON_ERROR_STOP=1',
    ]
    if args.dsn:
        command.extend(('--dbname', args.dsn))
    result = subprocess.run(
        command,
        input=sql,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, 'PGAPPNAME': 'evoke_index_promotion'},
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def query_json(args: argparse.Namespace, sql: str) -> Any:
    output = run_psql(args, sql)
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f'expected one JSON row, got {lines!r}')
    return json.loads(lines[0])


def catalog_snapshot(
    args: argparse.Namespace,
    relation: tuple[str, str],
) -> dict[str, Any] | None:
    schema, name = relation
    return query_json(
        args,
        f"""
        SELECT COALESCE((
            SELECT to_jsonb(snapshot)
            FROM (
                SELECT
                    c.oid::bigint AS oid,
                    n.nspname AS schema_name,
                    c.relname AS index_name,
                    i.indrelid::bigint AS table_oid,
                    table_namespace.nspname AS table_schema,
                    table_class.relname AS table_name,
                    am.amname AS access_method,
                    i.indisvalid AS valid,
                    i.indisready AS ready,
                    i.indislive AS live,
                    COALESCE(c.reloptions, ARRAY[]::text[]) AS reloptions,
                    pg_get_indexdef(c.oid) AS definition
                FROM pg_class AS c
                JOIN pg_namespace AS n ON n.oid = c.relnamespace
                JOIN pg_index AS i ON i.indexrelid = c.oid
                JOIN pg_am AS am ON am.oid = c.relam
                JOIN pg_class AS table_class ON table_class.oid = i.indrelid
                JOIN pg_namespace AS table_namespace
                  ON table_namespace.oid = table_class.relnamespace
                WHERE n.nspname = {sql_literal(schema)}
                  AND c.relname = {sql_literal(name)}
            ) AS snapshot
        ), 'null'::jsonb);
        """,
    )


def extension_schema(args: argparse.Namespace) -> str:
    value = query_json(
        args,
        """
        SELECT to_json(n.nspname)
        FROM pg_extension AS e
        JOIN pg_namespace AS n ON n.oid = e.extnamespace
        WHERE e.extname = 'evoke';
        """,
    )
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise RuntimeError('the evoke extension schema is unavailable')
    return value


def generation_snapshot(
    args: argparse.Namespace,
    relation: tuple[str, str],
) -> dict[str, Any]:
    schema = extension_schema(args)
    relation_literal = sql_literal('.'.join(relation))
    value = query_json(
        args,
        f"""
        SELECT {quote_ident(schema)}.evoke_index_generation_status_internal(
                {relation_literal}::regclass
            )::jsonb;
        """,
    )
    if not isinstance(value, dict):
        raise RuntimeError('invalid Evoke generation status')
    return value


def require_catalog_index(
    snapshot: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    if snapshot is None:
        raise RuntimeError(f'{label} index does not exist')
    if snapshot.get('access_method') != 'evoke':
        raise RuntimeError(f'{label} index is not an Evoke index')
    if not all(
        snapshot.get(field) is True
        for field in ('valid', 'ready', 'live')
    ):
        raise RuntimeError(f'{label} index is not valid, ready, and live')
    return snapshot


def generation_qualified(status: dict[str, Any]) -> bool:
    accelerator = status.get('semantic_accelerator')
    return bool(
        status.get('valid') is True
        and status.get('health') == 'ok'
        and status.get('runtime_contract_matches') is True
        and isinstance(accelerator, dict)
        and accelerator.get('state') == 'ready'
        and accelerator.get('forward_complete') is True
        and accelerator.get('scope_current') is True
        and status.get('generation_id')
    )


def file_evidence(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f'evidence must contain a JSON object: {path}')
    identity = {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }
    return value, identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def current_artifact_identity(path: Path) -> dict[str, str]:
    if path.is_symlink():
        return {'kind': 'symlink', 'target': os.readlink(path)}
    if not path.is_file():
        raise ValueError(f'release artifact is missing: {path}')
    return {'kind': 'file', 'sha256': sha256_file(path)}


def require_current_artifacts(
    artifacts: list[Any],
    label: str,
) -> None:
    if not artifacts:
        raise ValueError(f'release evidence has no {label} artifacts')
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError(f'invalid {label} artifact evidence')
        for side in ('staged', 'installed'):
            path_value = artifact.get(f'{side}_path')
            expected = artifact.get(side)
            if not isinstance(path_value, str) or not isinstance(
                expected,
                dict,
            ):
                raise ValueError(
                    f'incomplete {label} artifact evidence: '
                    f'{artifact.get("name")}'
                )
            actual = current_artifact_identity(Path(path_value))
            if actual != expected:
                raise ValueError(
                    f'{label} artifact changed after qualification: '
                    f'{path_value}'
                )


def validate_release_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError('promotion requires --release-evidence')
    document, identity = file_evidence(path)
    binding = document.get('package_binding')
    if (
        document.get('suite') != 'evoke_product_maturity'
        or document.get('passed') is not True
        or not isinstance(binding, dict)
        or binding.get('stable') is not True
    ):
        raise ValueError('release evidence is not a passing maturity suite')
    preflight = binding.get('preflight')
    postflight = binding.get('postflight')
    if not isinstance(preflight, dict) or not isinstance(postflight, dict):
        raise ValueError('release evidence has no package binding snapshots')
    if (
        preflight.get('passed') is not True
        or postflight.get('passed') is not True
        or not preflight.get('fingerprint')
        or preflight.get('fingerprint') != postflight.get('fingerprint')
    ):
        raise ValueError('release package binding is not stable')
    build_info = postflight.get('build_info')
    model_binding = postflight.get('model_binding')
    expected_onnxruntime = pinned_onnxruntime_version()
    if (
        not isinstance(build_info, dict)
        or build_info.get('Git tree') != 'clean'
        or not build_info.get('Git commit')
        or not build_info.get('Version')
        or build_info.get('ONNX Runtime') != expected_onnxruntime
        or build_info.get('ONNX Runtime linkage') != (
            'bundled in PostgreSQL pkglibdir'
        )
        or not isinstance(model_binding, dict)
        or model_binding.get('passed') is not True
        or not model_binding.get('model_id')
        or not model_binding.get('manifest_sha256')
    ):
        raise ValueError('release source or model identity is incomplete')
    require_current_artifacts(
        postflight.get('artifacts', []),
        'package',
    )
    require_current_artifacts(
        model_binding.get('artifacts', []),
        'model',
    )
    package_root = postflight.get('package_root')
    build_info_sha256 = postflight.get('build_info_sha256')
    if not isinstance(package_root, str) or not isinstance(
        build_info_sha256,
        str,
    ):
        raise ValueError('release BUILD-INFO identity is incomplete')
    build_info_path = Path(package_root) / 'BUILD-INFO.txt'
    if current_artifact_identity(build_info_path) != {
        'kind': 'file',
        'sha256': build_info_sha256,
    }:
        raise ValueError('release BUILD-INFO changed after qualification')
    identity.update({
        'fingerprint': postflight['fingerprint'],
        'git_commit': build_info['Git commit'],
        'extension_version': build_info['Version'],
        'onnxruntime_version': expected_onnxruntime,
        'model_id': model_binding['model_id'],
        'model_manifest_sha256': model_binding['manifest_sha256'],
    })
    return identity


def validate_qualification_evidence(
    paths: list[Path],
    candidate_name: str,
    release_version: str | None = None,
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError('promotion requires --qualification-evidence')
    identities = []
    observed_concurrency = set()
    generation_ids = set()
    commons_generation_sets: list[dict[str, str]] = []
    serial_rank_oracles: set[str] = set()
    for path in paths:
        document, identity = file_evidence(path)
        environment = document.get('environment')
        cases = document.get('cases')
        if (
            document.get('suite') != 'commons_query_matrix'
            or document.get('contract_version') != (
                COMMONS_MATRIX_CONTRACT_VERSION
            )
            or document.get('mode') != 'qualification'
            or document.get('passed') is not True
            or not isinstance(document.get('repeats'), int)
            or int(document['repeats']) < 2
            or not isinstance(environment, dict)
            or environment.get('qualified') is not True
            or environment.get('catalog_qualified') is not True
            or environment.get('onnxruntime_qualified') is not True
            or environment.get('schema_qualified') is not True
            or environment.get('generation_qualified') is not True
            or environment.get('planner_qualified') is not True
            or environment.get('runtime_qualified') is not True
            or environment.get('planner_capture_qualified') is not True
            or environment.get('query_metadata_qualified') is not True
            or not restart_evidence_qualified(environment)
            or environment.get('pubmed_index') != candidate_name
            or (
                release_version is not None
                and environment.get('extension_version') != release_version
            )
            or not isinstance(cases, list)
            or not cases
            or not all(case.get('passed') is True for case in cases)
        ):
            raise ValueError(f'unqualified Commons matrix: {path}')
        case_names = {
            str(case.get('name'))
            for case in cases
            if isinstance(case, dict) and case.get('name') is not None
        }
        if (
            len(cases) != len(REQUIRED_COMMONS_CASES)
            or case_names != REQUIRED_COMMONS_CASES
        ):
            raise ValueError(
                'Commons matrix does not cover the current contract case set: '
                f'{path}; count={len(cases)}; '
                f'missing={sorted(REQUIRED_COMMONS_CASES - case_names)}; '
                f'unexpected={sorted(case_names - REQUIRED_COMMONS_CASES)}'
            )
        concurrency = int(document.get('concurrency', 0))
        if not all(
            case_has_cold_and_warm_panel(case, concurrency)
            for case in cases
        ):
            raise ValueError(
                'Commons matrix lacks separate cold and warm panels: '
                f'{path}'
            )
        if not all(case_result_stability_qualified(case) for case in cases):
            raise ValueError(
                f'Commons matrix lacks stable repeated results: {path}'
            )
        observed_concurrency.add(concurrency)
        index_generations = commons_index_generations(
            environment,
            candidate_name,
        )
        index_status = environment['indexes'][candidate_name]
        generation_id = index_status.get('generation_id')
        if not generation_id:
            raise ValueError(f'missing candidate generation identity: {path}')
        generation_ids.add(str(generation_id))
        if concurrency == 1:
            rank_oracles = {
                str(case.get('name')): case['rank_oracle']
                for case in cases
                if isinstance(case.get('rank_oracle'), dict)
            }
            missing = REQUIRED_SERIAL_RANK_ORACLE_CASES - rank_oracles.keys()
            if missing or not all(
                oracle.get('passed') is True
                for oracle in rank_oracles.values()
            ):
                raise ValueError(
                    'serial rank oracle set did not pass: '
                    f'{path}; missing={sorted(missing)}'
                )
            serial_rank_oracles.update(rank_oracles)
        identity['concurrency'] = concurrency
        identity['generation_id'] = str(generation_id)
        identity['index_generations'] = index_generations
        commons_generation_sets.append(index_generations)
        identities.append(identity)
    if observed_concurrency != REQUIRED_CONCURRENCIES:
        raise ValueError(
            'qualification evidence must contain concurrency 1, 4, and 8; '
            f'got {sorted(observed_concurrency)}'
        )
    if len(identities) != len(REQUIRED_CONCURRENCIES):
        raise ValueError(
            'qualification evidence must contain exactly one matrix for '
            'each concurrency 1, 4, and 8'
        )
    if (
        len(generation_ids) != 1
        or not REQUIRED_SERIAL_RANK_ORACLE_CASES.issubset(
            serial_rank_oracles
        )
    ):
        raise ValueError('qualification evidence does not identify one generation')
    if any(
        generation_set != commons_generation_sets[0]
        for generation_set in commons_generation_sets[1:]
    ):
        raise ValueError(
            'qualification evidence does not identify one complete Commons '
            'generation set'
        )
    return identities


def validate_post_swap_evidence(
    path: Path | None,
    stable_name: str,
    release_version: str | None = None,
) -> dict[str, Any]:
    if path is None:
        raise ValueError('finalize requires --post-swap-evidence')
    document, identity = file_evidence(path)
    environment = document.get('environment')
    cases = document.get('cases')
    if (
        document.get('suite') != 'commons_query_matrix'
        or document.get('contract_version') != (
            COMMONS_MATRIX_CONTRACT_VERSION
        )
        or document.get('mode') != 'qualification'
        or document.get('passed') is not True
        or document.get('concurrency') != 1
        or not isinstance(document.get('repeats'), int)
        or int(document['repeats']) < 2
        or not isinstance(environment, dict)
        or environment.get('qualified') is not True
        or environment.get('catalog_qualified') is not True
        or environment.get('onnxruntime_qualified') is not True
        or environment.get('schema_qualified') is not True
        or environment.get('generation_qualified') is not True
        or environment.get('planner_qualified') is not True
        or environment.get('runtime_qualified') is not True
        or environment.get('planner_capture_qualified') is not True
        or environment.get('query_metadata_qualified') is not True
        or not restart_evidence_qualified(environment)
        or environment.get('pubmed_index') != stable_name
        or environment.get('pubmed_index_source') != 'stable'
        or (
            release_version is not None
            and environment.get('extension_version') != release_version
        )
        or not isinstance(cases, list)
        or not cases
        or not all(case.get('passed') is True for case in cases)
    ):
        raise ValueError('post-swap evidence is not a passing stable-name matrix')
    case_names = {
        str(case.get('name'))
        for case in cases
        if isinstance(case, dict) and case.get('name') is not None
    }
    if (
        len(cases) != len(REQUIRED_COMMONS_CASES)
        or case_names != REQUIRED_COMMONS_CASES
    ):
        raise ValueError(
            'post-swap matrix does not cover the current contract case set: '
            f'count={len(cases)}; '
            f'missing={sorted(REQUIRED_COMMONS_CASES - case_names)}; '
            f'unexpected={sorted(case_names - REQUIRED_COMMONS_CASES)}'
        )
    if not all(case_has_cold_and_warm_panel(case, 1) for case in cases):
        raise ValueError('post-swap matrix lacks separate cold and warm panels')
    if not all(case_result_stability_qualified(case) for case in cases):
        raise ValueError(
            'post-swap matrix lacks stable repeated results'
        )
    rank_oracles = {
        str(case.get('name')): case['rank_oracle']
        for case in cases
        if isinstance(case.get('rank_oracle'), dict)
    }
    missing = REQUIRED_SERIAL_RANK_ORACLE_CASES - rank_oracles.keys()
    if missing or not all(
        oracle.get('passed') is True for oracle in rank_oracles.values()
    ):
        raise ValueError(
            'post-swap serial rank oracle set did not pass: '
            f'missing={sorted(missing)}'
        )
    index_generations = commons_index_generations(environment, stable_name)
    index_status = environment['indexes'][stable_name]
    generation_id = index_status.get('generation_id')
    if not generation_id:
        raise ValueError('post-swap evidence has no stable generation identity')
    identity['generation_id'] = str(generation_id)
    identity['index_generations'] = index_generations
    return identity


def case_has_cold_and_warm_panel(
    case: dict[str, Any],
    concurrency: int,
) -> bool:
    first = case.get('first')
    warm = case.get('warm')
    return bool(
        concurrency > 0
        and isinstance(first, dict)
        and int(first.get('count', 0)) == concurrency
        and isinstance(warm, dict)
        and int(warm.get('count', 0)) >= concurrency
    )


def case_result_stability_qualified(case: dict[str, Any]) -> bool:
    first = case.get('first')
    warm = case.get('warm')
    stability = case.get('result_stability')
    if not all(isinstance(value, dict) for value in (
        first,
        warm,
        stability,
    )):
        return False
    expected_attempts = int(first.get('count', 0)) + int(
        warm.get('count', 0)
    )
    return bool(
        stability.get('passed') is True
        and expected_attempts > 0
        and int(stability.get('attempts', 0)) == expected_attempts
    )


def commons_index_generations(
    environment: dict[str, Any],
    pubmed_index: str,
) -> dict[str, str]:
    indexes = environment.get('indexes')
    expected = REQUIRED_COMMONS_FIXED_INDEXES | {pubmed_index}
    if not isinstance(indexes, dict) or set(indexes) != expected:
        observed = set(indexes) if isinstance(indexes, dict) else set()
        raise ValueError(
            'Commons matrix does not identify the complete index set: '
            f'missing={sorted(expected - observed)}; '
            f'unexpected={sorted(observed - expected)}'
        )
    generations: dict[str, str] = {}
    for index_name in sorted(expected):
        status = indexes[index_name]
        generation_id = (
            status.get('generation_id')
            if isinstance(status, dict)
            else None
        )
        if not generation_id:
            raise ValueError(
                f'missing generation identity for Commons index {index_name}'
            )
        generations[index_name] = str(generation_id)
    return generations


def restart_evidence_qualified(environment: dict[str, Any]) -> bool:
    requirement = environment.get('restart_requirement_seconds')
    age = environment.get('postmaster_age_seconds')
    start = environment.get('postmaster_start_epoch')
    return bool(
        environment.get('restart_qualified') is True
        and isinstance(requirement, (int, float))
        and 0.0 < float(requirement) <= MAX_RESTART_EVIDENCE_AGE_SECONDS
        and isinstance(age, (int, float))
        and 0.0 <= float(age) <= float(requirement)
        and isinstance(start, (int, float))
        and float(start) > 0.0
    )


def live_commons_index_generations(
    args: argparse.Namespace,
    pubmed_index: str,
) -> dict[str, str]:
    expected = REQUIRED_COMMONS_FIXED_INDEXES | {pubmed_index}
    generations: dict[str, str] = {}
    for index_name in sorted(expected):
        status = generation_snapshot(args, parse_relation(index_name))
        if not generation_qualified(status):
            raise RuntimeError(
                f'live Commons index is not qualified: {index_name}'
            )
        generations[index_name] = str(status['generation_id'])
    return generations


def validate_names(
    stable: tuple[str, str],
    candidate: tuple[str, str],
    rollback: tuple[str, str],
) -> None:
    if len({stable, candidate, rollback}) != 3:
        raise ValueError('stable, candidate, and rollback names must be distinct')
    if len({stable[0], candidate[0], rollback[0]}) != 1:
        raise ValueError('stable, candidate, and rollback must share one schema')


def rename_sql(source: tuple[str, str], destination: tuple[str, str]) -> str:
    return (
        f'ALTER INDEX {qualified_name(source)} '
        f'RENAME TO {quote_ident(destination[1])};'
    )


def identity_guard_sql(expected_oids: dict[str, tuple[str, int | None]]) -> str:
    checks = []
    for label, (name, expected_oid) in expected_oids.items():
        actual = f'to_regclass({sql_literal(name)})::oid'
        if expected_oid is None:
            condition = f'{actual} IS NULL'
        else:
            condition = f'{actual} = {int(expected_oid)}::oid'
        checks.append(
            f"IF NOT ({condition}) THEN RAISE EXCEPTION "
            f"'Evoke promotion identity changed: {label}'; END IF;"
        )
    return 'DO $evoke$ BEGIN ' + ' '.join(checks) + ' END $evoke$;'


def transaction_sql(statements: list[str], stable_name: str) -> str:
    lock_name = f'evoke-index-promotion:{stable_name}'
    return '\n'.join([
        'BEGIN;',
        (
            'SELECT pg_advisory_xact_lock('
            f'hashtextextended({sql_literal(lock_name)}, 0));'
        ),
        *statements,
        'COMMIT;',
    ])


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    stable = parse_relation(args.stable)
    candidate = parse_relation(args.candidate)
    rollback = parse_relation(args.rollback)
    validate_names(stable, candidate, rollback)
    if args.bootstrap and args.action == 'rollback':
        raise ValueError('--bootstrap is invalid for rollback')

    before = {
        'stable': catalog_snapshot(args, stable),
        'candidate': catalog_snapshot(args, candidate),
        'rollback': catalog_snapshot(args, rollback),
    }
    qualification_evidence: list[dict[str, Any]] = []
    post_swap_evidence: dict[str, Any] | None = None
    release_evidence: dict[str, Any] | None = None
    before_generation: dict[str, Any] | None = None
    after_generation: dict[str, Any] | None = None
    live_commons_generations: dict[str, str] | None = None
    stable_name = '.'.join(stable)
    candidate_name = '.'.join(candidate)

    if args.action == 'promote':
        release_evidence = validate_release_evidence(args.release_evidence)
        candidate_snapshot = require_catalog_index(
            before['candidate'],
            'candidate',
        )
        if before['rollback'] is not None:
            raise RuntimeError('rollback index already exists')
        stable_snapshot = before['stable']
        if args.bootstrap:
            if stable_snapshot is not None:
                raise RuntimeError(
                    'bootstrap requires the stable index name to be absent'
                )
        else:
            stable_snapshot = require_catalog_index(
                stable_snapshot,
                'stable',
            )
            if stable_snapshot['table_oid'] != candidate_snapshot['table_oid']:
                raise RuntimeError(
                    'stable and candidate indexes own different tables'
                )
        before_generation = generation_snapshot(args, candidate)
        if not generation_qualified(before_generation):
            raise RuntimeError('candidate generation is not qualified')
        qualification_evidence = validate_qualification_evidence(
            args.qualification_evidence,
            candidate_name,
            release_evidence['extension_version'],
        )
        live_commons_generations = live_commons_index_generations(
            args,
            candidate_name,
        )
        if live_commons_generations != qualification_evidence[0][
            'index_generations'
        ]:
            raise RuntimeError(
                'live Commons generations changed after qualification'
            )
        statements = [identity_guard_sql({
            'stable': (
                stable_name,
                None if args.bootstrap else stable_snapshot['oid'],
            ),
            'candidate': (candidate_name, candidate_snapshot['oid']),
            'rollback': ('.'.join(rollback), None),
        })]
        if not args.bootstrap:
            statements.append(rename_sql(stable, rollback))
        statements.append(rename_sql(candidate, stable))
        expected = {
            'stable_oid': candidate_snapshot['oid'],
            'rollback_oid': (
                None if args.bootstrap else stable_snapshot['oid']
            ),
            'candidate_oid': None,
        }
    elif args.action == 'rollback':
        stable_snapshot = require_catalog_index(before['stable'], 'stable')
        rollback_snapshot = require_catalog_index(
            before['rollback'],
            'rollback',
        )
        if before['candidate'] is not None:
            raise RuntimeError('candidate name must be free before rollback')
        if stable_snapshot['table_oid'] != rollback_snapshot['table_oid']:
            raise RuntimeError('stable and rollback indexes own different tables')
        statements = [
            identity_guard_sql({
                'stable': (stable_name, stable_snapshot['oid']),
                'candidate': (candidate_name, None),
                'rollback': ('.'.join(rollback), rollback_snapshot['oid']),
            }),
            rename_sql(stable, candidate),
            rename_sql(rollback, stable),
        ]
        expected = {
            'stable_oid': rollback_snapshot['oid'],
            'rollback_oid': None,
            'candidate_oid': stable_snapshot['oid'],
        }
    else:
        release_evidence = validate_release_evidence(args.release_evidence)
        stable_snapshot = require_catalog_index(before['stable'], 'stable')
        if before['candidate'] is not None:
            raise RuntimeError('candidate name must be absent before finalize')
        rollback_snapshot = before['rollback']
        if args.bootstrap:
            if rollback_snapshot is not None:
                raise RuntimeError(
                    'bootstrap finalization requires no rollback index'
                )
        else:
            rollback_snapshot = require_catalog_index(
                rollback_snapshot,
                'rollback',
            )
            if stable_snapshot['table_oid'] != rollback_snapshot['table_oid']:
                raise RuntimeError(
                    'stable and rollback indexes own different tables'
                )
        before_generation = generation_snapshot(args, stable)
        if not generation_qualified(before_generation):
            raise RuntimeError('stable generation is not qualified')
        post_swap_evidence = validate_post_swap_evidence(
            args.post_swap_evidence,
            stable_name,
            release_evidence['extension_version'],
        )
        live_commons_generations = live_commons_index_generations(
            args,
            stable_name,
        )
        if live_commons_generations != post_swap_evidence[
            'index_generations'
        ]:
            raise RuntimeError(
                'live Commons generations changed after post-swap qualification'
            )
        if (
            post_swap_evidence['generation_id']
            != str(before_generation.get('generation_id'))
        ):
            raise RuntimeError(
                'post-swap evidence identifies a different stable generation'
            )
        statements = [identity_guard_sql({
            'stable': (stable_name, stable_snapshot['oid']),
            'candidate': (candidate_name, None),
            'rollback': (
                '.'.join(rollback),
                None if args.bootstrap else rollback_snapshot['oid'],
            ),
        })]
        if not args.bootstrap:
            statements.append(f'DROP INDEX {qualified_name(rollback)};')
        expected = {
            'stable_oid': stable_snapshot['oid'],
            'rollback_oid': None,
            'candidate_oid': None,
        }

    sql = transaction_sql(statements, stable_name)
    if not args.dry_run:
        run_psql(args, sql)
    after = before if args.dry_run else {
        'stable': catalog_snapshot(args, stable),
        'candidate': catalog_snapshot(args, candidate),
        'rollback': catalog_snapshot(args, rollback),
    }
    if not args.dry_run:
        observed = {
            'stable_oid': after['stable']['oid'] if after['stable'] else None,
            'candidate_oid': (
                after['candidate']['oid'] if after['candidate'] else None
            ),
            'rollback_oid': (
                after['rollback']['oid'] if after['rollback'] else None
            ),
        }
        if observed != expected:
            raise RuntimeError(
                f'post-action identity mismatch: expected {expected}, got {observed}'
            )
        if args.action in ('promote', 'finalize'):
            after_generation = generation_snapshot(args, stable)
            if not generation_qualified(after_generation):
                raise RuntimeError('post-action stable generation is not qualified')
            if (
                before_generation is not None
                and after_generation.get('generation_id')
                != before_generation.get('generation_id')
            ):
                raise RuntimeError('promotion changed the candidate generation')

    evidence = {
        'suite': 'evoke_index_promotion',
        'action': args.action,
        'bootstrap': args.bootstrap,
        'dry_run': args.dry_run,
        'recorded_at_epoch': time.time(),
        'names': {
            'stable': stable_name,
            'candidate': candidate_name,
            'rollback': '.'.join(rollback),
        },
        'before': before,
        'after': after,
        'expected': expected,
        'before_generation': before_generation,
        'after_generation': after_generation,
        'qualification_evidence': qualification_evidence,
        'post_swap_evidence': post_swap_evidence,
        'release_evidence': release_evidence,
        'live_commons_generations': live_commons_generations,
        'transaction_sha256': hashlib.sha256(sql.encode()).hexdigest(),
    }
    write_evidence(args.output, evidence)
    print(json.dumps({
        'action': args.action,
        'dry_run': args.dry_run,
        'output': str(args.output),
        'expected': expected,
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
