#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = REPO_ROOT / 'ii42.control'
CONTROL_TEXT = CONTROL_PATH.read_text(encoding='utf-8')
VERSION_MATCH = re.search(
    r"^default_version = '([^']+)'$",
    CONTROL_TEXT,
    re.MULTILINE,
)
if VERSION_MATCH is None:
    raise RuntimeError('ii42.control has no default_version')
CURRENT_VERSION = VERSION_MATCH.group(1)
CURRENT_SQL_PATH = REPO_ROOT / f'sql/ii42--{CURRENT_VERSION}.sql'
MAKEFILE_PATH = REPO_ROOT / 'Makefile'
CMAKE_PATH = REPO_ROOT / 'CMakeLists.txt'
DOCKERFILE_PATH = (
    REPO_ROOT / 'packaging/docker/postgres18/Dockerfile'
)
RELEASE_ZIP_PATH = REPO_ROOT / 'scripts/build_release_zip.sh'
RELEASE_DOCKER_PATH = REPO_ROOT / 'scripts/build_release_docker_image.sh'
MILESTONE_MODEL_LOCK_PATH = REPO_ROOT / 'packaging/milestone-model.json'
MILESTONE_MODEL_NOTICE_PATH = (
    REPO_ROOT / 'packaging/MILESTONE-MODEL-NOTICE'
)
MILESTONE_MODEL_VALIDATOR_PATH = (
    REPO_ROOT / 'scripts/validate_milestone_model_checkout.py'
)
MILESTONE_MODEL_ARCHIVER_PATH = (
    REPO_ROOT / 'scripts/build_milestone_model_archive.py'
)
MILESTONE_MODEL_FETCHER_PATH = (
    REPO_ROOT / 'scripts/fetch_milestone_model.py'
)
PROJECT_LICENSE_PATH = REPO_ROOT / 'LICENSE'
SYNC_PUBLIC_WORKFLOW_PATH = REPO_ROOT / '.github/workflows/sync-public.yml'
ONNXRUNTIME_VERSION_PATH = REPO_ROOT / 'packaging/onnxruntime.version'
ONNXRUNTIME_CHECKSUM_PATH = REPO_ROOT / 'packaging/onnxruntime.sha256'
ONNXRUNTIME_INSTALLER_PATH = REPO_ROOT / 'scripts/install_onnxruntime_c.sh'
POSTGRES_APT_INSTALLER_PATH = REPO_ROOT / 'scripts/install_postgres_apt.sh'
AM_SOURCE_PATH = REPO_ROOT / 'src/ii42_am.c'
AM_BUILD_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_build.c'
AM_BUILD_HEADER_PATH = REPO_ROOT / 'src/ii42_am_build.h'
AM_HOT_FOLD_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_hot_fold.c'
AM_HOT_FOLD_HEADER_PATH = REPO_ROOT / 'src/ii42_am_hot_fold.h'
AM_MAINTENANCE_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_maintenance.c'
AM_MAINTENANCE_HEADER_PATH = REPO_ROOT / 'src/ii42_am_maintenance.h'
AM_META_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_meta.c'
AM_META_HEADER_PATH = REPO_ROOT / 'src/ii42_am_meta.h'
AM_MUTATION_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_mutation.c'
AM_MUTATION_HEADER_PATH = REPO_ROOT / 'src/ii42_am_mutation.h'
AM_OPTIONS_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_options.c'
AM_OPTIONS_HEADER_PATH = REPO_ROOT / 'src/ii42_am_options.h'
AM_PRELOAD_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_preload.c'
AM_PRELOAD_HEADER_PATH = REPO_ROOT / 'src/ii42_am_preload.h'
AM_RECLAMATION_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_reclamation.c'
AM_RECLAMATION_HEADER_PATH = REPO_ROOT / 'src/ii42_am_reclamation.h'
AM_SCHEDULER_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_scheduler.c'
AM_SCHEDULER_HEADER_PATH = REPO_ROOT / 'src/ii42_am_scheduler.h'
AM_SCAN_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_scan.c'
AM_SCAN_HEADER_PATH = REPO_ROOT / 'src/ii42_am_scan.h'
AM_SQL_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_sql.c'
AM_TEST_SUPPORT_SOURCE_PATH = REPO_ROOT / 'src/ii42_am_test_support.c'
AM_TEST_SUPPORT_HEADER_PATH = REPO_ROOT / 'src/ii42_am_test_support.h'
PG_COMMON_SOURCE_PATH = REPO_ROOT / 'src/ii42_pg_common.c'
PG_COMMON_HEADER_PATH = REPO_ROOT / 'src/ii42_pg_common.h'
SEMANTIC_SOURCE_PATH = REPO_ROOT / 'src/ii42_semantic.c'
SEMANTIC_HEADER_PATH = REPO_ROOT / 'src/ii42_semantic.h'
P2_RUNTIME_SOURCE_PATH = REPO_ROOT / 'src/ii42_p2_runtime.c'
RUNTIME_SERVICE_HEADER_PATH = REPO_ROOT / 'src/ii42_runtime_service.h'
ARCHIVED_DEVELOPMENT_RECORD_PATH = (
    REPO_ROOT
    / 'docs/archive/engineering/'
    'convergent-segmented-index-development-record.md'
)
ARCHIVED_RELEASE_READINESS_PATH = (
    REPO_ROOT
    / 'docs/archive/engineering/'
    'eventual-sae-release-readiness-history.md'
)
CONVERGENT_DESIGN_PATH = (
    REPO_ROOT / 'docs/convergent-segmented-index.md'
)
STORAGE_SOURCE_PATH = REPO_ROOT / 'src/ii42_storage.c'
STORAGE_HEADER_PATH = REPO_ROOT / 'src/ii42_storage.h'
QUERY_SOURCE_PATH = REPO_ROOT / 'src/ii42_query.c'
QUERY_HEADER_PATH = REPO_ROOT / 'src/ii42_query.h'
CORE_SOURCE_PATH = REPO_ROOT / 'src/ii42_core.c'
CORE_HEADER_PATH = REPO_ROOT / 'src/ii42_core.h'
SEGMENTS_SOURCE_PATH = REPO_ROOT / 'src/ii42_segments.c'
SEGMENTS_HEADER_PATH = REPO_ROOT / 'src/ii42_segments.h'
SEGMENT_PAGES_SOURCE_PATH = REPO_ROOT / 'src/ii42_segment_pages.c'
PAGE_QUERY_SOURCE_PATH = REPO_ROOT / 'src/ii42_page_query.c'
DOCUMENT_COW_SOURCE_PATH = REPO_ROOT / 'src/ii42_document_cow.c'
LIFECYCLE_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_unified_index_lifecycle_smoke.py'
)
CONVERGENT_SAE_LIFECYCLE_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_convergent_sae_lifecycle_smoke.py'
)
SAME_INDEX_CONCURRENCY_PATH = (
    REPO_ROOT / 'scripts/test_same_index_writer_concurrency_temp_pg.py'
)
CACHE_FAILURE_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_cache_failure_safety_temp_pg.py'
)
VACUUM_FRONTIER_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_convergent_vacuum_frontier_smoke.py'
)
STORAGE_LAYOUT_BOUNDARY_PATH = (
    REPO_ROOT / 'scripts/test_storage_layout_boundary.py'
)
METAPAGE_READ_BOUNDARY_PATH = (
    REPO_ROOT / 'scripts/test_metapage_read_boundary.py'
)
PAYLOAD_HEALTH_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_payload_health_corruption_smoke.py'
)
SCHEMA_SMOKE_PATH = REPO_ROOT / 'scripts/test_extension_schema_smoke.py'
REPLICATION_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_replication_lifecycle_smoke.py'
)
SOURCE_MIGRATION_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_psql_bm25s_source_migration_smoke.py'
)
MATURITY_SUITE_PATH = REPO_ROOT / 'scripts/run_product_maturity_suite.py'
BUMP_VERSION_PATH = REPO_ROOT / 'scripts/bump_extension_version.py'
REBUILD_INDEXES_PATH = REPO_ROOT / 'scripts/rebuild_ii42_indexes.py'
COMMONS_MATRIX_PATH = (
    REPO_ROOT / 'scripts/benchmark_commons_query_matrix.py'
)
PRIVILEGE_SMOKE_PATH = (
    REPO_ROOT / 'scripts/test_runtime_service_privilege_smoke.py'
)
MODEL_MANIFEST_ENTRYPOINTS = (
    REPO_ROOT / 'scripts/benchmark_ii42_product_path.py',
    REPO_ROOT / 'scripts/compile_p2_b1125_query.py',
    REPO_ROOT / 'scripts/create_p2_runtime_variant.py',
    REPO_ROOT / 'scripts/test_concurrent_ddl_lifecycle_smoke.py',
    REPO_ROOT / 'scripts/test_convergent_sae_lifecycle_smoke.py',
    REPO_ROOT / 'scripts/test_empty_unlogged_lifecycle_smoke.py',
    REPO_ROOT / 'scripts/test_model_lifecycle_medium_perf.py',
    REPO_ROOT / 'scripts/test_onnxruntime_provider_matrix.py',
    REPO_ROOT / 'scripts/test_onnxruntime_resource_soak.py',
    REPO_ROOT / 'scripts/test_unified_index_lifecycle_smoke.py',
    REPO_ROOT / 'scripts/test_replication_lifecycle_smoke.py',
    REPO_ROOT / 'scripts/test_runtime_service_privilege_smoke.py',
    REPO_ROOT / 'scripts/test_runtime_service_required_smoke.py',
    REPO_ROOT / 'scripts/test_runtime_service_restart_smoke.py',
    REPO_ROOT / 'scripts/test_runtime_service_temp_pg.py',
    REPO_ROOT / 'scripts/validate_p2_native_runtime_parity.py',
)
RESEARCH_SAE_ROOT = REPO_ROOT / 'docs/research-sae'
RESEARCH_SAE_README_PATH = RESEARCH_SAE_ROOT / 'README.md'
RESEARCH_ARTIFACT_README_PATH = RESEARCH_SAE_ROOT / 'artifacts/README.md'
RESEARCH_ARTIFACT_MANIFEST_PATH = (
    RESEARCH_SAE_ROOT / 'artifacts/manifest.jsonl'
)
RESEARCH_SOURCE_README_PATH = RESEARCH_SAE_ROOT / 'source/README.md'
RESEARCH_SOURCE_MANIFEST_PATH = RESEARCH_SAE_ROOT / 'source/manifest.jsonl'
RESEARCH_REPORT_ROOT = RESEARCH_SAE_ROOT / 'reports'

PRODUCT_DOCS = (
    'README.md',
    'docs/README.md',
    'docs/getting-started.md',
    'docs/api-reference.md',
    'docs/architecture-and-design.md',
    'docs/functions.md',
    'docs/index-parameters.md',
    'docs/query-semantics.md',
    'docs/testing-and-validation.md',
    'docs/upgrading.md',
    'docs/examples/semantic-index-quickstart.md',
    'docs/examples/semantic-index-operations.md',
    'docs/examples/semantic-query-api.md',
)

CURRENT_LIFECYCLE_DOCS = (
    'README.md',
    'docs/architecture-and-design.md',
    'docs/index-parameters.md',
    'docs/index-policy.md',
    'docs/query-semantics.md',
    'docs/maintenance-lifecycle.md',
)

RETIRED_LIFECYCLE_DOC_PHRASES = (
    'prepared and index-bound query values',
    'EVT-4` first-beta blocker',
    'Automatic-policy mutations compile complete unified rows',
    'Automatic semantic-enabled policies publish one complete',
    'not query-compatible in this test line',
)

SAE_EVENTUAL_ONLY_DOCS = (
    'README.md',
    'docs/README.md',
    'docs/architecture-and-design.md',
    'docs/index-parameters.md',
    'docs/index-policy.md',
    'docs/query-semantics.md',
    'docs/upgrading.md',
    'docs/examples/semantic-index-quickstart.md',
    'docs/examples/semantic-index-operations.md',
    'docs/examples/semantic-query-api.md',
    'docs/examples/semantic-runtime.md',
)

PAGE_NATIVE_V3_DOCS = (
    'README.md',
    'docs/README.md',
    'docs/api-reference.md',
    'docs/architecture-and-design.md',
    'docs/connection-memory.md',
    'docs/index-parameters.md',
    'docs/index-policy.md',
    'docs/query-semantics.md',
    'docs/shared-runtime-and-residency.md',
    'docs/testing-and-validation.md',
    'docs/examples/semantic-index-operations.md',
)

STALE_PAGE_NATIVE_DOC_PHRASES = (
    'arena owns mutable compiled unified-delta caches',
    'model sessions and mutable unified-delta caches',
    'mutable unified-delta caches are postmaster-owned',
    'queries traverse an exact shared unified dynamic-tail cache',
    'query path derives one exact unified delta cache',
    'semantic-enabled indexes normally attach only an exact shared unified '
    'delta cache',
    'mutable unified-tail cache has no ordinary backend-local',
    'sae realtime compiles complete unified records synchronously',
    'realtime mutations compile complete unified rows',
    '`sae=true` uses the same three `consistency` values',
)

UNSUPPORTED_SAE_INDEX_SQL_RE = re.compile(
    r'CREATE\s+INDEX[\s\S]{0,1200}?WITH\s*\('
    r'(?=[^)]*\bsae\s*=\s*true\b)'
    r'(?=[^)]*\bconsistency\s*=\s*[\'\"]?'
    r'(?:realtime|manual)[\'\"]?\b)',
    re.IGNORECASE,
)

REQUIRED_PRODUCT_FUNCTIONS = (
    'ii42_query',
    'ii42_index_options',
    'ii42_index_status',
    'ii42_index_maintain',
    'ii42_index_maintain_due',
)

RESTRICTED_QUERY_SIGNATURES = (
    'ii42_query_ids(regclass,int4[],int4,real[])',
    'ii42_query_tokens(regclass,text[],int4,real[])',
    'ii42_field_aware_query_tokens(regclass,text[],text[],real[],int4)',
    'ii42_field_aware_query(regclass,text,text[],real[],int4)',
    (
        'ii42_query_bm25_internal('
        'regclass,text,int4,real[],boolean,text[],boolean,boolean)'
    ),
    'ii42_prepared_query(regclass,text,boolean,text[],boolean,boolean)',
    'ii42_order_tokens(ii42_result_prepared_query)',
    'ii42_order_tokens(regclass,text,boolean,text[],boolean,boolean)',
    (
        'ii42_op_match_prepared_query('
        'text[],ii42_result_prepared_query)'
    ),
    (
        'ii42_op_match_prepared_query('
        'varchar[],ii42_result_prepared_query)'
    ),
    (
        'ii42_op_match_prepared_query_scalar('
        'text,ii42_result_prepared_query)'
    ),
    (
        'ii42_op_match_prepared_query_scalar('
        'varchar,ii42_result_prepared_query)'
    ),
    'ii42_match_prepared_query(text[],ii42_result_prepared_query)',
    'ii42_match_prepared_query(varchar[],ii42_result_prepared_query)',
    'ii42_match_prepared_query(text,ii42_result_prepared_query)',
    'ii42_match_prepared_query(varchar,ii42_result_prepared_query)',
    (
        'ii42_match_query('
        'text[],regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_match_query('
        'varchar[],regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_match_query('
        'text,regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_match_query('
        'varchar,regclass,text,boolean,text[],boolean,boolean)'
    ),
    'ii42_score_prepared_query(text[],ii42_result_prepared_query)',
    'ii42_score_prepared_query(varchar[],ii42_result_prepared_query)',
    'ii42_score_prepared_query(text,ii42_result_prepared_query)',
    'ii42_score_prepared_query(varchar,ii42_result_prepared_query)',
    (
        'ii42_score_query('
        'text[],regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_score_query('
        'varchar[],regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_score_query('
        'text,regclass,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_score_query('
        'varchar,regclass,text,boolean,text[],boolean,boolean)'
    ),
)

PUBLIC_COMPOSITION_SIGNATURES = (
    'ii42_fusion_weighted_query(ii42_result_prepared_query,real)',
    (
        'ii42_fusion_weighted_query('
        'regclass,text,real,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_fusion_weighted_queries('
        'regclass[],text,real[],boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_fusion_field_query('
        'text,ii42_result_fusion_weighted_query)'
    ),
    'ii42_fusion_field_query(text,ii42_result_prepared_query,real)',
    (
        'ii42_fusion_field_query('
        'text,regclass,text,real,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_fusion_field_queries('
        'text[],regclass[],text,real[],boolean,text[],boolean,boolean)'
    ),
    'ii42_fusion(ii42_result_hit[],real,ii42_result_hit[],real,int4)',
    (
        'ii42_fusion_query('
        'text[],regclass[],text,real[],int4,int4,real[],boolean,text[],'
        'boolean,boolean)'
    ),
    (
        'ii42_fusion_query('
        'regclass[],text,real[],int4,int4,real[],boolean,text[],boolean,'
        'boolean)'
    ),
    (
        'ii42_fusion_query_fields('
        'ii42_result_fusion_field_query[],int4,int4,real[])'
    ),
    (
        'ii42_fusion_query_weighted('
        'ii42_result_fusion_weighted_query[],int4,int4,real[])'
    ),
    (
        'ii42_fusion_query_weighted('
        'ii42_result_fusion_weighted_query[],text,int4,int4,real[])'
    ),
    'ii42_hybrid_candidate(text,tid,real,int4,real,text,text)',
    'ii42_hybrid_bm25_candidate(text,tid,real,int4,real,text)',
    'ii42_hybrid_vector_candidate(text,tid,real,int4,real,text)',
    (
        'ii42_hybrid_bm25_candidates('
        'text,regclass,text,real,int4,text,boolean,text[],boolean,boolean)'
    ),
    (
        'ii42_hybrid_fuse_candidates('
        'ii42_result_hybrid_candidate[],int4,text,real,real)'
    ),
)

DISALLOWED_SPLIT_API_PREFIXES = (
    'ii42_model_',
    'ii42_sae_',
    'ii42_index_model_',
)

DISALLOWED_SPLIT_LIFECYCLE_OPTIONS = (
    'generation_table',
    'generation_id',
    'semantic_overlay_table',
    'semantic_debt_table',
    'document_id_column',
    'document_identity_table',
)

RETIRED_PRODUCT_RELOPTIONS = (
    'auto_rebuild_threshold',
    'auto_rebuild_delta_bytes',
    'auto_rebuild_churn_ratio',
    'query_overlay_max_records',
    'query_overlay_max_bytes',
)

RETIRED_PRODUCT_GUCS = (
    'ii42.sae_delta_cache_wait_timeout',
    'ii42.sae_delta_cache_headroom_percent',
    'ii42.sae_local_delta_cache_max_records',
    'ii42.sae_local_delta_cache_max_bytes',
)

ROOT_RESEARCH_ARTIFACT_RE = re.compile(
    r'(?:ii42-m.*\.(?:json|jsonl|md|sha256|txt)|m[0-9].*\.json)'
)
RESEARCH_ARTIFACT_MANIFEST_SHA256 = (
    '50eff315e36899bed4f96560486ab68a7ac5b505bd2d557d445d70a629eaa65e'
)
RESEARCH_SOURCE_MANIFEST_SHA256 = (
    '70e0904f86b1cf15c753857e751f25b8df14151d5439a793036b580639c921cd'
)
RESEARCH_SCRIPT_RE = re.compile(
    r'(?:^research_|^test_research_|(?:^|_)m[0-9]{2,})',
    re.IGNORECASE,
)
VERSIONED_PRODUCT_SCRIPT_RE = re.compile(
    r'(?:^|[_-])v[0-9]+(?:[_-]|\.py$)',
    re.IGNORECASE,
)
RETIRED_SOURCE_PATHS = {
    Path('docs/convergent-segmented-index-design.md'),
    Path('docs/convergent-segmented-index-implementation-plan.md'),
    Path('docs/reliable-maintenance-preload-generations.md'),
    Path('docs/shared-generation-cache.md'),
    Path('docs/examples/model-backed-index-quickstart.md'),
    Path('docs/examples/model-backed-operations-guide.md'),
    Path('docs/examples/model-backed-query-api-reference.md'),
    Path('docs/examples/model-checkout-contract.md'),
    Path('docs/examples/model-runtime-contract.md'),
    Path('scripts/collect_ii42_official_beir15_comparison.py'),
    Path('scripts/benchmark_hybrid_fusion.py'),
    Path('scripts/summarize_p2_beir15_native_matrix.py'),
    Path('scripts/summarize_p2_mteb10_native_matrix.py'),
    Path('scripts/summarize_p2_product_matrix.py'),
    Path('scripts/test_bounded_fast_overlay_smoke.py'),
    Path('tests/test_native_query_filters.py'),
}

RETIRED_LOCAL_SOURCE_ROOTS = (
    '/Users/' 'leask/Documents/II/ii42',
)
RETIRED_RUNTIME_NAME = 'u' + 'bmx'
ACTIVE_TEXT_SUFFIXES = {
    '.c',
    '.control',
    '.h',
    '.md',
    '.py',
    '.sh',
    '.sql',
    '.toml',
    '.yaml',
    '.yml',
}
CURRENT_CONTRACT_DOCS = PRODUCT_DOCS + (
    'CHANGELOG.md',
    'SECURITY.md',
    'docs/performance/README.md',
)
OPEN_SOURCE_ROOT_FILES = (
    'CODE_OF_CONDUCT.md',
    'CONTRIBUTING.md',
    'LICENSE',
    'SECURITY.md',
    'SUPPORT.md',
)
MAINTAINER_LOCAL_PATHS = (
    '/Users/leask',
    '/Volumes/Betty',
    '/home/huoju',
    'spark-1:',
)
HISTORICAL_QUALIFICATION_PATH = (
    REPO_ROOT
    / 'docs/performance/reports/'
    'enlightenment-p2-production-qualification-2026-07-22.md'
)
HISTORICAL_QUALIFICATION_EVIDENCE_README = (
    REPO_ROOT
    / 'docs/performance/data/raw/'
    'enlightenment-p2-qualification-2026-07-22/README.md'
)
MARKDOWN_INLINE_LINK_RE = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
MARKDOWN_REFERENCE_LINK_RE = re.compile(
    r'^\s*\[[^\]]+\]:\s*(\S+)'
)
MARKDOWN_HTML_LINK_RE = re.compile(
    r'\b(?:href|src)=["\']([^"\']+)["\']'
)
MARKDOWN_HEADING_RE = re.compile(
    r'^\s{0,3}#{1,6}\s+(.+?)\s*$'
)
MARKDOWN_EXPLICIT_ANCHOR_RE = re.compile(
    r'\b(?:id|name)=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
RESEARCH_STAGE_RE = re.compile(r'm(\d{4})-m(\d{4})')
RESEARCH_REPORT_NUMBER_RE = re.compile(
    r'(?:^|-)m(\d{3,4})(?:[a-z]?(?:-|_)|\b)',
    re.IGNORECASE,
)
ROOT_NUMBERED_RESEARCH_RE = re.compile(
    r'^(?:ii42|sae)-m(\d{3,4})(?:-|_)',
    re.IGNORECASE,
)


def function_definition_count(sql: str, name: str) -> int:
    return len(re.findall(rf'CREATE FUNCTION {name}\(', sql))


def function_block(sql: str, name: str, next_name: str) -> str:
    match = re.search(
        rf'CREATE FUNCTION {name}\([\s\S]*?(?=CREATE FUNCTION {next_name}\()',
        sql,
    )
    if match is None:
        return ''
    return match.group(0)


def compact_sql(sql: str) -> str:
    return re.sub(r'\s+', '', sql).lower()


def check_sql_contract(errors: list[str]) -> None:
    sql = CURRENT_SQL_PATH.read_text(encoding='utf-8')
    catalog_contract = "ii42_catalog_v1"
    if (
        'CREATE FUNCTION ii42_catalog_contract_internal()' not in sql
        or f"SELECT '{catalog_contract}'::text" not in sql
        or (
            'REVOKE EXECUTE ON FUNCTION '
            'ii42_catalog_contract_internal() FROM PUBLIC;'
        ) not in sql
    ):
        errors.append(
            f'{CURRENT_SQL_PATH.name}: current catalog identity is missing '
            'or public'
        )
    for path in (COMMONS_MATRIX_PATH, REBUILD_INDEXES_PATH):
        content = path.read_text(encoding='utf-8')
        if f"CATALOG_CONTRACT = '{catalog_contract}'" not in content:
            errors.append(
                f'{path.name}: catalog identity does not match install SQL'
            )

    if 'DROP FUNCTION IF EXISTS' in sql:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: fresh install contains upgrade cleanup'
        )

    for name in REQUIRED_PRODUCT_FUNCTIONS:
        count = function_definition_count(sql, name)
        expected_count = 8 if name == 'ii42_query' else 1
        if count != expected_count:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: expected {expected_count} '
                f'{name} definitions, found {count}'
            )

    removed_sql_apis = (
        'ii42_search',
        'ii42_index_drop',
        'ii42_fast_path_advice',
        'ii42_fast_path_plan',
        'ii42_fast_path_explain',
        'ii42_ranked_query',
        'ii42_query_prepared',
        'ii42_filter_query',
        'ii42_result_ranked_query',
    )
    for name in removed_sql_apis:
        if name in sql:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: removed SQL API remains: {name}'
            )

    single_input_fusion = re.compile(
        r'CREATE\s+FUNCTION\s+ii42_fusion\s*\(\s*'
        r'hits\s+ii42_result_hit\[\]',
        re.IGNORECASE,
    )
    if single_input_fusion.search(sql):
        errors.append(
            f'{CURRENT_SQL_PATH.name}: single-input fusion wrapper remains'
        )

    required = (
        'Unified sparse posting access method for exact BM25 and '
        'model-backed search',
        'CREATE FUNCTION ii42_index_generation_status_internal(',
        "AS 'MODULE_PATHNAME', "
        "'ii42_index_generation_readiness_internal_c'",
        'CREATE FUNCTION ii42_index_generation_audit_internal(',
        "AS 'MODULE_PATHNAME', "
        "'ii42_index_generation_audit_internal_c'",
        "'payload_owner', 'index_relation'",
        "'lifecycle', 'postgresql_index'",
        "WHEN sae_enabled THEN 'semantic'",
        "ELSE 'bm25'",
        'REVOKE EXECUTE ON FUNCTION ii42_runtime_service_query_atoms(',
        'REVOKE EXECUTE ON FUNCTION '
        'ii42_runtime_service_atoms_batch_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_runtime_cache_clear()',
        'REVOKE EXECUTE ON FUNCTION ii42_index_touch_maintenance()',
        'REVOKE EXECUTE ON FUNCTION ii42_index_maintain_due(integer)',
        'REVOKE EXECUTE ON FUNCTION ii42_index_try_maintenance_lock(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_runtime_plan_internal(',
        'REVOKE EXECUTE ON FUNCTION '
        'ii42_index_checkout_validate_internal(',
        'REVOKE EXECUTE ON FUNCTION '
        'ii42_checkout_manifest_signature_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_options_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_runtime_signature_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_generation_status_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_generation_audit_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_index_generation_signature_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_encode_text_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_encode_document_batch_internal(',
        'REVOKE EXECUTE ON FUNCTION '
        'ii42_index_semantic_query_native_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_query_semantic_internal(',
        'REVOKE EXECUTE ON FUNCTION ii42_query_internal(',
        'FROM ii42_index_semantic_query_native_internal(',
        'field_names text[]',
        'field_weights real[]',
        'LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE',
        'PERFORM ii42_index_options(index_name);',
    )
    for needle in required:
        if needle not in sql:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: unified contract lacks {needle!r}'
            )
    sql_checkout_contracts = (
        'ii42_checkout_manifest_signature_internal(manifest)',
        "onnx_result->>'checkout_signature'",
        "'runtime_signature', runtime_signature",
        "encoded->>'runtime_signature'",
        "'$1, $2, $3, $4, $5, $6, $7, $8, $9, $10) AS q '",
        'ii42_index_runtime_signature_internal(index_name)',
        'ii42 model checkout changed during query encoding',
        'ii42 model checkout changed during batch encoding',
    )
    for contract in sql_checkout_contracts:
        if contract not in sql:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: checkout encoding guard is '
                f'missing {contract!r}'
            )

    if 'CREATE FUNCTION ii42_index_semantic_query_internal(' in sql:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: legacy semantic scorer is installed'
        )

    semantic_dispatch_start = sql.find(
        'CREATE FUNCTION ii42_query_semantic_internal('
    )
    semantic_dispatch_end = sql.find(
        'COMMENT ON FUNCTION ii42_query_semantic_internal(',
        semantic_dispatch_start,
    )
    if semantic_dispatch_start < 0 or semantic_dispatch_end < 0:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: semantic dispatch block is missing'
        )
    else:
        semantic_dispatch = sql[
            semantic_dispatch_start:semantic_dispatch_end
        ]
        for forbidden in (
            'ii42_index_generation_status_internal(index_name)',
            'semantic_executor',
            "ELSE 'ii42_index_semantic_query_internal'",
        ):
            if forbidden in semantic_dispatch:
                errors.append(
                    f'{CURRENT_SQL_PATH.name}: semantic dispatch retains '
                    f'legacy storage authority {forbidden!r}'
                )

    compact = compact_sql(sql)
    for signature in RESTRICTED_QUERY_SIGNATURES:
        required_acl = compact_sql(
            'REVOKE EXECUTE ON FUNCTION '
            f'{signature} FROM PUBLIC;'
        )
        if required_acl not in compact:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: diagnostic query remains public: '
                f'{signature}'
            )

    for signature in PUBLIC_COMPOSITION_SIGNATURES:
        forbidden_acl = compact_sql(
            'REVOKE EXECUTE ON FUNCTION '
            f'{signature} FROM PUBLIC;'
        )
        if forbidden_acl in compact:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: product composition API is '
                f'revoked from PUBLIC: {signature}'
            )

    for prefix in DISALLOWED_SPLIT_API_PREFIXES:
        if re.search(rf'CREATE FUNCTION {prefix}', sql):
            errors.append(
                f'{CURRENT_SQL_PATH.name}: retired API is still installed: '
                f'{prefix}*'
            )

    for option in RETIRED_PRODUCT_RELOPTIONS:
        if re.search(rf'\b{re.escape(option)}\b', sql):
            errors.append(
                f'{CURRENT_SQL_PATH.name}: retired reloption remains: '
                f'{option}'
            )
    for guc in RETIRED_PRODUCT_GUCS:
        if guc in sql:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: retired product GUC remains: '
                f'{guc}'
            )
    if "'delta_pressure'" in sql:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: legacy decoded-cache pressure remains '
            'in structured product status'
        )
    status_block = function_block(
        sql,
        'ii42_index_status',
        'ii42_index_audit',
    )
    if not status_block:
        errors.append(f'{CURRENT_SQL_PATH.name}: missing ii42_index_status')
    else:
        for stale_override in (
            "(details->>'pending_writes')::int8",
            "(details->>'pending_deletes')::int8",
        ):
            if stale_override in status_block:
                errors.append(
                    f'{CURRENT_SQL_PATH.name}: ii42_index_status rebuilds '
                    'generation debt from a separate details snapshot: '
                    f'{stale_override}'
                )
        for authoritative_projection in (
            "(generation#>>'{delta,upserts}')::int8",
            "(generation#>>'{delta,retirements}')::int8",
            "(generation#>>'{delta,records}')::int8",
            "(generation#>>'{delta,bytes}')::int8",
        ):
            if authoritative_projection not in status_block:
                errors.append(
                    f'{CURRENT_SQL_PATH.name}: ii42_index_status does not '
                    'project generation debt into details: '
                    f'{authoritative_projection}'
                )

    options_block = function_block(
        sql,
        'ii42_index_options_internal',
        'ii42_index_options',
    )
    query_block = function_block(
        sql,
        'ii42_query',
        'ii42_prepared_query',
    )
    maintain_block = function_block(
        sql,
        'ii42_index_maintain_due',
        'ii42_op_score_ids',
    )
    if not query_block:
        errors.append(f'{CURRENT_SQL_PATH.name}: missing ii42_query')
    elif 'SECURITY DEFINER' not in query_block:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: ii42_query cannot reach revoked '
            'internal helpers safely'
        )
    for block_name, block in (
        ('ii42_index_options_internal', options_block),
        ('ii42_index_maintain_due', maintain_block),
    ):
        if not block:
            errors.append(f'{CURRENT_SQL_PATH.name}: missing {block_name}')
            continue
        for option in DISALLOWED_SPLIT_LIFECYCLE_OPTIONS:
            if option in block:
                errors.append(
                    f'{CURRENT_SQL_PATH.name}: {block_name} still depends '
                    f'on {option}'
                )
        if 'ii42_model_semantic_' in block or 'ii42_sae_' in block:
            errors.append(
                f'{CURRENT_SQL_PATH.name}: {block_name} still invokes a '
                'split lifecycle family'
            )


def check_install_contract(errors: list[str]) -> None:
    makefile = MAKEFILE_PATH.read_text(encoding='utf-8')
    required_makefile_contract = (
        'II42_EXTENSION_VERSION := $(shell sed -n',
        'sql/ii42--$(II42_EXTENSION_VERSION).sql',
        'II42_ENABLE_ONNXRUNTIME ?= 1',
        'II42_ENABLE_ONNXRUNTIME must be 0 or 1',
        'II42_CLEAN_ONLY_GOALS := clean clean-ii42-runtime-server',
        '$(filter-out $(II42_CLEAN_ONLY_GOALS),$(MAKECMDGOALS))',
    )
    if not all(item in makefile for item in required_makefile_contract):
        errors.append(
            f'{MAKEFILE_PATH.name}: current-only install SQL contract is '
            'incomplete'
        )
    if 'II42_ENABLE_ONNXRUNTIME ?= auto' in makefile:
        errors.append(
            f'{MAKEFILE_PATH.name}: product builds may silently omit ONNX '
            'Runtime'
        )
    for rpath in ('-Wl,-rpath,@loader_path', r'-Wl,-rpath,\$$ORIGIN'):
        if rpath not in makefile:
            errors.append(
                f'{MAKEFILE_PATH.name}: missing relative runtime path '
                f'{rpath}'
            )
    versioned_sql = {
        path.name
        for path in (REPO_ROOT / 'sql').glob('ii42--*.sql')
    }
    if CURRENT_SQL_PATH.name not in versioned_sql:
        errors.append(
            f'sql/: missing current install SQL {CURRENT_SQL_PATH.name}'
        )
    expected_sql = {CURRENT_SQL_PATH.name}
    if versioned_sql != expected_sql:
        errors.append(
            'sql/: expected exactly one current install SQL and no beta '
            'upgrade scripts; found '
            f'{sorted(versioned_sql)}'
        )

    bump_version = BUMP_VERSION_PATH.read_text(encoding='utf-8')
    for required_bump_contract in (
        '--rename-current-sql',
        'current_sql.rename(next_sql)',
    ):
        if required_bump_contract not in bump_version:
            errors.append(
                f'{BUMP_VERSION_PATH.name}: versioned catalog bump '
                f'misses {required_bump_contract!r}'
            )
    for retired_bump_contract in (
        '--upgrade-sql',
        '--sql-unchanged',
        '--copy-current-sql',
        'shutil.copy',
    ):
        if retired_bump_contract in bump_version:
            errors.append(
                f'{BUMP_VERSION_PATH.name}: historical install path remains: '
                f'{retired_bump_contract!r}'
            )

    onnxruntime_version = ONNXRUNTIME_VERSION_PATH.read_text(
        encoding='utf-8',
    ).strip()
    if not onnxruntime_version:
        errors.append('packaging/onnxruntime.version is empty')
    checksum_lines = ONNXRUNTIME_CHECKSUM_PATH.read_text(
        encoding='utf-8',
    ).splitlines()
    required_archives = (
        'onnxruntime-linux-aarch64-',
        'onnxruntime-linux-x64-',
        'onnxruntime-osx-arm64-',
    )
    for archive_prefix in required_archives:
        if not any(archive_prefix in line for line in checksum_lines):
            errors.append(
                'packaging/onnxruntime.sha256 lacks '
                f'{archive_prefix}*'
            )
    for checksum_line in checksum_lines:
        checksum_parts = checksum_line.split()
        if len(checksum_parts) != 2:
            errors.append(
                'packaging/onnxruntime.sha256 contains a malformed entry: '
                f'{checksum_line!r}'
            )
            continue
        if onnxruntime_version not in checksum_parts[1]:
            errors.append(
                'packaging/onnxruntime.sha256 retains a non-current archive: '
                f'{checksum_parts[1]}'
            )
    installer = ONNXRUNTIME_INSTALLER_PATH.read_text(encoding='utf-8')
    if 'packaging/onnxruntime.version' not in installer:
        errors.append('ONNX Runtime installer ignores the pinned version')
    if 'packaging/onnxruntime.sha256' not in installer:
        errors.append('ONNX Runtime installer ignores pinned checksums')
    makefile_runtime = MAKEFILE_PATH.read_text(encoding='utf-8')
    for required_make_contract in (
        'II42_ONNXRUNTIME_REQUIRED_VERSION',
        'pkg-config --modversion libonnxruntime',
        'ifneq ($(ONNXRUNTIME_VERSION),'
        '$(II42_ONNXRUNTIME_REQUIRED_VERSION))',
    ):
        if required_make_contract not in makefile_runtime:
            errors.append(
                'Makefile does not fail closed on the pinned ONNX Runtime '
                f'version: {required_make_contract!r}'
            )
    postgres_installer = POSTGRES_APT_INSTALLER_PATH.read_text(
        encoding='utf-8',
    )
    if '    pkg-config \\\n' not in postgres_installer:
        errors.append(
            'PostgreSQL toolchain installer omits release-build pkg-config'
        )
    dockerfile = DOCKERFILE_PATH.read_text(encoding='utf-8')
    if 'install-onnxruntime-c' not in dockerfile:
        errors.append('Docker image duplicates or omits the runtime installer')
    release_zip = RELEASE_ZIP_PATH.read_text(encoding='utf-8')
    if not PROJECT_LICENSE_PATH.is_file():
        errors.append('repository lacks the project LICENSE')
    for required_release_item in (
        'runtime_stage_dir',
        '--onnxruntime-prefix',
        'II42_ONNXRUNTIME_PREFIX',
        'scripts/install_onnxruntime_c.sh',
        'onnxruntime_pkg_config_path',
        'EVOKE-LICENSE',
        'ONNXRUNTIME-LICENSE',
        'validate_milestone_model_checkout.py',
        'ii42/models/default',
        'MILESTONE-MODEL-NOTICE',
        'Git tree: ${git_tree_state}',
        '--allow-dirty',
        'refusing to build a release from a dirty Git worktree',
        'zip -qry',
    ):
        if required_release_item not in release_zip:
            errors.append(
                'release zip does not preserve the bundled runtime contract: '
                f'{required_release_item}'
            )
    release_docker = RELEASE_DOCKER_PATH.read_text(encoding='utf-8')
    for required_release_item in (
        '--allow-dirty',
        'refusing to build a release from a dirty Git worktree',
        'II42_GIT_COMMIT=${git_commit}',
        'II42_GIT_TREE_STATE=${git_tree_state}',
        'evoke_milestone_model=${model_checkout}',
        'II42_MODEL_MANIFEST_SHA256=${model_manifest_sha256}',
    ):
        if required_release_item not in release_docker:
            errors.append(
                'Docker release does not preserve clean provenance: '
                f'{required_release_item}'
            )
    for required_label in (
        'org.opencontainers.image.revision="${II42_GIT_COMMIT}"',
        'io.evoke.git-tree-state="${II42_GIT_TREE_STATE}"',
        'io.evoke.milestone-model.id="${II42_MODEL_ID}"',
    ):
        if required_label not in dockerfile:
            errors.append(
                f'Docker image lacks provenance label: {required_label}'
            )

    milestone_paths = (
        MILESTONE_MODEL_LOCK_PATH,
        MILESTONE_MODEL_NOTICE_PATH,
        MILESTONE_MODEL_VALIDATOR_PATH,
        MILESTONE_MODEL_ARCHIVER_PATH,
        MILESTONE_MODEL_FETCHER_PATH,
    )
    for milestone_path in milestone_paths:
        if not milestone_path.is_file():
            errors.append(
                f'{milestone_path.relative_to(REPO_ROOT)}: missing '
                'milestone model packaging contract'
            )
        elif (
            milestone_path.parent.name == 'scripts'
            and milestone_path.stat().st_mode & 0o111 == 0
        ):
            errors.append(
                f'{milestone_path.relative_to(REPO_ROOT)}: packaging '
                'entrypoint is not executable'
            )
    if 'II42_PACKAGED_MODEL_PATH' not in makefile:
        errors.append(
            f'{MAKEFILE_PATH.name}: bundled model location is not compiled '
            'into the extension'
        )
    model_lock = MILESTONE_MODEL_LOCK_PATH.read_text(encoding='utf-8')
    for contract in (
        '"manifest_sha256"',
        '"manifest_contract"',
        '"runtime_abi"',
        '"artifacts"',
    ):
        if contract not in model_lock:
            errors.append(
                'packaging/milestone-model.json lacks frozen contract '
                f'{contract}'
            )

    sql = CURRENT_SQL_PATH.read_text(encoding='utf-8')
    resolver_contract = (
        'configured_model_path',
        'environment_model_path',
        'packaged_model_path',
        "WHEN configured_model_path IS NOT NULL THEN 'index'",
        "WHEN environment_model_path IS NOT NULL THEN 'environment'",
        "WHEN packaged_model_path IS NOT NULL THEN 'package'",
    )
    if not all(contract in sql for contract in resolver_contract):
        errors.append(
            f'{CURRENT_SQL_PATH.name}: model checkout precedence is '
            'incomplete'
        )

    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    if f'--version {CURRENT_VERSION}' not in readme:
        errors.append('README Docker build version does not match control')

    if SYNC_PUBLIC_WORKFLOW_PATH.is_file():
        sync_public = SYNC_PUBLIC_WORKFLOW_PATH.read_text(encoding='utf-8')
        if 'PUBLIC_REPOSITORY: Intelligent-Internet/Evoke\n' not in sync_public:
            errors.append('public repository sync targets the wrong repository')
        if "vars.PUBLIC_REPO_SYNC_ENABLED == 'true'" not in sync_public:
            errors.append('public repository sync lacks an explicit enable gate')
        if 'PUBLIC_REPO_SYNC_TOKEN' not in sync_public:
            errors.append('public repository sync lacks an explicit token')
    else:
        # The public snapshot keeps its own publishing workflows, not sync.
        release = (REPO_ROOT / '.github/workflows/release.yml').read_text(
            encoding='utf-8'
        )
        if 'publish-release:' not in release or 'gh release create' not in release:
            errors.append('public repository lacks its release publisher')


def check_active_scripts(errors: list[str]) -> None:
    retired_call = re.compile(
        r'\b(?:SELECT|FROM|PERFORM|CALL)\s+'
        r'(?:[A-Za-z_][A-Za-z0-9_$]*\.)?'
        r'ii42_(?:model_|sae_generation_|index_model_)'
    )
    for path in sorted((REPO_ROOT / 'scripts').glob('*')):
        if not path.is_file() or path.suffix not in {'.py', '.sh'}:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name.startswith(('research_', 'test_research_')):
            continue
        text = path.read_text(encoding='utf-8')
        if retired_call.search(text):
            errors.append(
                f'{path.relative_to(REPO_ROOT)}: calls a retired split API'
            )
        for option in RETIRED_PRODUCT_RELOPTIONS:
            if re.search(rf'\b{re.escape(option)}\b', text):
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: uses retired reloption '
                    f'{option}'
                )
        for source_root in RETIRED_LOCAL_SOURCE_ROOTS:
            if source_root in text:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: references retired '
                    f'local source root {source_root}'
                )
        for local_path in MAINTAINER_LOCAL_PATHS:
            if local_path in text:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: embeds maintainer-local '
                    f'path {local_path}'
                )
    for path in MODEL_MANIFEST_ENTRYPOINTS:
        text = path.read_text(encoding='utf-8')
        for contract in (
            "'schema_version'",
            "'api_version'",
            "'evoke_model_v1'",
            "'evoke_p2_unified_text_atoms_v2'",
        ):
            if contract not in text:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: model entrypoint does '
                    f'not enforce current contract {contract}'
                )


def check_c_contract(errors: list[str]) -> None:
    source = AM_SOURCE_PATH.read_text(encoding='utf-8')
    same_index_concurrency = SAME_INDEX_CONCURRENCY_PATH.read_text(
        encoding='utf-8'
    )
    build_source = AM_BUILD_SOURCE_PATH.read_text(encoding='utf-8')
    build_header = AM_BUILD_HEADER_PATH.read_text(encoding='utf-8')
    hot_fold_source = AM_HOT_FOLD_SOURCE_PATH.read_text(encoding='utf-8')
    hot_fold_header = AM_HOT_FOLD_HEADER_PATH.read_text(encoding='utf-8')
    maintenance_source = AM_MAINTENANCE_SOURCE_PATH.read_text(
        encoding='utf-8'
    )
    maintenance_header = AM_MAINTENANCE_HEADER_PATH.read_text(
        encoding='utf-8'
    )
    meta_source = AM_META_SOURCE_PATH.read_text(encoding='utf-8')
    meta_header = AM_META_HEADER_PATH.read_text(encoding='utf-8')
    mutation_source = AM_MUTATION_SOURCE_PATH.read_text(encoding='utf-8')
    mutation_header = AM_MUTATION_HEADER_PATH.read_text(encoding='utf-8')
    options_source = AM_OPTIONS_SOURCE_PATH.read_text(encoding='utf-8')
    options_header = AM_OPTIONS_HEADER_PATH.read_text(encoding='utf-8')
    preload_source = AM_PRELOAD_SOURCE_PATH.read_text(encoding='utf-8')
    preload_header = AM_PRELOAD_HEADER_PATH.read_text(encoding='utf-8')
    reclamation_source = AM_RECLAMATION_SOURCE_PATH.read_text(
        encoding='utf-8'
    )
    reclamation_header = AM_RECLAMATION_HEADER_PATH.read_text(
        encoding='utf-8'
    )
    scheduler_source = AM_SCHEDULER_SOURCE_PATH.read_text(encoding='utf-8')
    scheduler_header = AM_SCHEDULER_HEADER_PATH.read_text(encoding='utf-8')
    scan_source = AM_SCAN_SOURCE_PATH.read_text(encoding='utf-8')
    scan_header = AM_SCAN_HEADER_PATH.read_text(encoding='utf-8')
    sql_source = AM_SQL_SOURCE_PATH.read_text(encoding='utf-8')
    test_support_header = AM_TEST_SUPPORT_HEADER_PATH.read_text(
        encoding='utf-8'
    )
    pg_common_source = PG_COMMON_SOURCE_PATH.read_text(encoding='utf-8')
    pg_common_header = PG_COMMON_HEADER_PATH.read_text(encoding='utf-8')
    semantic_source = SEMANTIC_SOURCE_PATH.read_text(encoding='utf-8')
    semantic_header = SEMANTIC_HEADER_PATH.read_text(encoding='utf-8')
    p2_runtime_source = P2_RUNTIME_SOURCE_PATH.read_text(encoding='utf-8')
    runtime_header = RUNTIME_SERVICE_HEADER_PATH.read_text(
        encoding='utf-8'
    )
    storage_source = STORAGE_SOURCE_PATH.read_text(encoding='utf-8')
    storage_header = STORAGE_HEADER_PATH.read_text(encoding='utf-8')
    query_source = QUERY_SOURCE_PATH.read_text(encoding='utf-8')
    query_header = QUERY_HEADER_PATH.read_text(encoding='utf-8')
    core_header = CORE_HEADER_PATH.read_text(encoding='utf-8')
    segments_source = SEGMENTS_SOURCE_PATH.read_text(encoding='utf-8')
    segments_header = SEGMENTS_HEADER_PATH.read_text(encoding='utf-8')
    document_cow_source = DOCUMENT_COW_SOURCE_PATH.read_text(
        encoding='utf-8'
    )
    required = (
        'ii42_am_semantic_builder_begin(',
        'ii42_am_semantic_builder_finish(',
        'ii42_am_lock_maintenance_xact(',
        'ii42_am_lock_writer_barrier_oid(',
        'ii42_am_schedule_background_maintenance(indexRelation)',
        'AND NOT COALESCE(c.reloptions, ARRAY[]::text[])',
        "@> ARRAY['consistency=manual']::text[]",
        'must be superuser to clear the ii42 runtime cache',
        'ii42.onnxruntime_intra_op_threads',
    )
    for needle in required:
        if needle not in source:
            errors.append(f'{AM_SOURCE_PATH.name}: missing {needle!r}')
    if "@> ARRAY['consistency=eventual']::text[]" in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: catalog reconciliation only covers '
            'eventual indexes'
        )

    vacuum_discovery_start = source.find(
        '\nstatic uint32\nii42_am_record_convergent_deletes('
    )
    vacuum_discovery_end = source.find(
        '\nstatic IndexBulkDeleteResult *\nii42_ambulkdelete(',
        vacuum_discovery_start,
    )
    if vacuum_discovery_start < 0 or vacuum_discovery_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: convergent VACUUM discovery not found'
        )
    else:
        vacuum_discovery = source[
            vacuum_discovery_start:vacuum_discovery_end
        ]
        lock_at = vacuum_discovery.find(
            'ii42_am_lock_maintenance_xact('
        )
        writer_pin_at = vacuum_discovery.find(
            'ii42_am_pin_maintenance_xact(indexRelation)'
        )
        root_read_at = vacuum_discovery.find(
            'ii42_am_read_meta(indexRelation, &meta)'
        )
        l0_read_at = vacuum_discovery.find(
            'ii42_segment_pages_load_l0_snapshot('
        )
        if not (
            0 <= lock_at < writer_pin_at < root_read_at < l0_read_at
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: convergent VACUUM must own '
                'maintenance before writer pin, root, and L0 discovery'
            )

    longjmp_ownership_required = (
        'struct ii42_am_convergent_completion_cleanup',
        'ii42_am_query_operator_cleanup_create(void)',
        'ii42_am_visibility_ctx visibility;',
        'ii42_am_semantic_builder contract;',
    )
    for needle in longjmp_ownership_required:
        if needle not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: longjmp-safe cleanup ownership is '
                f'missing {needle!r}'
            )
    reindex_start = source.find(
        '\nstatic void\nii42_am_reindex_relation('
    )
    reindex_end = source.find(
        '\ntypedef enum ii42_am_convergent_tail_cleanup_outcome',
        reindex_start,
    )
    if reindex_start < 0 or reindex_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: reindex cleanup block not found'
        )
    else:
        reindex_block = source[reindex_start:reindex_end]
        for needle in (
            'Relation volatile heapRelation = NULL;',
            'PG_FINALLY();',
            'table_close(heapRelation, AccessShareLock);',
            'ii42_am_unlock_generation_barrier(indexRelation);',
        ):
            if needle not in reindex_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: reindex longjmp cleanup is '
                    f'missing {needle!r}'
                )
    operator_start = source.find(
        'PG_FUNCTION_INFO_V1(ii42_match_query_tokens_op)'
    )
    operator_end = source.find(
        'PG_FUNCTION_INFO_V1(ii42_normalize_tokens_sql)',
        operator_start,
    )
    if operator_start < 0 or operator_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: query operator cleanup block not found'
        )
    else:
        operator_block = source[operator_start:operator_end]
        for stack_state in (
            'ii42_query query;',
            'char **doc_tokens;',
            'ii42_am_visibility_ctx visibility = {0};',
        ):
            if stack_state in operator_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: query operator keeps '
                    f'longjmp-unsafe stack state {stack_state!r}'
                )
        if operator_block.count(
            'cleanup = ii42_am_query_operator_cleanup_create();'
        ) != 4:
            errors.append(
                f'{AM_SOURCE_PATH.name}: not every query operator uses the '
                'stable cleanup owner'
            )

    maintenance_start = source.find(
        '\nstatic text *\nii42_am_try_maintain_index_oid('
    )
    maintenance_end = source.find(
        '\nPG_FUNCTION_INFO_V1(ii42_try_maintain_index)',
        maintenance_start,
    )
    if maintenance_start < 0 or maintenance_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: v3 maintenance selector not found'
        )
    else:
        maintenance_block = source[maintenance_start:maintenance_end]
        for needle in (
            'ii42_am_require_convergent_segment_storage(&initial_meta)',
            'ii42_am_rotate_convergent_active_l0(indexRelation)',
            'ii42_am_try_seal_pending_l0(',
            'ii42_am_complete_convergent_semantic(',
            'ii42_am_try_reclaim_retired_pages(',
            'ii42_am_try_compact_segments(',
            'reason=no_pending, mode=convergent_segment',
        ):
            if needle not in maintenance_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 maintenance selector is '
                    f'missing {needle!r}'
                )
        for forbidden in (
            'ii42_am_online_maintain_relation(',
            'ii42_am_reindex_relation(',
            'ii42_am_note_delta_record',
            'ii42_am_complete_pending_semantic(',
        ):
            if forbidden in maintenance_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 maintenance selector retains '
                    f'legacy dispatch {forbidden!r}'
                )

    insert_start = source.find('\nstatic bool\nii42_aminsert(')
    insert_end = source.find(
        '\nstatic void\nii42_aminsertcleanup(',
        insert_start,
    )
    if insert_start < 0 or insert_end < 0:
        errors.append(f'{AM_SOURCE_PATH.name}: v3 insert block not found')
    else:
        insert_block = source[insert_start:insert_end]
        for needle in (
            'ii42_am_require_convergent_segment_storage(&meta)',
            'ii42_am_append_convergent_l0_upsert(',
            'ii42_am_schedule_background_maintenance(indexRelation)',
        ):
            if needle not in insert_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 insert is missing {needle!r}'
                )
        for forbidden in (
            'ii42_am_note_delta_record',
            'ii42_am_queue_sae_mutation',
            'ii42_am_reindex_relation(',
        ):
            if forbidden in insert_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 insert retains legacy writer '
                    f'{forbidden!r}'
                )

    bulkdelete_start = source.find(
        '\nstatic IndexBulkDeleteResult *\nii42_ambulkdelete('
    )
    bulkdelete_end = source.find(
        '\nstatic IndexBulkDeleteResult *\nii42_amvacuumcleanup(',
        bulkdelete_start,
    )
    if bulkdelete_start < 0 or bulkdelete_end < 0:
        errors.append(f'{AM_SOURCE_PATH.name}: v3 bulk-delete block not found')
    else:
        bulkdelete_block = source[bulkdelete_start:bulkdelete_end]
        for needle in (
            'ii42_am_require_convergent_segment_storage(&meta)',
            'ii42_am_record_convergent_deletes(',
            'ii42_am_schedule_background_maintenance(info->index)',
        ):
            if needle not in bulkdelete_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 bulk-delete is missing '
                    f'{needle!r}'
                )
        if 'ii42_am_record_exact_deletes(' in bulkdelete_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: v3 bulk-delete retains the retired '
                'generation-relative tombstone writer'
            )
        if 'needs_refresh' in bulkdelete_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: v3 bulk-delete retains the dead '
                'VACUUM refresh statistic'
            )

    vacuum_start = bulkdelete_end
    vacuum_end = source.find(
        '\nstatic bool\nii42_amcanreturn(',
        vacuum_start,
    )
    if vacuum_start < 0 or vacuum_end < 0:
        errors.append(f'{AM_SOURCE_PATH.name}: v3 VACUUM cleanup not found')
    else:
        vacuum_block = source[vacuum_start:vacuum_end]
        for needle in (
            'ii42_am_require_convergent_segment_storage(&meta)',
            'ii42_am_get_stats(info->index, stats)',
        ):
            if needle not in vacuum_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 VACUUM cleanup is missing '
                    f'{needle!r}'
                )
        for forbidden in (
            'ii42_am_freeze_delta_record_xids(',
            'ii42_am_reindex_relation(',
        ):
            if forbidden in vacuum_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: v3 VACUUM cleanup retains '
                    f'legacy work {forbidden!r}'
                )

    root_store_start = mutation_source.find(
        '\nvoid\nii42_am_l0_store_root('
    )
    root_store_end = mutation_source.find(
        '\nXLogRecPtr\nii42_am_l0_rotate_active_locked(',
        root_store_start,
    )
    if root_store_start < 0 or root_store_end < 0:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 root writer not '
            'found'
        )
    else:
        root_store_block = mutation_source[
            root_store_start:root_store_end
        ]
        for needle in (
            'meta->pending_write_tuples = 0;',
            'meta->pending_delete_tuples = 0;',
        ):
            if needle not in root_store_block:
                errors.append(
                    f'{AM_MUTATION_SOURCE_PATH.name}: page-native root '
                    'writer does '
                    f'not clear legacy debt counter {needle!r}'
                )

    callback_start = source.find(
        '\nstatic void\nii42_am_xact_callback('
    )
    callback_end = source.find(
        '\nstatic bool\nii42_am_semantic_signature_valid(',
        callback_start,
    )
    if callback_start < 0 or callback_end < 0:
        errors.append(f'{AM_SOURCE_PATH.name}: transaction callback not found')
    else:
        callback_block = source[callback_start:callback_end]
        for required in (
            'ii42_am_touch_background_maintenance()',
            'ii42_am_clear_pinned_maintenances()',
        ):
            if required not in callback_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: transaction callback is missing '
                    f'current cleanup/wakeup authority {required!r}'
                )
        for forbidden in (
            'pending_maintenance_activity',
            'ii42_am_note_maintenance_activity(',
            'ii42_am_reindex_relation(',
            'ii42_am_publish_replacement_generation(',
            'ii42_am_flush_all_pending_sae_mutations()',
            'ii42_am_compile_pending_sae_batch(',
        ):
            if forbidden in callback_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: transaction callback retains '
                    f'foreground model or generation work {forbidden!r}'
                )

    if 'SetIntraOpNumThreads' not in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: missing ONNX CPU thread control'
        )

    if '#define II42_RUNTIME_SERVICE_RESTART_SECONDS 5' not in runtime_header:
        errors.append(
            f'{RUNTIME_SERVICE_HEADER_PATH.name}: runtime restart bound is '
            'missing'
        )
    if (
        'worker.bgw_restart_time = '
        'II42_RUNTIME_SERVICE_RESTART_SECONDS;' not in source
    ):
        errors.append(
            f'{AM_SOURCE_PATH.name}: runtime worker does not use the '
            'bounded restart interval'
        )
    if 'ii42_runtime_service_worker_is_ready(' not in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: stale worker PID guard is missing'
        )
    runtime_header_admission_contracts = (
        'uint32 request_kind;',
        'ProcNumber caller_proc_number;',
        'ProcNumber owner_proc_number;',
        'ProcNumber processing_owner_proc_number;',
    )
    runtime_source_admission_contracts = (
        'ii42_runtime_service_document_response_slots_locked(void)',
        'ii42_runtime_service_max_document_workers_locked(void)',
        'response->request_kind = 0;',
        '].request_kind = request_kind;',
        'ProcNumberGetProc(owner_proc_number);',
        'response->owner_proc_number == MyProcNumber',
        'ii42_runtime_service->workers[i].proc_number ==\n'
        '                    MyProcNumber',
        'worker->proc_number != expected_proc_number',
        '\\"document_response_slots_in_use\\":%u,',
        '\\"query_execution_lane_reserved\\":%s,',
    )
    for contract in runtime_header_admission_contracts:
        if contract not in runtime_header:
            errors.append(
                f'ii42 runtime ownership header is missing {contract!r}'
            )
    for contract in runtime_source_admission_contracts:
        if contract not in semantic_source:
            errors.append(
                f'ii42 runtime document admission is missing {contract!r}'
            )
    shared_preload_version = re.search(
        r'^#define II42_AM_PRELOAD_VERSION (\d+)$',
        preload_source,
        re.MULTILINE,
    )
    if (
        shared_preload_version is None
        or int(shared_preload_version.group(1)) < 19
    ):
        errors.append(
            f'{AM_PRELOAD_SOURCE_PATH.name}: shared arena ABI predates logical '
            'allocation tracking'
        )
    scheduler_start = scheduler_source.find(
        'typedef struct ii42_am_scheduler_control'
    )
    scheduler_end = scheduler_source.find(
        '} ii42_am_scheduler_control;',
        scheduler_start,
    )
    preload_start = preload_source.find(
        'typedef struct ii42_am_preload_control'
    )
    preload_end = preload_source.find(
        '} ii42_am_preload_control;',
        preload_start,
    )
    scheduler_body = (
        scheduler_source[scheduler_start:scheduler_end]
        if scheduler_start >= 0 and scheduler_end > scheduler_start
        else ''
    )
    preload_body = (
        preload_source[preload_start:preload_end]
        if preload_start >= 0 and preload_end > preload_start
        else ''
    )
    for required in (
        'active_background_workers',
        'work_hints[II42_AM_WORK_HINT_CAPACITY]',
        'reconcile_dbs[II42_AM_RECONCILE_DB_CAPACITY]',
        'semantic_telemetry[II42_AM_SEMANTIC_TELEMETRY_CAPACITY]',
        'posting_heat[II42_AM_POSTING_HEAT_CAPACITY]',
    ):
        if required not in scheduler_body:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: scheduler control is '
                'missing '
                f'{required!r}'
            )
    for forbidden in (
        'arena_size',
        'entry_capacity',
        'ii42_am_shared_preload_entry entries',
    ):
        if forbidden in scheduler_body:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: scheduler control '
                f'acquired cache authority {forbidden!r}'
            )
    for required in (
        'Size arena_size;',
        'Size used;',
        'uint32 entry_capacity;',
        'uint32 hash_capacity;',
        'ii42_am_preload_entry entries[FLEXIBLE_ARRAY_MEMBER];',
    ):
        if required not in preload_body:
            errors.append(
                f'{AM_PRELOAD_SOURCE_PATH.name}: preload control is missing '
                f'{required!r}'
            )
    for forbidden in (
        'active_background_workers',
        'work_hints[',
        'reconcile_dbs[',
        'semantic_telemetry[',
        'posting_heat[',
        'last_preload_warning_time',
    ):
        if forbidden in preload_body:
            errors.append(
                f'{AM_PRELOAD_SOURCE_PATH.name}: preload control reacquired '
                f'scheduler authority {forbidden!r}'
            )
    for required in (
        '#define II42_AM_SCHEDULER_MAGIC',
        '#define II42_AM_SCHEDULER_VERSION',
        '"ii42 scheduler state",',
        'static ii42_am_scheduler_control *ii42_scheduler = NULL;',
    ):
        if required not in scheduler_source:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: shared-state split is '
                'missing '
                f'{required!r}'
            )
    for required in (
        'TimestampTz maintenance_first_marked_at;',
        'ii42_am_scheduler_work_hint_reset_age(',
    ):
        if required not in scheduler_header:
            errors.append(
                f'{AM_SCHEDULER_HEADER_PATH.name}: periodic low-debt '
                f'contract is missing {required!r}'
            )
    for required in (
        '#define II42_AM_SCHEDULER_VERSION 3',
        '!maintenance_was_pending',
        'hint->maintenance_first_marked_at = GetCurrentTimestamp();',
        'ii42_am_scheduler_work_hint_reset_age(',
        'hint->accelerator_retry_after = retry_after;',
        'ii42_am_scheduler_work_hint_defer_accelerator(',
    ):
        if required not in scheduler_source:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: periodic low-debt '
                f'authority is missing {required!r}'
            )
    for required in (
        'ii42.maintenance_low_debt_interval_ms',
        'ii42_am_work_hint_low_debt_due(',
        'allow_periodic_low_debt && low_debt',
        'allow_periodic_low_debt && refresh.periodic_eligible',
        'candidate_out->action_class = !strict_due',
        'semantic_accelerator_strict_due_raw',
        'ii42_am_work_hint_accelerator_retry_allowed(hint_token)',
        'ii42_am_work_hint_defer_accelerator(index_oid)',
        'candidate->allow_accelerator_build',
        'ii42_am_maintenance_result_resets_low_debt_age(',
        '\\"refresh_max_age_ms\\\":null',
    ):
        if required not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: periodic low-debt scheduling is '
                f'missing {required!r}'
            )
    if 'RequestAddinShmemSpace(ii42_am_scheduler_shmem_size());' not in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: scheduler shmem request is missing'
        )
    if 'last_preload_warning_time' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: retired preload warning state remains'
        )
    exit_start = source.find(
        '\nstatic void\nii42_am_cache_shmem_exit('
    )
    exit_end = source.find(
        '\n\nstatic void\nii42_am_search_state_reset(',
        exit_start,
    )
    exit_block = (
        source[exit_start:exit_end]
        if exit_start >= 0 and exit_end > exit_start
        else ''
    )
    for required in (
        'ii42_am_local_posting_heat_query_count > 0',
        'ii42_am_posting_heat_flush_local();',
    ):
        if required not in exit_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: short-lived backends can discard '
                f'posting heat; missing {required!r}'
            )
    heat_start = source.find(
        '\nstatic void\nii42_am_posting_heat_observe_page_native('
    )
    heat_end = source.find(
        '\n\nstatic bool\nii42_am_posting_heat_candidate_snapshot(',
        heat_start,
    )
    heat_block = (
        source[heat_start:heat_end]
        if heat_start >= 0 and heat_end > heat_start
        else ''
    )
    for forbidden in (
        'ii42_am_posting_heat_namespace',
        'term_id >= lexical_term_limit',
    ):
        if forbidden in heat_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: page-native posting heat excludes '
                f'semantic atoms through {forbidden!r}'
            )
    identity_definition = 'bool\nii42_am_generation_identity_matches('
    identity_declaration = 'bool ii42_am_generation_identity_matches('
    if AM_SOURCE_PATH.read_text(encoding='utf-8').count(
        identity_definition
    ) != 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: generation identity returned to AM'
        )
    if AM_META_SOURCE_PATH.read_text(encoding='utf-8').count(
        identity_definition
    ) != 1:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: generation identity definition '
            'must be unique'
        )
    if AM_META_HEADER_PATH.read_text(encoding='utf-8').count(
        identity_declaration
    ) != 1:
        errors.append(
            f'{AM_META_HEADER_PATH.name}: generation identity declaration '
            'must be unique'
        )
    am_allocator_contracts = (
        'Size allocation_size;',
        'entry->mapped_size > entry->allocation_size',
        'entry->allocation_size >= payload_size',
        'allocation_size =\n'
        '            ii42_preload->entries[free_slot].allocation_size;',
        'ii42_preload->entries[free_slot].allocation_size = allocation_size;',
        '!entry->in_use && !ii42_am_preload_entry_has_block(entry) &&\n'
        '            free_slot < 0',
    )
    for contract in am_allocator_contracts:
        if contract not in preload_source:
            errors.append(
                f'{AM_PRELOAD_SOURCE_PATH.name}: shared arena allocator is '
                'missing '
                f'{contract!r}'
            )
    session_cache_identity_contracts = (
        '#define II42_CHECKOUT_VALIDATION_CACHE_LIMIT 16',
        'ii42_checkout_validation_cache['
        'II42_CHECKOUT_VALIDATION_CACHE_LIMIT]',
        'ii42_checkout_validation_cache_access_seq',
        'char checkout_signature[II42_CHECKOUT_SIGNATURE_HEX_LEN + 1];',
        'ii42_checkout_manifest_signature(manifest, checkout_signature);',
        'entry->checkout_signature,',
        'sizeof(entry->checkout_signature)',
        'ii42_checkout_validate_manifest(manifest, NULL, model_path);',
        '",\\"checkout_signature\\":"',
        'entry->document_mode == document_mode',
        'entry->document_mode = document_mode;',
        'DisableMemPattern(document)',
        'DisableCpuMemArena(document)',
        'worker->affinity_request_kind == request_kind',
        '\\"affinity_role\\":',
    )
    for contract in session_cache_identity_contracts:
        if contract not in semantic_source:
            errors.append(
                f'{SEMANTIC_SOURCE_PATH.name}: ONNX session cache identity '
                f'is missing {contract!r}'
            )
    compiler_cache_identity_contracts = (
        'char checkout_signature[II42_P2_CACHE_IDENTITY_BYTES];',
        'ii42_p2_tokenizer_cache->checkout_signature',
        'ii42_p2_compiler_cache->checkout_signature',
        'invalid P2 tokenizer cache identity',
        'invalid P2 compiler cache identity',
    )
    for contract in compiler_cache_identity_contracts:
        if contract not in p2_runtime_source:
            errors.append(
                f'{P2_RUNTIME_SOURCE_PATH.name}: runtime artifact cache '
                f'identity is missing {contract!r}'
            )
    checkout_generation_contracts = (
        "config->>'checkout_signature'",
        "config#>>'{index,runtime_signature}'",
        'ii42_am_semantic_builder_require_current_contract(',
        '"batch submission"',
        '"batch completion"',
        '"generation publication"',
        'checkout_signature = ii42_am_jsonb_object_get(',
        'builder->checkout_signature',
    )
    for contract in checkout_generation_contracts:
        if contract not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: checkout generation guard is '
                f'missing {contract!r}'
            )
    if (
        'ii42_runtime_service_outstanding_document_requests_locked'
        in semantic_source
    ):
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: document admission still uses '
            'pending/worker state instead of occupied response slots'
        )

    drain_start = source.find(
        '\nstatic void\nii42_am_semantic_builder_drain_one('
    )
    drain_end = source.find(
        '\nstatic void\nii42_am_semantic_builder_drain_all(',
        drain_start,
    )
    if drain_start < 0 or drain_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: semantic batch drain block not found'
        )
    elif 'SPI_' in source[drain_start:drain_end]:
        errors.append(
            f'{AM_SOURCE_PATH.name}: semantic batch drain regained an SPI '
            'rowset lifetime'
        )
    for lock_contract in (
        '#define II42_LWLOCK_TRANCHE_NAME "ii42"',
        '#define II42_LWLOCK_TRANCHE_COUNT 2',
        'extern LWLock *ii42_runtime_service_lock;',
        'extern Size ii42_runtime_service_shmem_size(void);',
        'extern void ii42_runtime_service_shmem_startup(LWLock *lock);',
    ):
        if lock_contract not in runtime_header:
            errors.append(
                f'{RUNTIME_SERVICE_HEADER_PATH.name}: missing shared-control '
                f'lock contract {lock_contract!r}'
            )
    for definition in (
        'ii42_runtime_service_control *ii42_runtime_service = NULL;',
        'LWLock *ii42_runtime_service_lock = NULL;',
    ):
        if definition in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: runtime shared state returned to AM'
            )
        if semantic_source.count(definition) != 1:
            errors.append(
                f'{SEMANTIC_SOURCE_PATH.name}: runtime shared-state owner '
                f'mismatch for {definition!r}'
            )
    for function, return_type in (
        ('ii42_runtime_service_shmem_size', 'Size'),
        ('ii42_runtime_service_shmem_startup', 'void'),
    ):
        definition = f'\n{return_type}\n{function}('
        if definition in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: runtime shmem authority returned '
                f'to AM: {function}'
            )
        if semantic_source.count(definition) != 1:
            errors.append(
                f'{SEMANTIC_SOURCE_PATH.name}: runtime shmem authority '
                f'mismatch for {function}'
            )
    if 'ii42_runtime_service->' in source or \
            'ii42_runtime_service_lock' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: runtime shared layout remains visible '
            'to the AM entry module'
        )
    for startup_contract in (
        'RequestAddinShmemSpace(ii42_runtime_service_shmem_size());',
        'ii42_runtime_service_shmem_startup(runtime_service_lock);',
    ):
        if source.count(startup_contract) != 1:
            errors.append(
                f'{AM_SOURCE_PATH.name}: runtime startup orchestration '
                f'mismatch for {startup_contract!r}'
            )
    if 'RequestNamedLWLockTranche(' not in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: add-in LWLock tranche is not requested'
        )
    if 'SplitDirectoriesString(raw_preload' not in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: shared preload membership is not '
            'parsed as a PostgreSQL library list'
        )
    if 'strstr(preload, "ii42")' in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: shared preload membership still '
            'uses substring matching'
        )
    for control_source, path in (
        (source, AM_SOURCE_PATH),
        (semantic_source, SEMANTIC_SOURCE_PATH),
        (runtime_header, RUNTIME_SERVICE_HEADER_PATH),
    ):
        if 'SpinLock' in control_source or 'slock_t' in control_source:
            errors.append(
                f'{path.name}: long-lived shared control still uses a '
                'spinlock'
            )

    if 'evoke_p2_unified_text_atoms_v2' not in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: current runtime ABI is missing'
        )
    for exact_manifest_contract in (
        "$1->'schema_version' = '1'::jsonb",
        "$1->>'api_version' = 'evoke_model_v1'",
    ):
        if exact_manifest_contract not in semantic_source:
            errors.append(
                f'{SEMANTIC_SOURCE_PATH.name}: model manifest does not require '
                f'{exact_manifest_contract!r}'
            )
    if re.search(
        r'ii42_(?:onnx_text_atoms|p2_unified_text_atoms)_v1',
        semantic_source,
    ):
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: intermediate runtime ABI remains'
        )
    if re.search(r'\bruntime\s*=\s*[\'"]mock[\'"]', semantic_source):
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: mock runtime branch remains'
        )
    for retired_semantic_pattern in (
        r'\bEATMH\d{3}\b',
        r'\bEATM_[A-Z0-9_]*(?:COMPACT|LEGACY|COMPAT)[A-Z0-9_]*\b',
        r'\bii42_semantic_[a-z0-9_]*'
        r'(?:compact|legacy|compat)[a-z0-9_]*\b',
        r'\bII42_AM_GENERATION_CONTRACT_'
        r'(?:LEGACY|COMPAT)[A-Z0-9_]*\b',
    ):
        if any(
            re.search(retired_semantic_pattern, contract_source)
            for contract_source in (source, semantic_source, semantic_header)
        ):
            errors.append(
                'current C contract retains an intermediate semantic format '
                f'matching {retired_semantic_pattern!r}'
            )
    eventual_completion_contracts = (
        'ii42_am_complete_convergent_semantic(',
        'ii42_am_semantic_frontier_load(',
        'ii42_am_semantic_frontier_classify(',
        'ii42_am_append_convergent_semantic_complete(',
        'ii42_am_append_convergent_semantic_quarantine(',
        'II42_L0_RECORD_SEMANTIC_COMPLETE',
        'II42_L0_RECORD_SEMANTIC_QUARANTINE',
        'semantic_completion_due',
        'II42_ACTIVE_L0_MAX_RECORDS',
        'II42_ACTIVE_L0_MAX_PAGES',
        'ii42 active L0 hard frontier is exhausted',
        '\\"semantic_completion\\":{\\"enabled\\":%s,',
        'mode=semantic_completion',
    )
    for contract in eventual_completion_contracts:
        owner_source = mutation_source if contract in (
            'II42_ACTIVE_L0_MAX_RECORDS',
            'II42_ACTIVE_L0_MAX_PAGES',
            'ii42 active L0 hard frontier is exhausted',
        ) else source
        owner_path = (
            AM_MUTATION_SOURCE_PATH
            if owner_source is mutation_source
            else AM_SOURCE_PATH
        )
        if contract not in owner_source:
            errors.append(
                f'{owner_path.name}: asynchronous eventual contract is '
                f'missing {contract!r}'
            )
    quarantine_projection_contracts = (
        'ii42_am_semantic_quarantine_scan_v3(',
        'ii42_segment_pages_visit_document_records(',
        'view->semantic_pending_since',
        'view->semantic_error_hash',
        'ii42_am_delta_record_states(',
    )
    for contract in quarantine_projection_contracts:
        if contract not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: exact v3 quarantine projection is '
                f'missing {contract!r}'
            )
    required_storage_contract = (
        '#define II42_STORAGE_CURRENT_VERSION 2U',
    )
    for contract in required_storage_contract:
        if contract not in storage_header:
            errors.append(
                f'{STORAGE_HEADER_PATH.name}: missing exact storage boundary '
                f'{contract!r}'
            )
    if 'version != II42_STORAGE_CURRENT_VERSION' not in storage_source:
        errors.append(
            f'{STORAGE_SOURCE_PATH.name}: current-only storage boundary '
            'is missing'
        )
    cache_dispatch_start = source.find(
        '\nstatic ii42_am_cache_entry *\n'
        'ii42_am_get_cached_index_internal('
    )
    cache_dispatch_end = source.find(
        '\nstatic ii42_am_cache_entry *\n'
        'ii42_am_get_cached_index(',
        cache_dispatch_start,
    )
    if cache_dispatch_start < 0 or cache_dispatch_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: current cache dispatch was not found'
        )
    else:
        cache_dispatch = source[cache_dispatch_start:cache_dispatch_end]
        for contract in (
            'ii42_am_require_convergent_segment_storage(&meta);',
            'return ii42_am_get_cached_segment_index(',
        ):
            if contract not in cache_dispatch:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: current cache dispatch is '
                    f'missing {contract!r}'
                )
    for path, storage_contract_source in (
        (STORAGE_SOURCE_PATH, storage_source),
        (AM_SOURCE_PATH, source),
    ):
        if re.search(
            r'\bII42_STORAGE_[A-Z0-9_]*(?:COMPAT|LEGACY)[A-Z0-9_]*\b',
            storage_contract_source,
        ):
            errors.append(
                f'{path.name}: historical storage decoder remains'
            )
    if re.search(
        r'\bII42_STORAGE_[A-Z0-9_]*(?:COMPAT|LEGACY)[A-Z0-9_]*\b',
        storage_header,
    ):
        errors.append(
            f'{STORAGE_HEADER_PATH.name}: historical storage version remains'
        )

    for path, query_contract_source in (
        (QUERY_SOURCE_PATH, query_source),
        (QUERY_HEADER_PATH, query_header),
    ):
        retired_query_pattern = (
            r'\b(?:QUERY_MODE|parse|build)_[a-z0-9_]*'
            r'(?:legacy|compat)[a-z0-9_]*\b'
        )
        if re.search(
            retired_query_pattern,
            query_contract_source,
            re.IGNORECASE,
        ):
            errors.append(
                f'{path.name}: classic BM25 query path still uses an '
                'ambiguous historical name'
            )

    ambuildempty = re.search(
        r'static void\s+ii42_ambuildempty\([^)]*\)\s*\{'
        r'(?P<body>[\s\S]*?)\n\}\n\nstatic ',
        source,
    )
    if ambuildempty is None:
        errors.append(f'{AM_SOURCE_PATH.name}: ii42_ambuildempty not found')
    else:
        empty_body = ambuildempty.group('body')
        if 'ii42_checkout_unified_contract(' not in empty_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: empty semantic init does not use '
                'the current unified checkout contract'
            )
        if 'ii42_am_semantic_builder_begin(' in empty_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: empty semantic init invokes the '
                'semantic builder'
            )

    retired_active_patterns = (
        r'\bii42_am_[a-z0-9_]*sql_semantic[a-z0-9_]*\b',
        r'\bii42_am_[a-z0-9_]*(?:semantic|model)_delta[a-z0-9_]*\b',
        r'\bii42_index_(?:model|semantic)_delta[a-z0-9_]*\b',
        r'\b(?:model|semantic)_delta_generation\b',
        r'\bII42_AM_SHARED_GENERATION_(?:SEMANTIC|MODEL)_DELTA\b',
        r'\bshared_preload_(?:semantic|model)_delta_entries\b',
        r'\bii42_am_publish_tail_generation[a-z0-9_]*\b',
        r'\bii42_am_semantic_bridge_[a-z0-9_]*\b',
        r'\bII42_AM_DESCRIPTOR_(?:MAGIC|VERSION)\b',
        r'\bii42_am_generation_descriptor[a-z0-9_]*\b',
        r'\bII42_AM_SHARED_GENERATION_'
        r'(?:BASE|LEXICAL_DELTA_CACHE|TOMBSTONE)\b',
        r'\bii42_am_shared_(?:lexical_delta_cache|tombstone)_state\b',
        r'\bii42_am_generation_wait_event\b',
        r'\bii42_am_shared_preload_waiter[a-z0-9_]*\b',
        r'\bii42_am_eventual_backlog_limits\b',
        r'\bii42_am_cache_release_shared_preload_leases\b',
        r'\bii42_am_shared_preload_note_admission_miss\b',
    )
    for retired_active_pattern in retired_active_patterns:
        if re.search(retired_active_pattern, source):
            errors.append(
                f'{AM_SOURCE_PATH.name}: split lifecycle remains, matching '
                f'{retired_active_pattern!r}'
            )

    for path, worker_source in (
        (AM_SOURCE_PATH, source),
        (SEMANTIC_SOURCE_PATH, semantic_source),
    ):
        for prefix in (
            'PG_FUNCTION_INFO_V1(ii42_model_',
            'PG_FUNCTION_INFO_V1(ii42_sae_',
            'PG_FUNCTION_INFO_V1(ii42_index_model_',
        ):
            if prefix in worker_source:
                errors.append(
                    f'{path.name}: retired C export remains: {prefix}'
                )
        if 'Template1DbOid' in worker_source:
            errors.append(
                f'{path.name}: background worker still pins template1'
            )

    makefile = MAKEFILE_PATH.read_text(encoding='utf-8')
    if 'ii42_sae_blockmax' in makefile:
        errors.append('Makefile: retired SAE implementation module remains')
    if makefile.count('src/ii42_am_build.o') != 1:
        errors.append('Makefile: build authority does not have one owner')
    if makefile.count('src/ii42_am_hot_fold.o') != 1:
        errors.append('Makefile: HOT_FOLD authority does not have one owner')
    if 'src/ii42_am_options.o' not in makefile:
        errors.append('Makefile: reloption authority module is missing')
    if makefile.count('src/ii42_am_meta.o') != 1:
        errors.append('Makefile: metapage authority does not have one owner')
    if makefile.count('src/ii42_am_preload.o') != 1:
        errors.append('Makefile: preload authority does not have one owner')
    if makefile.count('src/ii42_am_reclamation.o') != 1:
        errors.append('Makefile: reclamation authority has no sole owner')
    if makefile.count('src/ii42_am_scheduler.o') != 1:
        errors.append('Makefile: scheduler authority does not have one owner')
    if makefile.count('src/ii42_am_sql.o') != 1:
        errors.append('Makefile: SQL authority module does not have one owner')
    if makefile.count('src/ii42_am_test_support.o') != 1:
        errors.append('Makefile: test-support authority has no sole owner')

    if '#include "ii42_am_preload.h"' not in source:
        errors.append(f'{AM_SOURCE_PATH.name}: preload API is not included')
    if '#include "ii42_am_scheduler.h"' not in source:
        errors.append(f'{AM_SOURCE_PATH.name}: scheduler API is not included')
    if '#include "ii42_am_hot_fold.h"' not in source:
        errors.append(f'{AM_SOURCE_PATH.name}: HOT_FOLD API is not included')
    for forbidden in (
        'ii42_am_hot_fold_checksum(',
        'ii42_am_hot_fold_block_header_validate(',
        'ii42_am_hot_fold_block_validate(',
        'ii42_am_hot_fold_build_block(',
        'ii42_am_shared_hot_fold_attach(',
        'ii42_am_shared_hot_fold_publish(',
        '.private_header',
        '.private_terms',
        '.private_entries',
        '.private_lease',
        '->private_header',
        '->private_terms',
        '->private_entries',
        '->private_lease',
    ):
        if forbidden in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: HOT_FOLD authority leaked into AM: '
                f'{forbidden!r}'
            )
    for required in (
        '\nstatic uint64\nii42_am_hot_fold_checksum(',
        '\nstatic bool\nii42_am_hot_fold_block_header_validate(',
        '\nstatic bool\nii42_am_hot_fold_block_validate(',
        '\nstatic bool\nii42_am_hot_fold_build_block(',
        '\nbool\nii42_am_hot_fold_attach(',
        '\nbool\nii42_am_hot_fold_publish(',
        '\nvoid\nii42_am_hot_fold_view_release(',
    ):
        if hot_fold_source.count(required) != 1:
            errors.append(
                f'{AM_HOT_FOLD_SOURCE_PATH.name}: HOT_FOLD authority must '
                f'own one {required!r}'
            )
    for required in (
        'typedef struct ii42_am_hot_fold_header',
        'typedef struct ii42_am_hot_fold_term',
        'typedef struct ii42_am_hot_fold_entry',
        'typedef struct ii42_am_hot_fold_view',
        'ii42_am_hot_fold_attach(',
        'ii42_am_hot_fold_publish(',
    ):
        if hot_fold_header.count(required) != 1:
            errors.append(
                f'{AM_HOT_FOLD_HEADER_PATH.name}: HOT_FOLD typed boundary '
                f'must declare one {required!r}'
            )
    for forbidden in (
        'typedef struct ii42_am_preload_entry',
        'typedef struct ii42_am_preload_control',
        'ii42_preload->',
        'ii42_am_preload_publisher_active',
        'ii42_am_reserved_preload_slot',
    ):
        if forbidden in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: preload authority leaked into AM: '
                f'{forbidden!r}'
            )
    for required in (
        'typedef struct ii42_am_preload_entry',
        'typedef struct ii42_am_preload_control',
        'static ii42_am_preload_control *ii42_preload = NULL;',
        'static LWLock *ii42_preload_lock = NULL;',
        'ii42_am_preload_shmem_startup(',
        'ii42_am_preload_reserve(',
        'ii42_am_preload_attach(',
        'ii42_am_preload_status_snapshot(',
    ):
        if preload_source.count(required) != 1:
            errors.append(
                f'{AM_PRELOAD_SOURCE_PATH.name}: preload authority must own '
                f'one {required!r}'
            )
    for private_field in (
        '.private_token',
        '.private_payload',
        '.private_payload_size',
        '->private_token',
        '->private_payload',
        '->private_payload_size',
    ):
        for path, implementation_source in (
            (AM_SOURCE_PATH, source),
            (SEMANTIC_SOURCE_PATH, semantic_source),
        ):
            if private_field in implementation_source:
                errors.append(
                    f'{path.name}: preload handle internals escaped typed API: '
                    f'{private_field!r}'
                )
    if 'ii42_am_preload_cache_available(' not in semantic_source:
        errors.append(
            f'{SEMANTIC_SOURCE_PATH.name}: semantic runtime does not use '
            'preload availability API'
        )
    if 'ii42_shared_preload_available(' in semantic_header:
        errors.append(
            f'{SEMANTIC_HEADER_PATH.name}: retired AM preload wrapper remains'
        )
    for forbidden in (
        'typedef struct ii42_am_scheduler_control',
        'ii42_shared_scheduler',
        'ii42_shared_state_lock',
        '->work_hints[',
        '->reconcile_dbs[',
        '->semantic_telemetry[',
        '->posting_heat[',
    ):
        if forbidden in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: scheduler authority leaked into AM: '
                f'{forbidden!r}'
            )
    for required in (
        'typedef struct ii42_am_scheduler_control',
        'static ii42_am_scheduler_control *ii42_scheduler = NULL;',
        'static LWLock *ii42_scheduler_lock = NULL;',
        'ii42_am_scheduler_shmem_startup(',
        'ii42_am_scheduler_database_hint_snapshot(',
        'ii42_am_scheduler_status_snapshot(',
    ):
        if scheduler_source.count(required) != 1:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: scheduler authority must '
                f'own one {required!r}'
            )
    for forbidden in (
        'Max(maintenance_worker_limit, UINT32_C(1))',
        'Max(limit, UINT32_C(1))',
    ):
        if forbidden in scheduler_source:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: zero worker-limit '
                f'quiesce is overridden by {forbidden!r}'
            )
    for required in (
        'if (configured <= 0)',
        'worker_limit <= 0 ||',
        'ii42_am_scheduler_set_worker_limit((uint32) Max(newval, 0));',
        'II42_AM_DEFAULT_MAINTENANCE_WORKER_LIMIT,\n            0,',
        'ii42_am_forget_background_launch_stamp(InvalidOid);',
        'ii42_am_scheduler_database_hint_snapshot(',
    ):
        if required not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: maintenance admission quiesce '
                f'misses {required!r}'
            )
    for required in (
        'typedef struct ii42_am_scheduler_status',
        'ii42_am_scheduler_shmem_startup(',
        'ii42_am_scheduler_status_snapshot(',
    ):
        if scheduler_header.count(required) != 1:
            errors.append(
                f'{AM_SCHEDULER_HEADER_PATH.name}: scheduler typed boundary '
                f'must declare one {required!r}'
            )
    scheduler_process_retired = (
        'ii42_am_background_worker_phase',
        'II42_AM_WORKER_PHASE_',
        'ii42_am_maintenance_worker_counted_active',
        'ii42_am_maintenance_worker_launch_reserved',
        'ii42_am_maintenance_worker_exit_registered',
        'ii42_am_maintenance_launch_exit_registered',
        'ii42_am_current_worker_phase',
        'ii42_am_register_maintenance_launch_exit',
        'ii42_am_note_background_worker_phase',
        'ii42_am_register_maintenance_worker_exit',
        'ii42_am_release_current_maintenance_worker_launch',
        'ii42_am_note_maintenance_worker_started',
        'ii42_am_note_maintenance_worker_finished',
        'ii42_am_active_background_workers',
        'ii42_am_last_maintenance_db_oid',
        'ii42_am_set_last_maintenance_db_oid',
    )
    for retired in scheduler_process_retired:
        if retired in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: scheduler process authority leaked '
                f'into AM: {retired!r}'
            )
    for required in (
        'static bool ii42_scheduler_process_counted_active = false;',
        'static bool ii42_scheduler_process_launch_reserved = false;',
        'static bool '
        'ii42_scheduler_process_worker_exit_registered = false;',
        'static bool '
        'ii42_scheduler_process_launch_exit_registered = false;',
        'static ii42_am_scheduler_worker_phase '
        'ii42_scheduler_process_phase =',
    ):
        if scheduler_source.count(required) != 1:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: scheduler process state '
                f'must own one {required!r}'
            )
    scheduler_process_apis = (
        'ii42_am_scheduler_process_adopt_worker_launch',
        'ii42_am_scheduler_process_release_worker_launch',
        'ii42_am_scheduler_process_initialize',
        'ii42_am_scheduler_process_note_started',
        'ii42_am_scheduler_process_set_phase',
        'ii42_am_scheduler_process_note_finished',
    )
    for function in scheduler_process_apis:
        definition = re.compile(
            rf'\nvoid\n{re.escape(function)}\('
        )
        if len(definition.findall(scheduler_source)) != 1:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: process lifecycle must '
                f'own one definition of {function}'
            )
        if scheduler_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_SCHEDULER_HEADER_PATH.name}: process lifecycle must '
                f'declare one {function}'
            )
    adopt_start = scheduler_source.find(
        '\nvoid\nii42_am_scheduler_process_adopt_worker_launch(void)'
    )
    adopt_end = scheduler_source.find(
        '\nvoid\nii42_am_scheduler_status_snapshot(',
        adopt_start,
    )
    adopt_body = (
        scheduler_source[adopt_start:adopt_end]
        if adopt_start >= 0 and adopt_end > adopt_start
        else ''
    )
    for required in (
        'ii42_scheduler_process_launch_reserved = true;',
        'ii42_am_scheduler_process_note_started();',
        'before_shmem_exit(ii42_am_scheduler_process_worker_exit, 0);',
        'before_shmem_exit(ii42_am_scheduler_process_launch_exit, 0);',
    ):
        if required not in adopt_body:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: launch adoption must '
                f'hold bounded worker authority before database startup: '
                f'{required!r}'
            )
    scheduler_process_calls = {
        'ii42_am_scheduler_process_adopt_worker_launch(': 1,
        'ii42_am_scheduler_process_release_worker_launch(': 4,
        'ii42_am_scheduler_process_initialize(': 1,
        'ii42_am_scheduler_process_note_started(': 1,
        'ii42_am_scheduler_process_set_phase(': 2,
        'ii42_am_scheduler_process_note_finished(': 2,
    }
    for function, expected in scheduler_process_calls.items():
        if source.count(function) != expected:
            errors.append(
                f'{AM_SOURCE_PATH.name}: scheduler process call count for '
                f'{function!r} changed from {expected}'
            )
    scheduler_contract_counts = {
        'appname = "ii42 background";': 1,
        'appname = "ii42 preload";': 1,
        'appname = "ii42 maintenance";': 1,
        'activity = "idle";': 1,
        'activity = "auto-preload";': 1,
        'activity = "index maintenance";': 1,
        'before_shmem_exit(ii42_am_scheduler_process_worker_exit, 0);': 2,
        'before_shmem_exit(ii42_am_scheduler_process_launch_exit, 0);': 1,
    }
    for required, expected in scheduler_contract_counts.items():
        if scheduler_source.count(required) != expected:
            errors.append(
                f'{AM_SCHEDULER_SOURCE_PATH.name}: process lifecycle '
                f'contract must own {expected} occurrence(s) of '
                f'{required!r}'
            )
    for path, implementation_source in (
        (AM_SOURCE_PATH, source),
        (AM_OPTIONS_SOURCE_PATH, options_source),
        (SEMANTIC_SOURCE_PATH, semantic_source),
        (SEMANTIC_HEADER_PATH, semantic_header),
    ):
        for retired_internal_pattern in (
            r'\bii42_model_(?!v1\b)',
            r'\bii42_sae_[a-z0-9_]*\b',
            r'\bIi42Model[A-Za-z0-9_]*\b',
            r'\bII42_MODEL_[A-Z0-9_]*\b',
        ):
            if re.search(retired_internal_pattern, implementation_source):
                errors.append(
                    f'{path.name}: retired internal identifier remains, '
                    f'matching {retired_internal_pattern!r}'
                )

        for guc in RETIRED_PRODUCT_GUCS:
            if guc in implementation_source:
                errors.append(
                    f'{path.name}: retired product GUC remains: {guc}'
                )

    relopt_table = re.search(
        r'static const relopt_parse_elt ii42_relopt_elems\[\][\s\S]*?\n\};',
        options_source,
    )
    if relopt_table is None:
        errors.append(
            f'{AM_OPTIONS_SOURCE_PATH.name}: reloption table not found'
        )
    else:
        for option in DISALLOWED_SPLIT_LIFECYCLE_OPTIONS:
            if f'"{option}"' in relopt_table.group(0):
                errors.append(
                    f'{AM_OPTIONS_SOURCE_PATH.name}: sidecar reloption '
                    'remains: '
                    f'{option}'
                )
        for option in RETIRED_PRODUCT_RELOPTIONS:
            if f'"{option}"' in relopt_table.group(0):
                errors.append(
                    f'{AM_OPTIONS_SOURCE_PATH.name}: retired reloption '
                    'remains: '
                    f'{option}'
                )
    if 'static const relopt_parse_elt ii42_relopt_elems' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: duplicate reloption table remains'
        )
    for retired_options_owner in (
        'static relopt_kind ii42_relopt_kind',
        'static bool ii42_relopts_initialized',
        'ii42_am_validate_policy_reloptions(',
        'ii42_am_apply_reloption_defaults_and_flags(',
    ):
        if retired_options_owner in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: reloption ownership returned to '
                f'the AM entry module: {retired_options_owner}'
            )
    for option_projection in (
        'ii42_am_relation_has_explicit_reloptions',
        'ii42_am_get_consistency',
        'ii42_am_validate_relation_policy',
        'ii42_am_auto_preload_priority',
        'ii42_am_field_aware_enabled',
        'ii42_am_sae_enabled',
        'ii42_am_read_params',
        'ii42_am_read_text_policy',
    ):
        definition = re.compile(
            rf'\n(?:static )?(?:bool|int|void)\n'
            rf'{re.escape(option_projection)}\('
        )
        if len(definition.findall(options_source)) != 1:
            errors.append(
                f'{AM_OPTIONS_SOURCE_PATH.name}: option projection does '
                f'not have one owner: {option_projection}'
            )
        if definition.search(source):
            errors.append(
                f'{AM_SOURCE_PATH.name}: option projection returned to '
                f'the AM entry module: {option_projection}'
            )
    model_path_projection = re.compile(
        r'\nconst char \*\nii42_am_relation_model_path\('
    )
    if len(model_path_projection.findall(options_source)) != 1:
        errors.append(
            f'{AM_OPTIONS_SOURCE_PATH.name}: model-path projection does not '
            'have one owner'
        )
    if model_path_projection.search(source):
        errors.append(
            f'{AM_SOURCE_PATH.name}: model-path projection returned to the '
            'AM entry module'
        )
    for direct_option_field in (
        'options->sae',
        'options->consistency',
        'options->auto_preload',
        'options->field_aware',
        'options->method',
        'options->idf_method',
        'options->k1',
        'options->b',
        'options->delta',
        'options->create_empty_token',
    ):
        if direct_option_field in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: direct option projection remains: '
                f'{direct_option_field}'
            )
    if 'typedef struct ii42_am_options' in options_header:
        errors.append(
            f'{AM_OPTIONS_HEADER_PATH.name}: parsed reloption layout is public'
        )
    private_layout_patterns = (
        r'\bii42_am_options\s*\*',
        r'\bsizeof\s*\(\s*ii42_am_options\s*\)',
        r'\boffsetof\s*\(\s*ii42_am_options\s*,',
        r'\bGET_STRING_RELOPTION\s*\(',
        r'->rd_options\b',
    )
    for path in sorted((REPO_ROOT / 'src').glob('*.c')):
        if path == AM_OPTIONS_SOURCE_PATH:
            continue
        implementation_source = path.read_text(encoding='utf-8')
        for private_layout_pattern in private_layout_patterns:
            if re.search(private_layout_pattern, implementation_source):
                errors.append(
                    f'{path.name}: private reloption layout escaped options '
                    f'authority, matching {private_layout_pattern!r}'
                )
    for sql_symbol in ('ii42_score_ids_op', 'ii42_score_tokens_op'):
        info = f'PG_FUNCTION_INFO_V1({sql_symbol});'
        definition = re.compile(
            rf'\nDatum\n{re.escape(sql_symbol)}\(PG_FUNCTION_ARGS\)'
        )
        if sql_source.count(info) != 1 or len(definition.findall(sql_source)) != 1:
            errors.append(
                f'{AM_SQL_SOURCE_PATH.name}: immutable SQL operator does not '
                f'have one owner: {sql_symbol}'
            )
        if info in source or definition.search(source):
            errors.append(
                f'{AM_SOURCE_PATH.name}: immutable SQL operator returned to '
                f'the AM entry module: {sql_symbol}'
            )
    for sql_helper in (
        'ii42_am_text_array_element_type',
        'ii42_am_score_overlap_int4',
        'ii42_am_score_overlap_text',
    ):
        if sql_source.count(f'{sql_helper}(') < 1:
            errors.append(
                f'{AM_SQL_SOURCE_PATH.name}: immutable SQL helper is missing: '
                f'{sql_helper}'
            )
        if f'{sql_helper}(' in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: immutable SQL helper returned to the '
                f'AM entry module: {sql_helper}'
            )
    hybrid_symbol = 'ii42_hybrid_fuse_candidates'
    hybrid_info = f'PG_FUNCTION_INFO_V1({hybrid_symbol});'
    hybrid_definition = re.compile(
        rf'\nDatum\n{re.escape(hybrid_symbol)}\(PG_FUNCTION_ARGS\)'
    )
    if (
        sql_source.count(hybrid_info) != 1
        or len(hybrid_definition.findall(sql_source)) != 1
    ):
        errors.append(
            f'{AM_SQL_SOURCE_PATH.name}: value-only hybrid fusion does not '
            f'have one owner: {hybrid_symbol}'
        )
    for path in sorted((REPO_ROOT / 'src').glob('*.c')):
        if path == AM_SQL_SOURCE_PATH:
            continue
        if 'ii42_hybrid_' in path.read_text(encoding='utf-8'):
            errors.append(
                f'{path.name}: value-only hybrid fusion escaped the '
                'SQL-value module'
            )
    for hybrid_state in (
        'ii42_hybrid_candidate_state',
        'ii42_hybrid_source_state',
        'ii42_hybrid_hit_state',
        'ii42_hybrid_search_state',
    ):
        if sql_source.count(f'typedef struct {hybrid_state}') != 1:
            errors.append(
                f'{AM_SQL_SOURCE_PATH.name}: hybrid state does not have one '
                f'private owner: {hybrid_state}'
            )
    for hybrid_helper in (
        'ii42_hybrid_tid_text',
        'ii42_hybrid_parse_fusion',
        'ii42_hybrid_parse_normalizer',
        'ii42_hybrid_parse_direction',
        'ii42_hybrid_deform_candidate',
        'ii42_hybrid_prepare_candidates',
        'ii42_hybrid_deduplicate_candidates',
        'ii42_hybrid_build_hits',
        'ii42_hybrid_prepare_search_state',
    ):
        if sql_source.count(f'{hybrid_helper}(') < 1:
            errors.append(
                f'{AM_SQL_SOURCE_PATH.name}: value-only hybrid helper is '
                f'missing: {hybrid_helper}'
            )
    for forbidden_sql_dependency in (
        'Relation ',
        'index_open(',
        'table_open(',
        'rd_options',
        'LockRelation',
        'LWLock',
        'XLog',
        'GenericXLog',
        'Snapshot',
        'BackgroundWorker',
        'SPI_',
        'ii42_am_cache',
        'ii42_semantic',
        'ii42_runtime',
        'ii42_segment',
        'ii42_storage',
        'ii42_posting',
        'ii42_page',
        'ii42_query',
        '"ii42_am_options.h"',
        '"ii42_page_query.h"',
        '"ii42_runtime_service.h"',
        '"ii42_segment_pages.h"',
        '"ii42_semantic.h"',
        '"ii42_storage.h"',
    ):
        if forbidden_sql_dependency in sql_source:
            errors.append(
                f'{AM_SQL_SOURCE_PATH.name}: SQL-value module acquired '
                f'forbidden authority: {forbidden_sql_dependency!r}'
            )
    array_constructors = (
        'ii42_array_from_float4_values',
        'ii42_array_from_int4_values',
        'ii42_array_from_cstrings',
    )
    all_c_sources = {
        path: path.read_text(encoding='utf-8')
        for path in sorted((REPO_ROOT / 'src').glob('*.c'))
    }
    visibility_layout = 'typedef struct ii42_am_visibility_ctx'
    visibility_layout_owners = [
        path
        for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
        if visibility_layout in path.read_text(encoding='utf-8')
    ]
    if visibility_layout_owners != [AM_SCAN_HEADER_PATH]:
        owner_names = ', '.join(
            path.name for path in visibility_layout_owners
        )
        errors.append(
            f'{visibility_layout}: visibility context does not have one '
            f'header owner: {owner_names or "<none>"}'
        )
    scan_authorities = (
        ('ii42_am_visibility_begin', 'void', 5),
        ('ii42_am_visibility_begin_with_snapshot', 'void', 8),
        ('ii42_am_tid_visible', 'bool', 1),
        ('ii42_am_tid_visible_as', 'bool', 6),
        ('ii42_am_visibility_end', 'void', 7),
    )
    for authority, return_type, am_caller_count in scan_authorities:
        definition = re.compile(
            rf'\n{return_type}\n{re.escape(authority)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_SCAN_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{authority}: scan visibility authority does not have one '
                f'module owner: {owner_names or "<none>"}'
            )
        if scan_header.count(f'{authority}(') != 1:
            errors.append(
                f'{AM_SCAN_HEADER_PATH.name}: {authority} does not have one '
                'typed declaration'
            )
        if source.count(f'{authority}(') != am_caller_count:
            errors.append(
                f'{AM_SOURCE_PATH.name}: {authority} caller count changed '
                f'from {am_caller_count}'
            )
    for required in (
        'GetActiveSnapshot()',
        'table_index_fetch_begin(heap_relation)',
        'table_index_fetch_reset(visibility->fetch)',
        'table_index_fetch_tuple(',
        '&call_again,',
        '&all_dead)',
        '} while (call_again);',
        '&TTSOpsBufferHeapTuple',
        '*visible_tid_out = visibility->slot->tts_tid;',
        '*visible_tid_out = heap_tid;',
        'ExecDropSingleTupleTableSlot(visibility->slot);',
        'table_index_fetch_end(visibility->fetch);',
        'visibility->snapshot = NULL;',
    ):
        if required not in scan_source:
            errors.append(
                f'{AM_SCAN_SOURCE_PATH.name}: visibility protocol is '
                f'missing {required!r}'
            )
    if scan_source.count('ExecClearTuple(visibility->slot);') != 2:
        errors.append(
            f'{AM_SCAN_SOURCE_PATH.name}: visibility probes do not clear '
            'the tuple slot on both paths'
        )
    slot_release = scan_source.find(
        'ExecDropSingleTupleTableSlot(visibility->slot);'
    )
    fetch_release = scan_source.find(
        'table_index_fetch_end(visibility->fetch);'
    )
    if slot_release < 0 or fetch_release < 0 or slot_release > fetch_release:
        errors.append(
            f'{AM_SCAN_SOURCE_PATH.name}: visibility cleanup does not '
            'release the slot before the table fetch'
        )
    for forbidden in (
        'static ',
        'PG_FUNCTION_INFO',
        'BackgroundWorker',
        'SPI_',
        'GenericXLog',
        'XLog',
        'LockRelation',
        'ii42_am_build',
        'ii42_am_maintenance',
        'ii42_am_meta',
        'ii42_am_mutation',
        'ii42_am_options',
        'ii42_page_query',
        'ii42_posting',
        'ii42_runtime',
        'ii42_segment',
        'ii42_semantic',
        'ii42_storage',
    ):
        if forbidden in scan_source:
            errors.append(
                f'{AM_SCAN_SOURCE_PATH.name}: visibility slice acquired '
                f'forbidden authority {forbidden!r}'
            )
    if '#include "ii42_am_scan.h"' not in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: scan visibility contract is not included'
        )
    makefile = MAKEFILE_PATH.read_text(encoding='utf-8')
    if makefile.count('src/ii42_am_scan.o') != 1:
        errors.append('Makefile: scan visibility authority has no sole object')
    cmake = CMAKE_PATH.read_text(encoding='utf-8')
    if 'src/ii42_am_scan.c' in cmake:
        errors.append(
            'CMakeLists.txt: PostgreSQL scan authority entered core target'
        )
    maintenance_authorities = {
        'ii42_am_try_maintenance_lock': 2,
        'ii42_am_lock_maintenance_xact': 1,
        'ii42_am_try_accelerator_build_lock': 1,
        'ii42_am_accelerator_build_unlock': 1,
        'ii42_am_try_session_maintenance_lock': 3,
        'ii42_am_maintenance_lock_held': 1,
        'ii42_am_maintenance_unlock': 5,
        'ii42_am_session_maintenance_unlock': 4,
        'ii42_am_lock_append': 5,
        'ii42_am_unlock_append': 7,
        'ii42_am_lock_writer_barrier_oid': 3,
        'ii42_am_unlock_writer_barrier_oid': 4,
        'ii42_am_pin_maintenance_xact': 5,
        'ii42_am_pinned_maintenance_mode': 3,
        'ii42_am_pinned_maintenances_present': 1,
        'ii42_am_reparent_pinned_maintenances': 1,
        'ii42_am_clear_pinned_maintenances': 2,
        'ii42_am_maintenance_tracking_enabled': 5,
        'ii42_am_eventual_policy_enabled': 9,
        'ii42_am_automatic_policy_enabled': 1,
        'ii42_am_foreground_maintenance_enabled': 1,
        'ii42_am_maintenance_codec_error': 15,
    }
    maintenance_meta_callers = {
        'ii42_am_lock_append': 2,
        'ii42_am_unlock_append': 4,
        'ii42_am_pin_maintenance_xact': 1,
        'ii42_am_maintenance_tracking_enabled': 1,
    }
    maintenance_mutation_callers = {
        'ii42_am_lock_append': 2,
        'ii42_am_unlock_append': 2,
        'ii42_am_pin_maintenance_xact': 1,
        'ii42_am_maintenance_codec_error': 2,
    }
    maintenance_reclamation_callers = {
        'ii42_am_maintenance_codec_error': 1,
    }
    for authority, am_caller_count in maintenance_authorities.items():
        definition = re.compile(
            rf'\n(?:bool|void|LOCKMODE|LockAcquireResult)\n'
            rf'{re.escape(authority)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_MAINTENANCE_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{authority}: maintenance authority does not have one '
                f'module owner: {owner_names or "<none>"}'
            )
        if maintenance_header.count(f'{authority}(') != 1:
            errors.append(
                f'{AM_MAINTENANCE_HEADER_PATH.name}: {authority} does not '
                'have one typed declaration'
            )
        meta_caller_count = maintenance_meta_callers.get(authority, 0)
        mutation_caller_count = maintenance_mutation_callers.get(
            authority,
            0,
        )
        reclamation_caller_count = maintenance_reclamation_callers.get(
            authority,
            0,
        )
        expected_am_caller_count = (
            am_caller_count - meta_caller_count - mutation_caller_count -
            reclamation_caller_count
        )
        if source.count(f'{authority}(') != expected_am_caller_count:
            errors.append(
                f'{AM_SOURCE_PATH.name}: {authority} caller count changed '
                f'from {expected_am_caller_count}'
            )
        if meta_source.count(f'{authority}(') != meta_caller_count:
            errors.append(
                f'{AM_META_SOURCE_PATH.name}: {authority} caller count '
                f'changed from {meta_caller_count}'
            )
        if mutation_source.count(f'{authority}(') != mutation_caller_count:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: {authority} caller count '
                f'changed from {mutation_caller_count}'
            )
        if reclamation_source.count(
                f'{authority}(') != reclamation_caller_count:
            errors.append(
                f'{AM_RECLAMATION_SOURCE_PATH.name}: {authority} caller '
                f'count changed from {reclamation_caller_count}'
            )
    for layout in (
        '#define II42_AM_APPEND_LOCK_TAG UINT32_C(0x32534241)',
        '#define II42_AM_MAINTENANCE_LOCK_TAG UINT32_C(0x3253424D)',
        '#define II42_AM_WRITER_BARRIER_LOCK_TAG UINT32_C(0x32534242)',
        'typedef struct ii42_am_scoped_index',
        'static List *ii42_am_pinned_maintenances = NIL;',
    ):
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if layout in implementation_source
        ]
        if owners != [AM_MAINTENANCE_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{layout}: maintenance state does not have one owner: '
                f'{owner_names or "<none>"}'
            )
    for required in (
        'LockAcquire(&tag, ExclusiveLock, false, true)',
        'LockAcquire(&tag, ExclusiveLock, false, false)',
        'LockAcquire(&tag, ExclusiveLock, true, true)',
        'LockHeldByMe(&tag, ExclusiveLock, false)',
        'LockRelease(&tag, ExclusiveLock, false)',
        'LockRelease(&tag, ExclusiveLock, true)',
        'RelationGetRelid(index_relation)',
        'errmsg("ii42 append lock is not held")',
        'LockAcquire(&tag, lock_mode, false, dont_wait)',
        'LockRelease(&tag, lock_mode, false)',
        'MemoryContextSwitchTo(TopTransactionContext)',
        'GetCurrentSubTransactionId()',
        'entry->index_oid = InvalidOid;',
        'entry->subxid = parent_subid;',
        'ii42_am_pinned_maintenances = NIL;',
        'ii42_am_foreground_maintenance_enabled(index_relation)',
        '!ii42_am_sae_enabled(index_relation)',
        'ii42_am_validate_relation_policy(index_relation);',
        'II42_AM_CONSISTENCY_EVENTUAL',
        'II42_AM_CONSISTENCY_MANUAL',
        'II42_AM_CONSISTENCY_REALTIME',
        'failed to %s during ii42 segment maintenance',
        'Validation failed: %s.',
    ):
        if required not in maintenance_source:
            errors.append(
                f'{AM_MAINTENANCE_SOURCE_PATH.name}: transaction-scoped '
                f'maintenance authority is missing {required!r}'
            )
    for function in (
        'ii42_segment_object_ref_equal',
        'ii42_active_l0_frontier_equal',
    ):
        definition = re.compile(
            rf'\nbool\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [SEGMENTS_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: segment value comparison does not have one '
                f'codec owner: {owner_names or "<none>"}'
            )
        if segments_header.count(f'{function}(') != 1:
            errors.append(
                f'{SEGMENTS_HEADER_PATH.name}: {function} does not have '
                'one declaration'
            )
    identity_builder = 'ii42_segment_manifest_build_identity'
    identity_definition = re.compile(
        rf'\nii42_status\n{identity_builder}\('
    )
    identity_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if identity_definition.search(implementation_source)
    ]
    if identity_owners != [SEGMENTS_SOURCE_PATH]:
        owner_names = ', '.join(path.name for path in identity_owners)
        errors.append(
            f'{identity_builder}: identity-manifest builder does not have '
            f'one codec owner: {owner_names or "<none>"}'
        )
    if segments_header.count(f'{identity_builder}(') != 1:
        errors.append(
            f'{SEGMENTS_HEADER_PATH.name}: identity-manifest builder does '
            'not have one declaration'
        )
    if source.count(f'{identity_builder}(') != 4:
        errors.append(
            f'{AM_SOURCE_PATH.name}: identity-manifest caller count changed '
            'from 4'
        )
    saturating_add = 'ii42_u32_saturating_add'
    saturating_definition = re.compile(
        rf'\nuint32_t\n{saturating_add}\('
    )
    saturating_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if saturating_definition.search(implementation_source)
    ]
    if saturating_owners != [CORE_SOURCE_PATH]:
        owner_names = ', '.join(path.name for path in saturating_owners)
        errors.append(
            f'{saturating_add}: u32 saturation does not have one core '
            f'owner: {owner_names or "<none>"}'
        )
    if core_header.count(f'{saturating_add}(') != 1:
        errors.append(
            f'{CORE_HEADER_PATH.name}: u32 saturation does not have one '
            'declaration'
        )
    if source.count(f'{saturating_add}(') != 7:
        errors.append(
            f'{AM_SOURCE_PATH.name}: u32 saturation caller count changed '
            'from 7'
        )
    if meta_source.count(f'{saturating_add}(') != 4:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: u32 saturation caller count '
            'changed from 4'
        )
    saturating_u64_functions = (
        ('ii42_u64_saturating_add', 95, 3, 1, 4),
        ('ii42_u64_saturating_mul', 14, 6, 0, 0),
    )
    for function, am_callers, build_callers, meta_callers, scheduler_callers in (
        saturating_u64_functions
    ):
        definition = re.compile(
            rf'\nuint64_t\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [CORE_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: u64 saturation does not have one core '
                f'owner: {owner_names or "<none>"}'
            )
        if core_header.count(f'{function}(') != 1:
            errors.append(
                f'{CORE_HEADER_PATH.name}: u64 saturation does not have '
                f'one declaration: {function}'
            )
        caller_counts = (
            (AM_SOURCE_PATH, source, am_callers),
            (AM_BUILD_SOURCE_PATH, build_source, build_callers),
            (AM_META_SOURCE_PATH, meta_source, meta_callers),
            (
                AM_SCHEDULER_SOURCE_PATH,
                scheduler_source,
                scheduler_callers,
            ),
        )
        for path, implementation_source, expected in caller_counts:
            if implementation_source.count(f'{function}(') != expected:
                errors.append(
                    f'{path.name}: {function} caller count changed from '
                    f'{expected}'
                )
    for retired_name in (
        'ii42_am_segment_object_ref_equal',
        'ii42_am_l0_frontier_equal',
        'ii42_am_segment_maintenance_codec_error',
        'ii42_am_segment_identity_build_manifest',
        'ii42_am_saturating_add_u32',
        'ii42_am_meta_saturating_add_u32',
        'ii42_am_rebuild_estimate',
        'ii42_am_rebuild_estimate_add',
        'ii42_am_posting_heat_add_saturating',
        'ii42_am_posting_heat_multiply_saturating',
        'ii42_am_u64_saturating_add',
        'ii42_am_u64_saturating_mul',
    ):
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if retired_name in path.read_text(encoding='utf-8')
        ]
        if owners:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{retired_name}: retired AM-private helper remains in '
                f'{owner_names}'
            )
    for required in (
        'bool modified_delta = '
        'ii42_am_pinned_maintenances_present();',
        'ii42_am_reparent_pinned_maintenances(',
    ):
        if required not in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: transaction callback is missing '
                f'typed maintenance coordination {required!r}'
            )
    if source.count('ii42_am_clear_pinned_maintenances();') != 2:
        errors.append(
            f'{AM_SOURCE_PATH.name}: top-level transaction cleanup does not '
            'clear maintenance pins on both terminal paths'
        )
    for forbidden in (
        'ii42_am_pinned_maintenances =',
        'typedef struct ii42_am_scoped_index',
        'II42_AM_MAINTENANCE_LOCK_TAG',
        'II42_AM_WRITER_BARRIER_LOCK_TAG',
        'II42_AM_APPEND_LOCK_TAG',
    ):
        if forbidden in source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: retained maintenance-private '
                f'authority {forbidden!r}'
            )
    for forbidden in (
        'II42_AM_MAINTENANCE_WORKER_LOCK_TAG',
        'BackgroundWorker',
        'PG_FUNCTION_INFO',
        'SPI_',
        'Buffer ',
        'ReadBuffer',
        'GenericXLog',
        'XLog',
        'ii42_am_read_meta',
        'ii42_am_publish',
        'ii42_am_work_hint',
        'ii42_segment_',
        'ii42_page_query',
        'ii42_semantic',
        'ii42_runtime',
    ):
        if forbidden in maintenance_source:
            errors.append(
                f'{AM_MAINTENANCE_SOURCE_PATH.name}: lock/pin slice acquired '
                f'forbidden authority {forbidden!r}'
            )
    makefile = MAKEFILE_PATH.read_text(encoding='utf-8')
    if 'src/ii42_am_maintenance.o \\' not in makefile:
        errors.append('Makefile: maintenance module is not linked')
    fence_authority = 'ii42_am_acquire_convergent_reader_fence_internal'
    required_fence = 'ii42_am_acquire_convergent_reader_fence'
    if 'ii42_am_prepare_convergent_reuse_arena' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: retired-page reuse keeps the old '
            'non-owning helper contract'
        )
    if source.count(f'{fence_authority}(') != 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: reader-fence implementation remains '
            'in the entry module'
        )
    if reclamation_source.count(f'{fence_authority}(') != 2:
        errors.append(
            f'{AM_RECLAMATION_SOURCE_PATH.name}: reader-fence authority '
            'must have one definition and one narrow wrapper'
        )
    if any(
        'ii42_am_acquire_convergent_reuse_arena' in implementation_source
        for implementation_source in all_c_sources.values()
    ):
        errors.append(
            'online maintenance retains the obsolete long-lived reuse fence'
        )
    if source.count(f'{required_fence}(') != 7 or \
            reclamation_source.count(f'{required_fence}(') != 1:
        errors.append(
            'required reader fence must have one reclamation definition and '
            'seven AM callers'
        )
    if source.count(
        'volatile bool reuse_reader_fence_locked = false;'
    ) != 4:
        errors.append(
            f'{AM_SOURCE_PATH.name}: reuse callers do not retain four '
            'explicit reader-fence owners'
        )
    for function, return_type in (
        (required_fence, 'bool'),
        ('ii42_am_test_setting_enabled', 'bool'),
        ('ii42_am_test_pause_ms', 'void'),
        ('ii42_am_test_error_if_enabled', 'void'),
    ):
        definition = re.compile(
            rf'\n{return_type}\n{re.escape(function)}\('
        )
        expected_owner = (
            AM_RECLAMATION_SOURCE_PATH
            if function == required_fence
            else AM_TEST_SUPPORT_SOURCE_PATH
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [expected_owner]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: private authority owner mismatch: '
                f'{owner_names or "<none>"}'
            )
    for function, header in (
        (required_fence, reclamation_header),
        ('ii42_am_test_setting_enabled', test_support_header),
        ('ii42_am_test_pause_ms', test_support_header),
        ('ii42_am_test_error_if_enabled', test_support_header),
    ):
        if header.count(f'{function}(') != 1:
            errors.append(f'{function}: private declaration mismatch')
    for forbidden in (
        'ii42_am_mutation_publish',
        'ii42_am_preload_',
        'ii42_am_scheduler_',
        'ii42_am_work_hint_',
        'ii42_segment_pages_write_',
        'GenericXLog',
        'XLog',
        'BackgroundWorker',
        'SPI_',
    ):
        if forbidden in reclamation_source:
            errors.append(
                f'{AM_RECLAMATION_SOURCE_PATH.name}: reader-fence module '
                f'acquired forbidden authority {forbidden!r}'
            )
    reuse_cleanup = re.compile(
        r'if \(reuse_reader_fence_locked\)\s*\{\s*'
        r'UnlockRelation\(index_relation, AccessExclusiveLock\);\s*\}'
    )
    if len(reuse_cleanup.findall(source)) != 4:
        errors.append(
            f'{AM_SOURCE_PATH.name}: reuse callers do not release all four '
            'reader fences in cleanup'
        )
    fence_start = reclamation_source.find(
        f'\n{fence_authority}('
    )
    fence_end = reclamation_source.find(
        f'\nbool\n{required_fence}(',
        fence_start,
    )
    fence_body = (
        reclamation_source[fence_start:fence_end]
        if fence_start >= 0 and fence_end > fence_start
        else ''
    )
    for required in (
        'ConditionalLockRelation(',
        'ii42_segment_page_reuse_arena_build_retired(',
        'ii42_segment_page_reuse_arena_build_empty_fenced(',
        'ii42.test_reuse_reader_fence_pause_ms',
        'relation_locked = false;',
        'UnlockRelation(index_relation, AccessExclusiveLock);',
    ):
        if required not in fence_body:
            errors.append(
                f'{AM_RECLAMATION_SOURCE_PATH.name}: reuse lock-transfer '
                f'authority is missing {required!r}'
            )
    for required in (
        'def exercise_retired_page_reader_fence(',
        "SET ii42.test_prepared_cow_publication_pause_ms = '2000'",
        "SET ii42.test_reuse_reader_fence_pause_ms = '2000'",
        "SET ii42.test_convergent_root_snapshot_pause_ms ",
        "mode = 'AccessShareLock'",
        "and prepared_reader_ids == ['stable']",
        "and handoff_reused_blocks > 0",
        "normal_ids == oracle_ids == ['stable']",
    ):
        if required not in same_index_concurrency:
            errors.append(
                f'{SAME_INDEX_CONCURRENCY_PATH.name}: retired-page fence '
                f'gate is missing {required!r}'
            )
    retirement_append_start = mutation_source.find(
        '\nbool\nii42_am_mutation_append_l0_record('
    )
    retirement_append_end = len(mutation_source)
    retirement_append_body = (
        mutation_source[retirement_append_start:retirement_append_end]
        if (
            retirement_append_start >= 0 and
            retirement_append_end > retirement_append_start
        )
        else ''
    )
    for required in (
        'uint64 expected_l0_born_sequence',
        'current_source.version.born_sequence >',
        'expected_l0_born_sequence &&',
        'current_source.retirement.retirement_sequence != 0',
    ):
        if required not in retirement_append_body:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: L0 retirement '
                'incarnation guard '
                f'is missing {required!r}'
            )
    append_wrapper_start = source.find(
        '\nstatic bool\nii42_am_append_convergent_l0_record('
    )
    append_wrapper_end = source.find(
        '\nstatic int\nii42_am_cmp_l0_text_atom(',
        append_wrapper_start,
    )
    append_wrapper_body = (
        source[append_wrapper_start:append_wrapper_end]
        if (
            append_wrapper_start >= 0 and
            append_wrapper_end > append_wrapper_start
        )
        else ''
    )
    for required in (
        'volatile bool maintenance_due = false;',
        'ii42_am_mutation_append_l0_record(',
        'PG_FINALLY();',
        'if (maintenance_due)',
        'ii42_am_work_hint_mark(',
        'II42_AM_WORK_HINT_MAINTENANCE',
    ):
        if required not in append_wrapper_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: linked-L0 scheduling adapter is '
                f'missing {required!r}'
            )
    for required in (
        'bool *maintenance_due_out',
        '*maintenance_due_out = false;',
        '*maintenance_due_out = true;',
    ):
        if required not in retirement_append_body:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 mutation '
                f'outcome is missing {required!r}'
            )
    if retirement_append_body.count('*maintenance_due_out = true;') != 2:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 mutation does not '
            'preserve both urgent-maintenance outcomes'
        )
    for forbidden in (
        'ii42_am_work_hint_',
        'II42_AM_WORK_HINT_',
        'ii42_shared_preload',
        'BackgroundWorker',
        'SPI_',
    ):
        if forbidden in retirement_append_body:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 mutation '
                f'acquired scheduling authority: {forbidden!r}'
            )
    for forbidden in (
        'ReadBuffer',
        'LockBuffer',
        'GenericXLog',
        'ii42_segment_pages_',
        'ii42_am_lock_append',
    ):
        if forbidden in append_wrapper_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: linked-L0 scheduling adapter '
                f'retains mutation authority: {forbidden!r}'
            )
    cow_mutation = 'ii42_am_mutation_publish_cow_manifest'
    cow_mutation_start = mutation_source.find(
        f'\nbool\n{cow_mutation}('
    )
    cow_mutation_end = mutation_source.find(
        '\n\nXLogRecPtr\nii42_am_l0_rotate_active_locked(',
        cow_mutation_start,
    )
    cow_mutation_body = (
        mutation_source[cow_mutation_start:cow_mutation_end]
        if (
            cow_mutation_start >= 0 and
            cow_mutation_end > cow_mutation_start
        )
        else ''
    )
    cow_wrapper_start = source.find(
        '\nstatic bool\nii42_am_publish_cow_manifest('
    )
    cow_wrapper_end = source.find(
        '\nstatic ii42_am_pending_seal_outcome',
        cow_wrapper_start,
    )
    cow_wrapper_body = (
        source[cow_wrapper_start:cow_wrapper_end]
        if (
            cow_wrapper_start >= 0 and
            cow_wrapper_end > cow_wrapper_start
        )
        else ''
    )
    reclaim_start = source.find(
        '\nstatic ii42_am_retired_reclaim_outcome\n'
        'ii42_am_try_reclaim_retired_pages('
    )
    reclaim_end = source.find(
        '\nstatic bool\nii42_am_term_structural_fold_select(',
        reclaim_start,
    )
    reclaim_body = (
        source[reclaim_start:reclaim_end]
        if reclaim_start >= 0 and reclaim_end > reclaim_start
        else ''
    )
    cow_definition = re.compile(
        rf'\nbool\n{re.escape(cow_mutation)}\('
    )
    cow_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if cow_definition.search(implementation_source)
    ]
    if cow_owners != [AM_MUTATION_SOURCE_PATH]:
        owner_names = ', '.join(path.name for path in cow_owners)
        errors.append(
            f'{cow_mutation}: COW root publication does not have one '
            f'mutation owner: {owner_names or "<none>"}'
        )
    if mutation_header.count(f'{cow_mutation}(') != 1:
        errors.append(
            f'{AM_MUTATION_HEADER_PATH.name}: COW root publication does '
            'not have one declaration'
        )
    if source.count(f'{cow_mutation}(') != 1:
        errors.append(
            f'{AM_SOURCE_PATH.name}: COW root publication does not have '
            'one scheduling-adapter caller'
        )
    for required in (
        'ii42_am_lock_append(index_relation);',
        'ii42_am_read_meta(index_relation, &latest_meta);',
        'ii42_am_segment_read_root_from_meta(',
        'ii42_segment_object_ref_equal(',
        'ii42_active_l0_frontier_equal(',
        'ii42_segment_manifest_validate_published(',
        'ii42_segment_read_root_seal_pending(',
        'ii42_segment_read_root_replace_manifest(',
        'ReadBufferExtended(',
        'GenericXLogStart(index_relation)',
        'ii42_am_l0_store_root(next_meta_page, &next_root);',
        'XLogFlush(publish_lsn);',
        'ii42_segment_pages_publish_fsm_handoff(',
        'PG_FINALLY();',
        'ii42_am_unlock_append(index_relation);',
    ):
        if required not in cow_mutation_body:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: COW root publication '
                f'is missing {required!r}'
            )
    for forbidden in (
        'ii42_am_shared_preload_',
        'ii42_am_schedule_',
        'ii42_am_work_hint_',
        'BackgroundWorker',
        'SPI_',
    ):
        if forbidden in cow_mutation_body:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: COW root publication '
                f'acquired adjacent authority: {forbidden!r}'
            )
    for required in (
        'ii42_am_mutation_publish_cow_manifest(',
        'ii42_am_read_meta(index_relation, &current_meta);',
        'ii42_am_preload_rekey_generation(',
        'ii42_am_preload_retire_obsolete(',
        'ii42_am_schedule_auto_preload(index_relation);',
    ):
        if required not in cow_wrapper_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: COW publication adapter is '
                f'missing {required!r}'
            )
    for forbidden in (
        'ReadBuffer',
        'LockBuffer',
        'GenericXLog',
        'XLogFlush',
        'ii42_segment_pages_publish_fsm_handoff',
        'ii42_am_lock_append',
        'ii42_am_unlock_append',
    ):
        if forbidden in cow_wrapper_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: COW publication adapter retains '
                f'mutation authority: {forbidden!r}'
            )
    for required in (
        'void\nii42_am_preload_rekey_generation(',
        'ii42_am_preload_kind_rekey_safe(kind)',
        'ii42_am_preload_retire_entry_locked(entry);',
        'ii42_am_preload_hash_remove_locked(entry);',
        'entry->meta = *current_meta;',
        'ii42_am_preload_hash_insert_locked((int) index);',
        'ii42_am_preload_retire_obsolete_locked(',
    ):
        if required not in preload_source:
            errors.append(
                f'{AM_PRELOAD_SOURCE_PATH.name}: pure-reclamation preload '
                f'rekey is missing {required!r}'
            )
    rekey_safe_start = preload_source.find(
        '\nstatic bool\nii42_am_preload_kind_rekey_safe('
    )
    rekey_safe_end = preload_source.find(
        '\nstatic bool\nii42_am_preload_payload_size_valid(',
        rekey_safe_start,
    )
    rekey_safe_body = (
        preload_source[rekey_safe_start:rekey_safe_end]
        if rekey_safe_start >= 0 and rekey_safe_end > rekey_safe_start
        else ''
    )
    if 'II42_AM_PRELOAD_RESIDENT_FOLD' in rekey_safe_body:
        errors.append(
            f'{AM_PRELOAD_SOURCE_PATH.name}: root-embedded resident folds '
            'must be rebuilt rather than rekeyed after reclamation'
        )
    if 'ii42_am_preload_rekey_generation(' not in preload_header:
        errors.append(
            f'{AM_PRELOAD_HEADER_PATH.name}: pure-reclamation preload '
            'rekey has no declaration'
        )
    for required in (
        'ii42_segment_manifest_build_identity(',
        'ii42_segment_pages_write_cow_reclaim(',
        'false,\n                    &build_meta);',
    ):
        if required not in reclaim_body:
            errors.append(
                f'{AM_SOURCE_PATH.name}: pure reclamation does not preserve '
                f'preload authority: missing {required!r}'
            )
    if source.count('false,\n                    &build_meta);') != 1:
        errors.append(
            f'{AM_SOURCE_PATH.name}: only pure reclamation may rekey shared '
            'preload generations'
        )
    for function, return_type in (
        ('ii42_am_l0_require_meta_root', 'void'),
        ('ii42_am_l0_store_root', 'void'),
        ('ii42_am_l0_rotate_active_locked', 'XLogRecPtr'),
        ('ii42_am_mutation_append_l0_record', 'bool'),
    ):
        definition = re.compile(
            rf'\n{return_type}\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_MUTATION_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: linked-L0 writer does not have one mutation '
                f'owner: {owner_names or "<none>"}'
            )
        if mutation_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_MUTATION_HEADER_PATH.name}: linked-L0 writer '
                f'{function} does not have one declaration'
            )
    for required in (
        'def exercise_shared_retirement_sequence_guard(',
        'before_vacuum_records == 2',
        'after_first_vacuum_records == 3',
        'after_second_vacuum_records == after_first_vacuum_records',
        "before_seal_normal_ids == before_seal_oracle_ids == ['stable']",
        "after_seal_normal_ids == after_seal_oracle_ids == ['stable']",
    ):
        if required not in same_index_concurrency:
            errors.append(
                f'{SAME_INDEX_CONCURRENCY_PATH.name}: shared retirement '
                f'guard is missing {required!r}'
            )
    xid_classifier = 'ii42_am_delta_record_states'
    xid_classifier_definition = re.compile(
        rf'\nbool\n{re.escape(xid_classifier)}\('
    )
    xid_classifier_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if xid_classifier_definition.search(implementation_source)
    ]
    if xid_classifier_owners != [AM_MUTATION_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in xid_classifier_owners
        )
        errors.append(
            f'{xid_classifier}: linked-L0 XID classification does not have '
            f'one mutation-module owner: {owner_names or "<none>"}'
        )
    if mutation_header.count(f'{xid_classifier}(') != 1:
        errors.append(
            f'{AM_MUTATION_HEADER_PATH.name}: linked-L0 XID classifier does '
            'not have one declaration'
        )
    if source.count(f'{xid_classifier}(') != 5:
        errors.append(
            f'{AM_SOURCE_PATH.name}: linked-L0 XID classifier does not have '
            'the five frozen callers'
        )
    for layout in (
        '#define II42_AM_ABORTED_DELTA_XID BootstrapTransactionId',
        'typedef enum ii42_am_delta_xid_state',
    ):
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if layout in path.read_text(encoding='utf-8')
        ]
        if owners != [AM_MUTATION_HEADER_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{layout}: linked-L0 XID value does not have one mutation '
                f'header owner: {owner_names or "<none>"}'
            )
    rotation_limit = 'ii42_am_active_l0_rotation_record_limit'
    rotation_limit_definition = re.compile(
        rf'\nuint32\n{re.escape(rotation_limit)}\('
    )
    rotation_limit_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if rotation_limit_definition.search(implementation_source)
    ]
    if rotation_limit_owners != [AM_MUTATION_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in rotation_limit_owners
        )
        errors.append(
            f'{rotation_limit}: L0 rotation policy does not have one '
            f'mutation-module owner: {owner_names or "<none>"}'
        )
    if mutation_header.count(f'{rotation_limit}(') != 1:
        errors.append(
            f'{AM_MUTATION_HEADER_PATH.name}: L0 rotation policy does not '
            'have one declaration'
        )
    if mutation_source.count(f'{rotation_limit}(') != 4:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: L0 lifecycle does not have the '
            'three policy callers beside its definition'
        )
    if f'{rotation_limit}(' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: L0 rotation policy leaked back into '
            'the AM entry module'
        )
    for lifecycle_predicate in (
        'ii42_am_active_l0_checkpoint_due',
        'ii42_am_accelerator_refresh_record_limit',
    ):
        if mutation_header.count(f'{lifecycle_predicate}(') != 1:
            errors.append(
                f'{AM_MUTATION_HEADER_PATH.name}: lifecycle predicate '
                f'{lifecycle_predicate} does not have one declaration'
            )
        if mutation_source.count(f'{lifecycle_predicate}(') != 1:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: lifecycle predicate '
                f'{lifecycle_predicate} does not have one definition'
            )
    rotation_limit_start = mutation_source.find(
        f'\nuint32\n{rotation_limit}('
    )
    rotation_limit_end = mutation_source.find(
        f'\n\nbool\n{xid_classifier}(',
        rotation_limit_start,
    )
    if rotation_limit_start < 0 or rotation_limit_end < 0:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: L0 rotation policy closure '
            'not found'
        )
    else:
        rotation_limit_block = mutation_source[
            rotation_limit_start:rotation_limit_end
        ]
        for required in (
            '"ii42.test_convergent_l0_rotation_records"',
            'return II42_ACTIVE_L0_ROTATION_RECORDS;',
            'superuser_arg(GetOuterUserId())',
            'value > II42_ACTIVE_L0_MAX_RECORDS',
        ):
            if required not in rotation_limit_block:
                errors.append(
                    f'{AM_MUTATION_SOURCE_PATH.name}: L0 rotation policy '
                    f'is missing {required!r}'
                )
    xid_classifier_start = mutation_source.find(
        f'\nbool\n{xid_classifier}('
    )
    if xid_classifier_start < 0:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 XID classifier '
            'closure not found'
        )
    else:
        xid_classifier_end = mutation_source.find(
            '\n\nbool\nii42_am_l0_expected_source_matches(',
            xid_classifier_start,
        )
        xid_classifier_block = mutation_source[
            xid_classifier_start:xid_classifier_end
        ] if xid_classifier_end > xid_classifier_start else ''
        for required in (
            'record_count > 0',
            'TransactionIdIsNormal(record_xid)',
            '!TransactionIdIsCurrentTransactionId(record_xid)',
            'ii42_am_frozen_xid_audit_enabled()',
            'LWLockAcquire(XactTruncationLock, LW_SHARED)',
            'record_xid == II42_AM_ABORTED_DELTA_XID',
            'record_xid == FrozenTransactionId',
            'TransamVariables->oldestClogXid',
            'TransactionIdDidCommit(record_xid)',
            'TransactionIdDidAbort(record_xid)',
            'LWLockRelease(XactTruncationLock)',
            'return has_normal_xid;',
        ):
            if required not in xid_classifier_block:
                errors.append(
                    f'{AM_MUTATION_SOURCE_PATH.name}: linked-L0 XID '
                    f'classifier is missing {required!r}'
                )
    source_guard = 'ii42_am_l0_expected_source_matches'
    source_guard_definition = re.compile(
        rf'\nbool\n{re.escape(source_guard)}\('
    )
    source_guard_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if source_guard_definition.search(implementation_source)
    ]
    if source_guard_owners != [AM_MUTATION_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in source_guard_owners
        )
        errors.append(
            f'{source_guard}: L0 source-incarnation guard does not have one '
            f'mutation-module owner: {owner_names or "<none>"}'
        )
    if mutation_header.count(f'{source_guard}(') != 1:
        errors.append(
            f'{AM_MUTATION_HEADER_PATH.name}: L0 source-incarnation guard '
            'does not have one declaration'
        )
    if mutation_source.count(f'{source_guard}(') != 2:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: L0 source-incarnation guard '
            'does not have its definition and append caller'
        )
    if source.count(f'{source_guard}(') != 1:
        errors.append(
            f'{AM_SOURCE_PATH.name}: L0 source-incarnation guard does not '
            'have the sole VACUUM caller'
        )
    source_guard_start = mutation_source.find(
        f'\nbool\n{source_guard}('
    )
    if source_guard_start < 0:
        errors.append(
            f'{AM_MUTATION_SOURCE_PATH.name}: L0 source-incarnation guard '
            'closure not found'
        )
    else:
        source_guard_end = mutation_source.find(
            '\n\nstatic void\nii42_am_set_generic_page_content_len(',
            source_guard_start,
        )
        source_guard_block = mutation_source[
            source_guard_start:source_guard_end
        ] if source_guard_end > source_guard_start else ''
        for required in (
            'record_kind != II42_L0_RECORD_RETIRE',
            'ii42_document_cow_records_equal(expected, current)',
            'expected->version.document_slot',
            'expected->version.born_sequence',
            'expected->version.heap_block',
            'expected->version.heap_offset',
            'expected->version.document_length',
            'expected->version.semantic_input_fingerprint',
            'II42_DOCUMENT_FINGERPRINT_BYTES',
        ):
            if required not in source_guard_block:
                errors.append(
                    f'{AM_MUTATION_SOURCE_PATH.name}: L0 source-incarnation '
                    f'guard is missing {required!r}'
                )
        for forbidden in (
            'Relation ',
            'Buffer ',
            'LockBuffer',
            'MarkBufferDirty',
            'XLog',
            'ii42_am_read_meta',
            'ii42_am_append',
            'ii42_am_publish',
            'ii42_segment_pages_',
            'ii42_runtime',
            'ii42_model',
            'ii42_page_query',
        ):
            if forbidden in source_guard_block:
                errors.append(
                    f'{AM_MUTATION_SOURCE_PATH.name}: L0 source-incarnation '
                    f'guard acquired forbidden authority: {forbidden!r}'
                )
    for required in (
        '"ii42.test_require_frozen_delta_xids"',
        'GetConfigOptionByName(',
        'parse_bool(setting, &enabled)',
        'superuser_arg(GetOuterUserId())',
        'ERRCODE_INSUFFICIENT_PRIVILEGE',
    ):
        if required not in mutation_source:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: frozen-XID audit is '
                f'missing {required!r}'
            )
    for forbidden in (
        'ii42_am_work_hint_',
        'ii42_shared_preload',
        'ii42_am_schedule_',
        'ii42_am_rebuild_',
        'BackgroundWorker',
        'SPI_',
        'ii42_page_query',
        'ii42_semantic',
        'ii42_runtime',
        'ii42_model',
    ):
        if forbidden in mutation_source:
            errors.append(
                f'{AM_MUTATION_SOURCE_PATH.name}: mutation module acquired '
                f'forbidden adjacent authority: {forbidden!r}'
            )
    makefile = MAKEFILE_PATH.read_text(encoding='utf-8')
    if 'src/ii42_am_mutation.o \\' not in makefile:
        errors.append('Makefile: mutation module is not linked')
    for layout, expected_owner in (
        ('typedef enum ii42_am_rebuild_builder', AM_BUILD_HEADER_PATH),
        ('typedef struct ii42_am_rebuild_workload', AM_BUILD_HEADER_PATH),
        ('typedef struct ii42_am_rebuild_output', AM_BUILD_HEADER_PATH),
    ):
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if layout in path.read_text(encoding='utf-8')
        ]
        if owners != [expected_owner]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{layout}: build value does not have one header owner: '
                f'{owner_names or "<none>"}'
            )
    for function, return_type in (
        ('ii42_am_rebuild_memory_budget_choose', 'bool'),
        ('ii42_am_rebuild_builder_name', r'const char \*'),
        ('ii42_am_rebuild_output_release', 'void'),
    ):
        definition = re.compile(
            rf'\n{return_type}\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_BUILD_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: build admission does not have one owner: '
                f'{owner_names or "<none>"}'
            )
        if build_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_BUILD_HEADER_PATH.name}: build admission does not '
                f'have one declaration: {function}'
            )
    rebuild_output_start = build_header.find(
        'typedef struct ii42_am_rebuild_output'
    )
    rebuild_output_end = build_header.find(
        '} ii42_am_rebuild_output;',
        rebuild_output_start,
    )
    if rebuild_output_start < 0 or rebuild_output_end < 0:
        errors.append(
            f'{AM_BUILD_HEADER_PATH.name}: rebuild output layout not found'
        )
    else:
        rebuild_output_block = build_header[
            rebuild_output_start:
            rebuild_output_end + len('} ii42_am_rebuild_output;')
        ]
        for field in (
            'Oid source_type;',
            'ItemPointerData *doc_tids;',
            'size_t num_docs;',
            'uint8_t *index_bytes;',
            'size_t index_bytes_len;',
            'char semantic_signature[II42_AM_SEMANTIC_SIGNATURE_LEN + 1];',
            'ii42_index index;',
            'bool index_valid;',
            'bool segment_sae;',
            'BufFile *segment_semantic_postings;',
            'size_t segment_semantic_posting_count;',
            'uint8 *segment_semantic_input_fingerprints;',
            'size_t segment_semantic_input_fingerprint_count;',
            'double heap_tuples;',
            'double index_tuples;',
        ):
            if rebuild_output_block.count(field) != 1:
                errors.append(
                    f'{AM_BUILD_HEADER_PATH.name}: rebuild output field does '
                    f'not have one owner: {field!r}'
                )
    rebuild_release_start = build_source.find(
        '\nstatic void\n'
        'ii42_am_rebuild_output_release_materialized_sources('
    )
    rebuild_release_end = build_source.find(
        '\nvoid\nii42_am_rebuild_output_release(',
        rebuild_release_start,
    )
    public_release_end = build_source.find(
        '\n\n/* Prepare MAIN for one caller-serialized replacement',
        rebuild_release_start,
    )
    if (
        rebuild_release_start < 0 or
        rebuild_release_end < 0 or
        public_release_end < 0
    ):
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: rebuild output release not found'
        )
    else:
        rebuild_release_block = build_source[
            rebuild_release_start:public_release_end
        ]
        for required in (
            'if (output == NULL)',
            'pfree(output->doc_tids);',
            'free(output->index_bytes);',
            'BufFileClose(output->segment_semantic_postings);',
            'pfree(output->segment_semantic_input_fingerprints);',
            'if (output->index_valid)',
            'ii42_index_free(&output->index);',
            'ii42_am_rebuild_output_release_materialized_sources(output);',
            'memset(output, 0, sizeof(*output));',
        ):
            if required not in rebuild_release_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: rebuild output release is '
                    f'missing {required!r}'
                )
        for forbidden in (
            'Relation',
            'ii42_am_meta',
            'ii42_segment_pages',
            'ii42_runtime',
            'ii42_model',
            'LockRelation',
            'LWLock',
            'XLog',
            'GenericXLog',
            'ereport(',
            'palloc',
        ):
            if forbidden in rebuild_release_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: rebuild output release '
                    f'acquired forbidden authority: {forbidden!r}'
                )
    replacement_prepare = 'ii42_am_prepare_replacement_relation'
    replacement_prepare_definition = re.compile(
        rf'\nuint64\n{re.escape(replacement_prepare)}\('
    )
    replacement_prepare_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if replacement_prepare_definition.search(implementation_source)
    ]
    if replacement_prepare_owners != [AM_BUILD_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in replacement_prepare_owners
        )
        errors.append(
            f'{replacement_prepare}: replacement preparation does not have '
            f'one build-module owner: {owner_names or "<none>"}'
        )
    if build_header.count(f'{replacement_prepare}(') != 1:
        errors.append(
            f'{AM_BUILD_HEADER_PATH.name}: replacement preparation does not '
            'have one declaration'
        )
    if source.count(f'{replacement_prepare}(') != 1:
        errors.append(
            f'{AM_SOURCE_PATH.name}: replacement preparation does not have '
            'exactly one caller'
        )
    replacement_prepare_start = build_source.find(
        f'\nuint64\n{replacement_prepare}('
    )
    replacement_prepare_end = build_source.find(
        '\n\n/* Publish one checked rebuild bundle',
        replacement_prepare_start,
    )
    replacement_prepare_block = ''
    if replacement_prepare_start < 0 or replacement_prepare_end < 0:
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: replacement preparation closure '
            'not found'
        )
    else:
        replacement_prepare_block = build_source[
            replacement_prepare_start:replacement_prepare_end
        ]
        for required in (
            'ii42_am_meta_page empty_meta;',
            'uint64 rebuild_count;',
            'memset(&empty_meta, 0, sizeof(empty_meta));',
            'empty_meta.magic = II42_AM_MAGIC;',
            'empty_meta.version = II42_AM_VERSION;',
            'empty_meta.page_kind = II42_AM_PAGE_META;',
            'rebuild_count = ii42_am_next_rebuild_count(indexRelation);',
            'RelationTruncate(indexRelation, 0);',
            'ii42_am_write_new_page(',
            '&empty_meta,',
            'sizeof(empty_meta)',
            'return rebuild_count;',
        ):
            if required not in replacement_prepare_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: replacement preparation '
                    f'is missing {required!r}'
                )
        for forbidden in (
            'LockAcquire',
            'LockRelease',
            'LockRelation',
            'ii42_am_lock_generation_barrier',
            'ii42_am_runtime_contract_hash(',
            'ii42_am_publish_replacement_segments(',
            'ii42_am_write_init_page_at(',
            'ii42_am_next_cache_epoch(',
            'table_index_build_scan',
            'ii42_am_rebuild_memory_budget_choose(',
            'ii42_am_schedule',
            'ii42_am_preload',
            'ii42_am_maintenance',
            'ii42_am_convergent_mutation',
            'ambulkdelete',
            'ii42_page_query',
            'ii42_runtime',
            'ii42_model',
            'PG_TRY',
            'PG_FINALLY',
        ):
            if forbidden in replacement_prepare_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: replacement preparation '
                    f'acquired forbidden authority: {forbidden!r}'
                )
    rebuild_publisher = 'ii42_am_publish_replacement_segments'
    rebuild_publisher_definition = re.compile(
        rf'\nvoid\n{re.escape(rebuild_publisher)}\('
    )
    rebuild_publisher_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if rebuild_publisher_definition.search(implementation_source)
    ]
    if rebuild_publisher_owners != [AM_BUILD_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in rebuild_publisher_owners
        )
        errors.append(
            f'{rebuild_publisher}: rebuild publisher does not have one '
            f'build-module owner: {owner_names or "<none>"}'
        )
    if build_header.count(f'{rebuild_publisher}(') != 1:
        errors.append(
            f'{AM_BUILD_HEADER_PATH.name}: rebuild publisher does not have '
            'one declaration'
        )
    if source.count(f'{rebuild_publisher}(') != 2:
        errors.append(
            f'{AM_SOURCE_PATH.name}: rebuild publisher does not have exactly '
            'the MAIN and INIT callers'
        )
    rebuild_publisher_start = build_source.find(
        f'\nvoid\n{rebuild_publisher}('
    )
    rebuild_publisher_end = build_source.find(
        '\n\nstatic uint64\nii42_am_budget_headroom_limit(',
        rebuild_publisher_start,
    )
    rebuild_publisher_block = ''
    if rebuild_publisher_start < 0 or rebuild_publisher_end < 0:
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: rebuild publisher closure not '
            'found'
        )
    else:
        rebuild_publisher_block = build_source[
            rebuild_publisher_start:rebuild_publisher_end
        ]
        if '#define II42_AM_REBUILD_ID_SHIFT 11U' not in build_source:
            errors.append(
                f'{AM_BUILD_SOURCE_PATH.name}: rebuild ID namespace does '
                'not reserve the checked multi-segment range'
            )
        for required in (
            'const uint8 contract_hash[II42_SEGMENT_CONTRACT_HASH_BYTES]',
            'replacement == NULL || !replacement->index_valid',
            '(rebuild_count << II42_AM_REBUILD_ID_SHIFT) | UINT64_C(1);',
            'first_segment_id = manifest_id + 1;',
            'active_l0_segment_id = first_segment_id;',
            'memcpy(\n            manifest->contract_hash,',
            'ii42_initial_fold_stream_create(',
            'ii42_segment_query_contract_build(',
            'ii42_segment_pages_write_sealed_bundle_fork(',
            'ii42_segment_pages_write_streamed_initial_folded_bundle_fork(',
            'ii42_am_publish_rebuild_meta(',
            'PG_TRY();',
            'PG_FINALLY();',
            'ii42_initial_fold_stream_free(cleanup->fold_publish.stream);',
            'ii42_segment_query_contract_free(&cleanup->query_contract);',
            'ii42_segment_manifest_free(&cleanup->manifest);',
        ):
            if required not in rebuild_publisher_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: rebuild publisher is '
                    f'missing {required!r}'
                )
        for forbidden in (
            'ii42_am_runtime_contract_hash(',
            'RelationTruncate(',
            'ii42_am_lock_generation_barrier',
            'ii42_am_next_rebuild_count(',
            'ii42_am_next_cache_epoch(',
            'ii42_am_write_new_page(',
            'ii42_am_write_init_page_at(',
            'ii42_am_schedule',
            'ii42_am_preload',
            'ii42_runtime',
            'ii42_model',
            'ii42_segment_payload_partition_contiguous(',
            'rd_options',
            'GetConfigOption',
            'DefineCustom',
            'IndexScan',
            'ambulkdelete',
        ):
            if forbidden in rebuild_publisher_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: rebuild publisher '
                    f'acquired forbidden authority: {forbidden!r}'
                )
    main_publish_start = source.find(
        '\nstatic void\nii42_am_write_convergent_segment_relation('
    )
    main_publish_end = source.find(
        '\n\nstatic void\nii42_am_reindex_relation(',
        main_publish_start,
    )
    init_publish_start = source.find('\nstatic void\nii42_ambuildempty(')
    init_publish_end = source.find(
        '\n\nstatic char *\nii42_am_compile_semantic_documents_query(',
        init_publish_start,
    )
    for label, start, end, fork_setup, fork_number in (
        (
            'MAIN',
            main_publish_start,
            main_publish_end,
            f'{replacement_prepare}(',
            'MAIN_FORKNUM',
        ),
        (
            'INIT',
            init_publish_start,
            init_publish_end,
            'ii42_am_write_init_page_at(',
            'INIT_FORKNUM',
        ),
    ):
        if start < 0 or end < 0:
            errors.append(
                f'{AM_SOURCE_PATH.name}: {label} rebuild publisher caller '
                'closure not found'
            )
            continue
        caller_block = source[start:end]
        setup_at = caller_block.find(fork_setup)
        hash_at = caller_block.find(
            'ii42_am_runtime_contract_hash(indexRelation, contract_hash);'
        )
        publish_at = caller_block.find(f'{rebuild_publisher}(')
        if not (0 <= setup_at < hash_at < publish_at):
            errors.append(
                f'{AM_SOURCE_PATH.name}: {label} caller does not prepare '
                'its fork, derive its immutable hash, then publish'
            )
        for required in (
            'uint8 contract_hash[II42_SEGMENT_CONTRACT_HASH_BYTES];',
            fork_number,
            'replacement,\n            contract_hash,',
        ):
            if required not in caller_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: {label} caller is missing '
                    f'{required!r}'
                )
    if main_publish_start >= 0 and main_publish_end >= 0:
        main_publish_block = source[
            main_publish_start:main_publish_end
        ]
        lock_at = main_publish_block.find(
            'ii42_am_lock_generation_barrier(indexRelation);'
        )
        prepare_at = main_publish_block.find(f'{replacement_prepare}(')
        hash_at = main_publish_block.find(
            'ii42_am_runtime_contract_hash(indexRelation, contract_hash);'
        )
        publish_at = main_publish_block.find(f'{rebuild_publisher}(')
        unlock_at = main_publish_block.find(
            'ii42_am_unlock_generation_barrier(indexRelation);'
        )
        if not (
            0 <= lock_at < prepare_at < hash_at < publish_at < unlock_at
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: MAIN replacement lock, prepare, '
                'hash, publish, and unlock order changed'
            )
        for forbidden in (
            'ii42_am_meta_page empty_meta;',
            'empty_meta.magic = II42_AM_MAGIC;',
            'empty_meta.version = II42_AM_VERSION;',
            'empty_meta.page_kind = II42_AM_PAGE_META;',
            'RelationTruncate(indexRelation, 0);',
            'ii42_am_next_rebuild_count(indexRelation);',
            'ii42_am_write_new_page(',
        ):
            if forbidden in main_publish_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: MAIN replacement retained '
                    f'duplicate preparation: {forbidden!r}'
                )
    for retired_rebuild_output_name in (
        'ii42_am_replacement',
        'ii42_am_replacement_free',
    ):
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if retired_rebuild_output_name in path.read_text(encoding='utf-8')
        ]
        if owners:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{retired_rebuild_output_name}: retired rebuild output '
                f'name remains in {owner_names}'
            )
    fingerprint_predicate = 'ii42_document_fingerprint_is_zero'
    out_of_line_fingerprint_definition = re.compile(
        rf'\n(?:static\s+)?bool\n{fingerprint_predicate}\('
    )
    out_of_line_fingerprint_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if out_of_line_fingerprint_definition.search(implementation_source)
    ]
    if out_of_line_fingerprint_owners:
        owner_names = ', '.join(
            path.name for path in out_of_line_fingerprint_owners
        )
        errors.append(
            f'{fingerprint_predicate}: out-of-line fingerprint predicate '
            f'remains in {owner_names}'
        )
    fingerprint_definition = re.compile(
        rf'\nstatic inline bool\n{fingerprint_predicate}'
        rf'\(const uint8_t \*fingerprint\)\n\{{(?P<body>.*?)\n\}}\n',
        re.DOTALL,
    )
    fingerprint_matches = list(
        fingerprint_definition.finditer(segments_header)
    )
    if len(fingerprint_matches) != 1:
        errors.append(
            f'{SEGMENTS_HEADER_PATH.name}: fingerprint predicate does not '
            'have one static inline definition'
        )
    else:
        fingerprint_block = fingerprint_matches[0].group('body')
        for required_predicate_part in (
            'if (fingerprint == NULL)',
            'index < II42_DOCUMENT_FINGERPRINT_BYTES',
            'if (fingerprint[index] != 0)',
            'return false;',
            'return true;',
        ):
            if required_predicate_part not in fingerprint_block:
                errors.append(
                    f'{SEGMENTS_HEADER_PATH.name}: fingerprint predicate is '
                    f'missing {required_predicate_part!r}'
                )
        for forbidden_predicate_authority in (
            'ereport(',
            'palloc',
            'malloc',
            'Relation',
            'Lock',
            'XLog',
            'WAL',
            'ii42_am_',
            'ii42_runtime',
            'ii42_model',
        ):
            if forbidden_predicate_authority in fingerprint_block:
                errors.append(
                    f'{SEGMENTS_HEADER_PATH.name}: fingerprint predicate '
                    'acquired forbidden authority: '
                    f'{forbidden_predicate_authority!r}'
                )
    for retired_fingerprint_name in (
        'ii42_am_document_fingerprint_is_zero',
        'ii42_l0_fingerprint_is_zero',
        'ii42_document_cow_fingerprint_is_zero',
    ):
        retired_owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if retired_fingerprint_name in path.read_text(encoding='utf-8')
        ]
        if retired_owners:
            owner_names = ', '.join(path.name for path in retired_owners)
            errors.append(
                f'{retired_fingerprint_name}: retired fingerprint helper '
                f'remains in {owner_names}'
            )
    for caller_name, caller_source, expected_calls in (
        (AM_SOURCE_PATH.name, source, 3),
        (AM_BUILD_SOURCE_PATH.name, build_source, 1),
        (SEGMENTS_SOURCE_PATH.name, segments_source, 5),
        (DOCUMENT_COW_SOURCE_PATH.name, document_cow_source, 2),
    ):
        if caller_source.count(f'{fingerprint_predicate}(') != expected_calls:
            errors.append(
                f'{fingerprint_predicate}: expected {expected_calls} typed '
                f'uses in {caller_name}'
            )
    build_common_start = source.find(
        '\nstatic IndexBuildResult *\nii42_am_build_common('
    )
    build_common_end = source.find(
        '\n\nstatic IndexBuildResult *\nii42_ambuild(',
        build_common_start,
    )
    if build_common_start < 0 or build_common_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: normal rebuild output owner not found'
        )
    else:
        build_common_block = source[build_common_start:build_common_end]
        if build_common_block.count(
                'ii42_am_rebuild_output_release(&replacement);') != 1:
            errors.append(
                f'{AM_SOURCE_PATH.name}: normal rebuild output does not have '
                'one release'
            )
        if 'ii42_am_schedule_background_maintenance(indexRelation);' not in (
                build_common_block):
            errors.append(
                f'{AM_SOURCE_PATH.name}: bulk build does not schedule '
                'post-publication convergence'
            )

    fold_start = source.find(
        '\nstatic ii42_am_term_structural_fold_outcome\n'
        'ii42_am_try_term_fold('
    )
    fold_end = source.find(
        '\n\nstatic ii42_am_term_structural_fold_outcome\n'
        'ii42_am_try_structural_term_fold(',
        fold_start,
    )
    fold_block = (
        source[fold_start:fold_end]
        if fold_start >= 0 and fold_end > fold_start
        else ''
    )
    for required in (
        'ii42_am_term_structural_fold_load_term_extents(',
        'ii42_term_fold_bundle_advance_neutral_extents(',
        '&cleanup->next_fold,\n                NULL,\n                &cow_result',
        'II42_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED',
        'defer_fsm_handoff',
        'UnlockRelation(index_relation, AccessExclusiveLock);',
        'ii42_segment_pages_publish_fsm_handoff(',
    ):
        if required not in fold_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: structural term fold misses '
                f'page-native term-local input {required!r}'
            )
    if (
        '#define II42_AM_TERM_FOLD_TARGET_TAIL_EXTENTS \\\n'
        '    II42_AM_TERM_FOLD_MAX_NEW_EXTENTS'
    ) not in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: structural term fold does not retain '
            'the bounded new-extent tail'
        )
    if 'ii42_segment_pages_load_payload_range(' in fold_block:
        errors.append(
            f'{AM_SOURCE_PATH.name}: structural term fold materializes '
            'complete source segment payloads'
        )
    maintenance_counts_start = source.find(
        '\nstatic bool\nii42_am_maintenance_result_counts('
    )
    maintenance_counts_end = source.find(
        '\n\nstatic bool\n'
        'ii42_am_convergent_semantic_completion_due(',
        maintenance_counts_start,
    )
    maintenance_counts_block = (
        source[maintenance_counts_start:maintenance_counts_end]
        if maintenance_counts_start >= 0 and
        maintenance_counts_end > maintenance_counts_start
        else ''
    )
    if 'strstr(result_str, "maintained=true") != NULL' not in (
            maintenance_counts_block):
        errors.append(
            f'{AM_SOURCE_PATH.name}: no-progress maintenance can be counted '
            'as successful service'
        )
    defer_result_start = source.find(
        '\nstatic bool\n'
        'ii42_am_maintenance_result_defers_to_reconcile('
    )
    defer_result_end = source.find(
        '\n\nstatic bool\n'
        'ii42_am_convergent_semantic_completion_due(',
        defer_result_start,
    )
    defer_result_block = (
        source[defer_result_start:defer_result_end]
        if defer_result_start >= 0 and defer_result_end > defer_result_start
        else ''
    )
    for required in (
        'strstr(result_str, "maintained=false") != NULL',
        'strstr(result_str, "reason=accelerator_memory_budget") != NULL',
    ):
        if required not in defer_result_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: memory-blocked accelerator hints '
                f'miss periodic-reconcile classifier {required!r}'
            )
    worker_start = source.find(
        '\nstatic int\nii42_am_execute_maintenance_candidate('
    )
    worker_end = source.find(
        '\n\nPGDLLEXPORT void\nii42_maintenance_worker_main(',
        worker_start,
    )
    worker_block = (
        source[worker_start:worker_end]
        if worker_start >= 0 and worker_end > worker_start
        else ''
    )
    for required in (
        'ii42_am_maintenance_result_defers_to_reconcile(result)',
        'else if (deferred_to_reconcile && candidate->hinted)',
        'ii42_am_work_hint_clear_if_unchanged(',
    ):
        if required not in worker_block:
            errors.append(
                f'{AM_SOURCE_PATH.name}: memory-blocked accelerator hint '
                f'can busy-loop without {required!r}'
            )
    build_empty_start = source.find('\nstatic void\nii42_ambuildempty(')
    build_empty_end = source.find(
        '\n\nstatic char *\nii42_am_compile_semantic_documents_query(',
        build_empty_start,
    )
    if build_empty_start < 0 or build_empty_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: empty rebuild borrowed owner not found'
        )
    else:
        build_empty_block = source[build_empty_start:build_empty_end]
        for required in (
            'ii42_am_rebuild_output replacement;',
            'replacement.index = cleanup->index;',
            'ii42_index_init(&cleanup->index);',
            'ii42_am_rebuild_output_release(&replacement);',
            'ii42_index_free(&cleanup->index);',
        ):
            if required not in build_empty_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: empty rebuild borrowed owner is '
                    f'missing {required!r}'
                )
        if (
            build_empty_block.count(
                'ii42_am_rebuild_output_release(&replacement);'
            ) != 1
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: empty rebuild replacement does not '
                'have one release'
            )
    for constant in (
        'II42_AM_REBUILD_MEMORY_ESTIMATE_MULTIPLIER',
        'II42_AM_COMPACT_REBUILD_MEMORY_ESTIMATE_MULTIPLIER',
        'II42_AM_SPILL_REBUILD_MEMORY_ESTIMATE_MULTIPLIER',
        'II42_AM_STANDARD_REBUILD_BUDGET_HEADROOM_NUM',
        'II42_AM_STANDARD_REBUILD_BUDGET_HEADROOM_DEN',
        'II42_AM_COMPACT_REBUILD_BUDGET_HEADROOM_NUM',
        'II42_AM_COMPACT_REBUILD_BUDGET_HEADROOM_DEN',
        'II42_AM_STANDARD_REBUILD_MAX_PAYLOAD_BYTES',
        'II42_AM_COMPACT_REBUILD_MAX_PAYLOAD_BYTES',
        'II42_AM_SEMANTIC_BUILD_FIXED_WORKSPACE_BYTES',
    ):
        definition = re.compile(rf'^#define {constant}\b', re.MULTILINE)
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if definition.search(path.read_text(encoding='utf-8'))
        ]
        if owners != [AM_BUILD_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{constant}: build admission constant does not have one '
                f'owner: {owner_names or "<none>"}'
            )
    semantic_stream_adapter_start = build_source.find(
        '\nstatic ii42_status\nii42_am_semantic_posting_file_read('
    )
    semantic_stream_adapter_end = build_source.find(
        '\nstatic void\n'
        'ii42_am_rebuild_output_release_materialized_sources(',
        semantic_stream_adapter_start,
    )
    semantic_stream_adapter_block = ''
    if semantic_stream_adapter_start < 0 or semantic_stream_adapter_end < 0:
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: semantic stream adapter closure '
            'not found'
        )
    else:
        semantic_stream_adapter_block = build_source[
            semantic_stream_adapter_start:semantic_stream_adapter_end
        ]
    initial_fold_producer_start = build_source.find(
        '\nstatic ii42_status\nii42_am_initial_fold_publish_next('
    )
    initial_fold_producer_end = build_source.find(
        '\nvoid\nii42_am_rebuild_output_release(',
        initial_fold_producer_start,
    )
    initial_fold_producer_block = ''
    if initial_fold_producer_start < 0 or initial_fold_producer_end < 0:
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: streamed publication producer '
            'closure not found'
        )
    else:
        initial_fold_producer_block = build_source[
            initial_fold_producer_start:initial_fold_producer_end
        ]
        for required in (
            'ii42_initial_fold_stream_next(',
            'ii42_initial_fold_stream_free(publish->stream);',
            'ii42_am_rebuild_output_release_materialized_sources(',
            'publish->source_released = true;',
        ):
            if required not in initial_fold_producer_block:
                errors.append(
                    f'{AM_BUILD_SOURCE_PATH.name}: streamed publication '
                    f'producer is missing {required!r}'
                )
    pure_build_source = build_source
    for authority_block in (
        semantic_stream_adapter_block,
        initial_fold_producer_block,
        replacement_prepare_block,
        rebuild_publisher_block,
    ):
        if authority_block:
            pure_build_source = pure_build_source.replace(
                authority_block,
                '',
                1,
            )
    pure_build_source = '\n'.join(
        line
        for line in pure_build_source.splitlines()
        if line not in (
            '#include "ii42_am_meta.h"',
            '#include "ii42_segment_pages.h"',
        )
    )
    for forbidden_build_dependency in (
        'Relation',
        'MarkBufferDirty',
        'GenericXLog',
        'XLog',
        'LWLock',
        'LockRelation',
        'ii42_am_meta',
        'ii42_segment_',
        'ii42_runtime',
        'ii42_maintenance_rebuild_memory_budget_mb',
        'ereport(',
        'palloc',
    ):
        if forbidden_build_dependency in pure_build_source:
            errors.append(
                f'{AM_BUILD_SOURCE_PATH.name}: pure build admission acquired '
                f'forbidden authority: {forbidden_build_dependency!r}'
            )
    for forbidden_build_header_dependency in (
        'storage/bufmgr.h',
        'ii42_am_meta.h',
        'ii42_segment_pages.h',
        'ii42_runtime_service.h',
    ):
        if forbidden_build_header_dependency in build_header:
            errors.append(
                f'{AM_BUILD_HEADER_PATH.name}: build value contract exposes '
                f'forbidden authority: {forbidden_build_header_dependency!r}'
            )
    for constructor in array_constructors:
        definition = re.compile(
            rf'\nArrayType \*\n{re.escape(constructor)}\('
        )
        if len(definition.findall(pg_common_source)) != 1:
            errors.append(
                f'{PG_COMMON_SOURCE_PATH.name}: PostgreSQL array constructor '
                f'does not have one owner: {constructor}'
            )
        if pg_common_header.count(f'ArrayType *{constructor}(') != 1:
            errors.append(
                f'{PG_COMMON_HEADER_PATH.name}: PostgreSQL array constructor '
                f'does not have one declaration: {constructor}'
            )
        for path, implementation_source in all_c_sources.items():
            if path != PG_COMMON_SOURCE_PATH and definition.search(
                implementation_source
            ):
                errors.append(
                    f'{path.name}: PostgreSQL array constructor escaped '
                    f'common value authority: {constructor}'
                )
    for retired_constructor in (
        'ii42_am_float4_array_from_values',
        'ii42_am_int4_array_from_values',
        'ii42_am_text_array_from_cstrings',
    ):
        for path in sorted((REPO_ROOT / 'src').glob('*.[ch]')):
            if retired_constructor in path.read_text(encoding='utf-8'):
                errors.append(
                    f'{path.name}: retired AM-private array constructor '
                    f'remains: {retired_constructor}'
                )

    if '#include "ii42_am_meta.h"' not in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: checked metapage authority is not used'
        )
    meta_layout = 'typedef struct ii42_am_meta_page'
    meta_layout_owners = []
    for path in sorted((REPO_ROOT / 'src').glob('*.[ch]')):
        if meta_layout in path.read_text(encoding='utf-8'):
            meta_layout_owners.append(path)
    if meta_layout_owners != [AM_META_HEADER_PATH]:
        owners = ', '.join(path.name for path in meta_layout_owners)
        errors.append(
            'metapage wire layout does not have one header owner: '
            f'{owners or "<none>"}'
        )
    health_layout = 'typedef struct ii42_am_payload_health_state'
    health_layout_owners = []
    for path in sorted((REPO_ROOT / 'src').glob('*.[ch]')):
        if health_layout in path.read_text(encoding='utf-8'):
            health_layout_owners.append(path)
    if health_layout_owners != [AM_META_HEADER_PATH]:
        owners = ', '.join(path.name for path in health_layout_owners)
        errors.append(
            'payload-health value layout does not have one header owner: '
            f'{owners or "<none>"}'
        )
    debt_layout = 'typedef struct ii42_am_convergent_mutation_debt'
    debt_layout_owners = []
    for path in sorted((REPO_ROOT / 'src').glob('*.[ch]')):
        if debt_layout in path.read_text(encoding='utf-8'):
            debt_layout_owners.append(path)
    if debt_layout_owners != [AM_META_HEADER_PATH]:
        owners = ', '.join(path.name for path in debt_layout_owners)
        errors.append(
            'linked-L0 debt value layout does not have one header owner: '
            f'{owners or "<none>"}'
        )
    for constant in (
        'II42_AM_MAGIC',
        'II42_AM_VERSION',
        'II42_AM_PAGE_META',
        'II42_AM_FLAG_STALE',
        'II42_AM_FLAG_CORRUPT',
        'II42_AM_FLAG_REBUILD_REQUIRED',
        'II42_AM_FLAG_SEMANTIC_QUARANTINE',
        'II42_AM_STORAGE_CONVERGENT_SEGMENTS',
    ):
        definition = re.compile(rf'^#define {constant}\b', re.MULTILINE)
        owners = [
            path
            for path in sorted((REPO_ROOT / 'src').glob('*.[ch]'))
            if definition.search(path.read_text(encoding='utf-8'))
        ]
        if owners != [AM_META_HEADER_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{constant}: metapage constant does not have one header '
                f'owner: {owner_names or "<none>"}'
            )
    meta_read_functions = (
        'ii42_am_meta_uses_convergent_segment_storage',
        'ii42_am_require_convergent_segment_storage',
        'ii42_am_segment_read_root_from_meta',
        'ii42_am_relation_nblocks',
        'ii42_am_read_meta',
        'ii42_am_payload_health',
        'ii42_am_meta_has_pending_maintenance',
    )
    for function in meta_read_functions:
        definition = re.compile(
            rf'\n(?:bool|void|ii42_status|BlockNumber)\n'
            rf'{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: checked metapage read does not have one owner: '
                f'{owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: checked metapage read does not '
                f'have one declaration: {function}'
            )
    meta_accounting_functions = (
        'ii42_am_segment_object_bytes_add',
        'ii42_am_convergent_object_bytes',
    )
    for function in meta_accounting_functions:
        definition = re.compile(
            rf'\nuint64\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: root byte accounting does not have one owner: '
                f'{owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: root byte accounting does not '
                f'have one declaration: {function}'
            )
    accounting_start = meta_source.find(
        '\nuint64\nii42_am_segment_object_bytes_add('
    )
    accounting_end = meta_source.find(
        '\n\nvoid\nii42_am_convergent_mutation_debt_read(',
        accounting_start,
    )
    if accounting_start < 0 or accounting_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: root byte accounting closure not '
            'found'
        )
    else:
        accounting_block = meta_source[accounting_start:accounting_end]
        for required in (
            'return ii42_u64_saturating_add(left, right);',
            'ii42_am_segment_read_root_from_meta(meta, &root)',
            'root.published_block_high_watermark * BLCKSZ',
        ):
            if required not in accounting_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: root byte accounting is '
                    f'missing {required!r}'
                )
        for forbidden in (
            'ii42_am_convergent_mutation_debt',
            'ii42_am_schedule',
            'ii42_am_maintain',
            'ii42_runtime',
            'ii42_model',
            'rd_options',
            'GetConfigOption',
            'table_index_build_scan',
            'ii42_segment_pages_inventory_reachable',
            'ii42_segment_pages_load_sealed_manifest(',
            'retired_range',
            'recyclable',
            'RelationGetNumberOfBlocks',
            'MarkBufferDirty',
            'XLog',
            'ii42_page_query',
            'IndexScan',
            'ambulkdelete',
        ):
            if forbidden in accounting_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: root byte accounting '
                    f'acquired forbidden authority: {forbidden!r}'
                )
    debt_function = 'ii42_am_convergent_mutation_debt_read'
    debt_definition = re.compile(
        rf'\nvoid\n{re.escape(debt_function)}\('
    )
    debt_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if debt_definition.search(implementation_source)
    ]
    if debt_owners != [AM_META_SOURCE_PATH]:
        owner_names = ', '.join(path.name for path in debt_owners)
        errors.append(
            f'{debt_function}: linked-L0 debt projection does not have one '
            f'owner: {owner_names or "<none>"}'
        )
    if meta_header.count(f'{debt_function}(') != 1:
        errors.append(
            f'{AM_META_HEADER_PATH.name}: linked-L0 debt projection does '
            f'not have one declaration: {debt_function}'
        )
    debt_start = meta_source.find(
        '\nvoid\nii42_am_convergent_mutation_debt_read('
    )
    debt_end = meta_source.find(
        '\n\nstatic void\nii42_am_normalize_meta_storage(',
        debt_start,
    )
    if debt_start < 0 or debt_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: linked-L0 debt projection closure '
            'not found'
        )
    else:
        debt_block = meta_source[debt_start:debt_end]
        for required in (
            'ii42_u32_saturating_add(',
            'ii42_am_get_consistency(index_relation)',
            'II42_AM_CONSISTENCY_MANUAL',
            'meta->pending_write_tuples',
            'meta->pending_delete_tuples',
            'ii42_am_segment_read_root_from_meta(meta, &root)',
            'ii42_l0_storage_snapshot_init(&snapshot);',
            'ii42_segment_pages_load_l0_snapshot(',
            'II42_L0_RECORD_UPSERT',
            'II42_L0_RECORD_RETIRE',
            'II42_L0_RECORD_SEMANTIC_COMPLETE',
            'II42_L0_RECORD_SEMANTIC_QUARANTINE',
            'II42_L0_RECORD_INVALID',
            'debt_out->records = snapshot.record_count;',
            'debt_out->bytes = snapshot.payload_bytes;',
            'PG_TRY();',
            'PG_FINALLY();',
            'ii42_l0_storage_snapshot_free(&snapshot);',
        ):
            if required not in debt_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: linked-L0 debt projection '
                    f'is missing {required!r}'
                )
        for forbidden in (
            'ii42_am_append',
            'ii42_am_publish',
            'ii42_am_schedule',
            'ii42_am_try_maintain',
            'ii42_runtime',
            'ii42_model',
            'table_index_build_scan',
            'heap_beginscan',
            'MarkBufferDirty',
            'GenericXLog',
            'XLog',
            'LockRelation',
            'LWLock',
            'ii42_page_query',
            'ambulkdelete',
            'ii42_am_posting_heat',
        ):
            if forbidden in debt_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: linked-L0 debt projection '
                    f'acquired forbidden authority: {forbidden!r}'
                )
    rebuild_meta_publish = 'ii42_am_publish_rebuild_meta'
    rebuild_meta_publish_definition = re.compile(
        rf'\nvoid\n{re.escape(rebuild_meta_publish)}\('
    )
    rebuild_meta_publish_owners = [
        path
        for path, implementation_source in all_c_sources.items()
        if rebuild_meta_publish_definition.search(implementation_source)
    ]
    if rebuild_meta_publish_owners != [AM_META_SOURCE_PATH]:
        owner_names = ', '.join(
            path.name for path in rebuild_meta_publish_owners
        )
        errors.append(
            f'{rebuild_meta_publish}: rebuild metapage publication does not '
            f'have one owner: {owner_names or "<none>"}'
        )
    if meta_header.count(f'{rebuild_meta_publish}(') != 1:
        errors.append(
            f'{AM_META_HEADER_PATH.name}: rebuild metapage publication does '
            f'not have one declaration: {rebuild_meta_publish}'
        )
    if source.count(f'{rebuild_meta_publish}(') != 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: rebuild metapage publication caller '
            'escaped the build authority'
        )
    if build_source.count(f'{rebuild_meta_publish}(') != 1:
        errors.append(
            f'{AM_BUILD_SOURCE_PATH.name}: rebuild metapage publication '
            'does not have exactly one typed caller'
        )
    if 'ii42_am_publish_convergent_segment_meta' in source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: retired rebuild metapage writer remains'
        )
    rebuild_publish_start = meta_source.find(
        f'\nvoid\n{rebuild_meta_publish}('
    )
    rebuild_publish_end = meta_source.find(
        '\n\nbool\nii42_am_meta_has_pending_maintenance(',
        rebuild_publish_start,
    )
    meta_read_source = meta_source
    if rebuild_publish_start < 0 or rebuild_publish_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: rebuild metapage publication '
            'closure not found'
        )
    else:
        rebuild_publish_block = meta_source[
            rebuild_publish_start:rebuild_publish_end
        ]
        meta_read_source = (
            meta_source[:rebuild_publish_start] +
            meta_source[rebuild_publish_end:]
        )
        for required in (
            'fork_number != MAIN_FORKNUM',
            'fork_number != INIT_FORKNUM',
            'ii42_segment_read_root_serialize(',
            'root->published_block_high_watermark != (uint32) nblocks',
            'LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);',
            'START_CRIT_SECTION();',
            'MarkBufferDirty(meta_buffer);',
            'log_newpage_buffer(meta_buffer, false);',
            'XLogFlush(publish_lsn);',
        ):
            if required not in rebuild_publish_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: rebuild metapage '
                    f'publication is missing {required!r}'
                )
        for forbidden in (
            'RelationTruncate(',
            'ExtendBufferedRel(',
            'P_NEW',
            'GenericXLog',
            'ii42_am_lock_generation_barrier',
            'ii42_am_replacement',
            'ii42_segment_pages_',
            'ii42_runtime',
            'ii42_model',
            'ii42_am_schedule',
            'ii42_am_update_meta_flags',
        ):
            if forbidden in rebuild_publish_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: rebuild metapage '
                    f'publication acquired forbidden authority: {forbidden!r}'
                )
    meta_mutation_functions = (
        'ii42_am_note_maintenance_activity',
        'ii42_am_mark_stale',
        'ii42_am_get_stats',
    )
    for function in meta_mutation_functions:
        definition = re.compile(
            rf'\nvoid\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: metapage mutation/projection does not have '
                f'one owner: {owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: metapage mutation/projection '
                f'does not have one declaration: {function}'
            )
    meta_mutation_start = meta_source.find(
        '\nstatic void\nii42_am_update_meta_flags('
    )
    meta_mutation_end = meta_source.find(
        '\n\nuint64\nii42_am_next_rebuild_count(',
        meta_mutation_start,
    )
    if meta_mutation_start < 0 or meta_mutation_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: metapage mutation/projection '
            'closure not found'
        )
    else:
        meta_mutation_block = meta_source[
            meta_mutation_start:meta_mutation_end
        ]
        for required in (
            'ii42_am_lock_append(indexRelation);',
            'LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);',
            'ii42_am_mark_buffer_dirty_with_wal(indexRelation, buffer);',
            'ii42_am_maintenance_tracking_enabled(indexRelation)',
            'ii42_am_pin_maintenance_xact(indexRelation);',
            'ii42_u32_saturating_add(',
            'ii42_am_update_meta_flags(',
            'II42_AM_FLAG_STALE',
            'stats->num_index_tuples = meta.num_docs;',
        ):
            if required not in meta_mutation_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: metapage mutation/'
                    f'projection is missing {required!r}'
                )
        for forbidden in (
            'ii42_am_schedule',
            'ii42_am_try_maintain',
            'ii42_am_work_hint',
            'ii42_am_publish',
            'ii42_runtime',
            'ii42_model',
            'table_index_build_scan',
            'heap_beginscan',
            'ii42_page_query',
            'ii42_segment_pages_',
            'ambulkdelete',
        ):
            if forbidden in meta_mutation_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: metapage mutation/'
                    f'projection acquired forbidden authority: {forbidden!r}'
                )
        meta_read_source = meta_read_source.replace(
            meta_mutation_block,
            '',
        )
    if accounting_start >= 0 and accounting_end >= 0:
        meta_read_source = meta_read_source.replace(accounting_block, '')
    if debt_start >= 0 and debt_end >= 0:
        meta_read_source = meta_read_source.replace(debt_block, '')
    ordinal_functions = (
        (
            'ii42_am_next_rebuild_count',
            'uint64',
            build_source,
            AM_BUILD_SOURCE_PATH,
        ),
        (
            'ii42_am_next_cache_epoch',
            'uint16',
            source,
            AM_SOURCE_PATH,
        ),
    )
    for function, return_type, caller_source, caller_path in ordinal_functions:
        definition = re.compile(
            rf'\n{return_type}\n{re.escape(function)}\('
        )
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: rebuild ordinal projection does not have one '
                f'owner: {owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: rebuild ordinal projection '
                f'does not have one declaration: {function}'
            )
        if caller_source.count(f'{function}(') != 1:
            errors.append(
                f'{caller_path.name}: rebuild ordinal projection does not '
                f'have exactly one caller: {function}'
            )
    ordinal_start = meta_source.find(
        '\nuint64\nii42_am_next_rebuild_count('
    )
    ordinal_end = meta_source.find(
        '\n\nvoid\nii42_am_publish_rebuild_meta(',
        ordinal_start,
    )
    if ordinal_start < 0 or ordinal_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: rebuild ordinal projection closure '
            'not found'
        )
    else:
        ordinal_block = meta_source[ordinal_start:ordinal_end]
        for required in (
            'RelationGetNumberOfBlocks(indexRelation) == 0',
            'ii42_am_read_meta(indexRelation, &meta);',
            'meta.rebuild_count == UINT64_MAX',
            'return UINT64_MAX;',
            'return meta.rebuild_count + 1;',
            'meta.cache_epoch == UINT16_MAX',
            'return (uint16) (meta.cache_epoch + 1);',
        ):
            if required not in ordinal_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: rebuild ordinal projection '
                    f'is missing {required!r}'
                )
        for forbidden in (
            'RelationTruncate(',
            'RelationOpenSmgr(',
            'LockRelation',
            'LockAcquire',
            'LockBuffer',
            'MarkBufferDirty',
            'GenericXLog',
            'XLog',
            'ii42_am_publish',
            'ii42_am_schedule',
            'ii42_am_preload',
            'ii42_runtime',
            'ii42_model',
            'table_index_build_scan',
            'ambulkdelete',
            'ii42_page_query',
        ):
            if forbidden in ordinal_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: rebuild ordinal projection '
                    f'acquired forbidden authority: {forbidden!r}'
                )
        meta_read_source = meta_read_source.replace(ordinal_block, '')
    generation_barrier_tag = 'II42_AM_GENERATION_BARRIER_LOCK_TAG'
    if meta_source.count(
            f'#define {generation_barrier_tag} UINT32_C(0x32534255)') != 1:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: generation barrier tag changed'
        )
    for path, implementation_source in all_c_sources.items():
        if path == AM_META_SOURCE_PATH:
            continue
        if generation_barrier_tag in implementation_source:
            errors.append(
                f'{path.name}: generation barrier tag escaped root authority'
            )
    generation_barrier_functions = (
        'ii42_am_lock_generation_barrier',
        'ii42_am_unlock_generation_barrier',
    )
    for function in generation_barrier_functions:
        definition = re.compile(rf'\nvoid\n{re.escape(function)}\(')
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: generation barrier does not have one root '
                f'owner: {owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: generation barrier does not '
                f'have one declaration: {function}'
            )
        if source.count(f'{function}(') != 2:
            errors.append(
                f'{AM_SOURCE_PATH.name}: generation barrier caller count '
                f'changed: {function}'
            )
    for function in (
        'ii42_am_lock_generation_barrier_oid',
        'ii42_am_unlock_generation_barrier_oid',
    ):
        if meta_source.count(f'\nstatic void\n{function}(') != 1:
            errors.append(
                f'{AM_META_SOURCE_PATH.name}: private OID generation '
                f'barrier changed: {function}'
            )
        if function in meta_header:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: private OID generation '
                f'barrier escaped: {function}'
            )
        for path, implementation_source in all_c_sources.items():
            if path == AM_META_SOURCE_PATH:
                continue
            if function in implementation_source:
                errors.append(
                    f'{path.name}: private OID generation barrier escaped: '
                    f'{function}'
                )
    generation_barrier_start = meta_source.find(
        '\nstatic void\nii42_am_lock_generation_barrier_oid('
    )
    generation_barrier_end = meta_source.find(
        '\n\nvoid\nii42_am_mark_buffer_dirty_with_wal(',
        generation_barrier_start,
    )
    if generation_barrier_start < 0 or generation_barrier_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: generation barrier closure not '
            'found'
        )
    else:
        generation_barrier_block = meta_source[
            generation_barrier_start:generation_barrier_end
        ]
        for required in (
            'SET_LOCKTAG_ADVISORY(',
            'MyDatabaseId,',
            'II42_AM_GENERATION_BARRIER_LOCK_TAG,',
            'index_oid,',
            '(void) LockAcquire(&tag, ExclusiveLock, false, false);',
            'if (!LockRelease(&tag, ExclusiveLock, false))',
            'ii42 generation barrier is not held',
            'RelationGetRelid(indexRelation)',
        ):
            if required not in generation_barrier_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: generation barrier is '
                    f'missing {required!r}'
                )
        if generation_barrier_block.count('SET_LOCKTAG_ADVISORY(') != 2:
            errors.append(
                f'{AM_META_SOURCE_PATH.name}: generation barrier advisory '
                'key count changed'
            )
        for forbidden in (
            'II42_AM_WRITER_BARRIER_LOCK_TAG',
            'LockAcquire(&tag, ShareLock',
            'LockRelease(&tag, ShareLock',
            'ConditionalLock',
            'RelationTruncate(',
            'LockRelation',
            'LockBuffer',
            'MarkBufferDirty',
            'START_CRIT_SECTION',
            'XLog',
            'table_open(',
            'table_close(',
            'ii42_am_publish',
            'ii42_am_schedule',
            'ii42_am_preload',
            'ii42_am_maintenance',
            'ii42_runtime',
            'ii42_model',
            'ambulkdelete',
            'ii42_page_query',
        ):
            if forbidden in generation_barrier_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: generation barrier acquired '
                    f'forbidden authority: {forbidden!r}'
                )
        meta_read_source = meta_read_source.replace(
            generation_barrier_block,
            '',
        )
    if maintenance_source.count(
            '#define II42_AM_WRITER_BARRIER_LOCK_TAG '
            'UINT32_C(0x32534242)') != 1:
        errors.append(
            f'{AM_MAINTENANCE_SOURCE_PATH.name}: writer barrier ownership '
            'changed'
        )
    if 'II42_AM_WRITER_BARRIER_LOCK_TAG' in meta_source:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: writer barrier entered root '
            'authority'
        )
    page_write_functions = (
        'ii42_am_mark_buffer_dirty_with_wal',
        'ii42_am_write_page_at',
        'ii42_am_write_init_page_at',
        'ii42_am_write_new_page',
    )
    for function in page_write_functions:
        definition = re.compile(rf'\nvoid\n{re.escape(function)}\(')
        owners = [
            path
            for path, implementation_source in all_c_sources.items()
            if definition.search(implementation_source)
        ]
        if owners != [AM_META_SOURCE_PATH]:
            owner_names = ', '.join(path.name for path in owners)
            errors.append(
                f'{function}: page initialization primitive does not have '
                f'one owner: {owner_names or "<none>"}'
            )
        if meta_header.count(f'{function}(') != 1:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: page initialization primitive '
                f'does not have one declaration: {function}'
            )
    page_write_start = meta_source.find(
        '\nvoid\nii42_am_mark_buffer_dirty_with_wal('
    )
    page_write_end = meta_source.find(
        '\n\nvoid\nii42_am_payload_health(',
        page_write_start,
    )
    if page_write_start < 0 or page_write_end < 0:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: page initialization closure not '
            'found'
        )
    else:
        page_write_block = meta_source[page_write_start:page_write_end]
        for required in (
            'Assert(CritSectionCount > 0);',
            'if (RelationNeedsWAL(indexRelation))',
            'nblocks = RelationGetNumberOfBlocks(indexRelation);',
            'if (blkno < nblocks)',
            'else if (blkno == nblocks)',
            'P_NEW',
            'ii42 segment write skipped a block',
            'LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);',
            'nblocks = RelationGetNumberOfBlocksInFork(',
            'INIT_FORKNUM',
            'ii42 initialization fork write is not contiguous',
            'ExtendBufferedRel(',
            'EB_LOCK_FIRST | EB_SKIP_EXTENSION_LOCK',
            'unexpected ii42 initialization fork size',
            'START_CRIT_SECTION();',
            'PageInit(page, BLCKSZ, 0);',
            'memcpy(PageGetContents(page), contents, len);',
            'ii42_am_mark_buffer_dirty_with_wal(indexRelation, buffer);',
            'MarkBufferDirty(buffer);',
            'log_newpage_buffer(buffer, false);',
            'END_CRIT_SECTION();',
            'UnlockReleaseBuffer(buffer);',
            'ii42_am_write_page_at(',
        ):
            if required not in page_write_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: page initialization '
                    f'closure is missing {required!r}'
                )
        for forbidden in (
            'RelationTruncate(',
            'RelationOpenSmgr(',
            'LockRelation',
            'LockAcquire',
            'LockRelease',
            'table_open(',
            'table_close(',
            'ii42_am_publish',
            'ii42_am_schedule',
            'ii42_am_preload',
            'ii42_runtime',
            'ii42_model',
            'ambulkdelete',
            'ii42_page_query',
            'ii42_segment_pages_',
        ):
            if forbidden in page_write_block:
                errors.append(
                    f'{AM_META_SOURCE_PATH.name}: page initialization '
                    f'closure acquired forbidden authority: {forbidden!r}'
                )
        expected_am_calls = {
            'ii42_am_mark_buffer_dirty_with_wal': 0,
            'ii42_am_write_page_at': 0,
            'ii42_am_write_init_page_at': 1,
            'ii42_am_write_new_page': 0,
        }
        for function, expected_count in expected_am_calls.items():
            if source.count(f'{function}(') != expected_count:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: page initialization primitive '
                    f'caller count changed: {function}'
                )
        meta_read_source = meta_read_source.replace(page_write_block, '')
    normalize_definition = re.compile(
        r'\nstatic void\nii42_am_normalize_meta_storage\('
    )
    if len(normalize_definition.findall(meta_source)) != 1:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: metapage normalization is not '
            'private'
        )
    for path in sorted((REPO_ROOT / 'src').glob('*.[ch]')):
        if path == AM_META_SOURCE_PATH:
            continue
        if 'ii42_am_normalize_meta_storage' in path.read_text(
            encoding='utf-8'
        ):
            errors.append(
                f'{path.name}: private metapage normalization escaped its '
                'read authority'
            )
    if 'sizeof(ii42_am_meta_page)' not in meta_source:
        errors.append(
            f'{AM_META_SOURCE_PATH.name}: metapage size guard is missing'
        )
    for forbidden_meta_dependency in (
        'MarkBufferDirty',
        'GenericXLog',
        'XLog',
        'START_CRIT_SECTION',
        'LWLock',
        'LockRelation',
        'ii42_am_cache',
        'ii42_am_publish',
        'ii42_am_schedule',
        'ii42_am_write',
        'ii42_runtime',
        'ii42_model',
        'ii42_segment_read_root_serialize',
        'ii42_segment_publish',
        'ii42_segment_pages_',
        'ii42_am_convergent_mutation_debt',
        'ii42_am_rebuild',
        'ii42_document_cow',
    ):
        if forbidden_meta_dependency in meta_read_source:
            errors.append(
                f'{AM_META_SOURCE_PATH.name}: checked metapage reader '
                f'acquired forbidden authority: {forbidden_meta_dependency!r}'
            )
    for forbidden_meta_header_dependency in (
        'storage/bufmgr.h',
        'storage/bufpage.h',
        'ii42_page_query.h',
        'ii42_runtime_service.h',
        'ii42_segment_pages.h',
        'ii42_storage.h',
    ):
        if forbidden_meta_header_dependency in meta_header:
            errors.append(
                f'{AM_META_HEADER_PATH.name}: metapage wire contract exposes '
                f'forbidden authority: {forbidden_meta_header_dependency!r}'
            )


def check_lifecycle_smoke(errors: list[str]) -> None:
    smoke = LIFECYCLE_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        'sae = true',
        'consistency = realtime',
        'consistency = eventual',
        'ii42_index_maintain(',
        'DROP INDEX p2_mutable.docs_idx',
        'drop_removes_relation_and_preserves_source',
        'unified_generation_ready',
        "generation.get('atomic') is True",
        "generation.get('contract_signature')",
        'bm25_build_options_are_generation_bound',
        'bm25_contract_drift_requires_explicit_reindex',
        'aborted_dml_is_hidden_with_reclaimable_delta_debt',
        'savepoint_rollback_is_hidden_with_reclaimable_delta_debt',
        'savepoint_commit_is_immediately_visible_via_delta',
        'bulk_dml_is_lexical_first_and_worker_batched',
        'failed_subtransactions_release_pending_mutation_memory',
        'vacuum_tombstone_prevents_tid_reuse_resurrection',
        'skipped_index_cleanup_preserves_mvcc_results',
        'vacuum_records_generation_pinned_tombstones',
        'semantic_query_pins_checked_root',
        'shared_arena_accounts_exact_root_marker_only',
        'nonindexed_hot_update_preserves_generation',
        'bm25_aborted_dml_is_hidden_and_converges',
        'bm25_savepoint_rollback_is_hidden_and_converges',
        'VACUUM p2_mutable.docs',
        'REINDEX INDEX p2_mutable.docs_idx',
        'cold_restart_preserves_results',
        'crash_restart_preserves_results',
        'bm25_crash_restart_preserves_results',
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{LIFECYCLE_SMOKE_PATH.name}: missing gate {needle!r}'
            )
    convergent_sae_smoke = (
        CONVERGENT_SAE_LIFECYCLE_SMOKE_PATH.read_text(encoding='utf-8')
    )
    for gate in (
        'multicolumn_sae_supports_fused_and_field_aware_modes',
        'semantic_bmp_survives_immediate_crash_recovery',
        'page_native_exact_route_survives_cold_restart',
        'accelerator_query_memory_is_observable',
        'visibility_rank_attempts_are_observable',
    ):
        if gate not in convergent_sae_smoke:
            errors.append(
                f'{CONVERGENT_SAE_LIFECYCLE_SMOKE_PATH.name}: missing '
                f'convergent SSR lifecycle gate {gate!r}'
            )
    realtime_sae_pattern = re.compile(
        r'(?:sae\s*=\s*true[\s\S]{0,240}?'
        r'consistency\s*=\s*realtime|'
        r'consistency\s*=\s*realtime[\s\S]{0,240}?'
        r'sae\s*=\s*true)',
    )
    realtime_sae_fixtures = (
        LIFECYCLE_SMOKE_PATH,
        REPO_ROOT / 'scripts/benchmark_ii42_product_path.py',
        REPO_ROOT / 'scripts/test_concurrent_ddl_lifecycle_smoke.py',
        REPO_ROOT / 'scripts/test_onnxruntime_provider_matrix.py',
        REPO_ROOT / 'scripts/test_runtime_service_temp_pg.py',
    )
    for path in realtime_sae_fixtures:
        source = path.read_text(encoding='utf-8')
        if realtime_sae_pattern.search(source):
            errors.append(
                f'{path.name}: positive SAE realtime fixture remains'
            )
    for prefix in DISALLOWED_SPLIT_API_PREFIXES:
        if f'SELECT {prefix}' in smoke:
            errors.append(
                f'{LIFECYCLE_SMOKE_PATH.name}: calls retired API {prefix}'
            )


def check_cache_failure_smoke(errors: list[str]) -> None:
    smoke = CACHE_FAILURE_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        'ii42.test_query_operator_error_after_parse',
        'ii42.test_search_error_after_rank',
        'RSS_GROWTH_LIMIT_KIB',
        'expect_scan_fault',
        'L0 query retry did not return 20 hits',
        'L0 scan retry returned wrong count',
        'same-backend failure loop retained too much memory',
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{CACHE_FAILURE_SMOKE_PATH.name}: '
                f'missing failure-safety gate {needle!r}'
            )


def check_vacuum_frontier_smoke(errors: list[str]) -> None:
    smoke = VACUUM_FRONTIER_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        'ACTIVE_L0_MAX_RECORDS = 131_072',
        'VACUUM_COW_BATCH_RECORDS = 65_536',
        'DEFAULT_ROW_COUNT = 140_000',
        'ii42.test_convergent_vacuum_error_after_batch',
        'partial_vacuum_batch_is_failure_safe',
        'restart_preserves_converged_retirement',
        'post_frontier_insert_survives_restart',
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{VACUUM_FRONTIER_SMOKE_PATH.name}: '
                f'missing hard-frontier gate {needle!r}'
            )


def check_storage_layout_boundary(errors: list[str]) -> None:
    smoke = STORAGE_LAYOUT_BOUNDARY_PATH.read_text(encoding='utf-8')
    required_smoke = (
        'STORAGE_VERSION_OFFSET = PAGE_HEADER_SIZE + '
        'META_STORAGE_VERSION_OFFSET',
        'STORAGE_GENERATION_DELTA = 2',
        'STORAGE_CONVERGENT_SEGMENTS = 3',
        "'--no-data-checksums'",
        "'0A000'",
        "'unsupported ii42 index storage layout'",
        "'REINDEX the ii42 index to publish page-native v3 storage.'",
        "'background_due_count'",
        "'REINDEX INDEX docs_bm25_idx; CHECKPOINT;'",
    )
    for needle in required_smoke:
        if needle not in smoke:
            errors.append(
                f'{STORAGE_LAYOUT_BOUNDARY_PATH.name}: missing '
                'storage-boundary '
                f'gate {needle!r}'
            )

    metapage_smoke = METAPAGE_READ_BOUNDARY_PATH.read_text(
        encoding='utf-8'
    )
    required_metapage_smoke = (
        'META_MAGIC_OFFSET = PAGE_HEADER_SIZE',
        'META_TID_BYTES_OFFSET = PAGE_HEADER_SIZE + 24',
        'META_ROOT_OFFSET = PAGE_HEADER_SIZE + 153',
        "'invalid ii42 index metapage'",
        "'Validation failed: marked_corrupt.'",
        "'Validation failed: invalid_segment_read_root.'",
        "'Validation failed: segment_root_out_of_bounds.'",
        "'REINDEX INDEX docs_control_idx;'",
    )
    for needle in required_metapage_smoke:
        if needle not in metapage_smoke:
            errors.append(
                f'{METAPAGE_READ_BOUNDARY_PATH.name}: missing checked '
                f'metapage gate {needle!r}'
            )

    payload_health_smoke = PAYLOAD_HEALTH_SMOKE_PATH.read_text(
        encoding='utf-8'
    )
    required_payload_health_smoke = (
        "'--extension-libdir'",
        "'--extension-control-dir'",
        "'diagnostics_complete=false'",
        "details['pending_writes']",
        "generation.get('health_reason')",
        "status.get('blocker') != 'generation_invalid'",
        "'cannot preload corrupt ii42 generation'",
        "'invalid ii42 convergent segment payload'",
        "public.ii42_index_refresh('docs_bm25_idx')",
    )
    for needle in required_payload_health_smoke:
        if needle not in payload_health_smoke:
            errors.append(
                f'{PAYLOAD_HEALTH_SMOKE_PATH.name}: missing corrupt '
                f'operator-read gate {needle!r}'
            )

    source = AM_SOURCE_PATH.read_text(encoding='utf-8')
    prewarm_start = source.find(
        '\nstatic uint64\n'
        'ii42_am_prewarm_convergent_generation_pages('
    )
    prewarm_end = source.find(
        '\n\n\nstatic void\nii42_am_rebuild_workload_from_convergent(',
        prewarm_start,
    )
    if prewarm_start < 0 or prewarm_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: bounded preload boundary not found'
        )
    else:
        prewarm_block = source[prewarm_start:prewarm_end]
        for needle in (
            'ii42_segment_pages_load_sealed_manifest(',
            'ii42_segment_pages_inventory_reachable(',
            'ii42_segment_pages_load_maintenance_manifest(',
            'ii42_segment_pages_load_query_contract(',
            'ii42_am_prewarm_page_budget()',
        ):
            if needle not in prewarm_block:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: preload scope is missing '
                    f'{needle!r}'
                )
        exact_branch = prewarm_block.find(
            'if ((uint64) physical_blocks <= page_budget)'
        )
        bounded_branch = prewarm_block.find('\n        else\n', exact_branch)
        sealed_load = prewarm_block.find(
            'ii42_segment_pages_load_sealed_manifest('
        )
        maintenance_load = prewarm_block.find(
            'ii42_segment_pages_load_maintenance_manifest('
        )
        if not (
            0 <= exact_branch < sealed_load < bounded_branch <
            maintenance_load
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: bounded preload can run the '
                'index-sized sealed closure'
            )
    generation_signature_start = source.find(
        '\nDatum\nii42_index_generation_signature_internal_c('
    )
    generation_signature_end = source.find(
        '\nPG_FUNCTION_INFO_V1(ii42_index_details);',
        generation_signature_start,
    )
    if generation_signature_start < 0 or generation_signature_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: generation signature boundary not found'
        )
    else:
        generation_signature_block = source[
            generation_signature_start:generation_signature_end
        ]
        if 'ii42_segment_pages_load_maintenance_manifest(' not in (
            generation_signature_block
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: query generation signature must use '
                'the bounded manifest loader'
            )
        if 'ii42_segment_pages_load_sealed_manifest(' in (
            generation_signature_block
        ):
            errors.append(
                f'{AM_SOURCE_PATH.name}: query generation signature must not '
                'run a complete sealed-manifest closure scrub'
            )
    diagnostic_boundaries = (
        (
            '\nstatic Datum\nii42_am_convergent_generation_audit_datum(',
            '\nstatic Datum\nii42_am_generation_audit_datum(',
            (
                'ii42_segment_pages_load_sealed_manifest(',
                'ii42_am_convergent_mutation_debt_read(',
            ),
        ),
        (
            '\nii42_index_runtime_state_internal(',
            '\nPG_FUNCTION_INFO_V1(ii42_index_shared_preload_resident);',
            (
                'ii42_am_convergent_mutation_debt_read(',
                'ii42_am_rebuild_workload_from_convergent(',
            ),
        ),
        (
            '\nDatum\nii42_index_details(',
            '\nPG_FUNCTION_INFO_V1(ii42_index_policy_recommend);',
            (
                'ii42_am_convergent_object_bytes(',
                'ii42_am_convergent_mutation_debt_read(',
            ),
        ),
    )
    for start_marker, end_marker, traversals in diagnostic_boundaries:
        start = source.find(start_marker)
        end = source.find(end_marker, start)
        if start < 0 or end < 0:
            errors.append(
                f'{AM_SOURCE_PATH.name}: corrupt operator-read boundary '
                f'is missing: {start_marker.strip()!r}'
            )
            continue
        block = source[start:end]
        health_position = block.find('ii42_am_payload_health(')
        corrupt_branch_position = block.find('if (health.corrupt)')
        if health_position < 0 or corrupt_branch_position < health_position:
            errors.append(
                f'{AM_SOURCE_PATH.name}: corrupt operator-read boundary '
                f'does not fail safe: {start_marker.strip()!r}'
            )
            continue
        for traversal in traversals:
            traversal_position = block.find(traversal)
            if 0 <= traversal_position < corrupt_branch_position:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: {traversal} precedes corrupt '
                    f'health gate in {start_marker.strip()!r}'
                )
    if source.count('ii42_am_require_convergent_segment_storage(') < 17:
        errors.append(
            f'{AM_SOURCE_PATH.name}: v3 fail-closed guard no longer covers '
            'the installed product boundaries'
        )
    if source.count('ii42_am_require_exact_bm25_index_at_meta(') != 6:
        errors.append(
            f'{AM_SOURCE_PATH.name}: exact-BM25 product boundaries do not '
            'share one snapshot-meta guard'
        )
    background_start = source.find(
        '\nstatic bool\nii42_am_get_background_due_candidate('
    )
    background_end = source.find(
        '\nstatic int\nii42_am_maintenance_action_tier(',
        background_start,
    )
    if background_start < 0 or background_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: background maintenance selector missing'
        )
    else:
        background = source[background_start:background_end]
        for needle in (
            '!ii42_am_meta_uses_convergent_segment_storage(&meta)',
            'goto maintenance_candidate_done;',
            'ii42_am_convergent_structural_fold_due(',
            'semantic_accelerator_due ||',
            'structural_fold_due ||',
        ):
            if needle not in background:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: background selector can retry '
                    f'unsupported storage; missing {needle!r}'
                )
    preload_start = source.find(
        '\nstatic ii42_am_auto_preload_attempt\n'
        'ii42_am_try_auto_preload_index_oid('
    )
    preload_end = source.find(
        '\nstatic bool\nii42_am_select_auto_preload_index(',
        preload_start,
    )
    if (
        background_end >= 0 and
        preload_start >= 0 and
        'if (a->action_class != b->action_class)' in
        source[background_end:preload_start]
    ):
        errors.append(
            f'{AM_SOURCE_PATH.name}: action class can bypass the '
            'maintenance fairness cursor within one action tier'
        )
    if (
        background_end >= 0 and
        preload_start >= 0 and
        'a->urgent' in source[background_end:preload_start]
    ):
        errors.append(
            f'{AM_SOURCE_PATH.name}: pending L0 urgency can bypass '
            'cross-index maintenance fairness'
        )
    if preload_start < 0 or preload_end < 0:
        errors.append(
            f'{AM_SOURCE_PATH.name}: background preload selector missing'
        )
    else:
        preload = source[preload_start:preload_end]
        for needle in (
            'ii42_am_try_read_current_meta(indexRelation, &meta)',
            '!ii42_am_meta_uses_convergent_segment_storage(&meta)',
            '!health.rebuild_required',
            'goto auto_preload_done;',
        ):
            if needle not in preload:
                errors.append(
                    f'{AM_SOURCE_PATH.name}: background preload can retry '
                    f'unsupported storage; missing {needle!r}'
                )


def check_maturity_suite(errors: list[str]) -> None:
    suite = MATURITY_SUITE_PATH.read_text(encoding='utf-8')
    required = (
        'scripts/test_extension_regression_temp_pg.py',
        'scripts/test_extension_schema_smoke.py',
        'source-table BM25S migration smoke',
        'scripts/test_crash_recovery_smoke.py',
        'scripts/test_convergent_segment_read_smoke.py',
        'scripts/test_convergent_vacuum_frontier_smoke.py',
        'scripts/test_storage_layout_boundary.py',
        'scripts/test_metapage_read_boundary.py',
        'scripts/test_runtime_service_temp_pg.py',
        'scripts/test_transactional_delta_lifecycle.py',
        'scripts/test_sae_transaction_many_index_budget.py',
        'scripts/test_cache_failure_safety_temp_pg.py',
        'scripts/test_replication_lifecycle_smoke.py',
        'scripts/test_payload_health_corruption_smoke.py',
        'scripts/test_onnxruntime_resource_soak.py',
        'scripts/test_model_lifecycle_medium_perf.py',
        'scripts/benchmark_ii42_product_path.py',
    )
    for needle in required:
        if needle not in suite:
            errors.append(
                f'{MATURITY_SUITE_PATH.name}: missing product gate {needle!r}'
            )
    for legacy_gate in (
        'scripts/test_generation_waiter_temp_pg.py',
        'scripts/test_unified_delta_incremental_cache.py',
        'scripts/test_unified_delta_pressure_compaction.py',
        'scripts/benchmark_local_delta_cache_capacity.py',
        'scripts/benchmark_unified_delta_incremental_capacity.py',
    ):
        if legacy_gate in suite:
            errors.append(
                f'{MATURITY_SUITE_PATH.name}: legacy decoded-cache gate '
                f'remains active: {legacy_gate}'
            )
    for path in sorted((REPO_ROOT / 'scripts').glob('test_*.py')):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in suite:
            errors.append(
                f'{relative}: product test is orphaned from the canonical '
                'maturity suite'
            )


def check_arch3_legacy_runtime_absent(errors: list[str]) -> None:
    forbidden = (
        'ii42.test_legacy_v2_build',
        'ii42_am_runtime_state_unsupported_format',
        'storage=unsupported_format',
        'CREATE FUNCTION ii42_index_semantic_query_internal(',
        'ii42_index_semantic_query_internal',
        'SaeUnifiedDeltaCache',
        'Ii42UnifiedDeltaPreloadResult',
        'II42_AM_SHARED_GENERATION_UNIFIED_DELTA_CACHE',
        'ii42_unified_delta_cache_preload',
        'ii42_unified_delta_shared_',
        'ii42_unified_local_delta_cache',
        'Ii42UnifiedDeltaIdentity',
        'typedef struct Ii42UnifiedDelta\n',
        'ii42_unified_delta_load(',
        'ii42_unified_delta_load_snapshot',
        'ii42_unified_delta_free(',
        'ii42_am_unified_delta_load_prefix(',
        'ii42_am_unified_delta_identity_init(',
        'ii42_am_meta_uses_generation_delta_storage(',
        'Ii42UnifiedDeltaDocument',
        'ii42_am_semantic_pending_batch',
        'ii42_am_semantic_quarantine_state',
        'ii42_am_semantic_quarantine_scan(',
        'ii42_am_unified_delta_document_from_payload(',
        'ii42_am_semantic_frontier_tombstone',
        'typedef struct ii42_am_data_page',
        'ii42_am_delta_record_header',
        'ii42_am_unified_delta_header',
        'II42_AM_PAGE_DELTA',
        'Ii42SemanticReader',
        'Ii42SemanticDocumentCursor',
        'Ii42SemanticGenerationInfo',
        'ii42_semantic_reader_',
        'ii42_semantic_document_cursor_',
        'SaeEvidenceResidentPayload',
        'SaeEvidenceResidentReadFn',
        'ii42_unified_delta_snapshot_lock',
        'ii42_unified_delta_snapshot_unlock',
        'ii42_am_test_unified_delta_snapshot_pause',
        'ii42.test_unified_delta_snapshot_pause_ms',
        'ii42_am_delta_generation_identity_matches',
        'II42_AM_PAGE_GENERATION_DATA',
        'II42_AM_PAGE_SEMANTIC_DATA',
        'ii42_am_generation_data_page',
        'ii42_am_generation_page_header_size',
        'ii42_am_generation_page_max_payload',
        'ii42_am_semantic_metadata_valid',
        'ii42_am_persisted_generation_contract_version',
        'ii42_am_runtime_signature_for_generation',
        'ii42_am_semantic_payload_writer',
        'ii42_am_semantic_payload_write(',
        'ii42_am_semantic_doc_pair_read(',
        'ii42_am_semantic_lexical_pair_read(',
        'replacement->semantic_file',
        'replacement->semantic_bytes',
        'EATM_EVIDENCE_UNIFIED_MAGIC',
        'EATMH004',
        'ii42_semantic_build_empty_unified_payload',
        'ii42_semantic_write_unified_payload',
        'Ii42SemanticPairReadFn',
        'Ii42SemanticU32ReadFn',
        'Ii42SemanticLexicalPairReadFn',
        'Ii42SemanticOrdinalPairReadFn',
        'Ii42SemanticPayloadWriteFn',
        'Ii42SemanticUnifiedPayloadInput',
        'II42_AM_SEMANTIC_IMPACT_HEAD_SIZE',
        'ii42_am_semantic_head_pair',
        'ii42_am_semantic_head_add',
        'builder->head_pairs',
        'builder->head_sort',
        'builder->head_sort_desc',
        'builder->head_sort_input_slot',
        'builder->head_sort_output_slot',
        'ii42_am_ordinal_pair',
        'ii42_am_semantic_auxiliary_files',
        'ii42_am_semantic_build_lexical_atom_rows',
        'ii42_am_semantic_build_frontier_indexes',
        'ii42_am_semantic_auxiliary_files_close',
        'builder->semantic_frontier_pairs',
        'builder->semantic_frontier_starts',
        'builder->semantic_frontier_pair_count',
        'II42_AM_LEGACY_EVENTUAL_REBUILD_THRESHOLD',
        'II42_AM_LEGACY_SAE_EVENTUAL_REBUILD_DELTA_BYTES',
        'II42_AM_LEGACY_DELTA_OVERLAY_MAX_RECORDS',
        'II42_AM_LEGACY_DELTA_OVERLAY_MAX_BYTES',
        'II42_AM_LEGACY_SAE_DELTA_CACHE_HEADROOM_PERCENT',
        'ii42_generation_cache_',
        'shared_generation_cache_size',
        'overlay_full_scan_calls',
        'overlay_base_docs_scanned',
        'overlay_delta_docs_scanned',
        'delta_cache_builds',
        'delta_cache_hits',
        'delta_cache_misses',
        'delta_cache_decoded_postings',
        'generation.block',
        'generation.vocab_offsets',
        'delta_overlay_materialized',
        'index_owns_allocations',
        '->delta_overlay',
        'ii42_am_generation_ptr(',
        'ii42_am_can_use_unsigned_sparse_path(',
        'ii42_am_can_use_legacy_sparse_path(',
        'ii42_am_sparse_add_score(',
        'ii42_am_sparse_nonoccurrence_sum(',
        'ii42_am_rank_signed_sparse_scores(',
        'ii42_am_rank_signed_sparse_candidates',
        'ii42_am_rank_all_signed_sparse',
        'ii42_am_prepare_search_state_sparse(',
        'ii42_am_prepare_field_tokens_search_state_unsigned_sparse(',
        'ii42_am_prepare_field_tokens_search_state_sparse(',
        'init_filtered_ids',
        'next_zero_doc_id',
        'legacy_pos',
        'legacy_end',
        'II42_AM_STORAGE_GENERATION_DELTA',
        'ii42_am_block_ranges_overlap(',
        'unified_delta',
        'UNIFIED_DELTA',
        'cached->active_generation',
        'ii42_unified_delta_record_maybe_committed',
        'ii42_unified_delta_records_maybe_committed',
        'ii42_unified_delta_current_xact_modified',
        'meta.index_bytes_len > 0',
        'meta.index_bytes_len,\n            meta.semantic_bytes_len',
        'meta.semantic_data_pages',
        'meta.semantic_bytes_len',
        'meta.semantic_signature',
        'meta.delta_record_count',
        'meta.delta_bytes_len',
        'estimate_meta.tid_bytes_len',
        'estimate_meta.index_bytes_len',
        'meta->tid_bytes_len,\n            8',
        'meta->delta_bytes_len,\n            4',
    )
    paths = (
        AM_SOURCE_PATH,
        SEMANTIC_SOURCE_PATH,
        SEMANTIC_HEADER_PATH,
        CURRENT_SQL_PATH,
        REPO_ROOT / 'docs' / 'api-reference.md',
        REPO_ROOT / 'docs' / 'index-parameters.md',
        REPO_ROOT / 'docs' / 'index-policy.md',
        REPO_ROOT / 'docs' / 'shared-runtime-and-residency.md',
    )
    for path in paths:
        content = path.read_text(encoding='utf-8')
        for needle in forbidden:
            if needle in content:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: retired ARCH-3 runtime '
                    f'contract remains: {needle!r}'
                )

    am_source = AM_SOURCE_PATH.read_text(encoding='utf-8')
    if '#define II42_AM_SEMANTIC_QUERY_RESULT_COLS 15' not in am_source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: native semantic query contract is not '
            'the 15-column page-native contract'
        )
    mapper_source = am_source.split(
        'ii42_am_get_sae_lexical_mapper(Relation indexRelation)',
        maxsplit=1,
    )[-1].split(
        'ii42_am_semantic_builder_begin(',
        maxsplit=1,
    )[0]
    if 'ii42_segment_pages_load_maintenance_manifest(' not in mapper_source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: the SAE lexical mapper must use the '
            'bounded published-manifest loader'
        )
    if 'ii42_segment_pages_load_sealed_manifest(' in mapper_source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: the SAE lexical mapper validates the '
            'full COW closure on the query hot path'
        )
    page_query_source = PAGE_QUERY_SOURCE_PATH.read_text(encoding='utf-8')
    score_block_source = page_query_source.split(
        'ii42_page_query_score_block(',
        maxsplit=1,
    )[-1].split(
        'ii42_page_query_score_streaming(',
        maxsplit=1,
    )[0]
    if (
        'ii42_segment_pages_load_query_term_block_from_record('
        not in score_block_source
    ):
        errors.append(
            f'{PAGE_QUERY_SOURCE_PATH.name}: the streaming scorer re-reads '
            'posting block metadata already held by its cursor'
        )
    segment_pages_source = SEGMENT_PAGES_SOURCE_PATH.read_text(
        encoding='utf-8'
    )
    accelerator_source = (
        REPO_ROOT / 'src' / 'ii42_am_accelerator.c'
    ).read_text(encoding='utf-8')
    for required in (
        'ii42_segment_pages_init_preflight_fold_ref(',
        'ii42_segment_pages_term_retirements_equal(',
        'ii42_segment_page_reuse_arena handoff_arena;',
        'ii42_segment_page_reuse_arena staging_arena;',
        'ii42_segment_pages_set_staged_writes(',
        'II42_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED',
    ):
        if required not in segment_pages_source:
            errors.append(
                f'{SEGMENT_PAGES_SOURCE_PATH.name}: term fold lacks '
                f'append-only retirement preflight {required!r}'
            )
    term_fold_writer = segment_pages_source.split(
        'ii42_segment_pages_write_cow_term_fold(',
        maxsplit=1,
    )[-1].split(
        '\nii42_segment_cow_write_outcome\n'
        'ii42_segment_pages_write_cow_neutral_fold(',
        maxsplit=1,
    )[0]
    if 'ii42_segment_pages_load_sealed_manifest(' in term_fold_writer:
        errors.append(
            f'{SEGMENT_PAGES_SOURCE_PATH.name}: term-fold publication runs '
            'a corpus-wide COW closure under its reader fence'
        )
    if 'ii42_segment_pages_load_maintenance_manifest(' not in (
        term_fold_writer
    ):
        errors.append(
            f'{SEGMENT_PAGES_SOURCE_PATH.name}: term-fold publication lacks '
            'a bounded cold manifest identity check'
        )
    if segment_pages_source.count(
        'ii42_segment_pages_set_staged_writes('
    ) != 7:
        errors.append(
            f'{SEGMENT_PAGES_SOURCE_PATH.name}: all six online COW writers '
            'must inventory their staged closure pages'
        )
    impact_writer = am_source.split(
        'ii42_am_try_term_impact_specialization(',
        maxsplit=1,
    )[-1].split(
        '\nstatic ii42_am_term_structural_fold_outcome\n'
        'ii42_am_try_workload_term_fold(',
        maxsplit=1,
    )[0]
    impact_write = impact_writer.find(
        'ii42_segment_pages_write_cow_impact_fold('
    )
    impact_fence = impact_writer.find(
        'ii42_am_acquire_convergent_reader_fence('
    )
    if impact_write < 0 or impact_fence < 0 or impact_write > impact_fence:
        errors.append(
            f'{AM_SOURCE_PATH.name}: impact closure is not prepared before '
            'its publication-only reader fence'
        )
    if accelerator_source.count(
        'ii42_segment_pages_abandon_staged_write('
    ) != 1:
        errors.append(
            'ii42_am_accelerator.c: a busy publication fence does not '
            'recycle the prepared accelerator closure'
        )
    if 'context->page_validation_cache' not in segment_pages_source:
        errors.append(
            f'{SEGMENT_PAGES_SOURCE_PATH.name}: query range reads lack a '
            'bounded per-query page validation cache'
        )
    current_sql = CURRENT_SQL_PATH.read_text(encoding='utf-8')
    segment_smoke = (
        REPO_ROOT / 'scripts' / 'test_convergent_segment_read_smoke.py'
    ).read_text(encoding='utf-8')
    test_hooks = set(
        re.findall(
            r'PG_FUNCTION_INFO_V1\((ii42_test_[A-Za-z0-9_]+)\);',
            am_source,
        )
    )
    smoke_hooks = set(
        re.findall(r"['\"](ii42_test_[A-Za-z0-9_]+)['\"]", segment_smoke)
    )
    if test_hooks != smoke_hooks:
        errors.append(
            'ii42_am.c: white-box C hooks must exactly match the isolated '
            'page-native golden smoke; missing from smoke='
            f'{sorted(test_hooks - smoke_hooks)!r}, missing from AM='
            f'{sorted(smoke_hooks - test_hooks)!r}'
        )
    installed_test_hooks = sorted(
        hook for hook in test_hooks if hook in current_sql
    )
    if installed_test_hooks:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: installed SQL exposes white-box test '
            f'hooks: {installed_test_hooks!r}'
        )

    state_json_sql = current_sql.split(
        'CREATE FUNCTION ii42_index_runtime_state_json(index_name regclass)',
        maxsplit=1,
    )[-1].split(
        'COMMENT ON FUNCTION ii42_index_runtime_state_json(regclass)',
        maxsplit=1,
    )[0]
    if "AS 'MODULE_PATHNAME', 'ii42_index_runtime_state_json'" not in (
        state_json_sql
    ):
        errors.append(
            f'{CURRENT_SQL_PATH.name}: runtime state JSON is not bound '
            'directly to the C snapshot collector'
        )
    if 'regexp_match' in state_json_sql:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: runtime state JSON still parses '
            'diagnostic text with regular expressions'
        )

    design = CONVERGENT_DESIGN_PATH.read_text(encoding='utf-8')
    for stale_claim in (
        'preserving exact v2/v3 rows and scores',
        'The v2 gate interleaves the convergent index',
        'Legacy v2\nroots',
        'The first implementation checkpoint for this contract',
    ):
        if stale_claim in design:
            errors.append(
                f'{CONVERGENT_DESIGN_PATH.name}: stale ARCH-3 design claim '
                f'remains: {stale_claim!r}'
            )
    required_boundary = (
        'Any non-v3 root\nfails closed at installed query, mutation, '
        'maintenance, preload, and status\nboundaries.'
    )
    if required_boundary not in design:
        errors.append(
            f'{CONVERGENT_DESIGN_PATH.name}: non-v3 fail-closed boundary '
            'is missing'
        )

    retired_cache_programs = (
        'scripts/benchmark_generation_cache_churn.py',
        'scripts/test_generation_cache_smoke.py',
        'scripts/test_unified_delta_incremental_cache.py',
        'scripts/test_unified_delta_pressure_compaction.py',
        'scripts/benchmark_local_delta_cache_capacity.py',
        'scripts/benchmark_unified_delta_incremental_capacity.py',
    )
    for relative_path in retired_cache_programs:
        if (REPO_ROOT / relative_path).exists():
            errors.append(
                f'{relative_path}: retired decoded-cache program remains'
            )

    if 'II42_AM_MAINTENANCE_QUERY_VISIBILITY' in am_source:
        errors.append(
            f'{AM_SOURCE_PATH.name}: unreachable query-visibility '
            'maintenance class remains'
        )

    retired_decoded_contract = (
        'shared_preload_unified_delta_cache_entries',
        'shared_unified_delta_cache_current',
        'shared_unified_delta_cache_loading',
        'unified_delta_publish_due',
        'covered_delta_query_complete',
        'needs_background_convergence',
    )
    installed_contract_paths = (
        CURRENT_SQL_PATH,
        REPO_ROOT / 'docs' / 'api-reference.md',
        REPO_ROOT / 'docs' / 'shared-runtime-and-residency.md',
    )
    for path in installed_contract_paths:
        content = path.read_text(encoding='utf-8')
        for needle in retired_decoded_contract:
            if needle in content:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: retired decoded-cache '
                    f'contract remains: {needle!r}'
                )

    retired_page_native_status = (
        'share_eligible',
        'share_required',
        'descriptor_present',
        'descriptor_valid',
        'shared_preload_base_entries',
        'shared_preload_lexical_delta_cache_entries',
        'shared_preload_tombstone_entries',
        'shared_lexical_delta_cache_current',
        'shared_tombstone_current',
        'foreground_delta_offload_wakeups',
        'foreground_delta_local_builds',
        'generation_wait_prepares',
        'generation_wait_sleeps',
        'generation_wait_timeouts',
        'generation_wait_interrupts',
        'generation_wait_broadcasts',
        'shared_holder_effective',
        'admission_misses',
        'admission_miss_bytes',
        'last_admission_miss',
        'backlog_record_limit',
        'backlog_byte_limit',
    )
    page_native_status_paths = (
        CURRENT_SQL_PATH,
        REPO_ROOT / 'docs' / 'api-reference.md',
        REPO_ROOT / 'docs' / 'connection-memory.md',
        REPO_ROOT / 'docs' / 'functions.md',
        REPO_ROOT / 'docs' / 'index-policy.md',
        REPO_ROOT / 'docs' / 'shared-runtime-and-residency.md',
        REPO_ROOT / 'docs' / 'testing-and-validation.md',
        REPO_ROOT / 'docs' / 'examples' /
        'semantic-index-operations.md',
    )
    for path in page_native_status_paths:
        content = path.read_text(encoding='utf-8')
        for needle in retired_page_native_status:
            if needle in content:
                errors.append(
                    f'{path.relative_to(REPO_ROOT)}: retired page-native '
                    f'status remains: {needle!r}'
                )

    for needle in (
        'shared_preload_admission_misses=',
        'shared_preload_admission_miss_bytes=',
        'shared_preload_last_preload_admission_miss_',
    ):
        if needle in am_source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: retired preload admission status '
                f'remains: {needle!r}'
            )

    for needle in (
        'storage',
        'relation_blocks',
        'unified_warm_entries',
        'hot_fold_entries',
        'document_length_entries',
        'document_tid_lookup_entries',
        'hot_fold_current',
        'hot_fold_loading',
    ):
        if f'\\"{needle}\\"' not in am_source:
            errors.append(
                f'{AM_SOURCE_PATH.name}: page-native status field '
                f'is missing: {needle!r}'
            )

    lifecycle_smoke = (
        REPO_ROOT / 'scripts/test_convergent_segment_read_smoke.py'
    ).read_text(encoding='utf-8')
    if re.search(r'(?:\bv2\b|_v2\b|v2_)', lifecycle_smoke):
        errors.append(
            'test_convergent_segment_read_smoke.py: retired v2 fixture '
            'naming remains'
        )

    query_state_benchmark = (
        REPO_ROOT / 'scripts/benchmark_convergent_query_states.py'
    ).read_text(encoding='utf-8')
    if 'materialized' in query_state_benchmark:
        errors.append(
            'benchmark_convergent_query_states.py: retired materialized '
            'control naming remains'
            )


def check_rebuild_runner_storage_contract(errors: list[str]) -> None:
    runner = REBUILD_INDEXES_PATH.read_text(encoding='utf-8')
    required = (
        'REBUILD_PLAN_CONTRACT = 1',
        'def inventory_database(db):',
        'def require_refresh_plan_complete(',
        "'include_defs', index_inventory.include_defs",
        "predicate = (idx.get('predicate') or '').strip()",
        "include_sql = f\"INCLUDE ({', '.join(include_defs)})\\n\"",
        'semantic_runtime_preflight_before_refresh',
        "'DROP EXTENSION ii42;'",
        'ii42_catalog_contract_internal',
        'def current_generation_ready(db, idx):',
        "state->'generation'->>'storage' = 'convergent_segments'",
        "state->'generation'->>'payload_health' = 'ok'",
        "state->'generation'->>'rebuild_required'",
        'unsupported ii42 index metapage version',
        'unsupported ii42 index storage layout',
        'return current_generation_ready(db, idx)',
    )
    for contract in required:
        if contract not in runner:
            errors.append(
                f'{REBUILD_INDEXES_PATH.name}: missing current-storage '
                f'ready contract {contract!r}'
            )
    if 'DROP EXTENSION ii42 CASCADE' in runner:
        errors.append(
            f'{REBUILD_INDEXES_PATH.name}: destructive extension refresh '
            'still uses CASCADE instead of the audited atomic transition'
        )


def check_replication_smoke(errors: list[str]) -> None:
    smoke = REPLICATION_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        'update_and_insert_replay',
        'delete_and_update_replay',
        'UPDATE docs_bm25',
        'INSERT INTO docs_bm25',
        'DELETE FROM docs_bm25',
        'INSERT INTO docs_semantic',
        'UPDATE docs_semantic',
        'DELETE FROM docs_semantic',
        'REINDEX INDEX docs_bm25_idx',
        'REINDEX INDEX docs_semantic_idx',
        'same_relation_type_conversion_replay',
        'ALTER INDEX docs_convert_idx SET',
        'ALTER INDEX docs_convert_idx RESET (model_path)',
        'REINDEX INDEX docs_convert_idx',
        "'physical': 'bm25'",
        "'physical': 'semantic'",
        "summary['checks']['drop'] = 'ok'",
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{REPLICATION_SMOKE_PATH.name}: missing gate {needle!r}'
            )


def check_source_migration_smoke(errors: list[str]) -> None:
    smoke = SOURCE_MIGRATION_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        '--source-package-root',
        'CREATE EXTENSION psql_bm25s WITH SCHEMA public',
        'CREATE EXTENSION ii42 WITH SCHEMA ii42_ext',
        'public.psql_bm25s_search(',
        'public.psql_bm25s_maintain_index(',
        'CREATE INDEX CONCURRENTLY docs_new_idx',
        'side_by_side_score_parity',
        'pre_cutover_rollback',
        'dual_index_crud_parity',
        'DROP EXTENSION psql_bm25s',
        'source_rows_preserved',
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{SOURCE_MIGRATION_SMOKE_PATH.name}: missing migration '
                f'gate {needle!r}'
            )

    upgrading = (
        REPO_ROOT / 'docs/upgrading.md'
    ).read_text(encoding='utf-8')
    for needle in (
        'CREATE SCHEMA ii42_ext',
        'CREATE EXTENSION ii42 WITH SCHEMA ii42_ext',
        'cannot coexist in one schema',
    ):
        if needle not in upgrading:
            errors.append(
                f'docs/upgrading.md: missing migration boundary {needle!r}'
            )


def check_privilege_smoke(errors: list[str]) -> None:
    smoke = PRIVILEGE_SMOKE_PATH.read_text(encoding='utf-8')
    required = (
        'direct encoder should require source table SELECT privilege',
        'direct BM25 query should require source table SELECT privilege',
        'application role can execute diagnostic query functions',
        'application role can execute alternate top-k routes',
        'application role cannot execute product composition APIs',
        'fusion query should require source table SELECT privilege',
        'authorized product fusion query returned no rows',
        'authorized product hybrid query returned no rows',
        'application role should not execute direct BM25 diagnostics',
        'cache state should require source table SELECT privilege',
        'cache residency should require source table SELECT privilege',
        'index details should require source table SELECT privilege',
        'index policy should require source table SELECT privilege',
        'BM25 product search should reject row-level security',
        'model product search should reject row-level security',
        'application role should not use the internal semantic scorer',
        'application role should not use the build-only batch encoder',
        'non-owner application role unexpectedly refreshed index',
        'non-owner application role unexpectedly maintained index',
        'non-owner application role unexpectedly try-maintained index',
        'non-owner application role unexpectedly dropped index',
        'ordinary index owner could not search owned BM25 index',
        'ordinary index owner could not drop owned BM25 index',
    )
    for needle in required:
        if needle not in smoke:
            errors.append(
                f'{PRIVILEGE_SMOKE_PATH.name}: missing gate {needle!r}'
            )

    sql = CURRENT_SQL_PATH.read_text(encoding='utf-8')
    status_contract = (
        'CREATE FUNCTION ii42_index_status(index_name regclass)\n'
        'RETURNS jsonb\n'
        'LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE'
    )
    if status_contract not in sql:
        errors.append(
            f'{CURRENT_SQL_PATH.name}: status must use the guarded '
            'SECURITY DEFINER artifact-audit contract'
        )


def check_product_docs(errors: list[str]) -> None:
    required = (
        'ii42_query',
        'ii42_index_status',
    )
    for relative in PRODUCT_DOCS:
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f'{relative}: missing product documentation')
            continue
        text = path.read_text(encoding='utf-8')
        if 'ii42_search' in text:
            errors.append(f'{relative}: documents retired ii42_search API')
        for needle in required:
            if needle not in text:
                errors.append(f'{relative}: missing {needle}')
        for prefix in DISALLOWED_SPLIT_API_PREFIXES:
            if f'SELECT {prefix}' in text or f'FROM {prefix}' in text:
                errors.append(
                    f'{relative}: documents retired application API {prefix}'
                )
        for option in DISALLOWED_SPLIT_LIFECYCLE_OPTIONS:
            if f'`{option}`' in text or f'{option} =' in text:
                errors.append(
                    f'{relative}: documents retired sidecar option {option}'
                )
        for option in RETIRED_PRODUCT_RELOPTIONS:
            if f'`{option}`' in text or f'{option} =' in text:
                errors.append(
                    f'{relative}: documents retired reloption {option}'
                )
        for guc in RETIRED_PRODUCT_GUCS:
            if guc in text:
                errors.append(
                    f'{relative}: documents retired product GUC {guc}'
                )
        for local_path in MAINTAINER_LOCAL_PATHS:
            if local_path in text:
                errors.append(
                    f'{relative}: embeds maintainer-local reference '
                    f'{local_path}'
                )

    getting_started_path = REPO_ROOT / 'docs/getting-started.md'
    if getting_started_path.is_file():
        getting_started = getting_started_path.read_text(encoding='utf-8')
        onboarding_contracts = (
            ('USING ii42 (body)', 'single-column index example'),
            ('sae = true', 'semantic index example'),
            ('eventual-only', 'semantic consistency boundary'),
            ('ii42_index_status', 'readiness example'),
            ('ii42_index_try_maintain', 'maintenance example'),
            ('REINDEX INDEX', 'rebuild example'),
            ('DROP INDEX', 'native drop example'),
            ('multicolumn-indexes.md', 'multicolumn guide link'),
            ('field-aware-indexes.md', 'field-aware guide link'),
        )
        for needle, description in onboarding_contracts:
            if needle not in getting_started:
                errors.append(
                    'docs/getting-started.md: missing '
                    f'{description}: {needle!r}'
                )
        if 'ii42_index_drop' in getting_started:
            errors.append(
                'docs/getting-started.md: removed drop wrapper remains'
            )

    for relative in PRODUCT_DOCS:
        path = REPO_ROOT / relative
        if path.is_file() and 'ii42_index_drop' in path.read_text(
            encoding='utf-8'
        ):
            errors.append(
                f'{relative}: removed drop wrapper remains documented'
            )

    model_checkout_path = (
        REPO_ROOT / 'docs/examples/semantic-model-checkout.md'
    )
    if not model_checkout_path.is_file():
        errors.append('docs/examples/semantic-model-checkout.md: missing')
    else:
        model_checkout = model_checkout_path.read_text(encoding='utf-8')
        for contract in (
            'ii42_index_audit',
            'explicit_audit_required',
            'bounded readiness',
        ):
            if contract not in model_checkout:
                errors.append(
                    'docs/examples/semantic-model-checkout.md: missing '
                    f'status/audit boundary {contract!r}'
                )

    for relative in set(CURRENT_CONTRACT_DOCS + PAGE_NATIVE_V3_DOCS):
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        for guc in RETIRED_PRODUCT_GUCS:
            if guc in text:
                errors.append(
                    f'{relative}: documents retired product GUC {guc}'
                )

    for relative in CURRENT_LIFECYCLE_DOCS:
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f'{relative}: missing lifecycle documentation')
            continue
        text = path.read_text(encoding='utf-8')
        for phrase in RETIRED_LIFECYCLE_DOC_PHRASES:
            if phrase in text:
                errors.append(
                    f'{relative}: documents stale lifecycle contract '
                    f'{phrase!r}'
                )

    for relative in SAE_EVENTUAL_ONLY_DOCS:
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f'{relative}: missing SAE contract documentation')
            continue
        text = path.read_text(encoding='utf-8')
        if re.search(r'\beventual[- ]only\b', text, re.IGNORECASE) is None:
            errors.append(
                f'{relative}: does not state that SAE is eventual-only'
            )
        if UNSUPPORTED_SAE_INDEX_SQL_RE.search(text) is not None:
            errors.append(
                f'{relative}: contains an SAE realtime/manual CREATE INDEX '
                'example'
            )

    for relative in PAGE_NATIVE_V3_DOCS:
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f'{relative}: missing v3 memory documentation')
            continue
        text = path.read_text(encoding='utf-8')
        if 'page-native' not in text.lower():
            errors.append(
                f'{relative}: does not identify the page-native v3 path'
            )
        normalized = text.lower()
        for phrase in STALE_PAGE_NATIVE_DOC_PHRASES:
            if phrase in normalized:
                errors.append(
                    f'{relative}: documents stale v2 cache authority '
                    f'{phrase!r}'
                )

    retired_catalog_version = '0.' + '1.2'
    for relative in CURRENT_CONTRACT_DOCS:
        path = REPO_ROOT / relative
        if (
            path.is_file()
            and retired_catalog_version in path.read_text(encoding='utf-8')
        ):
            errors.append(
                f'{relative}: current product documentation promises a '
                'retired catalog version'
            )

    runtime_header = RUNTIME_SERVICE_HEADER_PATH.read_text(encoding='utf-8')
    runtime_version = re.search(
        r'^#define II42_RUNTIME_SERVICE_VERSION (\d+)$',
        runtime_header,
        re.MULTILINE,
    )
    design_contract = CONVERGENT_DESIGN_PATH.read_text(encoding='utf-8')
    if runtime_version is None:
        errors.append(
            f'{RUNTIME_SERVICE_HEADER_PATH.name}: runtime ABI version is '
            'missing'
        )
    elif (
        f'runtime ABI version {runtime_version.group(1)}'
        not in design_contract
    ):
        errors.append(
            f'{CONVERGENT_DESIGN_PATH.name}: runtime ABI contract '
            'does not match the current header'
        )

    historical_contracts = (
        (
            HISTORICAL_QUALIFICATION_PATH,
            'Historical status (superseded)',
        ),
        (
            HISTORICAL_QUALIFICATION_PATH,
            'support matrix for the current source package.',
        ),
        (
            HISTORICAL_QUALIFICATION_EVIDENCE_README,
            'immutable historical evidence archive',
        ),
    )
    for path, contract in historical_contracts:
        if (
            not path.is_file()
            or contract not in path.read_text(encoding='utf-8')
        ):
            errors.append(
                f'{path.relative_to(REPO_ROOT)}: historical evidence lacks '
                f'the current-only boundary {contract!r}'
            )


def normalized_local_link_target(
    raw_target: str,
) -> tuple[str, str] | None:
    target = raw_target.strip()
    if target.startswith('<') and '>' in target:
        target = target[1:target.index('>')]
    else:
        target = re.split(r'\s+["\']', target, maxsplit=1)[0]
    if target.startswith('//'):
        return None
    if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.IGNORECASE):
        return None
    path_target, separator, fragment = target.partition('#')
    path_target = unquote(path_target.split('?', 1)[0])
    fragment = unquote(fragment) if separator else ''
    if not path_target and not fragment:
        return None
    return path_target, fragment


def optional_research_source_link(source: Path, target: Path) -> bool:
    if source.parts[:3] != ('docs', 'research-sae', 'reports'):
        return False
    if not target.parts:
        return False
    name = target.name
    if target.parts[0] == 'scripts':
        return name.startswith(('ng', 'review_ng', 'requirements-ng'))
    if target.parts[0] == 'tests':
        return name.startswith(('test_ng', 'test_review_ng'))
    return False


def markdown_heading_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    in_fence = False
    fence_character = ''

    for line in path.read_text(encoding='utf-8').splitlines():
        fence_match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence_match is not None:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_character = marker
            elif marker == fence_character:
                in_fence = False
                fence_character = ''
            continue
        if in_fence:
            continue

        anchors.update(
            unquote(match.group(1))
            for match in MARKDOWN_EXPLICIT_ANCHOR_RE.finditer(line)
        )
        heading_match = MARKDOWN_HEADING_RE.match(line)
        if heading_match is None:
            continue
        heading = re.sub(r'\s+#+\s*$', '', heading_match.group(1))
        heading = re.sub(r'!?\[([^\]]+)\]\([^)]+\)', r'\1', heading)
        heading = re.sub(r'<[^>]+>', '', heading)
        heading = html.unescape(heading)
        heading = re.sub(r'[`*_~]', '', heading).strip().lower()
        slug = re.sub(r'[^\w\- ]', '', heading)
        slug = re.sub(r'\s+', '-', slug)
        if not slug:
            continue
        duplicate_count = slug_counts.get(slug, 0)
        slug_counts[slug] = duplicate_count + 1
        if duplicate_count:
            slug = f'{slug}-{duplicate_count}'
        anchors.add(slug)
    return anchors


def check_markdown_links(errors: list[str]) -> None:
    repository_root = REPO_ROOT.resolve()
    anchor_cache: dict[Path, set[str]] = {}
    markdown_paths = sorted(
        relative
        for relative in repository_inventory_paths()
        if relative.suffix.lower() == '.md'
    )
    for relative in markdown_paths:
        path = REPO_ROOT / relative
        in_fence = False
        fence_character = ''
        for line_number, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(),
            1,
        ):
            fence_match = re.match(r'^\s*(`{3,}|~{3,})', line)
            if fence_match is not None:
                marker = fence_match.group(1)[0]
                if not in_fence:
                    in_fence = True
                    fence_character = marker
                elif marker == fence_character:
                    in_fence = False
                    fence_character = ''
                continue
            if in_fence:
                continue

            raw_targets = [
                match.group(1)
                for match in MARKDOWN_INLINE_LINK_RE.finditer(line)
            ]
            reference_match = MARKDOWN_REFERENCE_LINK_RE.match(line)
            if reference_match is not None:
                raw_targets.append(reference_match.group(1))
            raw_targets.extend(
                match.group(1)
                for match in MARKDOWN_HTML_LINK_RE.finditer(line)
            )
            for raw_target in raw_targets:
                normalized_target = normalized_local_link_target(raw_target)
                if normalized_target is None:
                    continue
                target, fragment = normalized_target
                if Path(target).is_absolute():
                    errors.append(
                        f'{relative}:{line_number}: absolute local Markdown '
                        f'link is not release-portable: {raw_target!r}'
                    )
                    continue
                resolved = (
                    (path.parent / target).resolve()
                    if target
                    else path.resolve()
                )
                if not resolved.is_relative_to(repository_root):
                    errors.append(
                        f'{relative}:{line_number}: Markdown link escapes '
                        f'the repository: {raw_target!r}'
                    )
                elif not resolved.exists():
                    target_relative = resolved.relative_to(repository_root)
                    if optional_research_source_link(
                        relative,
                        target_relative,
                    ):
                        continue
                    errors.append(
                        f'{relative}:{line_number}: Markdown link target '
                        f'does not exist: {raw_target!r}'
                    )
                elif fragment and resolved.suffix.lower() == '.md':
                    if resolved not in anchor_cache:
                        anchor_cache[resolved] = markdown_heading_anchors(
                            resolved
                        )
                    anchors = anchor_cache[resolved]
                    if fragment not in anchors:
                        errors.append(
                            f'{relative}:{line_number}: Markdown link anchor '
                            f'does not exist: {raw_target!r}'
                        )
        if in_fence:
            errors.append(
                f'{relative}: Markdown code fence is not closed'
            )


def check_research_report_layout(errors: list[str]) -> None:
    for retired_name in (
        'research-artifacts',
        'research-reports',
        'research-source',
    ):
        retired_path = REPO_ROOT / 'docs' / retired_name
        if retired_path.exists():
            errors.append(
                f'{retired_path.relative_to(REPO_ROOT)}: retired SAE '
                'research root must remain absent'
            )

    root_contracts = (
        'single repository entrypoint',
        '[Reports](reports/README.md)',
        '[Artifacts](artifacts/README.md)',
        '[Source](source/README.md)',
        'Do not add another `docs/research-*` root.',
    )
    if not RESEARCH_SAE_README_PATH.is_file():
        errors.append('docs/research-sae/README.md: missing SAE research map')
    else:
        root_readme = RESEARCH_SAE_README_PATH.read_text(encoding='utf-8')
        for contract in root_contracts:
            if contract not in root_readme:
                errors.append(
                    'docs/research-sae/README.md: missing archive contract '
                    f'{contract!r}'
                )

    readme_path = RESEARCH_REPORT_ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    allowed_collections = {'designs', 'milestones', 'ng0001-ng0099'}

    for stage_path in sorted(RESEARCH_REPORT_ROOT.iterdir()):
        if not stage_path.is_dir():
            continue
        stage_match = RESEARCH_STAGE_RE.fullmatch(stage_path.name)
        if stage_match is None:
            if stage_path.name not in allowed_collections:
                errors.append(
                    f'{stage_path.relative_to(REPO_ROOT)}: unknown research '
                    'report collection'
                )
            continue
        start, end = map(int, stage_match.groups())
        reports = sorted(stage_path.glob('*.md'))
        table_row = (
            f'| [M{start:04d}-M{end:04d}]({stage_path.name}/) '
            f'| {len(reports)} |'
        )
        if table_row not in readme:
            errors.append(
                f'{readme_path.relative_to(REPO_ROOT)}: missing exact stage '
                f'inventory row {table_row!r}'
            )
        for report in reports:
            number_match = RESEARCH_REPORT_NUMBER_RE.search(report.name)
            if number_match is None:
                continue
            number = int(number_match.group(1))
            if not start <= number <= end:
                errors.append(
                    f'{report.relative_to(REPO_ROOT)}: M{number} is outside '
                    f'{stage_path.name}'
                )

    for report in sorted(RESEARCH_REPORT_ROOT.glob('*.md')):
        if report.name == 'README.md':
            continue
        number_match = ROOT_NUMBERED_RESEARCH_RE.match(report.name)
        if (
            number_match is not None
            and int(number_match.group(1)) >= 100
        ):
            errors.append(
                f'{report.relative_to(REPO_ROOT)}: numbered report must be '
                'stored in its stage directory'
            )

    for report in sorted(RESEARCH_REPORT_ROOT.rglob('*.md')):
        if report.stat().st_size == 0:
            errors.append(
                f'{report.relative_to(REPO_ROOT)}: empty research report'
            )


def check_engineering_plan_authority(errors: list[str]) -> None:
    retired_path = REPO_ROOT / 'docs/release-readiness-plan.md'
    if retired_path.exists():
        errors.append(
            'docs/release-readiness-plan.md: closed ledger must remain '
            'archived'
        )

    required_contracts = (
        (
            CONVERGENT_DESIGN_PATH,
            'current design authority',
        ),
        (
            ARCHIVED_DEVELOPMENT_RECORD_PATH,
            'closed stage-development record',
        ),
        (
            ARCHIVED_RELEASE_READINESS_PATH,
            'closed historical ledger',
        ),
        (
            RESEARCH_REPORT_ROOT / 'README.md',
            'not active engineering TODO authorities',
        ),
    )
    for path, contract in required_contracts:
        if (
            not path.is_file()
            or contract not in path.read_text(encoding='utf-8')
        ):
            errors.append(
                f'{path.relative_to(REPO_ROOT)}: missing planning authority '
                f'boundary {contract!r}'
            )

    planning_name = re.compile(
        r'(?:plan|todo|roadmap|checklist)',
        re.IGNORECASE,
    )
    active = {
        relative
        for relative in repository_inventory_paths()
        if relative.parts
        and relative.parts[0] == 'docs'
        and relative.suffix.lower() == '.md'
        and planning_name.search(relative.name) is not None
        and relative.parts[:2] != ('docs', 'archive')
        and relative.parts[:3] != ('docs', 'research-sae', 'reports')
    }
    expected = {
        Path('docs/product-roadmap.md'),
        Path('docs/model-planning.md'),
    }
    if active != expected:
        errors.append(
            'docs: active engineering planning documents must be exactly '
            f'{sorted(map(str, expected))}, found {sorted(map(str, active))}'
        )

    roadmap_path = REPO_ROOT / 'docs/product-roadmap.md'
    if roadmap_path.is_file():
        roadmap = roadmap_path.read_text(encoding='utf-8')
        for contract in (
            'current engineering planning authority',
            'completed the bounded current-only qualification',
            'evoke-v0.2.5-current-only-three-environment-rollout-2026-08-30.md',
            '[Model Planning](model-planning.md)',
        ):
            if contract not in roadmap:
                errors.append(
                    'docs/product-roadmap.md: missing current planning '
                    f'contract {contract!r}'
                )
        if 'Active Release Candidate: v0.2.5-rc1' in roadmap:
            errors.append(
                'docs/product-roadmap.md: retains completed rollout as an '
                'active release-candidate gate'
            )

    model_plan_path = REPO_ROOT / 'docs/model-planning.md'
    if model_plan_path.is_file():
        model_plan = model_plan_path.read_text(encoding='utf-8')
        for contract in (
            'subordinate to the',
            '[Product Roadmap](product-roadmap.md)',
            '[English](technical-report-evoke-model.md)',
            '[Traditional Chinese](technical-report-evoke-model-zh.md)',
        ):
            if contract not in model_plan:
                errors.append(
                    'docs/model-planning.md: missing subordinate planning '
                    f'contract {contract!r}'
                )


def repository_inventory_paths() -> tuple[Path, ...]:
    result = subprocess.run(
        [
            'git',
            '-C',
            str(REPO_ROOT),
            'ls-files',
            '--cached',
            '--others',
            '--exclude-standard',
            '-z',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return tuple(
            Path(value)
            for value in result.stdout.split('\0')
            if value and (REPO_ROOT / value).is_file()
        )
    return tuple(
        path.relative_to(REPO_ROOT)
        for path in REPO_ROOT.rglob('*')
        if path.is_file() and '.git' not in path.parts
    )


def check_repository_layout(errors: list[str]) -> None:
    for relative in OPEN_SOURCE_ROOT_FILES:
        if not (REPO_ROOT / relative).is_file():
            errors.append(f'{relative}: missing open-source project policy')

    for path in (
        RESEARCH_SAE_README_PATH,
        RESEARCH_ARTIFACT_README_PATH,
        RESEARCH_ARTIFACT_MANIFEST_PATH,
        RESEARCH_SOURCE_README_PATH,
        RESEARCH_SOURCE_MANIFEST_PATH,
    ):
        if not path.is_file():
            errors.append(
                f'{path.relative_to(REPO_ROOT)}: missing research archive '
                'contract'
            )

    if RESEARCH_ARTIFACT_MANIFEST_PATH.is_file():
        content = RESEARCH_ARTIFACT_MANIFEST_PATH.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != RESEARCH_ARTIFACT_MANIFEST_SHA256:
            errors.append(
                'docs/research-sae/artifacts/manifest.jsonl: digest does not '
                'match the frozen archive'
            )
        line_count = len(content.splitlines())
        if line_count != 1401:
            errors.append(
                'docs/research-sae/artifacts/manifest.jsonl: expected 1401 '
                f'artifacts, found {line_count}'
            )

    if RESEARCH_SOURCE_MANIFEST_PATH.is_file():
        content = RESEARCH_SOURCE_MANIFEST_PATH.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != RESEARCH_SOURCE_MANIFEST_SHA256:
            errors.append(
                'docs/research-sae/source/manifest.jsonl: digest does not match '
                'the frozen archive'
            )
        line_count = len(content.splitlines())
        if line_count != 1595:
            errors.append(
                'docs/research-sae/source/manifest.jsonl: expected 1595 source '
                f'files, found {line_count}'
            )

    for relative in repository_inventory_paths():
        if relative.name == '.DS_Store':
            errors.append(f'{relative}: operating-system metadata is tracked')
        if relative in RETIRED_SOURCE_PATHS:
            errors.append(f'{relative}: retired source path is tracked')
        if relative.parts and relative.parts[0] in {'runs', 'outputs'}:
            errors.append(
                f'{relative}: generated research output is tracked'
            )
        if (
            len(relative.parts) == 1
            and ROOT_RESEARCH_ARTIFACT_RE.fullmatch(relative.name)
        ):
            errors.append(
                f'{relative}: root-level research artifact is tracked'
            )
        if (
            relative.parts
            and relative.parts[0] == 'scripts'
            and RESEARCH_SCRIPT_RE.search(relative.name)
        ):
            errors.append(
                f'{relative}: historical research script is tracked'
            )
        if (
            relative.parts
            and relative.parts[0] == 'scripts'
            and VERSIONED_PRODUCT_SCRIPT_RE.search(relative.name)
        ):
            errors.append(
                f'{relative}: product script name encodes a retired '
                'implementation generation'
            )
        if relative.parts and relative.parts[0] == 'patches':
            errors.append(
                f'{relative}: one-off research patch is tracked'
            )
        if relative.suffix.lower() not in ACTIVE_TEXT_SUFFIXES:
            continue
        if relative.parts[:2] == ('docs', 'research-sae'):
            continue
        text = (REPO_ROOT / relative).read_text(
            encoding='utf-8',
            errors='replace',
        )
        shared_runtime_helper = (
            'from test_unified_index_lifecycle_smoke import (' in text
            and 'configure_cluster,' in text
            and 'configure_cluster(' in text
        )
        if (
            relative.parts
            and relative.parts[0] == 'scripts'
            and relative.suffix == '.py'
            and '--model-path' in text
            and 'initdb' in text
            and not shared_runtime_helper
            and (
                'shared_preload_libraries' not in text
                or 'shared_runtime_size' not in text
            )
        ):
            errors.append(
                f'{relative}: model-backed temporary PostgreSQL setup does '
                'not declare the required shared runtime and arena'
            )
        if re.search(
            rf'\b{RETIRED_RUNTIME_NAME}(?:_|\d)',
            text,
            re.IGNORECASE,
        ):
            errors.append(
                f'{relative}: retired runtime family '
                f'{RETIRED_RUNTIME_NAME}* remains outside the research '
                'archive'
            )


def main() -> int:
    errors: list[str] = []
    check_sql_contract(errors)
    check_install_contract(errors)
    check_active_scripts(errors)
    check_c_contract(errors)
    check_lifecycle_smoke(errors)
    check_cache_failure_smoke(errors)
    check_vacuum_frontier_smoke(errors)
    check_storage_layout_boundary(errors)
    check_maturity_suite(errors)
    check_arch3_legacy_runtime_absent(errors)
    check_rebuild_runner_storage_contract(errors)
    check_replication_smoke(errors)
    check_source_migration_smoke(errors)
    check_privilege_smoke(errors)
    check_product_docs(errors)
    check_markdown_links(errors)
    check_research_report_layout(errors)
    check_engineering_plan_authority(errors)
    check_repository_layout(errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print('product convergence inventory ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
