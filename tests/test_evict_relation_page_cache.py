from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'evict_relation_page_cache.py'
SPEC = importlib.util.spec_from_file_location(
    'evict_relation_page_cache',
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments(**overrides: object) -> SimpleNamespace:
    values = {
        'allow_page_cache_eviction': True,
        'expected_port': 56544,
        'max_resident_ratio': 0.01,
        'max_resident_mib': 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_requires_explicit_acknowledgement() -> None:
    with pytest.raises(ValueError, match='allow-page-cache-eviction'):
        MODULE.validate_args(
            arguments(allow_page_cache_eviction=False)
        )


def test_validation_rejects_hosts_without_posix_fadvise() -> None:
    if hasattr(MODULE.os, 'posix_fadvise'):
        pytest.skip('host provides POSIX_FADV_DONTNEED')
    with pytest.raises(RuntimeError, match='unavailable on this host'):
        MODULE.validate_args(arguments())


def test_runtime_ready_accepts_both_shared_query_tiers() -> None:
    resident = {
        'generation': {'auto_preload_priority': 100},
        'shared_preload': {
            'available': True,
            'loading': False,
            'resident_fold_current': True,
            'resident_fold_loading': False,
        },
    }
    metadata = {
        'generation': {'auto_preload_priority': 100},
        'shared_preload': {
            'available': True,
            'loading': False,
            'query_warm_marker_valid': True,
            'query_metadata_warm': True,
        },
    }

    assert MODULE.runtime_ready(resident) is True
    assert MODULE.runtime_ready(metadata) is True
    metadata['shared_preload']['query_metadata_warm'] = False
    assert MODULE.runtime_ready(metadata) is False


def test_publication_stability_requires_generation_and_relation() -> None:
    generation = {
        'generation_id': '1/2/3',
        'contract_signature': 'contract',
        'posting': {'signature': 'posting'},
        'primary': {'manifest_start_block': 42},
        'semantic_accelerator': {
            'source_manifest_id': '2',
            'builder_policy_id': 7,
        },
    }
    relation = {
        'oid': 42,
        'relkind': 'i',
        'access_method': 'evoke',
        'relative_path': 'base/1/42',
        'main_fork_bytes': 8192,
    }

    assert MODULE.publication_stable(
        generation,
        generation,
        relation,
        relation,
    ) is True
    assert MODULE.publication_stable(
        generation,
        {**generation, 'generation_id': '1/2/4'},
        relation,
        relation,
    ) is False
    assert MODULE.publication_stable(
        generation,
        generation,
        relation,
        {**relation, 'main_fork_bytes': 16384},
    ) is False


def test_catalog_text_normalizes_sql_ascii_values() -> None:
    assert MODULE.catalog_text(b'i') == 'i'
    assert MODULE.catalog_text(b'evoke') == 'evoke'
    assert MODULE.catalog_text('evoke') == 'evoke'


def test_residency_gate_requires_ratio_and_absolute_bound() -> None:
    qualified = {
        'resident_ratio': 0.005,
        'resident_bytes': 16 * 1024 * 1024,
    }

    assert MODULE.residency_qualified(
        qualified,
        0.01,
        64 * 1024 * 1024,
    ) is True
    assert MODULE.residency_qualified(
        {**qualified, 'resident_ratio': 0.02},
        0.01,
        64 * 1024 * 1024,
    ) is False
    assert MODULE.residency_qualified(
        {**qualified, 'resident_bytes': 128 * 1024 * 1024},
        0.01,
        64 * 1024 * 1024,
    ) is False


def test_relation_segment_order_rejects_gaps(tmp_path: Path) -> None:
    relative = 'base/1/42'
    base = tmp_path / relative
    base.parent.mkdir(parents=True)
    base.write_bytes(b'a')
    Path(str(base) + '.2').write_bytes(b'b')

    with pytest.raises(RuntimeError, match='sequence is incomplete'):
        MODULE.relation_segments(tmp_path, relative)


def test_relation_segment_order_accepts_postgresql_suffixes(
    tmp_path: Path,
) -> None:
    relative = 'base/1/42'
    base = tmp_path / relative
    base.parent.mkdir(parents=True)
    base.write_bytes(b'a')
    Path(str(base) + '.1').write_bytes(b'b')
    Path(str(base) + '.2').write_bytes(b'c')

    assert MODULE.relation_segments(tmp_path, relative) == [
        base,
        Path(str(base) + '.1'),
        Path(str(base) + '.2'),
    ]


def test_mincore_residency_and_eviction_on_regular_file(
    tmp_path: Path,
) -> None:
    if not hasattr(MODULE.os, 'posix_fadvise'):
        pytest.skip('POSIX_FADV_DONTNEED is unavailable on this host')
    path = tmp_path / 'relation'
    path.write_bytes(b'x' * (2 * 1024 * 1024))
    with path.open('rb') as handle:
        while handle.read(128 * 1024):
            pass

    before = MODULE.residency([path])
    MODULE.evict([path])
    after = MODULE.residency([path])

    assert before['resident_pages'] > 0
    assert after['resident_pages'] <= before['resident_pages']
