from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.create_p2_runtime_variant import create_variant


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_checkout(root: Path) -> None:
    scoring_path = root / 'calibration' / 'scoring_profile.json'
    contract_path = root / 'provenance' / 'p2_contract.json'
    write_json(
        scoring_path,
        {
            'document_compiler': {
                'semantic_budget_ratio_to_lexical': 1.125,
                'semantic_encoder': {'active_dims': 192},
            },
        },
    )
    write_json(contract_path, {'budget_ratio': 1.125})
    write_json(
        root / 'manifest.json',
        {
            'api_version': 'evoke_model_v1',
            'artifacts': {
                'p2_contract': {
                    'path': 'provenance/p2_contract.json',
                    'sha256': sha256_file(contract_path),
                },
                'scoring_profile': {
                    'path': 'calibration/scoring_profile.json',
                    'sha256': sha256_file(scoring_path),
                },
            },
            'model_id': 'source',
            'runtime_abi': 'evoke_p2_unified_text_atoms_v2',
            'schema_version': 1,
            'runtime_output': {
                'document_semantic_budget_ratio': 1.125,
                'document_semantic_max_atoms': 192,
            },
        },
    )


def test_budget_variant_updates_contract_and_digests(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'variant'
    make_checkout(source)

    summary = create_variant(
        source=source,
        output=output,
        model_id='budget-075',
        budget_ratio=0.75,
    )

    manifest = json.loads((output / 'manifest.json').read_text())
    scoring_path = output / 'calibration' / 'scoring_profile.json'
    contract_path = output / 'provenance' / 'p2_contract.json'
    scoring = json.loads(scoring_path.read_text())
    contract = json.loads(contract_path.read_text())
    assert summary['field'] == 'document_semantic_budget_ratio'
    assert manifest['model_id'] == 'budget-075'
    assert manifest['runtime_output'][
        'document_semantic_budget_ratio'
    ] == 0.75
    assert scoring['document_compiler'][
        'semantic_budget_ratio_to_lexical'
    ] == 0.75
    assert contract['budget_ratio'] == 0.75
    assert contract['lossy_variant']['source_value'] == 1.125
    assert manifest['artifacts']['scoring_profile']['sha256'] == (
        sha256_file(scoring_path)
    )
    assert manifest['artifacts']['p2_contract']['sha256'] == (
        sha256_file(contract_path)
    )
    source_manifest = json.loads((source / 'manifest.json').read_text())
    assert source_manifest['model_id'] == 'source'


def test_max_atoms_variant_changes_only_runtime_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'variant'
    make_checkout(source)

    create_variant(
        source=source,
        output=output,
        model_id='atoms-096',
        max_atoms=96,
    )

    manifest = json.loads((output / 'manifest.json').read_text())
    scoring = json.loads(
        (output / 'calibration' / 'scoring_profile.json').read_text(),
    )
    contract = json.loads(
        (output / 'provenance' / 'p2_contract.json').read_text(),
    )
    assert manifest['runtime_output']['document_semantic_max_atoms'] == 96
    assert manifest['runtime_output'][
        'document_semantic_budget_ratio'
    ] == 1.125
    assert scoring['document_compiler']['semantic_encoder'][
        'active_dims'
    ] == 96
    assert contract['budget_ratio'] == 1.125
    assert contract['lossy_variant']['field'] == (
        'document_semantic_max_atoms'
    )


def test_variant_rejects_no_op_and_invalid_source(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    make_checkout(source)
    with pytest.raises(ValueError, match='does not change'):
        create_variant(
            source=source,
            output=tmp_path / 'no-op',
            model_id='no-op',
            budget_ratio=1.125,
        )

    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = 0
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='source checkout'):
        create_variant(
            source=source,
            output=tmp_path / 'invalid',
            model_id='invalid',
            budget_ratio=0.75,
        )
