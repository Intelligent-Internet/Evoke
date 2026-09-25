from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_URL = 'https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1'
DOWNLOAD_DOC = REPO_ROOT / 'docs/examples/semantic-model-checkout.md'


def test_model_card_links_to_public_project_repository() -> None:
    card = (REPO_ROOT / 'packaging/huggingface/README.md').read_text()
    project_url = 'https://github.com/Intelligent-Internet/Evoke'
    github_links = re.findall(r'https://github\.com/[^\s)]+', card)
    assert set(github_links) == {
        project_url,
        f'{project_url}/blob/main/CONTRIBUTING.md',
        f'{project_url}/blob/main/docs/README.md',
    }
    for name in ('CONTRIBUTING.md', 'docs/README.md'):
        assert (REPO_ROOT / name).is_file()


def test_public_model_card_matches_frozen_content_lock() -> None:
    lock = json.loads(
        (REPO_ROOT / 'packaging/milestone-model.json').read_text(),
    )
    card = (REPO_ROOT / 'packaging/huggingface/README.md').read_text()
    assert 'license: apache-2.0' in card
    assert lock['upstream']['model'] in card
    for value in (
        lock['bundle_name'],
        lock['manifest_sha256'],
        lock['upstream']['revision'],
        lock['manifest_contract']['model_id'],
        lock['manifest_contract']['runtime_abi'],
    ):
        assert f'`{value}`' in card
    for artifact in lock['artifacts'].values():
        assert Path(artifact['path']).name in card
    ort_version = (
        (REPO_ROOT / 'packaging/onnxruntime.version').read_text().strip()
    )
    assert f'ONNX Runtime {ort_version}' in card
    assert 'not measurements of' in card
    assert 'exact model file inventory' in card


def test_download_instructions_pin_revision_checksum_and_validator() -> None:
    doc = DOWNLOAD_DOC.read_text()
    values = dict(re.findall(r"^(model_\w+)='([^']+)'$", doc, re.MULTILINE))
    lock = json.loads(
        (REPO_ROOT / 'packaging/milestone-model.json').read_text(),
    )
    card = (REPO_ROOT / 'packaging/huggingface/README.md').read_text()
    assert f'https://huggingface.co/{values["model_repo"]}' == HUB_URL
    assert re.fullmatch(r'[0-9a-f]{40}', values['model_revision'])
    assert re.fullmatch(r'[0-9a-f]{64}', values['model_sha256'])
    assert values['model_archive'] == f'{lock["bundle_name"]}.zip'
    assert f'`{values["model_sha256"]}`' in card
    assert '/resolve/${model_revision}/${model_archive}' in doc
    assert '--archive-sha256 "$model_sha256"' in doc
    assert '--output .artifacts/evoke-milestone-model' in doc
    assert 'scripts/validate_milestone_model_checkout.py' in doc
    assert '.cache/huggingface' in doc


def test_build_entrypoints_link_public_model_and_canonical_instructions() -> None:
    for name in ('README.md', 'CONTRIBUTING.md'):
        doc = (REPO_ROOT / name).read_text()
        assert HUB_URL in doc
        assert (
            'docs/examples/semantic-model-checkout.md'
            '#download-the-default-model'
        ) in doc
    for name in (
        'docs/technical-report-evoke-model.md',
        'docs/technical-report-evoke-model-zh.md',
    ):
        assert HUB_URL in (REPO_ROOT / name).read_text()
