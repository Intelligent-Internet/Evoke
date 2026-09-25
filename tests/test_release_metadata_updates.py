from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NON_RELEASE_PATHS = (
    ':(exclude,glob)**/*.md',
    ':(exclude)docs/',
    ':(exclude)tests/',
    ':(exclude).github/',
)


@pytest.mark.parametrize(('path', 'metadata_only'), (
    ('README.md', True),
    ('CONTRIBUTING.md', True),
    ('packaging/huggingface/README.md', True),
    ('docs/banner.png', True),
    ('tests/test_published_model_documentation.py', True),
    ('.github/workflows/prepare-release.yml', True),
    ('src/evoke_am.c', False),
    ('sql/evoke--0.2.5.sql', False),
    ('evoke.control', False),
    ('Makefile', False),
    ('scripts/build_release_zip.sh', False),
    ('packaging/milestone-model.json', False),
    ('packaging/onnxruntime.version', False),
    ('packaging/docker/postgres18/Dockerfile', False),
    ('LICENSE', False),
    ('packaging/MILESTONE-MODEL-NOTICE', False),
))
def test_metadata_exemption_preserves_release_inputs(
    tmp_path: Path, path: str, metadata_only: bool,
) -> None:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ['git', *args], cwd=tmp_path, check=True,
            capture_output=True, text=True,
        )

    git('init', '-q')
    git('config', 'user.name', 'Release test')
    git('config', 'user.email', 'release-test@example.invalid')
    git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'base')
    git('tag', 'v0.2.5')
    changed = tmp_path / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text('changed\n', encoding='utf-8')
    git('add', '--', path)
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'follow-up')
    result = subprocess.run(
        ['git', 'diff', '--quiet', 'v0.2.5^{commit}', 'HEAD', '--', '.',
         *NON_RELEASE_PATHS],
        cwd=tmp_path, check=False,
    )
    assert result.returncode == (0 if metadata_only else 1)


def test_public_metadata_exemption_is_guarded() -> None:
    workflow = (REPO_ROOT / '.github/workflows/prepare-release.yml').read_text()
    # The private repository owns a dry run; the public repository owns tagging.
    if not workflow.startswith('name: Prepare Release\n'):
        assert workflow.startswith('name: Release Branch Dry Run\n')
        return
    assert 'git merge-base --is-ancestor "${tag_name}" HEAD &&' in workflow
    assert 'gh release view "${tag_name}"' in workflow
    assert "--json isDraft --jq '.isDraft')\" = false ] &&" in workflow
    assert 'git diff --quiet "${tag_name}^{commit}" HEAD -- .' in workflow
    for path in NON_RELEASE_PATHS:
        assert f"'{path}'" in workflow
    assert 'already identifies different source; review a new version.' in workflow
