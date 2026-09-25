from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def public_promotion() -> str | None:
    path = ROOT / '.github/workflows/promote-release.yml'
    if not path.exists():
        release = (ROOT / '.github/workflows/release.yml').read_text()
        assert release.startswith('name: Release Dry Run\n')
        return None
    return path.read_text()


def test_promotion_rejects_incomplete_or_invalid_release_assets() -> None:
    workflow = public_promotion()
    if workflow is None:
        return
    match = re.search(
        r"jq -e --arg tag .*?'\n(.*?)\n\s*' release.json",
        workflow, re.DOTALL,
    )
    assert match is not None
    predicate = match.group(1)
    tag = 'v0.2.5'
    archives = [
        f'evoke-{tag}-linux-x86_64-pg17.zip',
        f'evoke-{tag}-linux-x86_64-pg18.zip',
        f'evoke-{tag}-docker-pg18-linux-x86_64.tar.gz',
    ]
    release = {
        'draft': False,
        'tag_name': tag,
        'assets': [
            {'name': name, 'state': 'uploaded', 'size': 100,
             'digest': 'sha256:' + 'a' * 64}
            for archive in archives
            for name in (archive, archive + '.sha256')
        ],
    }

    def valid(value: dict) -> bool:
        result = subprocess.run(
            ['jq', '-e', '--arg', 'tag', tag, predicate],
            input=json.dumps(value), text=True, capture_output=True,
            check=False,
        )
        return result.returncode == 0

    assert valid(release)
    for field, value in (('draft', True), ('tag_name', 'v0.4.14')):
        bad = copy.deepcopy(release)
        bad[field] = value
        assert not valid(bad)
    for field, value in (
        ('name', 'unexpected.zip'), ('state', 'starter'),
        ('size', 0), ('digest', None), ('digest', 'sha256:bad'),
    ):
        bad = copy.deepcopy(release)
        bad['assets'][0][field] = value
        assert not valid(bad)
    bad = copy.deepcopy(release)
    bad['assets'].pop()
    assert not valid(bad)
    bad = copy.deepcopy(release)
    bad['assets'].append(bad['assets'][0])
    assert not valid(bad)


def test_public_promotion_only_moves_aliases_after_identity_checks() -> None:
    workflow = public_promotion()
    if workflow is None:
        return
    release = (ROOT / '.github/workflows/release.yml').read_text()
    assert 'needs: [publish-release, build-docker-release-image]' in release
    assert 'uses: ./.github/workflows/promote-release.yml' in release
    assert '--generate-notes --verify-tag --latest=false' in release
    assert '--prerelease' not in release
    assert '--tag pg18' not in release
    assert 'image_digest: ${{ steps.publish.outputs.image_digest }}' in release
    assert 'group: promote-public-release' in workflow
    assert 'max_by(.published_at).tag_name' in workflow
    assert 'test "${newest}" = "${TAG_NAME}"' in workflow
    assert 'test "${digest}" = "${EXPECTED_DIGEST}"' in workflow
    assert 'org.opencontainers.image.revision' in workflow
    assert 'org.opencontainers.image.source' in workflow
    assert 'io.evoke.git-tree-state' in workflow
    assert '--prefer-index=false' in workflow
    assert 'for tag in "pg18-${TAG_NAME}" pg18 latest;' in workflow
    assert '--prerelease=false --latest' in workflow
    assert '.assets | map({id,name,digest,size}) | sort_by(.id)' in workflow
    assert workflow.index('test "${digest}"') < workflow.index(
        'docker buildx imagetools create',
    )
    assert workflow.index('test "${actual}"') < workflow.index('gh release edit')
    for forbidden in ('docker build ', 'gh release upload', 'git push',
                      'build_release_zip.sh', 'build_release_docker_image.sh'):
        assert forbidden not in workflow
