from pathlib import Path

import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_builders_reject_dirty_trees_by_default() -> None:
    for relative_path in (
        'scripts/build_release_zip.sh',
        'scripts/build_release_docker_image.sh',
    ):
        text = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
        assert '--allow-dirty' in text
        assert 'git_tree_state' in text
        assert (
            'refusing to build a release from a dirty Git worktree'
            in text
        )


def test_pgxs_bitcode_does_not_dirty_release_worktrees() -> None:
    gitignore = (REPO_ROOT / '.gitignore').read_text(encoding='utf-8')

    assert '/src/*.bc' in gitignore.splitlines()


def test_release_zip_carries_project_and_runtime_licenses() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')

    assert (REPO_ROOT / 'LICENSE').is_file()
    assert 'EVOKE-LICENSE' in script
    assert 'ONNXRUNTIME-LICENSE' in script
    assert 'MODEL-LICENSE' in script
    assert 'MILESTONE-MODEL-NOTICE' in script
    assert '${runtime_prefix}/share/licenses/onnxruntime/LICENSE' in script
    assert '${runtime_prefix}/LICENSE' in script
    assert '${runtime_prefix}/LICENSE.txt' in script


def test_macos_release_loads_its_bundled_runtime() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')

    assert 'install_name_tool -change' in script
    assert 'packaged_dependency="@loader_path/' in script
    assert 'does not load its bundled ONNX Runtime' in script


def test_release_builders_require_locked_milestone_model() -> None:
    zip_script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')
    docker_script = (
        REPO_ROOT / 'scripts/build_release_docker_image.sh'
    ).read_text(encoding='utf-8')
    dockerfile = (
        REPO_ROOT / 'packaging/docker/postgres18/Dockerfile'
    ).read_text(encoding='utf-8')

    for script in (zip_script, docker_script):
        assert '--model-checkout' in script
        assert 'II42_MILESTONE_MODEL_CHECKOUT' in script
        assert 'validate_milestone_model_checkout.py' in script
    assert 'ii42/models/default' in zip_script
    assert 'ii42_milestone_model' in docker_script
    assert 'ii42/models/default' in dockerfile


def test_release_zip_checksum_is_portable() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')

    assert 'zip_basename="$(basename "$zip_path")"' in script
    assert 'cd "$dist_dir"' in script
    assert 'sha256sum "$zip_basename"' in script
    assert 'shasum -a 256 "$zip_basename"' in script
    assert 'sha256sum "$zip_path"' not in script
    assert 'shasum -a 256 "$zip_path"' not in script
    assert 'zip -T "$zip_path"' in script


def test_docker_build_contexts_include_the_ort_version_lock() -> None:
    for relative in (
        'packaging/docker/postgres18/Dockerfile',
        'packaging/docker/runtime-gpu-aarch64/Dockerfile',
    ):
        dockerfile = (REPO_ROOT / relative).read_text(encoding='utf-8')
        copy = 'COPY packaging/onnxruntime.version ./packaging/onnxruntime.version'
        assert copy in dockerfile
        first_build = dockerfile.index('make ', dockerfile.index('WORKDIR /src'))
        assert dockerfile.index(copy) < first_build


def test_release_workflow_checksums_resolve_from_dist(tmp_path: Path) -> None:
    import hashlib

    dist = tmp_path / 'dist'
    dist.mkdir()
    archive = dist / 'ii42-test.zip'
    archive.write_bytes(b'checksum regression fixture')
    checksum = archive.with_suffix('.zip.sha256')
    checksum.write_text(
        f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n',
        encoding='utf-8',
    )
    workflow = (REPO_ROOT / '.github/workflows/release.yml').read_text(
        encoding='utf-8'
    )
    expected = 'sha256sum -c "$(basename "${checksum}")"'
    assert expected in workflow
    # Run the same shell fragment with a basename-only release checksum.
    subprocess.run(
        ['bash', '-euc', f'checksum=dist/{checksum.name}\ncd dist\n{expected}'],
        cwd=tmp_path,
        check=True,
    )


def test_release_zip_preflights_archive_tools_before_compilation() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')
    first_make = script.index('make -C "$repo_root" clean')

    assert script.index('command -v zip') < first_make
    assert script.index('command -v sha256sum') < first_make
    assert script.index('command -v shasum') < first_make


def test_release_zip_owns_the_pinned_onnxruntime_sdk() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')

    assert '--onnxruntime-prefix' in script
    assert 'II42_ONNXRUNTIME_PREFIX' in script
    assert 'packaging/onnxruntime.version' in script
    assert 'scripts/install_onnxruntime_c.sh' in script
    assert 'onnxruntime_pkg_config_path' in script
    assert 'export PKG_CONFIG_PATH=' in script
    assert 'ONNX Runtime SDK prefix: ${onnxruntime_prefix}' in script


def test_release_zip_records_postgres_installation_identity() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_zip.sh'
    ).read_text(encoding='utf-8')

    for field in (
        'Operating system: ${os_name}',
        'Architecture: ${arch}',
        'PostgreSQL version: ${pg_version_text}',
        'PostgreSQL pkglibdir: ${pkglibdir}',
        'PostgreSQL sharedir: ${pg_sharedir}',
    ):
        assert field in script


def test_docker_release_records_git_provenance() -> None:
    script = (
        REPO_ROOT / 'scripts/build_release_docker_image.sh'
    ).read_text(encoding='utf-8')
    dockerfile = (
        REPO_ROOT / 'packaging/docker/postgres18/Dockerfile'
    ).read_text(encoding='utf-8')

    assert 'II42_GIT_COMMIT=${git_commit}' in script
    assert 'II42_GIT_TREE_STATE=${git_tree_state}' in script
    assert (
        'org.opencontainers.image.revision="${II42_GIT_COMMIT}"'
        in dockerfile
    )
    assert (
        'io.evoke.git-tree-state="${II42_GIT_TREE_STATE}"'
        in dockerfile
    )


def test_ci_runs_complete_source_syntax_and_python_tests() -> None:
    workflow = (
        REPO_ROOT / '.github/workflows/ci.yml'
    ).read_text(encoding='utf-8')

    assert 'python3 -m compileall -q scripts tests' in workflow
    assert "-type f -name '*.sh' -print0" in workflow
    assert 'xargs -0 -n1 bash -n' in workflow
    assert 'python3 -m pytest -q $(' in workflow
    assert "! -name 'test_ng*.py'" in workflow
    assert "! -name 'test_prepare_ng*.py'" in workflow
    assert "! -name 'test_review_ng*.py'" in workflow
    assert 'workflow_dispatch:' in workflow


def test_ci_installs_declared_python_test_dependencies() -> None:
    installer = (
        REPO_ROOT / 'scripts/install_postgres_apt.sh'
    ).read_text(encoding='utf-8')

    for package in (
        'python3-numpy',
        'python3-pytest',
        'python3-scipy',
    ):
        assert package in installer


def test_ci_preserves_onnxruntime_discovery_during_install() -> None:
    workflow = (
        REPO_ROOT / '.github/workflows/ci.yml'
    ).read_text(encoding='utf-8')

    assert 'sudo --preserve-env=PKG_CONFIG_PATH make install' in workflow
    assert workflow.count('II42_ENABLE_ONNXRUNTIME=1') >= 3


def test_docker_smokes_wait_for_the_final_tcp_server() -> None:
    for name in ('ci.yml', 'prepare-release.yml', 'release.yml'):
        workflow = (REPO_ROOT / '.github/workflows' / name).read_text()
        readiness_checks = [
            line.strip() for line in workflow.splitlines()
            if 'pg_isready ' in line
        ]
        if name != 'prepare-release.yml':
            assert readiness_checks
        for check in readiness_checks:
            assert 'pg_isready -h 127.0.0.1 -U postgres' in check
