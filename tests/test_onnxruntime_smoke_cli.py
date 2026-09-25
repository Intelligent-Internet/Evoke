from __future__ import annotations

from pathlib import Path

import pytest

from scripts.test_onnxruntime_smoke import (
    CLUSTER_ROLE,
    extension_control_root,
    initdb_command,
    psql_command,
)


def test_extension_control_root_accepts_share_directory(
    tmp_path: Path,
) -> None:
    extension_dir = tmp_path / 'extension'
    extension_dir.mkdir()
    (extension_dir / 'evoke.control').write_text(
        "default_version = '0.2.0'\n",
        encoding='utf-8',
    )

    assert extension_control_root(tmp_path) == tmp_path.resolve()


def test_extension_control_root_accepts_extension_directory(
    tmp_path: Path,
) -> None:
    extension_dir = tmp_path / 'extension'
    extension_dir.mkdir()
    (extension_dir / 'evoke.control').write_text(
        "default_version = '0.2.0'\n",
        encoding='utf-8',
    )

    assert extension_control_root(extension_dir) == tmp_path.resolve()


def test_extension_control_root_rejects_missing_control(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match='evoke.control'):
        extension_control_root(tmp_path)


def test_smoke_uses_one_explicit_cluster_role(tmp_path: Path) -> None:
    initdb = tmp_path / 'initdb'
    psql = tmp_path / 'psql'
    data_dir = tmp_path / 'data'
    socket_dir = tmp_path / 'socket'

    assert CLUSTER_ROLE == 'postgres'
    assert initdb_command(initdb, data_dir)[-2:] == ['-U', CLUSTER_ROLE]
    assert psql_command(psql, socket_dir, 55444)[4:6] == [
        '-U',
        CLUSTER_ROLE,
    ]
