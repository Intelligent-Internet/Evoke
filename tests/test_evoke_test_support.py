import os
import shutil
from pathlib import Path

import pytest

from scripts.evoke_test_support import (
    UNIX_SOCKET_PATH_MAX,
    create_short_socket_root,
    ensure_temp_root,
)


def test_ensure_temp_root_creates_missing_nested_directory(
    tmp_path: Path,
) -> None:
    root = ensure_temp_root(tmp_path / 'nested' / 'clusters')

    assert root == (tmp_path / 'nested' / 'clusters').resolve()
    assert root.is_dir()


def test_ensure_temp_root_rejects_regular_file(tmp_path: Path) -> None:
    path = tmp_path / 'not-a-directory'
    path.write_text('not a directory', encoding='utf-8')

    with pytest.raises(NotADirectoryError, match='temporary root'):
        ensure_temp_root(path)


def test_create_short_socket_root_fits_postgresql_limit() -> None:
    root = create_short_socket_root('evoke_test_socket_')
    try:
        socket_path = root / 's' / '.s.PGSQL.65535'
        assert len(os.fsencode(socket_path)) <= UNIX_SOCKET_PATH_MAX
    finally:
        shutil.rmtree(root, ignore_errors=True)
