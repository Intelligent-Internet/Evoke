#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIX_SOCKET_PATH_MAX = 103


def extension_control_root(path: Path) -> Path:
    """Return the PostgreSQL share root containing an extension directory."""
    resolved = path.expanduser().resolve()
    if (resolved / 'extension' / 'evoke.control').is_file():
        return resolved
    if (resolved / 'evoke.control').is_file():
        return resolved.parent
    raise FileNotFoundError(
        'evoke.control is missing below extension control path: '
        f'{resolved}'
    )


def ensure_temp_root(path: Path) -> Path:
    """Create and return a usable root for isolated test clusters."""
    root = path.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f'temporary root is not a directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_short_socket_root(prefix: str) -> Path:
    """Allocate a short path suitable for PostgreSQL Unix sockets."""
    root = Path(tempfile.mkdtemp(prefix=prefix, dir='/tmp'))
    socket_path = root / 's' / '.s.PGSQL.65535'
    if len(os.fsencode(socket_path)) > UNIX_SOCKET_PATH_MAX:
        shutil.rmtree(root, ignore_errors=True)
        raise RuntimeError(
            'temporary socket path exceeds the PostgreSQL pathname limit: '
            f'{socket_path}'
        )
    return root


def vacuum_with_session_maintenance_lock(
    connection: psycopg.Connection[Any],
    table_name: str,
    *,
    index_cleanup: bool | None = None,
) -> None:
    """Run VACUUM through the session that owns a maintenance test gate."""
    parts = table_name.split('.')
    if len(parts) not in (1, 2) or not all(parts):
        raise ValueError(f'invalid table name: {table_name}')

    if index_cleanup is None:
        command = sql.SQL('VACUUM {}').format(sql.Identifier(*parts))
    else:
        option = sql.SQL('ON' if index_cleanup else 'OFF')
        command = sql.SQL('VACUUM (INDEX_CLEANUP {}) {}').format(
            option,
            sql.Identifier(*parts),
        )

    prior_autocommit = connection.autocommit
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(command)
    finally:
        connection.autocommit = prior_autocommit
