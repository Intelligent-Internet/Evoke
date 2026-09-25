from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import bump_extension_version


def test_install_sql_rename_requires_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        ['bump_extension_version.py', '--version', '0.2.1'],
    )

    with pytest.raises(RuntimeError, match='--rename-current-sql'):
        bump_extension_version.main()


def test_bump_renames_the_only_install_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control = tmp_path / 'evoke.control'
    sql_dir = tmp_path / 'sql'
    sql_dir.mkdir()
    control.write_text(
        "default_version = '0.2.0'\n",
        encoding='utf-8',
    )
    current_sql = sql_dir / 'evoke--0.2.0.sql'
    current_sql.write_text('-- current install\n', encoding='utf-8')
    monkeypatch.setattr(bump_extension_version, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'bump_extension_version.py',
            '--version',
            '0.2.1',
            '--rename-current-sql',
        ],
    )

    assert bump_extension_version.main() == 0

    assert not current_sql.exists()
    assert (sql_dir / 'evoke--0.2.1.sql').read_text(
        encoding='utf-8',
    ) == '-- current install\n'
    assert sorted(path.name for path in sql_dir.glob('evoke--*.sql')) == [
        'evoke--0.2.1.sql',
    ]
    assert "default_version = '0.2.1'" in control.read_text(
        encoding='utf-8',
    )


def test_bump_rejects_retired_upgrade_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control = tmp_path / 'evoke.control'
    sql_dir = tmp_path / 'sql'
    sql_dir.mkdir()
    control.write_text("default_version = '0.2.0'\n", encoding='utf-8')
    (sql_dir / 'evoke--0.2.0.sql').write_text(
        '-- current install\n',
        encoding='utf-8',
    )
    (sql_dir / 'evoke--0.1.9--0.2.0.sql').write_text(
        '-- retired upgrade\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(bump_extension_version, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'bump_extension_version.py',
            '--version',
            '0.2.1',
            '--rename-current-sql',
        ],
    )

    with pytest.raises(RuntimeError, match='exactly one current install SQL'):
        bump_extension_version.main()


def test_bump_rejects_multiple_install_sql_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control = tmp_path / 'evoke.control'
    sql_dir = tmp_path / 'sql'
    sql_dir.mkdir()
    control.write_text("default_version = '0.2.0'\n", encoding='utf-8')
    (sql_dir / 'evoke--0.2.0.sql').write_text(
        '-- current install\n',
        encoding='utf-8',
    )
    (sql_dir / 'evoke--0.1.9.sql').write_text(
        '-- stale install\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(bump_extension_version, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'bump_extension_version.py',
            '--version',
            '0.2.1',
            '--rename-current-sql',
        ],
    )

    with pytest.raises(RuntimeError, match='exactly one current install SQL'):
        bump_extension_version.main()
