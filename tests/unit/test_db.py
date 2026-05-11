"""Tests for database initialization."""

from __future__ import annotations

from pathlib import Path


def test_init_db_creates_all_five_tables(tmp_db: Path) -> None:
    from db import init_db

    conn = init_db(str(tmp_db))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = sorted(row[0] for row in cursor.fetchall())
    conn.close()

    assert "api_tokens" in tables
    assert "hitl_requests" in tables
    assert "sudo_grants" in tables
    assert "tool_hitl_settings" in tables
    assert "users" in tables


def test_init_db_idempotent(tmp_db: Path) -> None:
    from db import init_db

    conn1 = init_db(str(tmp_db))
    conn1.close()
    # Calling again should not raise
    conn2 = init_db(str(tmp_db))
    conn2.close()


def test_init_db_creates_directory(tmp_path: Path) -> None:
    from db import init_db

    nested = tmp_path / "a" / "b" / "c" / "test.db"
    conn = init_db(str(nested))
    conn.close()
    assert nested.exists()
