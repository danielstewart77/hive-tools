"""Shared test fixtures for hive-tools."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary database path (file does not exist yet)."""
    return tmp_path / "test.db"


@pytest.fixture()
def db_conn(tmp_db: Path) -> Generator[sqlite3.Connection, None, None]:
    """Return an initialized database connection."""
    from db import init_db

    conn = init_db(str(tmp_db))
    yield conn
    conn.close()
