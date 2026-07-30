"""Shared test fixtures for hive-tools."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _block_outbound_telegram() -> Generator[None, None, None]:
    """Keep HITL approval prompts inside the test run.

    Any request to a HITL-gated endpoint whose mode is ``on`` creates an
    approval request and calls ``notify_daniel``, which POSTs to the real
    Telegram API using the bot token in the environment. A test suite that
    exercises gated endpoints therefore pages a human — once per test, with
    no context explaining why.

    Tests that assert on notification behaviour patch ``httpx.post``
    themselves and are unaffected; this only stops the ones that never meant
    to send anything.
    """
    with patch("hitl.notify_daniel", return_value=None):
        yield


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
