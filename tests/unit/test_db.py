"""Tests for database initialization."""

from __future__ import annotations

import sqlite3
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


async def test_many_concurrent_writes_on_the_sqlite_half_all_land(
    db_conn: sqlite3.Connection,
) -> None:
    """The revert lever has to be at least as sound as the code it replaced.

    There is one `sqlite3.Connection` and a `sqlite3.Connection` is not safe
    to use from two threads at once whatever `check_same_thread` says. Run on
    the default thread pool, two statements interleave on it and return each
    other's rows: measured through the live routes, fifty concurrent writes
    produced fourteen errors - `bad parameter or other API misuse`, `cannot
    start a transaction within a transaction`, `another row available` - and
    three valid bearer tokens rejected with a 401, because the token read
    came back with another request's row. The synchronous code this replaced
    was serialised on the event loop by construction.
    """
    import asyncio

    import store

    at = 1_800_000_000
    count = 60
    results = await asyncio.gather(*[
        store.create_request(
            f"concurrent-{index}", "gmail_send_email", "skippy", "{}",
            at + index, at + 300, "POST", "/gmail/send", "{}")
        for index in range(count)
    ], return_exceptions=True)

    raised = [r for r in results if isinstance(r, BaseException)]
    assert raised == []
    assert db_conn.execute(
        "SELECT count(*) AS c FROM hitl_requests "
        "WHERE id LIKE 'concurrent-%'").fetchone()["c"] == count


async def test_many_concurrent_reads_answer_about_their_own_row(
    db_conn: sqlite3.Connection,
) -> None:
    """The quiet half of the same fault. An interleaved read does not only
    raise - it comes back with another statement's row, which on the token
    path is a valid credential refused and on the login path is one person's
    session resolved to another's account."""
    import asyncio

    import store

    at = 1_800_000_000
    for index in range(40):
        await store.create_token(f"token-{index}", f"digest-{index}", at)

    found = await asyncio.gather(*[
        store.live_token(f"digest-{index}") for index in range(40)
    ])

    assert [row["name"] if row else None for row in found] == [
        f"token-{index}" for index in range(40)]
