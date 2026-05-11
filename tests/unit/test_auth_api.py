"""Tests for bearer token auth functions."""

from __future__ import annotations

import hashlib
import sqlite3
import time


def test_generate_token_returns_raw_and_stores_hash(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token

    raw = generate_token(db_conn, "ada")
    assert isinstance(raw, str)
    assert len(raw) > 20

    row = db_conn.execute("SELECT token_hash FROM api_tokens WHERE name = ?", ("ada",)).fetchone()
    assert row is not None
    assert row["token_hash"] == hashlib.sha256(raw.encode()).hexdigest()


def test_generate_token_hash_is_sha256(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token

    raw = generate_token(db_conn, "bob")
    expected_hash = hashlib.sha256(raw.encode()).hexdigest()

    row = db_conn.execute("SELECT token_hash FROM api_tokens WHERE name = ?", ("bob",)).fetchone()
    assert row["token_hash"] == expected_hash


def test_validate_token_returns_mind_name(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token, validate_token

    raw = generate_token(db_conn, "ada")
    result = validate_token(db_conn, raw)
    assert result == "ada"


def test_validate_token_invalid_returns_none(db_conn: sqlite3.Connection) -> None:
    from auth_api import validate_token

    result = validate_token(db_conn, "not-a-real-token")
    assert result is None


def test_validate_token_revoked_returns_none(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token, revoke_token, validate_token

    raw = generate_token(db_conn, "ada")
    row = db_conn.execute("SELECT id FROM api_tokens WHERE name = ?", ("ada",)).fetchone()
    revoke_token(db_conn, row["id"])

    result = validate_token(db_conn, raw)
    assert result is None


def test_validate_token_updates_last_used_at(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token, validate_token

    raw = generate_token(db_conn, "ada")

    row_before = db_conn.execute(
        "SELECT last_used_at FROM api_tokens WHERE name = ?", ("ada",)
    ).fetchone()
    assert row_before["last_used_at"] is None

    validate_token(db_conn, raw)

    row_after = db_conn.execute(
        "SELECT last_used_at FROM api_tokens WHERE name = ?", ("ada",)
    ).fetchone()
    assert row_after["last_used_at"] is not None
    assert row_after["last_used_at"] >= int(time.time()) - 5


def test_revoke_token_sets_revoked_at(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token, revoke_token

    generate_token(db_conn, "ada")
    row = db_conn.execute("SELECT id FROM api_tokens WHERE name = ?", ("ada",)).fetchone()

    result = revoke_token(db_conn, row["id"])
    assert result is True

    row_after = db_conn.execute(
        "SELECT revoked_at FROM api_tokens WHERE id = ?", (row["id"],)
    ).fetchone()
    assert row_after["revoked_at"] is not None


def test_list_tokens_returns_all(db_conn: sqlite3.Connection) -> None:
    from auth_api import generate_token, list_tokens, revoke_token

    generate_token(db_conn, "ada")
    generate_token(db_conn, "bob")

    # Revoke one
    row = db_conn.execute("SELECT id FROM api_tokens WHERE name = ?", ("bob",)).fetchone()
    revoke_token(db_conn, row["id"])

    tokens = list_tokens(db_conn)
    assert len(tokens) == 2
    names = {t["name"] for t in tokens}
    assert names == {"ada", "bob"}
