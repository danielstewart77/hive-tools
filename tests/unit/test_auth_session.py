"""Tests for session-based authentication (management UI)."""

from __future__ import annotations

import sqlite3

import pytest


def test_hash_password_returns_prefixed_string() -> None:
    from auth_session import hash_password

    result = hash_password("test123")
    assert result.startswith("bcrypt$") or result.startswith("pbkdf2$")


def test_hash_password_empty_raises() -> None:
    from auth_session import hash_password

    with pytest.raises(ValueError):
        hash_password("")


def test_verify_password_correct() -> None:
    from auth_session import hash_password, verify_password

    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed) is True


def test_verify_password_wrong() -> None:
    from auth_session import hash_password, verify_password

    hashed = hash_password("mypassword")
    assert verify_password("wrong", hashed) is False


async def test_create_user_and_get_by_username(db_conn: sqlite3.Connection) -> None:
    from auth_session import create_user, get_user_by_username

    user = await create_user("daniel", "pass123")
    assert user["username"] == "daniel"

    fetched = await get_user_by_username("daniel")
    assert fetched is not None
    assert fetched["username"] == "daniel"


async def test_create_user_duplicate_raises(db_conn: sqlite3.Connection) -> None:
    from auth_session import create_user

    await create_user("daniel", "pass123")
    with pytest.raises(sqlite3.IntegrityError):
        await create_user("daniel", "pass456")


def test_create_session_token_and_read_back() -> None:
    from auth_session import create_session_token, read_session_token

    user = {"id": 1, "username": "daniel"}
    token = create_session_token(user)
    payload = read_session_token(token)
    assert payload is not None
    assert payload["user_id"] == 1


def test_read_session_token_invalid_returns_none() -> None:
    from auth_session import read_session_token

    result = read_session_token("garbage.token.value")
    assert result is None


async def test_verify_user_credentials_valid(db_conn: sqlite3.Connection) -> None:
    from auth_session import create_user, verify_user_credentials

    await create_user("daniel", "pass123")
    user = await verify_user_credentials("daniel", "pass123")
    assert user is not None
    assert user["username"] == "daniel"


async def test_verify_user_credentials_disabled_returns_none(
    db_conn: sqlite3.Connection,
) -> None:
    import time

    from auth_session import create_user, verify_user_credentials

    await create_user("daniel", "pass123")
    db_conn.execute(
        "UPDATE users SET disabled_at = ? WHERE username = ?",
        (int(time.time()), "daniel"),
    )
    db_conn.commit()

    result = await verify_user_credentials("daniel", "pass123")
    assert result is None


def test_session_cookie_name_is_ht_session() -> None:
    from auth_session import SESSION_COOKIE_NAME

    assert SESSION_COOKIE_NAME == "ht_session"
