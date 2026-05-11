"""Session-based authentication for the HiveTools management UI.

Adapted from Spark to Bloom's auth.py, with cookie name and salt
changed for HiveTools.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from typing import Any

from fastapi import HTTPException, Request

from config import SECRET_KEY

try:
    import bcrypt as _bcrypt
except ImportError:  # pragma: no cover
    _bcrypt = None  # type: ignore[assignment]

try:
    from itsdangerous import BadSignature, BadTimeSignature, URLSafeTimedSerializer
except ImportError:  # pragma: no cover
    BadSignature = BadTimeSignature = Exception  # type: ignore[assignment,misc]
    URLSafeTimedSerializer = None  # type: ignore[assignment,misc]


SESSION_COOKIE_NAME = "ht_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _secret_key() -> str:
    import os

    return os.getenv("HT_SECRET_KEY", SECRET_KEY)


def hash_password(password: str) -> str:
    """Hash a password with bcrypt when available, otherwise PBKDF2."""
    if not password:
        raise ValueError("password must not be empty")

    password_bytes = password.encode("utf-8")
    if _bcrypt is not None:
        hashed = _bcrypt.hashpw(password_bytes, _bcrypt.gensalt())
        return f"bcrypt${hashed.decode('utf-8')}"

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, 200_000)
    return "pbkdf2$200000$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored hash."""
    if not password_hash:
        return False

    password_bytes = password.encode("utf-8")
    if password_hash.startswith("bcrypt$") and _bcrypt is not None:
        stored = password_hash.split("$", 1)[1].encode("utf-8")
        return _bcrypt.checkpw(password_bytes, stored)

    if password_hash.startswith("pbkdf2$"):
        _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, int(rounds))
        return hmac.compare_digest(actual, expected)

    return False


def create_user(
    db: sqlite3.Connection,
    username: str,
    password: str,
    *,
    is_admin: bool = False,
) -> dict[str, Any]:
    """Create a console user."""
    normalized = username.strip().lower()
    if not normalized:
        raise ValueError("username must not be empty")

    password_hash = hash_password(password)
    now = int(time.time())
    db.execute(
        "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
        (normalized, password_hash, int(is_admin), now),
    )
    db.commit()
    return get_user_by_username(db, normalized)  # type: ignore[return-value]


def get_user_by_username(
    db: sqlite3.Connection, username: str
) -> dict[str, Any] | None:
    normalized = username.strip().lower()
    row = db.execute(
        "SELECT id, username, password_hash, is_admin, created_at, disabled_at "
        "FROM users WHERE username = ?",
        (normalized,),
    ).fetchone()
    return dict(row) if row else None


def get_user_by_id(db: sqlite3.Connection, user_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT id, username, password_hash, is_admin, created_at, disabled_at "
        "FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def verify_user_credentials(
    db: sqlite3.Connection, username: str, password: str
) -> dict[str, Any] | None:
    user = get_user_by_username(db, username)
    if not user or user.get("disabled_at"):
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def _make_serializer() -> URLSafeTimedSerializer | None:  # type: ignore[return]
    if URLSafeTimedSerializer is None:
        return None
    return URLSafeTimedSerializer(_secret_key(), salt="hive-tools-session")


def create_session_token(user: dict[str, Any]) -> str:
    payload = {"user_id": user["id"], "username": user["username"]}
    serializer = _make_serializer()
    if serializer is not None:
        return serializer.dumps(payload)

    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_b64 = base64.urlsafe_b64encode(body).decode("ascii")
    signature = hmac.new(
        _secret_key().encode("utf-8"),
        body_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{body_b64}.{signature}"


def read_session_token(token: str) -> dict[str, Any] | None:
    serializer = _make_serializer()
    if serializer is not None:
        try:
            return serializer.loads(token, max_age=SESSION_MAX_AGE)
        except (BadSignature, BadTimeSignature):
            return None

    try:
        body_b64, signature = token.split(".", 1)
    except ValueError:
        return None

    expected = hmac.new(
        _secret_key().encode("utf-8"),
        body_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        body = base64.urlsafe_b64decode(body_b64.encode("ascii"))
        return json.loads(body.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


def get_current_user_from_request(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    session = read_session_token(token)
    if not session:
        return None
    db: sqlite3.Connection = request.app.state.db
    user = get_user_by_id(db, int(session["user_id"]))
    if not user or user.get("disabled_at"):
        return None
    return user


def require_auth(request: Request) -> dict[str, Any]:
    user = get_current_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def set_session_cookie(response: Any, user: dict[str, Any]) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        create_session_token(user),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
    )


def clear_session_cookie(response: Any) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME)
