"""Bearer token authentication for the API surface."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from typing import Any

from fastapi import HTTPException, Request


def generate_token(db: sqlite3.Connection, name: str) -> str:
    """Generate a new API token, store its SHA-256 hash, and return the raw token."""
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    now = int(time.time())
    db.execute(
        "INSERT INTO api_tokens (name, token_hash, created_at) VALUES (?, ?, ?)",
        (name, token_hash, now),
    )
    db.commit()
    return raw


def validate_token(db: sqlite3.Connection, raw_token: str) -> str | None:
    """Validate a raw token and return the mind name, or None if invalid/revoked."""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    row = db.execute(
        "SELECT id, name FROM api_tokens WHERE token_hash = ? AND revoked_at IS NULL",
        (token_hash,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())
    db.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (now, row["id"]))
    db.commit()
    return row["name"]


def revoke_token(db: sqlite3.Connection, token_id: int) -> bool:
    """Revoke a token by setting its revoked_at timestamp."""
    now = int(time.time())
    db.execute("UPDATE api_tokens SET revoked_at = ? WHERE id = ?", (now, token_id))
    db.commit()
    return True


def list_tokens(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all tokens (active and revoked)."""
    rows = db.execute(
        "SELECT id, name, created_at, last_used_at, revoked_at FROM api_tokens ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


async def require_api_token(request: Request) -> str:
    """FastAPI dependency: validate bearer token from Authorization header.

    Returns the caller mind name on success, raises 401 on failure.
    Accepts 'Internal <INTERNAL_SECRET>' for HITL replay calls.
    """
    from config import INTERNAL_SECRET

    auth_header = request.headers.get("Authorization", "")

    # Internal HITL replay bypass
    if auth_header == f"Internal {INTERNAL_SECRET}":
        return "internal"

    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    raw_token = auth_header[7:]
    db: sqlite3.Connection = request.app.state.db
    name = validate_token(db, raw_token)
    if name is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked token")

    return name
