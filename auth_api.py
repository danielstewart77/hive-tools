"""Bearer token authentication for the API surface."""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

import store
from fastapi import HTTPException, Request


async def generate_token(name: str) -> str:
    """Generate a new API token, store its SHA-256 hash, and return the raw token."""
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    await store.create_token(name, token_hash, int(time.time()))
    return raw


async def validate_token(raw_token: str) -> str | None:
    """Validate a raw token and return the mind name, or None if invalid/revoked."""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    row = await store.live_token(token_hash)
    if row is None:
        return None

    await store.record_token_use(int(row["id"]), int(time.time()))
    return str(row["name"])


async def revoke_token(token_id: int) -> bool:
    """Revoke a token, reporting whether this call was the one that did it.

    A token already revoked answers False rather than True: the count is the
    arbitration, and a caller told it succeeded twice cannot tell a
    revocation it performed from one it merely repeated.
    """
    return bool(await store.revoke_token(token_id, int(time.time())))


async def list_tokens() -> list[dict[str, Any]]:
    """Return all tokens (active and revoked), never their digests."""
    return await store.tokens()


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
    name = await validate_token(raw_token)
    if name is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked token")

    return name
