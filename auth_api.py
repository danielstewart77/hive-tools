"""Bearer token authentication for the API surface."""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

import store
from fastapi import HTTPException, Request


#: The holder of the one credential that may decide an approval over the API.
#: Not a mind: the surface bots hold it, and send it when the operator taps
#: Approve or Deny. Minted like any other token, under this name, and it is
#: the name that gives it the power - so it does nothing else (see
#: `require_api_token`), and a mind's token, which may make gated calls,
#: never decides one.
APPROVER_HOLDER = "hitl-approver"

#: What a mind is told when it tries to decide a gate.
#: The sentence that matters first: the surface bot shows the first 120
#: characters of a refusal, and "still pending" is what the operator needs.
MIND_CANNOT_DECIDE = (
    "A mind cannot decide its own gate; the request is still pending. "
    "The token sent is {holder}'s, and a mind's token may make gated calls "
    "but never approve or deny one. Decide on the approval page, or from a "
    "bot holding the approver credential.")


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


async def generate_approver_token() -> str:
    """Mint the approver credential: a token held by `APPROVER_HOLDER`."""
    return await generate_token(APPROVER_HOLDER)


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

    The approver credential is refused here with 403. It exists to decide
    approvals and for nothing else: a credential that could also make gated
    calls could ask for something and then approve its own request.
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
    if name == APPROVER_HOLDER:
        raise HTTPException(
            status_code=403,
            detail="The approver credential decides approvals "
                   "(POST /hitl/{id}/respond) and does nothing else.")

    return name


async def require_approver(request: Request) -> str:
    """FastAPI dependency for deciding an approval over the API.

    Only the approver credential passes. A mind's token - any mind's, the
    operator mind's included - is refused with 403 before anything is
    decided, so the request stays pending; so is hive-tools' own internal
    replay secret, which replays a decision and never makes one. The other
    way to decide is Daniel's sign-in on the approval page, which does not
    come through here.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        if auth_header.startswith("Internal "):
            raise HTTPException(
                status_code=403,
                detail="hive-tools' internal secret replays a decision and "
                       "never makes one; the request is still pending.")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    name = await validate_token(auth_header[7:])
    if name is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked token")
    if name != APPROVER_HOLDER:
        raise HTTPException(status_code=403,
                            detail=MIND_CANNOT_DECIDE.format(holder=name))
    return name
