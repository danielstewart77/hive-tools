"""HITL (Human-in-the-Loop) gate dependency and helpers for HiveTools."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import uuid
from typing import Any

from fastapi import Depends, Request

from auth_api import require_api_token

log = logging.getLogger(__name__)

HITL_EXPIRY_SECONDS = 300  # 5 minutes


class HITLPendingException(Exception):
    """Raised when a tool call requires human approval.

    Use a custom exception instead of HTTPException so the response body
    is returned flat ({"hitl_request_id": ..., "status": "pending"}) rather
    than wrapped under the FastAPI {"detail": ...} envelope.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"HITL approval required: {request_id}")


def _check_hitl(
    db: sqlite3.Connection,
    tool_name: str,
    caller_mind: str,
    params_json: str = "{}",
    tool_method: str | None = None,
    tool_path: str | None = None,
    tool_body: str | None = None,
) -> str:
    """Core HITL logic. Returns caller_mind if allowed, raises HITLPendingException otherwise."""
    row = db.execute(
        "SELECT mode, sudo_timeout_minutes FROM tool_hitl_settings WHERE tool_name = ?",
        (tool_name,),
    ).fetchone()

    if row is None:
        return caller_mind

    mode = row["mode"]

    if mode == "off":
        return caller_mind

    if mode == "sudo":
        now = int(time.time())
        grant = db.execute(
            "SELECT id FROM sudo_grants "
            "WHERE tool_name = ? AND caller_mind = ? AND expires_at > ?",
            (tool_name, caller_mind, now),
        ).fetchone()
        if grant is not None:
            return caller_mind

    # mode == "on" or "sudo" without valid grant — create HITL request
    request_id = str(uuid.uuid4())
    now = int(time.time())
    expires_at = now + HITL_EXPIRY_SECONDS

    db.execute(
        "INSERT INTO hitl_requests "
        "(id, tool_name, caller_mind, params_json, tool_method, tool_path, tool_body, status, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
        (request_id, tool_name, caller_mind, params_json, tool_method, tool_path, tool_body, now, expires_at),
    )
    db.commit()

    try:
        notify_daniel(tool_name, caller_mind, params_json, request_id)
    except Exception:
        log.warning("HITL notification failed for %s", request_id, exc_info=True)

    raise HITLPendingException(request_id)


async def _replay_request_async(method: str, path: str, body: str) -> None:
    """Re-execute the original request as a background task after HITL approval."""
    from config import INTERNAL_SECRET

    try:
        import httpx

        url = f"http://127.0.0.1:9421{path}"
        headers = {
            "Authorization": f"Internal {INTERNAL_SECRET}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            await client.request(method, url, content=body, headers=headers, timeout=30)
        log.info("HITL replay succeeded: %s %s", method, path)
    except Exception:
        log.warning("HITL replay failed: %s %s", method, path, exc_info=True)


def hitl_gate(tool_name: str):
    """Factory that returns a FastAPI dependency for HITL checks."""

    async def _check(
        request: Request,
        caller_mind: str = Depends(require_api_token),
    ) -> str:
        # Internal replay calls bypass the gate entirely
        if caller_mind == "internal":
            return caller_mind

        db: sqlite3.Connection = request.app.state.db
        body_bytes = await request.body()
        return _check_hitl(
            db,
            tool_name,
            caller_mind,
            tool_method=request.method,
            tool_path=str(request.url.path),
            tool_body=body_bytes.decode("utf-8", errors="replace"),
        )

    return _check


def get_hitl_status(db: sqlite3.Connection, request_id: str) -> dict[str, Any] | None:
    """Get the current status of a HITL request, accounting for expiry."""
    row = db.execute(
        "SELECT id, tool_name, caller_mind, status, created_at, expires_at, decided_at "
        "FROM hitl_requests WHERE id = ?",
        (request_id,),
    ).fetchone()

    if row is None:
        return None

    status = row["status"]
    now = int(time.time())

    # If still pending but expired, mark as expired
    if status == "pending" and now > row["expires_at"]:
        status = "expired"

    return {
        "hitl_request_id": row["id"],
        "status": status,
        "tool_name": row["tool_name"],
        "caller_mind": row["caller_mind"],
        "decided_at": row["decided_at"],
    }


def respond_to_hitl(
    db: sqlite3.Connection, request_id: str, action: str
) -> dict[str, Any] | None:
    """Approve or deny a HITL request."""
    row = db.execute(
        "SELECT id, tool_name, caller_mind, status, expires_at, tool_method, tool_path, tool_body "
        "FROM hitl_requests WHERE id = ?",
        (request_id,),
    ).fetchone()

    if row is None:
        return None

    if row["status"] != "pending":
        return {"error": "already_resolved", "current_status": row["status"]}

    now = int(time.time())
    if now > row["expires_at"]:
        return {"error": "already_resolved", "current_status": "expired"}

    new_status = "approved" if action == "approve" else "denied"
    db.execute(
        "UPDATE hitl_requests SET status = ?, decided_at = ? WHERE id = ?",
        (new_status, now, request_id),
    )

    # Replay data returned to caller — fired as async background task by the route handler
    replay: dict[str, str] | None = None
    if new_status == "approved":
        r_method = row["tool_method"]
        r_path = row["tool_path"]
        r_body = row["tool_body"]
        if r_method and r_path and r_body is not None:
            replay = {"method": r_method, "path": r_path, "body": r_body}

    # If approved and tool is in sudo mode, create a sudo grant
    if new_status == "approved":
        setting = db.execute(
            "SELECT mode, sudo_timeout_minutes FROM tool_hitl_settings WHERE tool_name = ?",
            (row["tool_name"],),
        ).fetchone()
        if setting and setting["mode"] == "sudo":
            timeout_minutes = setting["sudo_timeout_minutes"]
            db.execute(
                "INSERT INTO sudo_grants (tool_name, caller_mind, granted_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    row["tool_name"],
                    row["caller_mind"],
                    now,
                    now + timeout_minutes * 60,
                ),
            )

    db.commit()

    result = {
        "hitl_request_id": request_id,
        "status": new_status,
        "tool_name": row["tool_name"],
        "caller_mind": row["caller_mind"],
        "decided_at": now,
    }
    if replay:
        result["_replay"] = replay
    return result


def notify_daniel(
    tool_name: str,
    caller_mind: str,
    params_json: str,
    request_id: str,
) -> None:
    """Send a Telegram notification with inline keyboard buttons."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_OWNER_CHAT_ID")

    if not bot_token or not chat_id:
        log.warning(
            "HITL notification skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_OWNER_CHAT_ID not set"
        )
        return

    try:
        import httpx

        _TOOL_ICONS = {
            "gmail": "📧",
            "calendar": "📅",
            "docker": "🐳",
            "linkedin": "💼",
        }
        prefix = tool_name.split("_")[0]
        icon = _TOOL_ICONS.get(prefix, "🔧")
        label = tool_name.replace(f"{prefix}_", "").replace("_", " ").title()

        text = f"{icon} *{label}*\nRequested by: {caller_mind}"

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": [
                    [{"text": "✅ Approve", "callback_data": f"hitl_approve_{request_id}"}],
                    [{"text": "❌ Deny",    "callback_data": f"hitl_deny_{request_id}"}],
                ]
            },
        }

        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json=payload,
            timeout=10,
        )
    except Exception:
        log.warning("Failed to send HITL Telegram notification", exc_info=True)
