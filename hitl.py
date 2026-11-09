"""HITL (Human-in-the-Loop) gate dependency and helpers for HiveTools."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from fastapi import Depends, Request

import store
from auth_api import require_api_token

log = logging.getLogger(__name__)

HITL_EXPIRY_SECONDS = 300  # 5 minutes

# Auto-registration: every hitl_gate("tool_name") call records the name here
# at import time. sync_tool_registry() upserts these into tool_hitl_settings
# at startup so the control panel automatically lists every gated tool with
# a sane default mode. No manual seed step required for new tools.
_REGISTERED_TOOLS: set[str] = set()

AUTO_REGISTER_DEFAULT_MODE = "on"
AUTO_REGISTER_DEFAULT_SUDO_TIMEOUT_MINUTES = 15

#: The only two things a person can do to a pending request. Declared, because
#: the decision used to be `"approved" if action == "approve" else "denied"` -
#: so `{"action": "approved"}`, the natural past-tense mistake, and `"yes"`,
#: and an empty string, all denied the request irrevocably and answered 200
#: with `denied`. A caller told "denied" has no reason to think it meant
#: approve.
ACTIONS = {"approve": "approved", "deny": "denied"}


class UnknownAction(ValueError):
    """Raised for a decision that is neither approve nor deny."""


class UncapturableCall(ValueError):
    """Raised when a gated call cannot be stored as the call it was.

    A NUL byte is the case that exists. `errors="replace"` fixes invalid
    UTF-8 and leaves `\x00` alone, because a NUL *is* valid UTF-8; SQLite
    stored it and Postgres refuses it outright, so the same call was queued
    on one store and 500ed on the other.

    Refused at the route rather than escaped on the way in. An escaped body
    is a different body, and the whole purpose of capturing it is that the
    replay re-makes the call the person approved - so a gate that quietly
    rewrote it would show Daniel one call and run another. A caller whose
    body cannot be stored faithfully is told so, now, instead of being
    queued against a replay that cannot be honoured.
    """


def _capturable(text: str | None, what: str) -> str | None:
    """The captured text, or a refusal naming what cannot be stored."""
    if text is not None and "\x00" in text:
        raise UncapturableCall(
            f"this call's {what} holds a NUL byte, which the hive's database "
            "cannot store in a text column. The gate captures the call so an "
            "approval can replay it exactly, and a body stored with the NUL "
            "removed would replay a different call - so the call is refused "
            "here rather than queued against a replay that cannot be "
            "honoured.")
    return text


class HITLPendingException(Exception):
    """Raised when a tool call requires human approval.

    Use a custom exception instead of HTTPException so the response body
    is returned flat ({"hitl_request_id": ..., "status": "pending"}) rather
    than wrapped under the FastAPI {"detail": ...} envelope.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"HITL approval required: {request_id}")


async def _check_hitl(
    tool_name: str,
    caller_mind: str,
    params_json: str = "{}",
    tool_method: str | None = None,
    tool_path: str | None = None,
    tool_body: str | None = None,
) -> str:
    """Core HITL logic. Returns caller_mind if allowed, raises HITLPendingException otherwise."""
    # Checked here rather than at each gated route, so a route added later
    # cannot ship without it. Before the mode is read, because a call that
    # cannot be captured is refused whether or not it would have been gated
    # - a tool whose gate is off today and on tomorrow must not change
    # whether the same body is accepted.
    params_json = _capturable(params_json, "parameters") or "{}"
    tool_body = _capturable(tool_body, "body")

    row = await store.setting(tool_name)

    if row is None:
        return caller_mind

    mode = row["mode"]

    if mode == "off":
        return caller_mind

    if mode == "sudo":
        now = int(time.time())
        grant = await store.active_grant(tool_name, caller_mind, now)
        if grant is not None:
            return caller_mind

    # mode == "on" or "sudo" without valid grant — create HITL request
    request_id = str(uuid.uuid4())
    now = int(time.time())
    expires_at = now + HITL_EXPIRY_SECONDS

    await store.create_request(
        request_id, tool_name, caller_mind, params_json, now, expires_at,
        tool_method, tool_path, tool_body,
    )

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
            response = await client.request(
                method, url, content=body, headers=headers, timeout=30)
        # What the replay answered, not merely that it answered. An approved
        # gmail send whose replay 500s used to be logged as a success beside
        # a `replay_scheduled=True` from the route - so nothing anywhere
        # learned that the tool Daniel approved never ran.
        if response.is_success:
            log.info("HITL replay succeeded: %s %s -> %d",
                     method, path, response.status_code)
        else:
            log.warning(
                "HITL replay was refused: %s %s -> %d. The request is "
                "recorded as approved and the tool did not run; it has to be "
                "re-made rather than re-approved, because a decided request "
                "cannot be decided again. Response: %s",
                method, path, response.status_code, response.text[:500])
    except Exception:
        log.warning("HITL replay failed: %s %s", method, path, exc_info=True)


def hitl_gate(tool_name: str):
    """Factory that returns a FastAPI dependency for HITL checks.

    Side effect: records `tool_name` in `_REGISTERED_TOOLS` at import time so
    `sync_tool_registry` can upsert it into `tool_hitl_settings` at startup.
    """
    _REGISTERED_TOOLS.add(tool_name)

    async def _check(
        request: Request,
        caller_mind: str = Depends(require_api_token),
    ) -> str:
        # Internal replay calls bypass the gate entirely
        if caller_mind == "internal":
            return caller_mind

        body_bytes = await request.body()
        return await _check_hitl(
            tool_name,
            caller_mind,
            tool_method=request.method,
            tool_path=str(request.url.path),
            tool_body=body_bytes.decode("utf-8", errors="replace"),
        )

    return _check


# Default HITL settings: tool_name -> mode
_DEFAULT_HITL_SETTINGS: list[tuple[str, str, int]] = [
    # Calendar read tools -- off
    ("calendar_list_events", "off", 15),
    ("calendar_get_event", "off", 15),
    ("calendar_list", "off", 15),
    ("calendar_check_availability", "off", 15),
    # Calendar write tools -- on
    ("calendar_create_event", "on", 15),
    ("calendar_quick_add", "on", 15),
    ("calendar_update_event", "on", 15),
    ("calendar_delete_event", "on", 15),
    ("calendar_invite", "on", 15),
    # Gmail read tools -- off
    ("gmail_read_emails", "off", 15),
    ("gmail_get_email", "off", 15),
    ("gmail_list_labels", "off", 15),
    # Gmail write tools -- on
    ("gmail_send_email", "on", 15),
    ("gmail_reply_email", "on", 15),
    ("gmail_delete_email", "on", 15),
    ("gmail_move_email", "on", 15),
    # LinkedIn -- on
    ("linkedin_post", "on", 15),
    # Docker mutating -- on
    ("docker_compose_up", "on", 15),
    ("docker_compose_restart", "on", 15),
    ("docker_compose_down", "on", 15),
    # Docker read-only -- off
    ("docker_compose_logs", "off", 15),
    ("docker_compose_status", "off", 15),
    ("docker_list_containers", "off", 15),
    ("docker_list_networks", "off", 15),
    # Browser -- on for actions that introduce external content or change page state
    ("browser_navigate", "on", 15),
    ("browser_click", "on", 15),
    ("browser_type", "on", 15),
    ("browser_web_search", "on", 15),
    # Ollama generic LLM endpoints — read-only side effects, no approval needed
    ("ollama_structured", "off", 15),
    ("ollama_chat", "off", 15),
    # Schema mutation — always HITL-gated. Adding new node/edge types is a
    # durable schema decision and should never auto-execute.
    ("schema_new_type", "on", 15),
    # Training pipeline — off, because the only caller is the console and a
    # console click is already a human act taken behind a login. Gating them
    # would page the same person who just pressed the button, and the panel
    # would sit on a pending response with nothing to show.
    ("training_curate", "off", 15),
    ("training_export", "off", 15),
    ("training_train", "off", 15),
    ("training_deploy", "off", 15),
]


async def seed_default_hitl_settings() -> None:
    """Give each shipped tool its mode only if it has none.

    Insert-if-absent, and that is the point: this runs on every boot, so a
    replacing write would re-arm every approval gate the operator had turned
    off, on every deploy, silently.
    """
    for tool_name, mode, timeout in _DEFAULT_HITL_SETTINGS:
        await store.seed_setting(tool_name, mode, timeout)


async def sync_tool_registry(
    default_mode: str = AUTO_REGISTER_DEFAULT_MODE,
    default_sudo_timeout_minutes: int = AUTO_REGISTER_DEFAULT_SUDO_TIMEOUT_MINUTES,
) -> int:
    """Upsert any tool name discovered at hitl_gate import time.

    Existing rows in tool_hitl_settings are left alone (insert-if-absent)
    so the shipped modes in seed_default_hitl_settings and user-edited modes
    both survive. Returns the number of new rows inserted.
    """
    inserted = 0
    for name in sorted(_REGISTERED_TOOLS):
        inserted += await store.seed_setting(
            name, default_mode, default_sudo_timeout_minutes)
    if inserted:
        log.info("sync_tool_registry: registered %d new gated tool(s)", inserted)
    return inserted


def effective_status(row: dict[str, Any], now: int) -> str:
    """What a request's status actually is at `now`.

    Nothing ever writes `expired` to a row: the expiry is a rule applied
    when the row is read. One function for it rather than the rule inline at
    each reader, because the approvals page had its own copy - which is to
    say it had none, and drew working Approve and Deny buttons on eleven
    requests that had been past their expiry for up to ninety-six days.
    """
    if row["status"] == "pending" and now > row["expires_at"]:
        return "expired"
    return str(row["status"])


async def get_hitl_status(request_id: str) -> dict[str, Any] | None:
    """Get the current status of a HITL request, accounting for expiry."""
    row = await store.request(request_id)

    if row is None:
        return None

    now = int(time.time())
    status = effective_status(row, now)

    return {
        "hitl_request_id": row["id"],
        "status": status,
        "tool_name": row["tool_name"],
        "caller_mind": row["caller_mind"],
        "decided_at": row["decided_at"],
    }


async def respond_to_hitl(
    request_id: str, action: str
) -> dict[str, Any] | None:
    """Approve or deny a HITL request.

    The decision is one statement guarded on the request still being pending
    and not yet expired, so two people reaching the queue at the same moment
    produce one decision. What it changed is what it hands back: a request
    that was already settled, or that expired while the page was open,
    changes nothing and is reported as already resolved rather than being
    overwritten.

    The decision and the sudo grant it may earn are **one** unit of work.
    Separately committed they leave an end state nothing can recover from: a
    grant that fails after the decision lands leaves the request saying
    approved, the call it authorised never replayed (the route fires the
    replay after this returns, so an exception loses it), and the request
    un-decidable for ever because the decision is guarded on `pending`. The
    page would show one approval and the operator would have no way back.
    """
    try:
        new_status = ACTIONS[action]
    except KeyError:
        raise UnknownAction(
            f"{action!r} is not a decision. A request is approved or denied, "
            f"and nothing else: {', '.join(sorted(ACTIONS))}. Anything "
            "unrecognised used to be taken as a denial, so the past-tense "
            "spelling of approve denied the request and answered 200.") \
            from None
    now = int(time.time())

    async with store.transaction():
        row = await store.decide_request(request_id, new_status, now, now)

        if row is None:
            # Nothing was decided. Say which of the three reasons it was,
            # because "no such request" and "somebody got there first" are
            # different sentences, and an expired one is neither.
            existing = await store.request(request_id)
            if existing is None:
                return None
            if existing["status"] != "pending":
                return {"error": "already_resolved",
                        "current_status": existing["status"]}
            return {"error": "already_resolved", "current_status": "expired"}

        # Replay data returned to the caller — fired as a background task by
        # the route handler, after this unit has committed.
        replay: dict[str, str] | None = None
        if new_status == "approved":
            r_method = row.get("tool_method")
            r_path = row.get("tool_path")
            r_body = row.get("tool_body")
            if r_method and r_path and r_body is not None:
                replay = {"method": r_method, "path": r_path, "body": r_body}

            setting = await store.setting(row["tool_name"])
            if setting and setting["mode"] == "sudo":
                timeout_minutes = int(setting["sudo_timeout_minutes"])
                await store.grant_sudo(
                    row["tool_name"], row["caller_mind"], now,
                    now + timeout_minutes * 60)

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
