"""Management UI routes for HiveTools."""

from __future__ import annotations

from typing import Any

import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import store
from auth_api import generate_token, list_tokens, revoke_token
from auth_session import get_current_user_from_request
from config import BASE_DIR
from hitl import _replay_request_async, respond_to_hitl

router = APIRouter()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---- Tools ----


_METHOD_ORDER = {"GET": 0, "POST": 1, "PATCH": 2, "DELETE": 3}


def _infer_method(tool_name: str) -> str:
    """Derive HTTP method from semantic tool name."""
    # Strip group prefix (e.g. "calendar_", "gmail_")
    parts = tool_name.split("_", 1)
    stem = parts[1] if len(parts) > 1 else tool_name

    if any(stem.startswith(p) for p in ("list", "get", "check", "read", "logs", "status")):
        return "GET"
    if any(stem.startswith(p) for p in ("delete", "down", "remove")):
        return "DELETE"
    if any(stem.startswith(p) for p in ("update", "move")):
        return "PATCH"
    return "POST"


@router.get("/tools", response_class=HTMLResponse)
async def tools_page(request: Request) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    rows = await store.settings()
    tools = [dict(r) | {"method": _infer_method(r["tool_name"])} for r in rows]
    tools.sort(key=lambda t: (t["tool_name"].split("_")[0], _METHOD_ORDER.get(t["method"], 9), t["tool_name"]))

    return templates.TemplateResponse(request, "tools.html", {"user": user, "tools": tools})


@router.post("/tools/settings")
async def tools_settings_save(
    request: Request,
    tool_name: str = Form(...),
    mode: str = Form(...),
    sudo_timeout_minutes: int = Form(15),
) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    await store.set_setting(tool_name, mode, sudo_timeout_minutes)
    return RedirectResponse(url="/tools", status_code=303)


# ---- Tokens ----


@router.get("/tokens", response_class=HTMLResponse)
async def tokens_page(request: Request) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    tokens = await list_tokens()
    return templates.TemplateResponse(request, "tokens.html", {"user": user, "tokens": tokens})


@router.post("/tokens/create", response_class=HTMLResponse)
async def tokens_create(
    request: Request,
    name: str = Form(...),
) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    raw_token = await generate_token(name)
    tokens = await list_tokens()
    return templates.TemplateResponse(
        request,
        "tokens.html",
        {"user": user, "tokens": tokens, "new_token": raw_token, "new_token_name": name},
    )


@router.post("/tokens/{token_id}/revoke")
async def tokens_revoke(
    request: Request,
    token_id: int,
) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    await revoke_token(token_id)
    return RedirectResponse(url="/tokens", status_code=303)


# ---- HITL UI ----


@router.get("/hitl-ui", response_class=HTMLResponse)
async def hitl_page(request: Request) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    rows = await store.recent_requests()
    requests_list = [dict(r) for r in rows]
    return templates.TemplateResponse(
        request, "hitl.html", {"user": user, "requests": requests_list}
    )


@router.post("/hitl-ui/{request_id}/respond")
async def hitl_ui_respond(
    request: Request,
    request_id: str,
    action: str = Form(...),
) -> Any:
    user = await get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    result = await respond_to_hitl(request_id, action) or {}
    replay = result.pop("_replay", None)
    if replay:
        asyncio.create_task(_replay_request_async(**replay))
    return RedirectResponse(url="/hitl-ui", status_code=303)
