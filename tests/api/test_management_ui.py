"""API tests for the management UI routes."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth_api import generate_token
from auth_session import create_session_token, create_user


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    from server import create_app

    return create_app(db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    # Entered, not merely constructed: the store opens in the app's
    # lifespan, and a client that never enters it is an app whose database
    # was never opened.
    with TestClient(app, follow_redirects=False) as entered:
        yield entered


@pytest.fixture()
def authed_client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Client with session cookie set, on an app that has been started."""
    with TestClient(app, follow_redirects=False) as entered:
        user = asyncio.run(create_user("daniel", "pass123", is_admin=True))
        entered.cookies.set("ht_session", create_session_token(user))
        yield entered


def test_dashboard_requires_auth(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code in (302, 303)


def test_dashboard_with_auth_returns_html(authed_client: TestClient) -> None:
    resp = authed_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_tools_page_lists_registered_routes(authed_client: TestClient) -> None:
    resp = authed_client.get("/tools")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_tools_page_save_updates_hitl_mode(
    app: FastAPI, authed_client: TestClient
) -> None:
    resp = authed_client.post(
        "/tools/settings",
        data={"tool_name": "gmail_send_email", "mode": "on", "sudo_timeout_minutes": "15"},
    )
    assert resp.status_code in (200, 302, 303)

    db = app.state.db
    row = db.execute(
        "SELECT mode FROM tool_hitl_settings WHERE tool_name = ?",
        ("gmail_send_email",),
    ).fetchone()
    assert row is not None
    assert row["mode"] == "on"


def test_tokens_page_lists_tokens(app: FastAPI, authed_client: TestClient) -> None:
    asyncio.run(generate_token("ada"))
    resp = authed_client.get("/tokens")
    assert resp.status_code == 200
    assert "ada" in resp.text


def test_tokens_create_shows_raw_token_once(authed_client: TestClient) -> None:
    resp = authed_client.post(
        "/tokens/create",
        data={"name": "bob"},
    )
    assert resp.status_code == 200
    # The raw token should be displayed in the response
    assert "bob" in resp.text


def test_tokens_revoke_sets_revoked_at(
    app: FastAPI, authed_client: TestClient
) -> None:
    asyncio.run(generate_token("ada"))
    row = app.state.db.execute(
        "SELECT id FROM api_tokens WHERE name = ?", ("ada",)
    ).fetchone()
    token_id = row["id"]

    resp = authed_client.post(f"/tokens/{token_id}/revoke")
    assert resp.status_code in (200, 302, 303)

    row = app.state.db.execute(
        "SELECT revoked_at FROM api_tokens WHERE id = ?", (token_id,)
    ).fetchone()
    assert row["revoked_at"] is not None


def test_hitl_page_lists_pending_requests(
    app: FastAPI, authed_client: TestClient
) -> None:
    db = app.state.db
    request_id = str(uuid.uuid4())
    now = int(time.time())
    db.execute(
        "INSERT INTO hitl_requests (id, tool_name, caller_mind, params_json, status, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (request_id, "test_tool", "ada", "{}", "pending", now, now + 300),
    )
    db.commit()

    resp = authed_client.get("/hitl-ui")
    assert resp.status_code == 200
    assert request_id in resp.text


def test_hitl_page_approve_button_works(
    app: FastAPI, authed_client: TestClient
) -> None:
    db = app.state.db
    request_id = str(uuid.uuid4())
    now = int(time.time())
    db.execute(
        "INSERT INTO hitl_requests (id, tool_name, caller_mind, params_json, status, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (request_id, "test_tool", "ada", "{}", "pending", now, now + 300),
    )
    db.commit()

    resp = authed_client.post(
        f"/hitl-ui/{request_id}/respond",
        data={"action": "approve"},
    )
    assert resp.status_code in (200, 302, 303)

    row = db.execute(
        "SELECT status FROM hitl_requests WHERE id = ?", (request_id,)
    ).fetchone()
    assert row["status"] == "approved"
