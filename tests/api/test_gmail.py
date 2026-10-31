"""API tests for Gmail endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth_api import generate_token


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    from server import create_app

    return create_app(db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    # Entered, not merely constructed: the store opens in the app's
    # lifespan, and a client that never enters it is an app whose database
    # was never opened.
    with TestClient(app, raise_server_exceptions=False) as entered:
        yield entered


@pytest.fixture()
def auth_header(client: TestClient) -> dict[str, str]:
    raw = asyncio.run(generate_token("ada"))
    return {"Authorization": f"Bearer {raw}"}


def _mock_gmail_service() -> MagicMock:
    service = MagicMock()
    service.users().messages().list().execute.return_value = {
        "messages": [{"id": "msg1"}],
        "resultSizeEstimate": 1,
    }
    service.users().messages().get().execute.return_value = {
        "id": "msg1",
        "threadId": "thread1",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@test.com"},
                {"name": "Subject", "value": "Test Subject"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "dGVzdCBib2R5"},
        },
        "snippet": "test snippet",
        "labelIds": ["INBOX"],
    }
    service.users().labels().list().execute.return_value = {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
        ]
    }
    return service


def test_read_emails_requires_auth(client: TestClient) -> None:
    resp = client.get("/gmail/messages?query=test")
    assert resp.status_code == 401


@patch("tools.gmail._get_gmail_service")
def test_read_emails_with_auth_returns_messages(
    mock_svc: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_svc.return_value = _mock_gmail_service()
    resp = client.get("/gmail/messages?query=test", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "messages" in data


@patch("tools.gmail._get_gmail_service")
def test_read_emails_hitl_off(
    mock_svc: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_svc.return_value = _mock_gmail_service()
    resp = client.get("/gmail/messages?query=test", headers=auth_header)
    assert resp.status_code == 200


@patch("tools.gmail._get_gmail_service")
def test_get_email_returns_full_message(
    mock_svc: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_svc.return_value = _mock_gmail_service()
    resp = client.get("/gmail/messages/msg1", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "msg1"


@patch("tools.gmail._get_gmail_service")
def test_list_labels_returns_array(
    mock_svc: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_svc.return_value = _mock_gmail_service()
    resp = client.get("/gmail/labels", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_send_email_hitl_on_returns_202(
    app: FastAPI,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    db = app.state.db
    db.execute(
        "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
        ("gmail_send_email", "on"),
    )
    db.commit()

    resp = client.post(
        "/gmail/send",
        json={"to": "test@example.com", "subject": "hi", "body": "hello"},
        headers=auth_header,
    )
    assert resp.status_code == 202


def test_reply_email_hitl_on_returns_202(
    app: FastAPI,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    db = app.state.db
    db.execute(
        "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
        ("gmail_reply_email", "on"),
    )
    db.commit()

    resp = client.post(
        "/gmail/reply",
        json={"message_id": "msg1", "body": "reply text"},
        headers=auth_header,
    )
    assert resp.status_code == 202


def test_delete_email_hitl_on_returns_202(
    app: FastAPI,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    db = app.state.db
    db.execute(
        "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
        ("gmail_delete_email", "on"),
    )
    db.commit()

    resp = client.post(
        "/gmail/trash",
        json={"message_id": "msg1"},
        headers=auth_header,
    )
    assert resp.status_code == 202
