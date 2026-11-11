"""What an approved call did, reported back to the mind that asked.

The gated call is made through the real route, approved through the real
route, and replayed by the real `hitl.py` onto the scratch Postgres. Only
the transport is stubbed: the far side of the replayed HTTP call, where it
leaves this process, and the HTTP to hive-comms.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import hitl
from auth_api import generate_token
from tests.scratch import run

ADA_ID = "565e5a66-d20c-4266-872a-3268c4c894fc"
COMMS = "http://comms.test:8424"


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    from server import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as entered:
        yield entered


@pytest.fixture()
def ada(client: TestClient) -> dict[str, str]:
    return {"Authorization": f"Bearer {run(generate_token('ada'))}"}


@pytest.fixture()
def comms(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """hive-comms' far side: a registry holding Ada, and a message sink."""
    monkeypatch.setenv("COMMS_URL", COMMS)
    monkeypatch.setenv("COMMS_BEARER_TOKEN", "service-token")
    posted: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/broker/minds":
            return httpx.Response(200, json=[
                {"id": "39369381-0700-4203-b742-f6c9555adcc7",
                 "name": "dragoman"},
                {"id": ADA_ID, "name": "ada"},
            ])
        if request.method == "POST" and request.url.path == "/broker/messages":
            posted.append(request)
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "status": "dispatched",
                "conversation_id": body["conversation_id"],
                "message_id": body["message_id"]})
        return httpx.Response(404, json={"error": "no such route"})

    monkeypatch.setattr(hitl, "_COMMS_TRANSPORT", httpx.MockTransport(handler),
                        raising=False)
    return posted


def _tool_answers(monkeypatch: pytest.MonkeyPatch,
                  answer: Callable[[httpx.Request], httpx.Response]) -> None:
    """The replayed call's far side answers with `answer`."""
    monkeypatch.setattr(hitl, "_REPLAY_TRANSPORT", httpx.MockTransport(answer),
                        raising=False)


def _approved_label_request(client: TestClient,
                            ada: dict[str, str]) -> str:
    """Ada asks for a label, which is gated, and Daniel approves it."""
    asked = client.post("/gmail/labels", json={"name": "Receipts/2026"},
                        headers=ada)
    assert asked.status_code == 202
    request_id = asked.json()["hitl_request_id"]
    decided = client.post(f"/hitl/{request_id}/respond",
                          json={"action": "approve"}, headers=ada)
    assert decided.status_code == 200
    run(hitl.settle_replays(timeout=10))
    return request_id


def test_an_approved_replay_that_succeeds_reports_success_and_the_tools_body(
        client: TestClient, ada: dict[str, str], comms: list[httpx.Request],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _tool_answers(monkeypatch, lambda request: httpx.Response(200, json={
        "id": "Label_42", "name": "Receipts/2026", "created": True}))

    request_id = _approved_label_request(client, ada)
    status = client.get(f"/hitl/{request_id}", headers=ada)

    assert status.status_code == 200
    answer = status.json()
    assert answer["status"] == "approved"
    assert answer["outcome"]["success"] is True
    assert answer["outcome"]["http_status"] == 200
    assert answer["outcome"]["result"]["id"] == "Label_42"


@pytest.mark.parametrize("answer, error", [
    # A tool that caught its own failure answers 200 and says so.
    (lambda request: httpx.Response(
        200, json={"error": "Gmail not authenticated"}),
     "Gmail not authenticated"),
    (lambda request: httpx.Response(
        500, json={"detail": "label service exploded"}),
     "label service exploded"),
], ids=["200-with-error", "500"])
def test_an_approved_replay_that_fails_reports_failure_with_the_tools_error(
        client: TestClient, ada: dict[str, str], comms: list[httpx.Request],
        monkeypatch: pytest.MonkeyPatch,
        answer: Callable[[httpx.Request], httpx.Response], error: str) -> None:
    _tool_answers(monkeypatch, answer)

    request_id = _approved_label_request(client, ada)
    outcome = client.get(f"/hitl/{request_id}", headers=ada).json()["outcome"]

    assert outcome["success"] is False
    assert error in outcome["error"]


def test_a_replay_that_never_gets_an_answer_reports_failure(
        client: TestClient, ada: dict[str, str], comms: list[httpx.Request],
        monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _tool_answers(monkeypatch, refused)

    request_id = _approved_label_request(client, ada)
    outcome = client.get(f"/hitl/{request_id}", headers=ada).json()["outcome"]

    assert outcome["success"] is False
    assert outcome["http_status"] is None
    assert "connection refused" in outcome["error"]


def test_the_outcome_is_sent_once_to_the_asking_mind_through_comms(
        client: TestClient, ada: dict[str, str], comms: list[httpx.Request],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _tool_answers(monkeypatch, lambda request: httpx.Response(200, json={
        "id": "Label_42", "name": "Receipts/2026", "created": True}))

    request_id = _approved_label_request(client, ada)

    assert len(comms) == 1
    sent: dict[str, Any] = json.loads(comms[0].content)
    assert comms[0].headers["Authorization"] == "Bearer service-token"
    assert sent["to"] == ADA_ID
    assert sent["from"] == ADA_ID
    assert sent["metadata"]["request_type"] == "hitl_result"
    assert sent["metadata"]["hitl_request_id"] == request_id
    assert sent["metadata"]["tool_name"] == "gmail_create_label"
    assert sent["metadata"]["success"] is True
    assert "Label_42" in sent["metadata"]["result"]
    assert request_id in sent["content"] and "Label_42" in sent["content"]


def test_a_message_comms_refuses_never_loses_the_stored_outcome(
        client: TestClient, ada: dict[str, str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMS_URL", COMMS)
    monkeypatch.setenv("COMMS_BEARER_TOKEN", "service-token")
    monkeypatch.setattr(hitl, "_COMMS_TRANSPORT", httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": "down"})),
        raising=False)
    _tool_answers(monkeypatch, lambda request: httpx.Response(200, json={
        "id": "Label_42", "name": "Receipts/2026", "created": True}))

    request_id = _approved_label_request(client, ada)
    outcome = client.get(f"/hitl/{request_id}", headers=ada).json()["outcome"]

    assert outcome["success"] is True
    assert outcome["result"]["id"] == "Label_42"
