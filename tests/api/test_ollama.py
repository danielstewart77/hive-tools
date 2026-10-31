"""API tests for the /ollama/models endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    from server import create_app

    return create_app(db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    # Entered, not merely constructed: the store opens in the app's
    # lifespan, and a client that never enters it is an app whose database
    # was never opened.
    with TestClient(app) as entered:
        yield entered


@pytest.fixture()
def auth_headers(app: FastAPI) -> dict[str, str]:
    from auth_api import generate_token

    raw = asyncio.run(generate_token("ada"))
    return {"Authorization": f"Bearer {raw}"}


def _fake_tags_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", "x"))


def test_models_returns_flat_shape(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    tags_payload = {
        "models": [
            {
                "name": "gpt-oss:20b-32k",
                "size": 12345,
                "modified_at": "2026-05-18T00:00:00Z",
                "details": {
                    "family": "gpt-oss",
                    "parameter_size": "20B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }
    with patch("tools.ollama.httpx.AsyncClient") as ctor:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=_fake_tags_response(tags_payload))
        ctor.return_value.__aenter__.return_value = instance

        resp = client.get("/ollama/models", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert "ollama_base_url" in body
    assert "default_model" in body
    assert body["models"] == [
        {
            "name": "gpt-oss:20b-32k",
            "size": 12345,
            "modified_at": "2026-05-18T00:00:00Z",
            "family": "gpt-oss",
            "parameter_size": "20B",
            "quantization": "Q4_K_M",
        }
    ]


def test_models_returns_502_when_ollama_unreachable(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    with patch("tools.ollama.httpx.AsyncClient") as ctor:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctor.return_value.__aenter__.return_value = instance

        resp = client.get("/ollama/models", headers=auth_headers)

    assert resp.status_code == 502
    assert "ollama unreachable" in resp.json()["detail"]


def test_models_requires_auth(client: TestClient) -> None:
    resp = client.get("/ollama/models")
    assert resp.status_code == 401


def _fake_systemone_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "x"))


def test_the_relay_sends_the_callers_own_state_and_questions_verbatim(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """The caller owns the question; this route must not compose one."""
    sent: dict[str, Any] = {}
    answer = {"answers": {"shape_1": {"type": "choice", "choice": "a_bucket"}}}

    async def fake_post(self: Any, url: str, json: dict[str, Any]) -> httpx.Response:
        sent["url"] = url
        sent["json"] = json
        return _fake_systemone_response(answer)

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/ollama/systemone",
            headers=auth_headers,
            json={
                "model": "pplx-decider:27b",
                "state": {"events": [{"id": 1, "raw": "a line"}]},
                "questions": {"shape_1": {"type": "choice", "criteria": {"a_bucket": "d"}}},
                "keep_alive": "15m",
            },
        )

    assert resp.status_code == 200
    assert resp.json() == answer
    assert sent["url"].endswith("/v1/systemone")
    assert sent["json"]["model"] == "pplx-decider:27b"
    assert sent["json"]["state"]["events"][0]["raw"] == "a line"
    assert sent["json"]["questions"]["shape_1"]["criteria"] == {"a_bucket": "d"}
    assert sent["json"]["keep_alive"] == "15m"


def test_a_call_naming_no_model_is_refused_rather_than_given_a_default(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """A substituted model is a placement nobody can account for."""
    calls: list[str] = []

    async def fake_post(self: Any, url: str, json: dict[str, Any]) -> httpx.Response:
        calls.append(url)
        return _fake_systemone_response({"answers": {}})

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/ollama/systemone",
            headers=auth_headers,
            json={"state": {}, "questions": {"q": {}}},
        )

    assert resp.status_code == 400
    assert calls == []


def test_a_call_asking_nothing_is_refused(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    calls: list[str] = []

    async def fake_post(self: Any, url: str, json: dict[str, Any]) -> httpx.Response:
        calls.append(url)
        return _fake_systemone_response({"answers": {}})

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/ollama/systemone",
            headers=auth_headers,
            json={"model": "m", "state": {}, "questions": {}},
        )

    assert resp.status_code == 400
    assert calls == []


def test_an_unreachable_ollama_is_a_bad_gateway_not_a_silent_empty_answer(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    async def fake_post(self: Any, url: str, json: dict[str, Any]) -> httpx.Response:
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=fake_post):
        resp = client.post(
            "/ollama/systemone",
            headers=auth_headers,
            json={"model": "m", "state": {}, "questions": {"q": {"type": "choice"}}},
        )

    assert resp.status_code == 502


def test_the_relay_refuses_a_caller_with_no_token(client: TestClient) -> None:
    resp = client.post(
        "/ollama/systemone",
        json={"model": "m", "state": {}, "questions": {"q": {}}},
    )

    assert resp.status_code in (401, 403)
