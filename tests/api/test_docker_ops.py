"""API tests for Docker Ops endpoints."""

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


def test_docker_status_requires_auth(client: TestClient) -> None:
    resp = client.get("/docker/status?project=hive_mind")
    assert resp.status_code == 401


def test_docker_compose_up_hitl_on_returns_202(
    app: FastAPI,
    client: TestClient,
    auth_header: dict[str, str],
    tmp_path,
) -> None:
    db = app.state.db
    db.execute(
        "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
        ("docker_compose_up", "on"),
    )
    db.commit()

    # The path must be a real directory or _validate_path short-circuits before HITL.
    resp = client.post(
        f"/docker/up?path={tmp_path}",
        headers=auth_header,
    )
    assert resp.status_code == 202


@patch("tools.docker_ops._run_subprocess")
def test_docker_compose_logs_returns_output(
    mock_run: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
    tmp_path,
) -> None:
    mock_run.return_value = (0, "log line 1\nlog line 2", "")
    resp = client.get(f"/docker/logs?path={tmp_path}", headers=auth_header)
    assert resp.status_code == 200


def test_docker_invalid_project_returns_error(
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    resp = client.get("/docker/status?path=/no/such/path/xyz", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


@patch("tools.docker_ops._run_subprocess")
def test_docker_list_containers_returns_array(
    mock_run: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_run.return_value = (0, '{"Names":"test","Image":"img","State":"running","Status":"Up","Ports":""}\n', "")
    resp = client.get("/docker/containers", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "containers" in data


@patch("tools.docker_ops._run_subprocess")
def test_docker_list_networks_returns_array(
    mock_run: MagicMock,
    client: TestClient,
    auth_header: dict[str, str],
) -> None:
    mock_run.return_value = (0, '{"Name":"bridge","Driver":"bridge","Scope":"local"}\n', "")
    resp = client.get("/docker/networks", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "networks" in data
