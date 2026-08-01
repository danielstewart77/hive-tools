"""API tests for the FastAPI application skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def app(tmp_path: Path) -> FastAPI:
    from server import create_app

    return create_app(db_path=str(tmp_path / "test.db"))


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_health_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_app_initializes_db_on_startup(app: FastAPI) -> None:
    assert hasattr(app.state, "db")
    assert app.state.db is not None
