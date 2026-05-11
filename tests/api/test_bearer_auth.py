"""API tests for bearer token authentication."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from auth_api import generate_token, require_api_token
from db import init_db


@pytest.fixture()
def app_with_auth(tmp_path: Path) -> FastAPI:
    db_path = tmp_path / "test.db"
    app = FastAPI()
    conn = init_db(str(db_path))
    app.state.db = conn

    @app.get("/protected")
    async def protected(caller: str = Depends(require_api_token)):
        return {"caller": caller}

    return app


@pytest.fixture()
def client(app_with_auth: FastAPI) -> TestClient:
    return TestClient(app_with_auth)


def test_endpoint_without_auth_returns_401(client: TestClient) -> None:
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_endpoint_with_invalid_token_returns_401(client: TestClient) -> None:
    resp = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401


def test_endpoint_with_valid_token_returns_200(
    app_with_auth: FastAPI, client: TestClient
) -> None:
    db = app_with_auth.state.db
    raw = generate_token(db, "ada")

    resp = client.get("/protected", headers={"Authorization": f"Bearer {raw}"})
    assert resp.status_code == 200
    assert resp.json()["caller"] == "ada"
