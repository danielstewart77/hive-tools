"""API tests for bearer token authentication."""

from __future__ import annotations

from contextlib import asynccontextmanager

from collections.abc import Generator

from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import store
from auth_api import generate_token, require_api_token
from tests.scratch import run


@pytest.fixture()
def app_with_auth(tmp_path: Path) -> FastAPI:
    """A bare app around the token dependency.

    Not `create_app`: the point here is the dependency in isolation. The
    store still has to be opened for it, and it is opened in this app's own
    lifespan for the same reason the real one is.
    """
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await store.open()
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/protected")
    async def protected(caller: str = Depends(require_api_token)):
        return {"caller": caller}

    return app


@pytest.fixture()
def client(app_with_auth: FastAPI) -> Generator[TestClient, None, None]:
    # Entered, not merely constructed: the store opens in the app's
    # lifespan, and a client that never enters it is an app whose database
    # was never opened.
    with TestClient(app_with_auth) as entered:
        yield entered


def test_endpoint_without_auth_returns_401(client: TestClient) -> None:
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_endpoint_with_invalid_token_returns_401(client: TestClient) -> None:
    resp = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401


def test_endpoint_with_valid_token_returns_200(client: TestClient) -> None:
    raw = run(generate_token("ada"))

    resp = client.get("/protected", headers={"Authorization": f"Bearer {raw}"})
    assert resp.status_code == 200
    assert resp.json()["caller"] == "ada"
