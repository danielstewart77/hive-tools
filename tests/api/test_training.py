"""API tests for the /training router.

The router is a transport over hive_mind's stateless pipeline CLI, so these
tests pin the transport contract: arguments are translated faithfully, the
CLI's JSON is passed through unchanged, a missing mount is a clear 503
rather than a stack trace, and launching a fine-tune is never the default.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

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


@pytest.fixture()
def auth_headers(app: FastAPI) -> dict[str, str]:
    from auth_api import generate_token

    raw = generate_token(app.state.db, "ada")
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture(autouse=True)
def hitl_off(app: FastAPI):
    """Run the gated endpoints directly.

    Curate, export and train are HITL-gated in production — each fires a
    Telegram approval on first call and returns 202. That behaviour is the
    HITL suite's to test; here it would mask the argument translation these
    tests exist to pin.
    """
    for tool in ("training_curate", "training_export", "training_train"):
        app.state.db.execute(
            "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
            (tool, "off"),
        )
    app.state.db.commit()


@pytest.fixture()
def fake_cli(tmp_path: Path, monkeypatch):
    """Point the router at a CLI path that exists, and capture invocations."""
    import tools.training as training

    hive_dir = tmp_path / "hive_mind"
    cli = hive_dir / training.CLI_RELATIVE
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("# stub\n")
    monkeypatch.setattr(training, "HIVE_MIND_DIR", hive_dir)

    calls: list[list[str]] = []

    def _completed(payload: dict, returncode: int = 0):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=json.dumps(payload), stderr=""
        )

    return {"calls": calls, "completed": _completed, "dir": hive_dir}


def _patch_run(fake_cli, payload, returncode=0):
    def _fake_run(command, **kwargs):
        fake_cli["calls"].append(command)
        return fake_cli["completed"](payload, returncode)

    return patch("tools.training.subprocess.run", side_effect=_fake_run)


def _args(fake_cli) -> list[str]:
    """CLI arguments of the last invocation, without interpreter and path."""
    return fake_cli["calls"][-1][2:]


def test_status_passes_the_cli_payload_through(client, auth_headers, fake_cli):
    payload = {"total": 8809, "by_flag": {"keep": 3024}, "exists": True}
    with _patch_run(fake_cli, payload):
        response = client.get("/training/status", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == payload
    assert _args(fake_cli) == ["status"]


def test_curate_translates_every_knob(client, auth_headers, fake_cli):
    with _patch_run(fake_cli, {"kept": 3024}):
        response = client.post(
            "/training/curate",
            headers=auth_headers,
            json={
                "keep_per_cluster": 5,
                "min_length_tokens": 100,
                "exclude_secret_rows": True,
                "harness": ["codex"],
            },
        )
    assert response.status_code == 200
    args = _args(fake_cli)
    assert args[0] == "curate"
    assert "--keep-per-cluster" in args and args[args.index("--keep-per-cluster") + 1] == "5"
    assert args[args.index("--min-length-tokens") + 1] == "100"
    assert "--exclude-secret-rows" in args
    assert args[args.index("--harness") + 1] == "codex"


def test_curate_reset_sends_only_the_reset_flag(client, auth_headers, fake_cli):
    with _patch_run(fake_cli, {"reset_rows": 8809}):
        response = client.post(
            "/training/curate", headers=auth_headers, json={"reset": True}
        )
    assert response.status_code == 200
    assert _args(fake_cli) == ["curate", "--reset"]


def test_curate_rejects_a_nonsense_cluster_budget(client, auth_headers, fake_cli):
    response = client.post(
        "/training/curate", headers=auth_headers, json={"keep_per_cluster": 0}
    )
    assert response.status_code == 422


def test_export_translates_mode_and_filters(client, auth_headers, fake_cli):
    with _patch_run(fake_cli, {"train_examples": 2840}):
        response = client.post(
            "/training/export",
            headers=auth_headers,
            json={
                "mode": "reasoning",
                "name": "v1",
                "eval_fraction": 0.1,
                "require_reasoning": True,
                "include_system_prompt": False,
            },
        )
    assert response.status_code == 200
    args = _args(fake_cli)
    assert args[args.index("--mode") + 1] == "reasoning"
    assert args[args.index("--name") + 1] == "v1"
    assert "--require-reasoning" in args
    assert "--no-system-prompt" in args


def test_export_defaults_to_randomizing_credentials(client, auth_headers, fake_cli):
    """Same default as the CLI: a surrogate, never a real value by accident."""
    with _patch_run(fake_cli, {"train_examples": 1}):
        client.post("/training/export", headers=auth_headers, json={"name": "raw"})
    args = _args(fake_cli)
    assert args[args.index("--secrets") + 1] == "randomize"


def test_export_forwards_the_chosen_secret_policy(client, auth_headers, fake_cli):
    for policy in ("keep", "redact"):
        with _patch_run(fake_cli, {"train_examples": 1}):
            client.post(
                "/training/export",
                headers=auth_headers,
                json={"name": "shareable", "secrets": policy},
            )
        args = _args(fake_cli)
        assert args[args.index("--secrets") + 1] == policy


def test_export_rejects_an_unknown_secret_policy(client, auth_headers, fake_cli):
    response = client.post(
        "/training/export", headers=auth_headers, json={"secrets": "yolo"}
    )
    assert response.status_code == 422


def test_export_rejects_an_unknown_mode(client, auth_headers, fake_cli):
    response = client.post(
        "/training/export", headers=auth_headers, json={"mode": "vibes"}
    )
    assert response.status_code == 422


def test_train_defaults_to_a_dry_run(client, auth_headers, fake_cli):
    """A launch takes the whole GPU; it must never be the fallthrough."""
    with _patch_run(fake_cli, {"can_run": False, "blockers": ["gpu busy"]}):
        response = client.post(
            "/training/train", headers=auth_headers, json={"train_file": "/x/train.jsonl"}
        )
    assert response.status_code == 200
    assert "--dry-run" in _args(fake_cli)


def test_train_omits_dry_run_only_when_explicitly_disabled(
    client, auth_headers, fake_cli
):
    with _patch_run(fake_cli, {"launched": True}):
        client.post(
            "/training/train",
            headers=auth_headers,
            json={"train_file": "/x/train.jsonl", "dry_run": False},
        )
    assert "--dry-run" not in _args(fake_cli)


def test_runs_clamps_the_limit_and_validates_kind(client, auth_headers, fake_cli):
    with _patch_run(fake_cli, {"runs": []}):
        client.get("/training/runs?limit=9999", headers=auth_headers)
    assert _args(fake_cli)[_args(fake_cli).index("--limit") + 1] == "200"

    response = client.get("/training/runs?kind=nonsense", headers=auth_headers)
    assert response.status_code == 400


def test_missing_repo_mount_is_a_clear_503(client, auth_headers, tmp_path, monkeypatch):
    import tools.training as training

    monkeypatch.setattr(training, "HIVE_MIND_DIR", tmp_path / "absent")
    response = client.get("/training/status", headers=auth_headers)
    assert response.status_code == 503
    assert "mounted" in response.json()["detail"]


def test_cli_failure_surfaces_its_json_error(client, auth_headers, fake_cli):
    with _patch_run(fake_cli, {"error": "database is locked"}, returncode=1):
        response = client.get("/training/status", headers=auth_headers)
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "database is locked"


def test_non_json_output_is_reported_not_swallowed(client, auth_headers, fake_cli):
    def _fake_run(command, **kwargs):
        fake_cli["calls"].append(command)
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="ModuleNotFoundError: core", stderr=""
        )

    with patch("tools.training.subprocess.run", side_effect=_fake_run):
        response = client.get("/training/status", headers=auth_headers)
    assert response.status_code == 500
    assert "non-JSON" in response.json()["detail"]


def test_timeout_is_reported_as_gateway_timeout(client, auth_headers, fake_cli):
    def _fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1800)

    with patch("tools.training.subprocess.run", side_effect=_fake_run):
        response = client.get("/training/status", headers=auth_headers)
    assert response.status_code == 504


def test_mutating_endpoints_are_hitl_gated(client, auth_headers, app, fake_cli):
    """With the gate on, a curate request waits for approval instead of running."""
    app.state.db.execute(
        "INSERT OR REPLACE INTO tool_hitl_settings (tool_name, mode) VALUES (?, ?)",
        ("training_curate", "on"),
    )
    app.state.db.commit()
    with _patch_run(fake_cli, {"kept": 1}):
        response = client.post("/training/curate", headers=auth_headers, json={})
    assert response.status_code == 202
    assert fake_cli["calls"] == []


def test_every_endpoint_requires_a_token(client):
    assert client.get("/training/status").status_code in (401, 403)
    assert client.post("/training/curate", json={}).status_code in (401, 403)
    assert client.post("/training/export", json={}).status_code in (401, 403)
    assert (
        client.post("/training/train", json={"train_file": "/x"}).status_code
        in (401, 403)
    )
