# Plan — REQ-038: Ollama classification endpoint in hive-tools

> Adds `POST /ollama/classify` to the existing hive-tools FastAPI service.
> Derived from `/home/daniel/Storage/Dev/spark_to_bloom/src/backlog/memory-system-implementation-skippy.md` REQ-038.

## Working environment

- Project root: `/home/daniel/Storage/hive-tools/`
- venv/binary path: hive-tools runs in Docker; for local pytest use the system python (or the existing test runner). `pyproject.toml` configures pytest, mypy, ruff. `requirements.txt` already lists `httpx`.
- Auth pattern: every router endpoint requires bearer auth via `Depends(require_api_token)` (called inside `hitl_gate`).
- HITL pattern: `caller: str = Depends(hitl_gate("<tool_name>"))` injects the caller mind name and gates on the row in `tool_hitl_settings`.
- Test pattern: see `tests/api/test_linkedin.py` for the canonical router-test fixture set (app, client, auth_header).

## Scope

**Add** a new tool router `tools/ollama.py` that exposes `POST /ollama/classify`.
The endpoint takes a transcript chunk plus a list of data-class specs and returns
the Ollama model's classification verdict as JSON.

The Ollama server is already running on the local network at
`http://192.168.4.64:11434`. Use the `/api/generate` HTTP API with
`format=json` and `stream=false` so the response is a single JSON object.

Default model: `gpt-oss:20b-32k` (overridable via `OLLAMA_CLASSIFY_MODEL` env var).
This model is installed on the shared Ollama server and was selected for the
Skippy memory-system bake-off (REQ-036/037). Spec'd choice — do not swap
without updating this doc, `memory-system-implementation-skippy.md`, and the
playbook (`memory-system-fixes-skippy.md`).
Default base URL: `http://192.168.4.64:11434` (overridable via `OLLAMA_BASE_URL`).

## Steps

### 1. Create `tools/ollama.py`

```python
"""Ollama classification router for hive-tools.

Exposes POST /ollama/classify — wraps Ollama's /api/generate with format=json
so the model returns a structured classification verdict.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hitl import hitl_gate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ollama", tags=["ollama"])

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.4.64:11434")
OLLAMA_CLASSIFY_MODEL = os.environ.get("OLLAMA_CLASSIFY_MODEL", "gpt-oss:20b-32k")
CLASSIFY_TIMEOUT_SECONDS = 60


class DataClassSpec(BaseModel):
    class_name: str
    body: str


class ClassifyRequest(BaseModel):
    chunk: str
    data_class_specs: list[DataClassSpec]


@router.post("/classify")
async def classify(
    body: ClassifyRequest,
    caller: str = Depends(hitl_gate("ollama_classify")),
) -> dict[str, Any]:
    """Classify a transcript chunk against the given data-class specs.

    Returns the model's JSON verdict as a dict. The model is instructed to
    emit one of the data-class names (or null) plus a short reason and the
    suggested action (save-vector / save-graph / notify / discard).
    """
    prompt = _build_prompt(body.chunk, body.data_class_specs)
    try:
        async with httpx.AsyncClient(timeout=CLASSIFY_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_CLASSIFY_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        log.exception("Ollama classify call failed")
        raise HTTPException(status_code=502, detail=f"ollama call failed: {e}") from e

    payload = resp.json()
    raw = payload.get("response", "")
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ollama returned non-JSON response: %r", raw[:200])
        return {
            "data_class": None,
            "reason": "ollama returned non-JSON",
            "action": "discard",
            "raw_response": raw,
        }
    return verdict


def _build_prompt(chunk: str, specs: list[DataClassSpec]) -> str:
    spec_text = "\n\n".join(f"## {s.class_name}\n{s.body}" for s in specs)
    return (
        "Classify the following content against the data class specs below. "
        "Return JSON with exactly these keys: "
        "{\"data_class\": <name>|null, \"reason\": <short>, "
        "\"action\": save-vector|save-graph|notify|discard}\n\n"
        f"### Content\n{chunk}\n\n### Specs\n{spec_text}"
    )
```

### 2. Register the router in `server.py`

Add a try/except include block alongside the others (gmail, linkedin, etc.):

```python
try:
    from tools.ollama import router as ollama_router

    app.include_router(ollama_router)
except ImportError:
    pass
```

Place it after the `browser` block and before the `ui` block.

### 3. Seed the HITL setting in `db.py`

Append to `_DEFAULT_HITL_SETTINGS`:

```python
# Ollama classification — read-only side effects, no approval needed
("ollama_classify", "off", 15),
```

Place after the docker block. The seed is idempotent (`INSERT OR IGNORE`), so
existing deployments will pick this up on next boot.

### 4. Tests — `tests/api/test_ollama.py`

Mirror the linkedin/gmail test patterns. Cover:

1. **Auth required** — POST without bearer returns 401.
2. **HITL off (default after seeding) → 200** — endpoint returns JSON verdict.
   Mock `httpx.AsyncClient.post` to return a canned response payload
   `{"response": "{\\"data_class\\": \\"preference\\", \\"reason\\": \\"...\\", \\"action\\": \\"save-vector\\"}"}`.
   Assert response body parses to expected dict.
3. **HITL on → 202** — set `tool_hitl_settings.mode='on'`, POST returns 202 with `hitl_request_id`.
4. **Ollama returns non-JSON → graceful fallback** — mock returns `{"response": "garbage text"}`. Endpoint returns
   `{"data_class": None, "action": "discard", "raw_response": "garbage text", ...}` with status 200.
5. **Ollama HTTP error → 502** — mock raises `httpx.HTTPError`. Endpoint returns 502.

Test fixtures:

```python
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

@pytest.fixture()
def app(tmp_path: Path):
    from server import create_app
    return create_app(db_path=str(tmp_path / "test.db"))

@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)

@pytest.fixture()
def auth_header(app):
    from auth_api import generate_token
    raw = generate_token(app.state.db, "skippy")
    return {"Authorization": f"Bearer {raw}"}

SAMPLE_REQ = {
    "chunk": "Daniel said his birthday is March 14",
    "data_class_specs": [
        {"class_name": "person", "body": "Facts about people..."},
        {"class_name": "timed-event", "body": "Dated events..."},
    ],
}
```

Use `patch("tools.ollama.httpx.AsyncClient")` to inject an async mock with a
context manager that returns a Response-like object whose `.json()` returns
the canned payload and `.raise_for_status()` is a no-op.

### 5. Container rebuild

The hive-tools service runs in Docker. After the code is in place locally:
- `docker compose -f /home/daniel/Storage/Dev/hive_mind/docker-compose.yml build hive-tools`
- `docker compose -f /home/daniel/Storage/Dev/hive_mind/docker-compose.yml up -d hive-tools`

(This is a follow-up step after code-genius PASS — the rebuild requires the user's
docker access. Document in STATE that the rebuild is pending.)

## Validation

```
cd /home/daniel/Storage/hive-tools
python -m pytest tests/api/test_ollama.py tests/unit/test_db.py tests/unit/test_hitl_defaults.py -v
python -m ruff check tools/ollama.py db.py server.py tests/api/test_ollama.py
python -m mypy tools/ollama.py
```

Expect: all new tests pass; pre-existing tests still pass; ruff clean; mypy clean
on the new file (the rest of hive-tools may have pre-existing mypy noise that is
outside this scope).

## Out of scope

- Any change to Skippy or hive-mind — REQ-038 is purely hive-tools.
- The `/auto-remember` skill that calls this endpoint — that's REQ-034..REQ-037,
  separate batch.
- Any change to the existing Ollama deployment.
- Container rebuild — post-PASS step, not part of this batch.
