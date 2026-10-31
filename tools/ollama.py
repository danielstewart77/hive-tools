"""Ollama generic-LLM router for hive-tools.

Three endpoints, model-agnostic:

- POST /ollama/structured — structured output. Caller supplies a JSON schema;
  the schema is passed straight to Ollama's `format` field (Ollama v0.5+
  schema-constrained sampling). Output is guaranteed valid JSON conforming
  to the schema. Same contract as OpenAI's `response_format.json_schema`.

- POST /ollama/chat — conversational text output. Caller supplies a prompt
  (and optional system prompt). Returns the model's text response verbatim.

- GET /ollama/models — list models the configured Ollama instance is serving.
  Powers each mind's `/models` slash command. Hive-tools is the single
  wrapper around Ollama regardless of where Ollama runs (host, container,
  remote), so every mind asks the same shape and gets the same answer.

The chat/structured endpoints accept `model` (defaults to
OLLAMA_DEFAULT_MODEL env or `gpt-oss:20b-32k`). HITL gating is
per-endpoint.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth_api import require_api_token
from hitl import hitl_gate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ollama", tags=["ollama"])

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "gpt-oss:20b-32k")
OLLAMA_TIMEOUT_SECONDS = 90


class StructuredRequest(BaseModel):
    prompt: str
    json_schema: dict[str, Any] = Field(alias="schema")
    model: str | None = None
    system: str | None = None

    model_config = {"populate_by_name": True}


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None
    system: str | None = None


async def _ollama_chat(
    *, model: str, prompt: str, system: str | None, fmt: dict[str, Any] | str | None
) -> dict[str, Any]:
    """Call Ollama's /api/chat with messages format.

    /api/chat (not /api/generate) is required for thinking models like
    gpt-oss — it implements the two-request dance that produces valid
    structured output. /api/generate emits empty responses for thinking
    models when a format constraint is active. See ollama/ollama PR #12460
    (merged into chat) and #14288 (open for generate).
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if fmt is not None:
        body["format"] = fmt
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=body)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        log.exception("Ollama call failed")
        raise HTTPException(status_code=502, detail=f"ollama call failed: {e}") from e
    return resp.json()


@router.post("/structured")
async def structured(
    body: StructuredRequest,
    caller: str = Depends(hitl_gate("ollama_structured")),
) -> dict[str, Any]:
    """Schema-constrained generation. Returns the parsed JSON object directly.

    On non-conforming output (rare with schema-constrained sampling but
    possible if the model fails), returns the raw string under `_raw` and
    an `_error` marker so callers can debug.
    """
    del caller
    model = body.model or OLLAMA_DEFAULT_MODEL
    payload = await _ollama_chat(
        model=model, prompt=body.prompt, system=body.system, fmt=body.json_schema
    )
    raw = payload.get("message", {}).get("content", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ollama structured response was not valid JSON: %r", raw[:300])
        return {"_raw": raw, "_error": "non-json response from schema-constrained call"}


@router.post("/chat")
async def chat(
    body: ChatRequest,
    caller: str = Depends(hitl_gate("ollama_chat")),
) -> dict[str, str]:
    """Plain-text generation. Returns `{"response": "<text>"}`."""
    del caller
    model = body.model or OLLAMA_DEFAULT_MODEL
    payload = await _ollama_chat(
        model=model, prompt=body.prompt, system=body.system, fmt=None
    )
    return {"response": payload.get("message", {}).get("content", "")}


@router.get("/models")
async def models(caller: str = Depends(require_api_token)) -> dict[str, Any]:
    """List the models served by the configured Ollama instance.

    Read-only metadata; no HITL gate. Each mind's `/models` slash-command
    calls this so the user can see what's available without poking the
    Ollama server directly. Wraps Ollama's `/api/tags`.

    Returns:
        {
            "ollama_base_url": "http://...",
            "default_model": "gpt-oss:20b-32k",
            "models": [
                {"name": "...", "size": 12345, "modified_at": "...", "family": "..."},
                ...
            ]
        }

    On Ollama unreachable: 502 with the error detail. Empty model list is
    not an error — Ollama just hasn't pulled anything yet.
    """
    del caller
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError as e:
        log.exception("Ollama /api/tags failed")
        raise HTTPException(status_code=502, detail=f"ollama unreachable: {e}") from e

    raw = resp.json().get("models", [])
    flat = [
        {
            "name": m.get("name", ""),
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
            "family": (m.get("details") or {}).get("family"),
            "parameter_size": (m.get("details") or {}).get("parameter_size"),
            "quantization": (m.get("details") or {}).get("quantization_level"),
        }
        for m in raw
    ]
    return {
        "ollama_base_url": OLLAMA_BASE_URL,
        "default_model": OLLAMA_DEFAULT_MODEL,
        "models": flat,
    }
