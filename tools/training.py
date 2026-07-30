"""Training pipeline router for hive-tools.

The training corpus, the curation engine and the fine-tune launcher all live
in the hive_mind repo, because that is where the capture hook writes and
where the data contract is kept. The console lives in its own container and
cannot import any of it. This router is the seam between them.

It does not reimplement the pipeline. It shells out to hive_mind's stateless
CLI — ``tools/stateless/training_pipeline/training_pipeline.py``, stdlib
only by design — and passes the JSON straight back. Every behavioural
decision therefore stays in one place, next to the data and its tests, and
this file stays a transport.

Endpoints mirror the CLI subcommands:

- ``GET  /training/status``  — corpus counts, GPU state, latest run per kind
- ``GET  /training/runs``    — ledger history
- ``POST /training/curate``  — run a curation pass (or reset verdicts)
- ``POST /training/export``  — write train/eval JSONL
- ``POST /training/train``   — plan or launch a LoRA fine-tune

Curation and export are minutes-long on the real corpus, so both run in a
worker thread with a generous timeout rather than blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth_api import require_api_token
from hitl import hitl_gate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

HIVE_MIND_DIR = Path(
    os.environ.get("HIVE_MIND_DIR", "/home/daniel/Storage/Dev/hive_mind")
)
CLI_RELATIVE = Path("tools/stateless/training_pipeline/training_pipeline.py")

# Curation walks every row and hashes it; export renders several thousand
# turns to JSONL. Both are minutes, not seconds, on the live corpus.
LONG_TIMEOUT_SECONDS = 1_800
SHORT_TIMEOUT_SECONDS = 120


def _python_binary() -> str:
    """Prefer hive_mind's venv, fall back to whatever interpreter exists.

    The CLI is stdlib-only precisely so this fallback is safe: if the venv
    has not been created on this host, a bare python3 still runs it.
    """
    venv_python = HIVE_MIND_DIR / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _cli_path() -> Path:
    return HIVE_MIND_DIR / CLI_RELATIVE


def _run_cli(args: list[str], timeout: int) -> dict[str, Any]:
    """Invoke the pipeline CLI and return its JSON object.

    The CLI's contract is one JSON object on stdout and a non-zero exit on
    failure. A non-JSON body means the interpreter itself failed (missing
    mount, syntax error), which is worth surfacing verbatim rather than
    swallowing into a generic 500.
    """
    cli = _cli_path()
    if not cli.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"training pipeline CLI not found at {cli}. "
                "Is the hive_mind repo mounted into hive-tools?"
            ),
        )
    command = [_python_binary(), str(cli), *args]
    log.info("training.cli.invoke cmd=%s", shlex.join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=str(HIVE_MIND_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504, detail=f"training command timed out after {timeout}s"
        ) from None
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise HTTPException(
            status_code=500,
            detail=(completed.stderr or "no output from training CLI").strip()[:2000],
        )
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail=f"non-JSON output from training CLI: {stdout[:1000]}"
        ) from None
    if completed.returncode != 0:
        raise HTTPException(status_code=500, detail=payload)
    return payload


async def _run_cli_async(args: list[str], timeout: int) -> dict[str, Any]:
    return await asyncio.to_thread(_run_cli, args, timeout)


class CurateRequest(BaseModel):
    keep_per_cluster: int = Field(
        3,
        ge=1,
        le=1000,
        description=(
            "How many turns survive per near-duplicate cluster. The single "
            "most consequential knob: 1 discards genuine variety in a "
            "recurring task, unbounded lets scheduled skills dominate."
        ),
    )
    min_length_tokens: int = Field(60, ge=0)
    max_length_tokens: int = Field(32_000, ge=512)
    max_tool_error_ratio: float = Field(0.6, ge=0.0, le=1.0)
    require_tool_call: bool = False
    exclude_secret_rows: bool = False
    harness: list[str] | None = None
    reset: bool = Field(
        False, description="Return every row to pending instead of curating."
    )


class ExportRequest(BaseModel):
    mode: str = Field("stripped", pattern="^(stripped|reasoning)$")
    name: str = Field("latest", pattern=r"^[A-Za-z0-9._-]{1,64}$")
    out_dir: str | None = None
    harness: list[str] | None = None
    require_reasoning: bool = False
    eval_fraction: float = Field(0.05, ge=0.0, lt=1.0)
    include_system_prompt: bool = True
    max_tool_result_chars: int = Field(8_000, ge=0)
    secrets: str = Field(
        "keep",
        pattern="^(keep|randomize|redact)$",
        description=(
            "How credentials leave the corpus. keep (default) writes the "
            "real values, because this dataset trains a locally-hosted "
            "model that needs them. randomize substitutes a same-length, "
            "same-shape surrogate so the model learns what a token looks "
            "like without learning a real one — the right choice when a "
            "dataset leaves this machine. redact writes placeholders, which "
            "teaches the model to emit a slug where a live token belongs."
        ),
    )


class TrainRequest(BaseModel):
    train_file: str
    eval_file: str | None = None
    out_dir: str | None = None
    name: str = Field("hive-harness-lora", pattern=r"^[A-Za-z0-9._-]{1,64}$")
    base_model: str | None = None
    image: str | None = None
    lora_rank: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(1e-4, gt=0, lt=1)
    epochs: int = Field(2, ge=1, le=20)
    max_sequence_length: int = Field(8_192, ge=512)
    dry_run: bool = Field(
        True,
        description=(
            "Default true. A launch consumes the whole GPU, so starting one "
            "is an explicit act rather than the fallthrough."
        ),
    )


@router.get("/status", dependencies=[Depends(require_api_token)])
async def training_status() -> dict[str, Any]:
    """Corpus counts, curation breakdown, GPU state, latest run per kind."""
    return await _run_cli_async(["status"], SHORT_TIMEOUT_SECONDS)


@router.get("/runs", dependencies=[Depends(require_api_token)])
async def training_runs(kind: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Ledger history, newest first."""
    if kind and kind not in {"curate", "export", "train"}:
        raise HTTPException(status_code=400, detail=f"unknown run kind {kind!r}")
    args = ["runs", "--limit", str(max(1, min(limit, 200)))]
    if kind:
        args += ["--kind", kind]
    return await _run_cli_async(args, SHORT_TIMEOUT_SECONDS)


@router.post("/curate")
async def training_curate(
    body: CurateRequest,
    caller: str = Depends(hitl_gate("training_curate")),
) -> dict[str, Any]:
    """Run a curation pass, writing verdicts in place. Never deletes rows."""
    del caller
    if body.reset:
        return await _run_cli_async(["curate", "--reset"], LONG_TIMEOUT_SECONDS)
    args = [
        "curate",
        "--keep-per-cluster",
        str(body.keep_per_cluster),
        "--min-length-tokens",
        str(body.min_length_tokens),
        "--max-length-tokens",
        str(body.max_length_tokens),
        "--max-tool-error-ratio",
        str(body.max_tool_error_ratio),
    ]
    if body.require_tool_call:
        args.append("--require-tool-call")
    if body.exclude_secret_rows:
        args.append("--exclude-secret-rows")
    for harness in body.harness or []:
        args += ["--harness", harness]
    return await _run_cli_async(args, LONG_TIMEOUT_SECONDS)


@router.post("/export")
async def training_export(
    body: ExportRequest,
    caller: str = Depends(hitl_gate("training_export")),
) -> dict[str, Any]:
    """Render kept rows to train/eval JSONL with credentials redacted."""
    del caller
    args = [
        "export",
        "--mode",
        body.mode,
        "--name",
        body.name,
        "--eval-fraction",
        str(body.eval_fraction),
        "--max-tool-result-chars",
        str(body.max_tool_result_chars),
    ]
    if body.out_dir:
        args += ["--out-dir", body.out_dir]
    if body.require_reasoning:
        args.append("--require-reasoning")
    if not body.include_system_prompt:
        args.append("--no-system-prompt")
    args += ["--secrets", body.secrets]
    for harness in body.harness or []:
        args += ["--harness", harness]
    return await _run_cli_async(args, LONG_TIMEOUT_SECONDS)


@router.post("/train")
async def training_train(
    body: TrainRequest,
    caller: str = Depends(hitl_gate("training_train")),
) -> dict[str, Any]:
    """Plan a fine-tune, or launch one when ``dry_run`` is false."""
    del caller
    args = [
        "train",
        "--train-file",
        body.train_file,
        "--name",
        body.name,
        "--lora-rank",
        str(body.lora_rank),
        "--learning-rate",
        str(body.learning_rate),
        "--epochs",
        str(body.epochs),
        "--max-sequence-length",
        str(body.max_sequence_length),
    ]
    if body.eval_file:
        args += ["--eval-file", body.eval_file]
    if body.out_dir:
        args += ["--out-dir", body.out_dir]
    if body.base_model:
        args += ["--base-model", body.base_model]
    if body.image:
        args += ["--image", body.image]
    if body.dry_run:
        args.append("--dry-run")
    return await _run_cli_async(args, SHORT_TIMEOUT_SECONDS)
