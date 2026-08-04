"""GPU arbitration around a fine-tune run.

This host has one RTX A6000. It serves inference, runs two TTS stacks, and
drives the monitor. A LoRA run needs most of it, so a launch has to take
the card, hold it, and give it back — including when nobody is awake and
including when this process dies in the middle.

Three things make that non-trivial, and each one is why a piece of this
module exists rather than being a couple of ``docker stop`` calls in the
route handler:

**The journal is written before anything is stopped.** The launch happens
inside a request whose CLI subprocess is killed at a fixed timeout, and
this container is redeployed regularly. If the record of what to restore
lived only in memory, one badly-timed restart would leave the machine mute
until someone noticed by talking to it. The journal is on the one writable
mount, fsynced, and written first.

**Ollama cannot be stopped from here.** It is a host systemd unit; this
process is unprivileged in a container with a Docker socket and nothing
else. What is reachable is a per-model unload over HTTP, which frees the
card *now* and prevents nothing — lucent embeds on every memory write, so
the model is back within seconds. The eviction pump therefore keeps
unloading on a short interval, but only until the trainer has visibly
claimed the card. After that the trainer owns the memory and Ollama's next
load simply fails or falls to CPU, which is the slow-at-night the operator
accepted. Pumping past that point would be pointless work against a
process that can no longer win.

**"Gone" and "cannot tell" are different answers.** A Docker socket that
errors is not a dead trainer. Reading it as one restores the voice stack on
top of a live run and reports a completion that did not happen, so an
unreadable daemon holds the watch open instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: Containers that hold VRAM and are expendable for the duration of a run.
#: Overridable because the set is a property of the host, not of training.
YIELD_CONTAINERS = [
    name.strip()
    for name in os.environ.get(
        "TRAINING_YIELD_CONTAINERS", "hive-mind-voice,hive-mind-voice-kokoro"
    ).split(",")
    if name.strip()
]

#: Lives on a writable mount, not in the read-only repo, and not in /tmp —
#: a restart must be able to find it.
JOURNAL_PATH = Path(
    os.environ.get(
        "TRAINING_GPU_JOURNAL",
        "/home/daniel/Storage/Dev/hive_mind/data/gpu_arbitration.json",
    )
)

OLLAMA_URL = os.environ.get("TRAINING_OLLAMA_URL", "http://host.docker.internal:11434")

#: How often the watcher asks whether the trainer is still alive.
POLL_SECONDS = float(os.environ.get("TRAINING_WATCH_POLL_SECONDS", "20"))

#: How often Ollama is re-evicted while the trainer is still loading, and
#: how long that pump runs before giving up on ever seeing an allocation.
EVICT_SECONDS = float(os.environ.get("TRAINING_EVICT_POLL_SECONDS", "10"))
EVICT_MAX_SECONDS = float(os.environ.get("TRAINING_EVICT_MAX_SECONDS", "5400"))

#: A trainer is considered to have claimed the card once it holds this much.
TRAINER_CLAIMED_MIB = int(os.environ.get("TRAINING_CLAIMED_MIB", "4096"))

DOCKER_TIMEOUT = 90

STATE_RUNNING = "running"
STATE_STOPPED = "stopped"
STATE_MISSING = "missing"
STATE_UNKNOWN = "unknown"


class DockerUnreachable(RuntimeError):
    """The daemon did not answer. Not a fact about any container."""


# ------------------------------------------------------------------ docker


def _docker(args: list[str], timeout: int = DOCKER_TIMEOUT) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise DockerUnreachable(str(exc)) from exc


def container_state(name: str) -> str:
    """``running``, ``stopped`` or ``missing``. Raises if docker is mute."""
    completed = _docker(["inspect", "--format", "{{.State.Status}}", name])
    if completed.returncode != 0:
        stderr = (completed.stderr or "").lower()
        if "no such object" in stderr or "no such container" in stderr:
            return STATE_MISSING
        raise DockerUnreachable(completed.stderr.strip()[:400])
    status = completed.stdout.strip()
    # 'restarting' and 'paused' are not running, but they are also not a
    # container someone deliberately stopped — treating them as stopped
    # would have restore start something that was already coming up.
    return STATE_RUNNING if status in {"running", "restarting"} else STATE_STOPPED


def stop_container(name: str) -> bool:
    return _docker(["stop", name]).returncode == 0


def start_container(name: str) -> bool:
    return _docker(["start", name]).returncode == 0


def trainer_container(output_name: str) -> str:
    return f"hive-trainer-{output_name}"


def trainer_status(output_name: str) -> dict[str, Any]:
    """Liveness and, once it has exited, how it ended.

    The launcher deliberately does not pass ``--rm``: the container is the
    only place an exit code survives, and a run that is reported without
    one is a run whose failure looks exactly like its success.
    """
    name = trainer_container(output_name)
    completed = _docker(
        ["inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", name]
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").lower()
        if "no such object" in stderr or "no such container" in stderr:
            return {"state": STATE_MISSING, "exit_code": None}
        raise DockerUnreachable(completed.stderr.strip()[:400])
    parts = completed.stdout.split()
    status = parts[0] if parts else ""
    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
    if status in {"running", "created", "restarting"}:
        return {"state": STATE_RUNNING, "exit_code": None}
    return {"state": STATE_STOPPED, "exit_code": exit_code}


def remove_trainer(output_name: str) -> None:
    """Drop the exited container once its outcome has been read."""
    try:
        _docker(["rm", "-f", trainer_container(output_name)])
    except DockerUnreachable:
        log.warning("training.gpu.trainer_rm_failed name=%s", output_name)


def trainer_vram_mib() -> int:
    """VRAM held by compute processes on the card, in MiB.

    Used only to answer "has the trainer actually allocated yet", which is
    what ends the eviction pump. nvidia-smi being absent answers zero, and
    the pump then runs its full budget rather than stopping early.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return 0
    if completed.returncode != 0:
        return 0
    total = 0
    for line in completed.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            total += int(value)
    return total


# ------------------------------------------------------------------ ollama


async def loaded_models(url: str = OLLAMA_URL) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{url}/api/ps")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("training.gpu.ollama_ps_failed error=%s", exc)
        return []
    return [
        entry.get("model") or entry.get("name")
        for entry in payload.get("models", [])
        if entry.get("model") or entry.get("name")
    ]


async def evict_models(url: str = OLLAMA_URL) -> list[str]:
    """Unload every currently-resident model. Returns what was evicted.

    ``keep_alive: 0`` on a generate call with no prompt is Ollama's unload
    verb. It frees the card at that instant and reserves nothing — which
    is why the caller pumps it rather than calling it once.
    """
    models = await loaded_models(url)
    evicted: list[str] = []
    if not models:
        return evicted
    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            try:
                await client.post(
                    f"{url}/api/generate", json={"model": model, "keep_alive": 0}
                )
                evicted.append(model)
            except httpx.HTTPError as exc:
                log.warning("training.gpu.evict_failed model=%s error=%s", model, exc)
    return evicted


# ----------------------------------------------------------------- journal


def read_journal(path: Path | None = None) -> dict[str, Any] | None:
    path = path or JOURNAL_PATH
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_journal(entry: dict[str, Any], path: Path | None = None) -> None:
    """Persist and fsync, so a crash one line later still leaves a record."""
    path = path or JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp")
    with staging.open("w") as handle:
        json.dump(entry, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def clear_journal(path: Path | None = None) -> None:
    path = path or JOURNAL_PATH
    try:
        path.unlink()
    except OSError:
        pass


def run_in_flight(path: Path | None = None) -> dict[str, Any] | None:
    """The journal entry of a run still holding the card, if there is one.

    A stale journal whose trainer is gone is not in flight — otherwise one
    crashed run would refuse every future launch as "GPU busy" forever.
    """
    entry = read_journal(path)
    if not entry:
        return None
    try:
        status = trainer_status(entry.get("output_name", ""))
    except DockerUnreachable:
        # Cannot tell. Refusing a launch is the recoverable answer;
        # starting a second trainer onto a busy card is not.
        return entry
    return entry if status["state"] == STATE_RUNNING else None


# --------------------------------------------------------------- preflight


async def preflight(
    output_name: str,
    run_id: str = "",
    containers: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Free the card and record what has to be given back.

    Prior state is captured *before* the first stop, and the journal is
    written before it too: a container the operator had already stopped
    must not be started later on their behalf, and a crash between the
    stop and the record would strand the machine mute.
    """
    containers = containers if containers is not None else YIELD_CONTAINERS
    prior: dict[str, str] = {}
    for name in containers:
        try:
            prior[name] = container_state(name)
        except DockerUnreachable as exc:
            log.warning("training.gpu.state_unknown container=%s error=%s", name, exc)
            prior[name] = STATE_UNKNOWN

    entry = {
        "run_id": run_id,
        "output_name": output_name,
        "started_at": int(time.time()),
        "prior": prior,
        # Only what was running is ours to restore.
        "restore": [name for name, state in prior.items() if state == STATE_RUNNING],
        "restored": False,
    }
    write_journal(entry, path)

    stopped: list[str] = []
    for name in entry["restore"]:
        try:
            if stop_container(name):
                stopped.append(name)
        except DockerUnreachable as exc:
            log.warning("training.gpu.stop_failed container=%s error=%s", name, exc)
    entry["stopped"] = stopped
    entry["evicted"] = await evict_models()
    write_journal(entry, path)
    log.info(
        "training.gpu.preflight output_name=%s stopped=%s evicted=%s",
        output_name,
        stopped,
        entry["evicted"],
    )
    return entry


async def restore(entry: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Give back exactly what preflight took. Retries, then reports.

    A restore that quietly fails is the worst outcome available here: the
    machine stays mute and every surface still looks healthy.
    """
    failures: list[str] = []
    started: list[str] = []
    for name in entry.get("restore", []):
        for attempt in range(3):
            try:
                if start_container(name):
                    started.append(name)
                    break
            except DockerUnreachable as exc:
                log.warning("training.gpu.start_failed container=%s error=%s", name, exc)
            await asyncio.sleep(2 * (attempt + 1))
        else:
            failures.append(name)
    entry["restored"] = True
    entry["restored_at"] = int(time.time())
    entry["restore_failures"] = failures
    if failures:
        # Keep the journal: reconcile on the next start will try again.
        entry["restored"] = False
        write_journal(entry, path)
    else:
        clear_journal(path)
    log.info("training.gpu.restore started=%s failures=%s", started, failures)
    return {"started": started, "failures": failures}


# ------------------------------------------------------------------- watch


async def evict_until_claimed(
    poll_seconds: float = EVICT_SECONDS,
    max_seconds: float = EVICT_MAX_SECONDS,
    claimed_mib: int = TRAINER_CLAIMED_MIB,
) -> bool:
    """Keep Ollama off the card until the trainer has taken it.

    Returns True if an allocation was observed. The window that matters is
    between the unload and the trainer's first CUDA allocation — minutes,
    while the image and the base weights load — because after that the
    trainer holds the memory and Ollama can no longer take it.
    """
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if trainer_vram_mib() >= claimed_mib:
            return True
        await evict_models()
        await asyncio.sleep(poll_seconds)
    return False


async def watch(
    entry: dict[str, Any],
    notify=None,
    poll_seconds: float = POLL_SECONDS,
    path: Path | None = None,
) -> dict[str, Any]:
    """Wait for the trainer to end, then restore and report how it ended."""
    output_name = entry.get("output_name", "")
    asyncio.ensure_future(evict_until_claimed())
    exit_code: int | None = None
    while True:
        try:
            status = trainer_status(output_name)
        except DockerUnreachable as exc:
            # Not an answer. Hold the watch rather than restoring onto a
            # trainer that may well still be running.
            log.warning("training.gpu.docker_unreachable error=%s", exc)
            await asyncio.sleep(poll_seconds)
            continue
        if status["state"] == STATE_RUNNING:
            await asyncio.sleep(poll_seconds)
            continue
        exit_code = status["exit_code"]
        break

    if exit_code is not None:
        remove_trainer(output_name)
    outcome = "succeeded" if exit_code == 0 else "failed"
    if exit_code is None:
        # Container vanished before anyone read it — someone removed it by
        # hand, or an older launch used --rm. Say so rather than guessing.
        outcome = "ended, outcome unknown"
    result = await restore(entry, path)
    if notify is not None:
        await safe_notify(
            notify,
            f"training run {output_name} {outcome}"
            + (f" (exit {exit_code})" if exit_code is not None else "")
            + (
                "; voice is back"
                if not result["failures"]
                else f"; RESTORE FAILED for {', '.join(result['failures'])}"
            ),
        )
    return {"outcome": outcome, "exit_code": exit_code, **result}


async def safe_notify(notify, message: str) -> None:
    """A failed notification must never take the restore down with it."""
    try:
        outcome = notify(message)
        if asyncio.iscoroutine(outcome):
            await outcome
    except Exception as exc:  # noqa: BLE001 - transport of any kind may fail
        log.warning("training.gpu.notify_failed error=%s", exc)


async def notify_owner(message: str) -> bool:
    """Tell the operator directly over Telegram.

    Deliberately not routed through the gateway: these messages are about
    the machine being in an unusual state, and the gateway is one of the
    things that may be down or restarting when that is true. The bot
    token and chat id are already in this container's environment.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_OWNER_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("training.gpu.notify_unconfigured")
        return False
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("training.gpu.notify_failed error=%s", exc)
        return False
    return True


async def reconcile(notify=None, path: Path | None = None) -> dict[str, Any]:
    """Pick the watch back up after this process restarted.

    A journal is only ever present between a preflight and its restore, so
    finding one means a run was in flight when we died. A trainer still
    alive gets its watcher back; a trainer gone gets restored immediately.
    A machine with no journal restores nothing — containers the operator
    stopped themselves are none of our business.
    """
    entry = read_journal(path)
    if not entry:
        return {"action": "nothing-to-do"}
    try:
        status = trainer_status(entry.get("output_name", ""))
    except DockerUnreachable:
        asyncio.ensure_future(watch(entry, notify=notify, path=path))
        return {"action": "resumed-watch", "reason": "docker unreachable"}
    if status["state"] == STATE_RUNNING:
        asyncio.ensure_future(watch(entry, notify=notify, path=path))
        return {"action": "resumed-watch"}
    result = await watch(entry, notify=notify, poll_seconds=0, path=path)
    return {"action": "restored", **result}
