"""Unit tests for GPU arbitration around a fine-tune run.

Docker and nvidia-smi are subprocess calls and Ollama is HTTP, so the seams
are patched — but every test here is written to fail against an
implementation that merely *issues* the right calls. What is asserted is
the state the machine is left in: which containers are up, whether the
journal survives, and whether a run that ended is reported as having ended
the way it actually did.
"""

from __future__ import annotations

import json

import pytest

from tools import training_gpu as gpu


class FakeDocker:
    """A Docker daemon with container states you can move underneath it."""

    def __init__(self, states: dict[str, str], unreachable: bool = False):
        self.states = dict(states)
        self.unreachable = unreachable
        self.calls: list[list[str]] = []
        self.start_failures: set[str] = set()

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        if self.unreachable:
            raise gpu.DockerUnreachable("dial unix /var/run/docker.sock: permission denied")
        verb = args[0]
        if verb == "inspect":
            name = args[-1]
            if name not in self.states:
                return _Completed(1, "", f"Error: No such object: {name}")
            state = self.states[name]
            if "{{.State.ExitCode}}" in args[2]:
                status, _, code = state.partition(":")
                return _Completed(0, f"{status} {code or 0}", "")
            return _Completed(0, state.partition(":")[0], "")
        if verb == "stop":
            self.states[args[-1]] = "exited"
            return _Completed(0, "", "")
        if verb == "start":
            if args[-1] in self.start_failures:
                return _Completed(1, "", "start failed")
            self.states[args[-1]] = "running"
            return _Completed(0, "", "")
        if verb == "rm":
            self.states.pop(args[-1], None)
            return _Completed(0, "", "")
        return _Completed(0, "", "")


class _Completed:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def journal(tmp_path):
    return tmp_path / "gpu_arbitration.json"


@pytest.fixture()
def no_ollama(monkeypatch):
    """Ollama absent by default; tests that care patch it explicitly."""
    evicted: list[str] = []

    async def fake_evict(url=None):
        evicted.append("called")
        return ["qwen3-embedding:8b"]

    monkeypatch.setattr(gpu, "evict_models", fake_evict)
    return evicted


# ------------------------------------------------------------- requirement 1


async def test_preflight_stops_the_voice_stack_and_evicts_ollama(
    monkeypatch, journal, no_ollama
):
    docker = FakeDocker({"voice": "running", "kokoro": "running"})
    monkeypatch.setattr(gpu, "_docker", docker)

    entry = await gpu.preflight("run-a", containers=["voice", "kokoro"], path=journal)

    assert docker.states == {"voice": "exited", "kokoro": "exited"}
    assert no_ollama == ["called"]
    assert entry["restore"] == ["voice", "kokoro"]


async def test_the_journal_is_written_before_anything_is_stopped(
    monkeypatch, journal, no_ollama
):
    """A crash between the stop and the record leaves the machine mute."""
    seen = {}

    docker = FakeDocker({"voice": "running"})

    def watching_docker(args, timeout=None):
        if args[0] == "stop":
            seen["journal_at_stop_time"] = journal.exists()
        return docker(args, timeout)

    monkeypatch.setattr(gpu, "_docker", watching_docker)
    await gpu.preflight("run-a", containers=["voice"], path=journal)

    assert seen["journal_at_stop_time"] is True


# ------------------------------------------------------------- requirement 4


async def test_restore_starts_only_what_preflight_stopped(monkeypatch, journal, no_ollama):
    """A container the operator had already stopped stays stopped."""
    docker = FakeDocker({"voice": "running", "kokoro": "exited"})
    monkeypatch.setattr(gpu, "_docker", docker)

    entry = await gpu.preflight("run-a", containers=["voice", "kokoro"], path=journal)
    result = await gpu.restore(entry, path=journal)

    assert result["started"] == ["voice"]
    assert docker.states["kokoro"] == "exited"
    assert not journal.exists()


async def test_prior_state_is_captured_before_the_stop_not_after(
    monkeypatch, journal, no_ollama
):
    """Capturing after the stops would read everything as already-down."""
    docker = FakeDocker({"voice": "running", "kokoro": "running"})
    monkeypatch.setattr(gpu, "_docker", docker)

    entry = await gpu.preflight("run-a", containers=["voice", "kokoro"], path=journal)

    assert sorted(entry["restore"]) == ["kokoro", "voice"]


# ------------------------------------------------------------ requirement 13


async def test_a_failed_restore_keeps_the_journal_and_reports_it(
    monkeypatch, journal, no_ollama
):
    """Silence here means a mute machine that looks healthy everywhere."""
    docker = FakeDocker({"voice": "running"})
    docker.start_failures = {"voice"}
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu.asyncio, "sleep", _instant)

    entry = await gpu.preflight("run-a", containers=["voice"], path=journal)
    result = await gpu.restore(entry, path=journal)

    assert result["failures"] == ["voice"]
    assert journal.exists()
    assert json.loads(journal.read_text())["restored"] is False


# ------------------------------------------------------------- requirement 5


async def test_a_crashed_trainer_restores_and_is_reported_as_failed(
    monkeypatch, journal, no_ollama
):
    docker = FakeDocker({"voice": "exited", "hive-trainer-run-a": "exited:137"})
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu, "evict_until_claimed", _noop_coro)
    messages: list[str] = []

    entry = {"output_name": "run-a", "restore": ["voice"]}
    result = await gpu.watch(entry, notify=messages.append, poll_seconds=0, path=journal)

    assert result["outcome"] == "failed"
    assert result["exit_code"] == 137
    assert docker.states["voice"] == "running"
    assert "failed" in messages[0]


async def test_a_clean_exit_is_reported_as_succeeded(monkeypatch, journal, no_ollama):
    docker = FakeDocker({"voice": "exited", "hive-trainer-run-a": "exited:0"})
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu, "evict_until_claimed", _noop_coro)
    messages: list[str] = []

    entry = {"output_name": "run-a", "restore": ["voice"]}
    result = await gpu.watch(entry, notify=messages.append, poll_seconds=0, path=journal)

    assert result["outcome"] == "succeeded"
    assert "succeeded" in messages[0]
    assert "voice is back" in messages[0]


# ------------------------------------------------------------- requirement 6


async def test_reconcile_restores_when_the_trainer_is_gone(monkeypatch, journal, no_ollama):
    docker = FakeDocker({"voice": "exited"})
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu, "evict_until_claimed", _noop_coro)
    gpu.write_journal({"output_name": "run-a", "restore": ["voice"]}, journal)

    outcome = await gpu.reconcile(path=journal)

    assert outcome["action"] == "restored"
    assert docker.states["voice"] == "running"
    assert not journal.exists()


async def test_reconcile_waits_when_the_trainer_is_still_alive(
    monkeypatch, journal, no_ollama
):
    docker = FakeDocker({"voice": "exited", "hive-trainer-run-a": "running"})
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu, "evict_until_claimed", _noop_coro)
    gpu.write_journal({"output_name": "run-a", "restore": ["voice"]}, journal)

    outcome = await gpu.reconcile(path=journal)

    assert outcome["action"] == "resumed-watch"
    assert docker.states["voice"] == "exited"
    assert journal.exists()


async def test_a_host_with_no_journal_restores_nothing(monkeypatch, journal):
    """Containers the operator stopped themselves are none of our business."""
    docker = FakeDocker({"voice": "exited"})
    monkeypatch.setattr(gpu, "_docker", docker)

    outcome = await gpu.reconcile(path=journal)

    assert outcome["action"] == "nothing-to-do"
    assert docker.states["voice"] == "exited"


async def test_an_unreachable_docker_is_not_read_as_a_finished_run(
    monkeypatch, journal, no_ollama
):
    """Restoring onto a live trainer is the failure this prevents."""
    docker = FakeDocker({"voice": "exited"}, unreachable=True)
    monkeypatch.setattr(gpu, "_docker", docker)
    monkeypatch.setattr(gpu, "evict_until_claimed", _noop_coro)
    gpu.write_journal({"output_name": "run-a", "restore": ["voice"]}, journal)

    outcome = await gpu.reconcile(path=journal)

    assert outcome["action"] == "resumed-watch"
    assert docker.states["voice"] == "exited"


# ------------------------------------------------------------ requirement 11


async def test_a_live_run_is_in_flight_and_a_dead_one_is_not(monkeypatch, journal):
    monkeypatch.setattr(gpu, "JOURNAL_PATH", journal)
    gpu.write_journal({"output_name": "run-a", "restore": []}, journal)

    monkeypatch.setattr(gpu, "_docker", FakeDocker({"hive-trainer-run-a": "running"}))
    assert gpu.run_in_flight(journal) is not None

    monkeypatch.setattr(gpu, "_docker", FakeDocker({"hive-trainer-run-a": "exited:0"}))
    assert gpu.run_in_flight(journal) is None


# --------------------------------------------------------- eviction pumping


async def test_the_eviction_pump_stops_once_the_trainer_has_the_card(monkeypatch):
    """Pumping past the allocation is work against a process that lost."""
    readings = iter([0, 0, 30_000])
    monkeypatch.setattr(gpu, "trainer_vram_mib", lambda: next(readings))
    calls: list[int] = []

    async def fake_evict(url=None):
        calls.append(1)
        return []

    monkeypatch.setattr(gpu, "evict_models", fake_evict)
    monkeypatch.setattr(gpu.asyncio, "sleep", _instant)

    assert await evict_once(gpu) is True
    assert len(calls) == 2


async def evict_once(module):
    return await module.evict_until_claimed(poll_seconds=0, max_seconds=60)


async def test_a_failing_notification_never_blocks_the_restore(monkeypatch):
    def explode(_message):
        raise RuntimeError("telegram is down")

    await gpu.safe_notify(explode, "anything")


async def _instant(_seconds):
    return None


async def _noop_coro(*args, **kwargs):
    return True
