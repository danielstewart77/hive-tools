"""Shared test fixtures for hive-tools."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _block_outbound_telegram() -> Generator[None, None, None]:
    """Keep HITL approval prompts inside the test run.

    Any request to a HITL-gated endpoint whose mode is ``on`` creates an
    approval request and calls ``notify_daniel``, which POSTs to the real
    Telegram API using the bot token in the environment. A test suite that
    exercises gated endpoints therefore pages a human — once per test, with
    no context explaining why.

    The training launcher has its own notifier — ``training_gpu.notify_owner``
    talks to Telegram directly rather than through HITL — and the launch path
    arms a watcher holding a reference to it. Emptying the bot credentials for
    the whole run makes that notifier short-circuit at its own unconfigured
    check, which covers every notification path rather than the two we
    happen to know the names of.

    Tests that assert on notification behaviour patch ``httpx.post``
    themselves and are unaffected; this only stops the ones that never meant
    to send anything.
    """
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_OWNER_CHAT_ID": ""}):
        with patch("hitl.notify_daniel", return_value=None):
            yield


@pytest.fixture(autouse=True)
def _no_real_docker_from_the_gpu_arbiter(tmp_path: Path) -> Generator[None, None, None]:
    """Keep the arbiter's side effects off this machine.

    ``training_gpu`` stops the voice containers, evicts Ollama's models and
    writes a journal at a path baked in from the environment. A test that
    reaches the launch path without patching those does all of it for real,
    to the host running the tests. Anything genuinely exercising docker
    patches ``_docker`` or ``trainer_status`` itself and never gets here.
    """
    import tools.training_gpu as training_gpu

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "training_gpu shelled out to docker during a test; patch _docker "
            "or trainer_status in the test that needs it"
        )

    with patch.object(training_gpu.subprocess, "run", _refuse):
        with patch.object(training_gpu, "JOURNAL_PATH", tmp_path / "gpu_arbitration.json"):
            yield


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary database path (file does not exist yet)."""
    return tmp_path / "test.db"


@pytest.fixture()
def db_conn(tmp_db: Path) -> Generator[sqlite3.Connection, None, None]:
    """Open the store on a temporary database and hand back its connection.

    The operations are module-level and async now, so a unit test needs the
    store open rather than a connection of its own. The connection yielded
    is the one the store is using, so a test that sets up a row directly
    cannot set up state the operations would not see.
    """
    import asyncio

    import store

    asyncio.run(store.open(str(tmp_db)))
    connection = store.sqlite_connection()
    assert connection is not None
    try:
        yield connection
    finally:
        asyncio.run(store.close())


def pytest_pyfunc_call(pyfuncitem):
    """Run coroutine tests without pulling in a plugin for it.

    The suite is otherwise synchronous, and one async module is not worth a
    dependency whose default event-loop scoping would then apply to every
    test in the repo.
    """
    import asyncio
    import inspect

    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    asyncio.run(func(**kwargs))
    return True
