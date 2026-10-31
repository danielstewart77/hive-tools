"""Do the two halves of the store answer the same thing?

`store.py` has two backends and the whole claim made for keeping both is
that unsetting one environment variable returns the service to the behaviour
it had. Nothing was testing that. The mutation reviewer made the point
sharply: `dsn_configured()` could be hard-wired to `False` and
`on_postgres()` to `True` - "`HIVE_DATABASE_URL` set means Postgres" broken
outright - with the suite entirely green, because no test anywhere set that
variable.

`hive_data`'s own suite guards each `tools` operation against a real
Postgres. What no suite guarded is the *wiring*: which `hive_data` function
each store operation calls, with which argument order, and which key it
reads off the result (`["rows"]`, `["row"]`). That rested on reading
`store.py`.

So every test here runs one sequence twice - once against a temporary
SQLite file, once against a scratch Postgres database this module creates
and drops - and compares what came back. A difference is the finding; the
value itself is secondary.

Skipped, cleanly, when no Postgres is named. `HIVE_DATA_TEST_ADMIN_DSN` says
where one is, with no default, because a default would be a credential in a
tracked file.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import pytest

import store

ADMIN_DSN = os.environ.get("HIVE_DATA_TEST_ADMIN_DSN", "").strip()

NO_SERVER = (
    "HIVE_DATA_TEST_ADMIN_DSN is unset. It names a Postgres this suite may "
    "create and drop a database on. Without it the two halves cannot be "
    "compared and these tests are skipped rather than passing on one half."
)

#: The schemas to apply to the scratch database, as applied, vendored in the
#: access layer. Read from the installed package rather than copied here, so
#: a schema change arrives instead of being silently out of date.
#:
#: `core.sql` as well as `tools.sql`, although hive-tools touches no `core`
#: table: the access layer refuses to open a pool against a database with no
#: `core` schema at all, on the grounds that such a database is not this
#: hive's. The scratch database has to be a believable one.
def _schemas() -> list[str]:
    import hive_data

    directory = Path(hive_data.__file__).parent / "schema"
    return [(directory / name).read_text()
            for name in ("core.sql", "tools.sql")]


async def _on_sqlite(sequence: Callable[[], Awaitable[Any]],
                     db_path: Path) -> Any:
    """Run a sequence against a temporary SQLite file."""
    os.environ.pop(store.DSN_VAR, None)
    await store.open(str(db_path))
    try:
        assert not store.on_postgres()
        return await sequence()
    finally:
        await store.close()


async def _on_postgres(sequence: Callable[[], Awaitable[Any]]) -> Any:
    """Run the same sequence against a scratch Postgres database."""
    import asyncpg

    name = f"hive_tools_parity_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    base, _, _ = ADMIN_DSN.rpartition("/")
    dsn = f"{base}/{name}"
    try:
        target = await asyncpg.connect(dsn)
        try:
            for schema in _schemas():
                await target.execute(schema)
        finally:
            await target.close()
        os.environ[store.DSN_VAR] = dsn
        await store.open()
        try:
            assert store.on_postgres()
            return await sequence()
        finally:
            await store.close()
    finally:
        os.environ.pop(store.DSN_VAR, None)
        admin = await asyncpg.connect(ADMIN_DSN)
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()", name)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()


def both_halves(sequence: Callable[[], Awaitable[Any]],
                db_path: Path) -> tuple[Any, Any]:
    """What each half answered, for the same sequence of operations.

    Run in separate event loops, one after the other, because each half
    opens and closes its own store and the Postgres pool belongs to the loop
    it was opened on.
    """
    if not ADMIN_DSN:
        pytest.skip(NO_SERVER)
    on_file = asyncio.run(_on_sqlite(sequence, db_path))
    on_server = asyncio.run(_on_postgres(sequence))
    return on_file, on_server


# ---------------------------------------------------------------------------


def test_the_two_halves_decide_a_request_the_same_way(tmp_path: Path) -> None:
    """The decision, its guard, and what it hands back to replay. Three
    things the commit changed deliberately, on the one path where the two
    halves running different SQL would be invisible."""
    async def sequence() -> list[Any]:
        at = 1_800_000_000
        await store.create_request(
            "req-live", "docker_compose_up", "skippy", "{}", at, at + 300,
            "POST", "/docker/up", '{"path": "/opt/thing"}')
        await store.create_request(
            "req-stale", "docker_compose_up", "skippy", "{}", at, at + 300,
            "POST", "/docker/up", "{}")

        won = await store.decide_request("req-live", "approved", at + 1, at)
        lost = await store.decide_request("req-live", "denied", at + 2, at)
        expired = await store.decide_request(
            "req-stale", "approved", at + 999, at + 999)
        missing = await store.decide_request("nope", "approved", at, at)
        return [won, lost, expired, missing,
                await store.request("req-live"),
                await store.request("req-stale")]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    # And the behaviour itself, so this is not two wrong answers agreeing.
    won, lost, expired, missing, live, stale = on_file
    assert won is not None and won["tool_body"] == '{"path": "/opt/thing"}'
    assert lost is None and expired is None and missing is None
    assert live["status"] == "approved"
    assert stale["status"] == "pending"


def test_the_two_halves_revoke_a_token_the_same_way(tmp_path: Path) -> None:
    """The count is the arbitration, and it is the thing a caller branches
    on - so the two halves disagreeing about it is a revocation reported as
    a repeat, or the other way round."""
    async def sequence() -> list[Any]:
        at = 1_800_000_000
        await store.create_token("console", "digest-one", at)
        token = await store.live_token("digest-one")
        first = await store.revoke_token(token["id"], at + 1)
        second = await store.revoke_token(token["id"], at + 2)
        return [token["name"], first, second,
                await store.live_token("digest-one"),
                await store.revoke_token(9999, at + 3)]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    assert on_file == ["console", 1, 0, None, 0]


def test_the_two_halves_seed_a_mode_the_same_way(tmp_path: Path) -> None:
    """Insert-if-absent on both, and the count is which tools were new. A
    replacing write on either half re-arms every approval gate the operator
    turned off, on every restart."""
    async def sequence() -> list[Any]:
        first = await store.seed_setting("gmail_send_email", "on", 15)
        await store.set_setting("gmail_send_email", "off", 60)
        second = await store.seed_setting("gmail_send_email", "on", 15)
        return [first, second, await store.setting("gmail_send_email"),
                await store.setting("never_registered"),
                await store.settings()]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    first, second, edited, absent, listing = on_file
    assert [first, second] == [1, 0]
    assert edited == {"tool_name": "gmail_send_email", "mode": "off",
                      "sudo_timeout_minutes": 60}
    assert absent is None
    assert listing == [edited]


def test_the_two_halves_list_the_queue_with_the_same_columns(
        tmp_path: Path) -> None:
    """The row shape the approvals page renders. The captured bodies are not
    in it: on the live data the newest fifty rows carry megabytes of them,
    some holding credentials, and the page shows none of it. The replay
    reads them by name instead."""
    async def sequence() -> list[Any]:
        at = 1_800_000_000
        for index in range(3):
            await store.create_request(
                f"req-{index}", "gmail_send_email", "skippy",
                '{"secret": "hunter2"}', at + index, at + 300,
                "POST", "/gmail/send", '{"token": "hunter2"}')
        listing = await store.recent_requests()
        return [sorted(listing[0]), [row["id"] for row in listing],
                [row["id"] for row in await store.recent_requests(limit=2)]]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    columns, ids, capped = on_file
    # Every column of a request except the two the access layer declares
    # private - the captured body, which is what holds the credentials.
    assert columns == ["caller_mind", "created_at", "decided_at",
                       "expires_at", "id", "status", "tool_method",
                       "tool_name", "tool_path"]
    # Newest first, and the tie-break is the id so the same list comes back
    # twice. Three requests one second apart, so no tie here - the ordering
    # that matters is the direction.
    assert ids == ["req-2", "req-1", "req-0"]
    assert capped == ["req-2", "req-1"]


def test_the_two_halves_grant_sudo_the_same_way(tmp_path: Path) -> None:
    """The window's boundary. A grant expiring at this instant has run out,
    which is strictly-after on both halves or a mind is let through a gate
    one second after its grant died."""
    async def sequence() -> list[Any]:
        at = 1_800_000_000
        await store.grant_sudo("docker_compose_up", "skippy", at, at + 900)
        live = await store.active_grant("docker_compose_up", "skippy", at)
        boundary = await store.active_grant(
            "docker_compose_up", "skippy", at + 900)
        other_mind = await store.active_grant(
            "docker_compose_up", "mordecai", at)
        return [live["expires_at"], boundary, other_mind]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    assert on_file == [1_800_000_000 + 900, None, None]


def test_the_two_halves_read_a_login_the_same_way(tmp_path: Path) -> None:
    """Every authenticated page load resolves a cookie to a user id and reads
    the login back, so a disagreement here is every session on the service."""
    async def sequence() -> list[Any]:
        at = 1_800_000_000
        created = await store.create_user("daniel", "argon2-hash", 1, at)
        by_name = await store.user_by_username("daniel")
        by_id = await store.user_by_id(created["id"])
        return [sorted(created), created["username"], created["is_admin"],
                by_name == created, by_id == created,
                await store.user_by_username("nobody"),
                await store.user_by_id(9999),
                await store.count_users()]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    columns, username, is_admin, by_name, by_id, no_name, no_id, count = \
        on_file
    assert columns == ["created_at", "disabled_at", "id", "is_admin",
                       "password_hash", "username"]
    assert (username, is_admin) == ("daniel", 1)
    assert by_name and by_id
    assert no_name is None and no_id is None
    assert count == 1


def test_a_unit_of_work_is_undone_together_on_both_halves(
        tmp_path: Path) -> None:
    """The construct the decision depends on. A transaction that committed
    anyway on one half would leave that half with the half-applied state the
    unit exists to prevent, and the suite could not tell."""
    class Deliberate(Exception):
        pass

    async def sequence() -> list[Any]:
        at = 1_800_000_000
        try:
            async with store.transaction():
                await store.create_request(
                    "rolled-back", "gmail_send_email", "skippy", "{}", at,
                    at + 300, "POST", "/gmail/send", "{}")
                await store.grant_sudo("gmail_send_email", "skippy", at,
                                       at + 900)
                raise Deliberate
        except Deliberate:
            pass

        async with store.transaction():
            await store.create_request(
                "committed", "gmail_send_email", "skippy", "{}", at,
                at + 300, "POST", "/gmail/send", "{}")
        return [await store.request("rolled-back") is None,
                await store.active_grant("gmail_send_email", "skippy", at),
                await store.request("committed") is not None]

    on_file, on_server = both_halves(sequence, tmp_path / "parity.db")
    assert on_file == on_server
    assert on_file == [True, None, True]
