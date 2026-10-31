"""Where hive-tools' data lives, as one async surface with two backends.

Nothing else in hive-tools opens a database or writes SQL. Every route calls
a named operation here, and which store answers is one value in the
environment: `HIVE_DATABASE_URL` set means the hive's Postgres through
`hive_data`, unset means this host's own `hivetools.db` as before.

Two backends rather than one, for the length of the migration only. The
SQLite file is left on disk and reachable, so a service that misbehaves on
Postgres is reverted by unsetting one variable and restarting rather than by
restoring a backup. Retiring the SQLite half is its own story and does not
happen until Postgres has run in anger.

Async, because `hive_data` is async only and a synchronous face would mean
every query existing twice forever. The SQLite backend therefore runs its
statements off the event loop: the file is on local disk and the statements
are small, but blocking the loop for any of them would stall every other
request the process is serving.

Off the loop on **one** thread, not on the default pool. There is a single
`sqlite3.Connection` and a `sqlite3.Connection` is not safe to use from two
threads at once whatever `check_same_thread` says: two statements interleaved
on it return each other's rows, raise `bad parameter or other API misuse`,
and - measured through the live routes - reject a valid bearer token with a
401 because the token read came back with another request's row. The whole
point of this half is to be the revert lever, so it has to be at least as
sound as the synchronous code it replaced, which was serialised on the event
loop by construction. One worker restores exactly that, and it is also what
makes `check_same_thread=False` honest: the connection is touched by one
thread for its whole life.

The composite rules - when an approval has expired, whether a decision
grants sudo, what a replay needs - are not here. They live in `hitl.py`,
once, on top of these operations, so the two backends cannot disagree about
them.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, TypeVar

#: The variable that decides which store answers. Named rather than derived,
#: because the revert is an operator action and has to be one obvious edit.
DSN_VAR = "HIVE_DATABASE_URL"

#: The newest approvals the queue page shows. The page has no paging, so a
#: larger number is a longer page rather than more information.
QUEUE_LIMIT = 50

Row = dict[str, Any]

_sqlite: sqlite3.Connection | None = None
_postgres = False
#: The one thread every SQLite statement runs on. See the module docstring.
_worker: ThreadPoolExecutor | None = None
#: Whether this store was opened and then closed, as against never opened.
#: The two are different sentences: the first is a straggling background task
#: reaching a store the lifespan has already shut down, which is ordinary at
#: shutdown, and the second is a route running outside the app.
_was_open = False

#: The unit of work the current task is inside, if any. A context variable
#: rather than a parameter on thirty operations, so a call site reads the same
#: whether or not it is in a transaction - which is what keeps the two halves
#: from diverging at every caller.
_unit: contextvars.ContextVar[Any] = contextvars.ContextVar("unit",
                                                            default=None)

T = TypeVar("T")


def on_postgres() -> bool:
    """Which store is answering. For the status the health route reports."""
    return _postgres


def dsn_configured() -> bool:
    return bool(os.environ.get(DSN_VAR, "").strip())


async def open(db_path: str | None = None) -> None:
    """Open whichever store this process is pointed at.

    Called from the application's lifespan rather than at import, because
    opening a pool is async and because a module that connected on import
    would do it again in every test that imported it.
    """
    global _sqlite, _postgres, _worker, _was_open
    _was_open = True
    if dsn_configured():
        import hive_data

        await hive_data.connect()
        _postgres = True
        _sqlite = None
        return
    from db import init_db

    _postgres = False
    _worker = ThreadPoolExecutor(max_workers=1,
                                 thread_name_prefix="hivetools-sqlite")
    _sqlite = await _off_loop(lambda: init_db(db_path))


async def close() -> None:
    global _sqlite, _postgres, _worker
    if _postgres:
        import hive_data

        await hive_data.close(drain_timeout=5.0)
        _postgres = False
        return
    connection, _sqlite = _sqlite, None
    worker, _worker = _worker, None
    if connection is not None and worker is not None:
        await asyncio.get_running_loop().run_in_executor(
            worker, connection.close)
    if worker is not None:
        worker.shutdown(wait=True)


def sqlite_connection() -> sqlite3.Connection | None:
    """The SQLite connection this store is using, or None on Postgres.

    For the length of the migration only, and for one caller: the test suite
    sets up rows directly rather than through a route. It is the same
    connection the operations use, so a test cannot set up state the store
    would not see. It goes when the SQLite half goes.
    """
    return _sqlite


def _unopened() -> RuntimeError:
    """Why there is no store, said as the thing that actually happened.

    A store that was opened and closed is a straggling background task - the
    fire-and-forget replay, the training watcher - reaching a database after
    the lifespan shut it down, which is ordinary at shutdown. A store that
    was never opened is a route running outside the app. Folding the two
    together sends somebody hunting the second when it was the first.
    """
    if _was_open:
        return RuntimeError(
            "hive-tools' store is closed. The lifespan shut it down, so this "
            "is work that outlived the application - a background replay or a "
            "watcher still running after shutdown - rather than a route "
            "reaching a database too early.")
    return RuntimeError(
        "hive-tools' store was never opened. The application's lifespan "
        "opens it; a caller reaching a database before that is a route "
        "running outside the app.")


def _connection() -> sqlite3.Connection:
    if _sqlite is None:
        raise _unopened()
    return _sqlite


async def _off_loop(run: Callable[[], T]) -> T:
    """Run one piece of SQLite work on the store's single worker thread."""
    if _worker is None:
        raise _unopened()
    return await asyncio.get_running_loop().run_in_executor(_worker, run)


async def _one(sql: str, args: tuple[Any, ...] = ()) -> Row | None:
    def run() -> Row | None:
        row = _connection().execute(sql, args).fetchone()
        return dict(row) if row is not None else None

    return await _off_loop(run)


async def _all(sql: str, args: tuple[Any, ...] = ()) -> list[Row]:
    def run() -> list[Row]:
        return [dict(r) for r in _connection().execute(sql, args).fetchall()]

    return await _off_loop(run)


async def _write(sql: str, args: tuple[Any, ...] = ()) -> int:
    def run() -> int:
        connection = _connection()
        # No `commit()`. The connection is in autocommit mode, so a
        # statement outside a transaction has already landed, and one inside
        # the store's own unit of work is the unit's to commit.
        return connection.execute(sql, args).rowcount

    return await _off_loop(run)


class transaction:
    """Several operations, all of them or none, on whichever store answers.

    Both halves, because a construct that only worked on one of them would
    be a caller that reads the same and behaves differently depending on an
    environment variable - which is the one thing the revert lever must not
    be.

    It exists because deciding an approval is not one write. The decision,
    the mode it reads and the sudo grant it may insert were three statements
    under one commit before this store did, and committed separately they
    leave an end state nobody can recover from: the request says approved,
    the call it authorised was never replayed, and because the decision is
    guarded on `pending` it can never be approved again.

    A class rather than an `@asynccontextmanager`, and the reason is the
    access layer refusing the generator version outright. `hive_data`
    refuses to open a unit of work inside an async generator, because a
    generator yields to its caller while holding one pooled connection
    inside an open transaction and a caller that abandons it - a closed
    browser tab on a streaming reply - leaves the transaction to unwind
    whenever the event loop's asyncgen finalizer reaches it. A generator
    wrapper around that unit is the same hazard with this module's name on
    it, and the guard could not tell the two apart because there is nothing
    to tell apart. `__aenter__` and `__aexit__` are ordinary coroutines and
    there is no generator to finalize.

    The unit travels in a context variable, so the operations inside need no
    extra argument and every call site reads the same in or out of one. A
    task created inside a unit inherits it - `hive_data.Unit` refuses to be
    used from another task, which is the protection wanted rather than a
    limitation.
    """

    def __init__(self) -> None:
        self._scope: Any = None
        self._token: Any = None

    async def __aenter__(self) -> None:
        if _unit.get() is not None:
            raise RuntimeError(
                "hive-tools' store does not nest transactions. SQLite has no "
                "savepoint here and the inner block would commit the outer "
                "one's work on its way out, so a failure after it would roll "
                "back nothing. Open one unit at the top of the operation.")
        if _postgres:
            import hive_data

            self._scope = hive_data.transaction()
            self._token = _unit.set(await self._scope.__aenter__())
            return None
        await _off_loop(lambda: _connection().execute("BEGIN IMMEDIATE"))
        self._token = _unit.set(_SQLITE_UNIT)
        return None

    async def __aexit__(self, kind: Any, value: Any, trace: Any) -> bool:
        # The context variable is reset before the commit or the rollback, so
        # the statement that ends the unit is not itself run as part of one.
        _unit.reset(self._token)
        if self._scope is not None:
            return bool(await self._scope.__aexit__(kind, value, trace))
        await _off_loop(
            lambda: _connection().execute(
                "ROLLBACK" if kind is not None else "COMMIT"))
        return False


#: What the context variable holds on the SQLite half. The connection is
#: module state, so there is nothing to carry - but "a unit is open" still
#: has to be distinguishable from "no unit", which is what this sentinel is.
_SQLITE_UNIT = object()


def _pg_unit() -> Any:
    """The `hive_data` unit this task is inside, or None."""
    unit = _unit.get()
    return None if unit is _SQLITE_UNIT else unit


# -------------------------------------------------------------------- users

async def count_users() -> int:
    if _postgres:
        from hive_data import tools

        return await tools.count_users(unit=_pg_unit())
    row = await _one("SELECT COUNT(*) AS cnt FROM users")
    return int(row["cnt"]) if row else 0


async def user_by_username(username: str) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.user(username, unit=_pg_unit())
    return await _one(
        "SELECT id, username, password_hash, is_admin, created_at, "
        "disabled_at FROM users WHERE username = ?", (username,))


async def user_by_id(user_id: int) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.user_by_id(user_id, unit=_pg_unit())
    return await _one(
        "SELECT id, username, password_hash, is_admin, created_at, "
        "disabled_at FROM users WHERE id = ?", (user_id,))


async def create_user(username: str, password_hash: str, is_admin: int,
                      created_at: int) -> Row | None:
    if _postgres:
        from hive_data import tools

        await tools.create_user(username, password_hash, is_admin,
                                created_at, unit=_pg_unit())
    else:
        await _write(
            "INSERT INTO users (username, password_hash, is_admin, "
            "created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, is_admin, created_at))
    return await user_by_username(username)


# ------------------------------------------------------------------- tokens

async def create_token(name: str, token_hash: str, created_at: int) -> None:
    if _postgres:
        from hive_data import tools

        await tools.create_token(name, token_hash, created_at,
                                 unit=_pg_unit())
        return
    await _write(
        "INSERT INTO api_tokens (name, token_hash, created_at) "
        "VALUES (?, ?, ?)", (name, token_hash, created_at))


async def live_token(token_hash: str) -> Row | None:
    """The token a digest belongs to, while it is still valid."""
    if _postgres:
        from hive_data import tools

        return await tools.live_token(token_hash, unit=_pg_unit())
    return await _one(
        "SELECT id, name FROM api_tokens "
        "WHERE token_hash = ? AND revoked_at IS NULL", (token_hash,))


async def record_token_use(token_id: int, at: int) -> None:
    if _postgres:
        from hive_data import tools

        await tools.record_token_use(token_id, at, unit=_pg_unit())
        return
    await _write("UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
                 (at, token_id))


async def tokens() -> list[Row]:
    """Every token, newest first, and never the digests."""
    if _postgres:
        from hive_data import tools

        return await tools.tokens(unit=_pg_unit())
    return await _all(
        "SELECT id, name, created_at, last_used_at, revoked_at "
        "FROM api_tokens ORDER BY created_at DESC")


async def revoke_token(token_id: int, at: int) -> int:
    """Revoke a token if it is not revoked already, reporting what changed.

    `IS NULL`, not `= NULL`: the second is never true, so the statement
    would match nothing, report zero, and leave the bearer authenticating
    while the page said it was gone.
    """
    if _postgres:
        from hive_data import tools

        return int((await tools.revoke_token(
            token_id, at, unit=_pg_unit()))["rows"])
    return await _write(
        "UPDATE api_tokens SET revoked_at = ? "
        "WHERE id = ? AND revoked_at IS NULL", (at, token_id))


# ----------------------------------------------------------------- settings

async def settings() -> list[Row]:
    if _postgres:
        from hive_data import tools

        return await tools.hitl_settings(unit=_pg_unit())
    return await _all(
        "SELECT tool_name, mode, sudo_timeout_minutes "
        "FROM tool_hitl_settings ORDER BY tool_name")


async def setting(tool_name: str) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.hitl_setting(tool_name, unit=_pg_unit())
    return await _one(
        "SELECT tool_name, mode, sudo_timeout_minutes "
        "FROM tool_hitl_settings WHERE tool_name = ?", (tool_name,))


async def set_setting(tool_name: str, mode: str,
                      sudo_timeout_minutes: int) -> None:
    """The operator's own edit: replace whatever is there."""
    if _postgres:
        from hive_data import tools

        await tools.set_hitl_setting(tool_name, mode, sudo_timeout_minutes,
                                     unit=_pg_unit())
        return
    await _write(
        "INSERT INTO tool_hitl_settings (tool_name, mode, "
        "sudo_timeout_minutes) VALUES (?, ?, ?) "
        "ON CONFLICT(tool_name) DO UPDATE SET mode = excluded.mode, "
        "sudo_timeout_minutes = excluded.sudo_timeout_minutes",
        (tool_name, mode, sudo_timeout_minutes))


async def seed_setting(tool_name: str, mode: str,
                       sudo_timeout_minutes: int) -> int:
    """Give a tool a mode only if it has none, reporting whether it was new.

    Insert-if-absent, deliberately. This runs on every boot for every gated
    tool, and an operator who turned a gate off expects it to stay off
    across a restart.
    """
    if _postgres:
        from hive_data import tools

        return int((await tools.seed_hitl_setting(
            tool_name, mode, sudo_timeout_minutes,
            unit=_pg_unit()))["rows"])
    return await _write(
        "INSERT OR IGNORE INTO tool_hitl_settings (tool_name, mode, "
        "sudo_timeout_minutes) VALUES (?, ?, ?)",
        (tool_name, mode, sudo_timeout_minutes))


# ---------------------------------------------------------------- approvals

#: Every column of a request except the two the access layer declares
#: private: `params_json` and `tool_body`, which are the captured body of
#: every gated call a person was asked to approve. On the live data the
#: newest fifty rows carry megabytes of them, some holding credentials, and
#: the queue page shows none of it. The replay reads them by name.
#:
#: Named once, and `tool_method` and `tool_path` are in it because the
#: Postgres half returns them - every non-private column - and these two
#: reads were handing back seven columns on one half and nine on the other.
#: Nothing today renders either, so the page is unchanged; what changes is
#: that a caller reading a row off the store gets the same row whichever
#: half answered, which is the entire claim made for keeping both.
_REQUEST_COLUMNS = (
    "SELECT id, tool_name, caller_mind, status, created_at, expires_at, "
    "decided_at, tool_method, tool_path FROM hitl_requests")


async def create_request(request_id: str, tool_name: str, caller_mind: str,
                         params_json: str, created_at: int, expires_at: int,
                         tool_method: str | None, tool_path: str | None,
                         tool_body: str | None) -> None:
    if _postgres:
        from hive_data import tools

        await tools.create_request(
            request_id, tool_name, caller_mind, params_json, created_at,
            expires_at, tool_method=tool_method, tool_path=tool_path,
            tool_body=tool_body, unit=_pg_unit())
        return
    await _write(
        "INSERT INTO hitl_requests (id, tool_name, caller_mind, params_json, "
        "tool_method, tool_path, tool_body, status, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
        (request_id, tool_name, caller_mind, params_json, tool_method,
         tool_path, tool_body, created_at, expires_at))


async def request(request_id: str) -> Row | None:
    """One request, without the call it captured."""
    if _postgres:
        from hive_data import tools

        return await tools.request(request_id, unit=_pg_unit())
    return await _one(_REQUEST_COLUMNS + " WHERE id = ?", (request_id,))


async def recent_requests(limit: int = QUEUE_LIMIT) -> list[Row]:
    """The queue the approvals page renders: newest first, capped.

    Without the captured bodies. On the live data the newest fifty rows
    carry megabytes of them, some holding credentials, and the page shows
    none of it.
    """
    if _postgres:
        from hive_data import tools

        return await tools.recent_requests(limit=limit, unit=_pg_unit())
    # `id DESC` under `created_at DESC`, matching the Postgres listing. The
    # timestamp is whole seconds, so two requests made in the same second tie
    # - the live table has such a pair inside the newest fifty - and SQL
    # leaves the order of equal keys undefined. The two halves ordering a tie
    # the same way by accident is not the halves agreeing.
    return await _all(
        _REQUEST_COLUMNS + " ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,))


async def decide_request(request_id: str, status: str, decided_at: int,
                         now: int) -> Row | None:
    """Decide a request still waiting at `now`, and hand back the call.

    The guard is in the statement on both backends. A request already
    decided, or past its expiry, changes nothing and answers None - so two
    people reaching the queue at the same moment produce one decision and
    the loser is told it lost.
    """
    if _postgres:
        from hive_data import tools

        written = await tools.decide_request(
            request_id, status, decided_at, now, unit=_pg_unit())
        return written["row"] if written["rows"] else None

    def run() -> Row | None:
        connection = _connection()
        cursor = connection.execute(
            "UPDATE hitl_requests SET status = ?, decided_at = ? "
            "WHERE id = ? AND status = 'pending' AND expires_at >= ? "
            "RETURNING id, tool_name, caller_mind, params_json, tool_method, "
            "tool_path, tool_body",
            (status, decided_at, request_id, now))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    return await _off_loop(run)


# ------------------------------------------------------------------- grants

async def active_grant(tool_name: str, caller_mind: str,
                       now: int) -> Row | None:
    """A sudo grant still live at `now`, for this mind and this tool."""
    if _postgres:
        from hive_data import tools

        return await tools.active_grant(tool_name, caller_mind, now,
                                        unit=_pg_unit())
    return await _one(
        "SELECT id, tool_name, caller_mind, granted_at, expires_at "
        "FROM sudo_grants WHERE tool_name = ? AND caller_mind = ? "
        "AND expires_at > ?", (tool_name, caller_mind, now))


async def grant_sudo(tool_name: str, caller_mind: str, granted_at: int,
                     expires_at: int) -> None:
    if _postgres:
        from hive_data import tools

        await tools.grant_sudo(tool_name, caller_mind, granted_at,
                               expires_at, unit=_pg_unit())
        return
    await _write(
        "INSERT INTO sudo_grants (tool_name, caller_mind, granted_at, "
        "expires_at) VALUES (?, ?, ?, ?)",
        (tool_name, caller_mind, granted_at, expires_at))
