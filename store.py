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
statements in a worker thread: the file is on local disk and the statements
are small, but blocking the event loop for any of them would stall every
other request the process is serving.

The composite rules - when an approval has expired, whether a decision
grants sudo, what a replay needs - are not here. They live in `hitl.py`,
once, on top of these operations, so the two backends cannot disagree about
them.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from typing import Any

#: The variable that decides which store answers. Named rather than derived,
#: because the revert is an operator action and has to be one obvious edit.
DSN_VAR = "HIVE_DATABASE_URL"

#: The newest approvals the queue page shows. The page has no paging, so a
#: larger number is a longer page rather than more information.
QUEUE_LIMIT = 50

Row = dict[str, Any]

_sqlite: sqlite3.Connection | None = None
_postgres = False


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
    global _sqlite, _postgres
    if dsn_configured():
        import hive_data

        await hive_data.connect()
        _postgres = True
        _sqlite = None
        return
    from db import init_db

    _postgres = False
    _sqlite = await asyncio.to_thread(init_db, db_path)


async def close() -> None:
    global _sqlite, _postgres
    if _postgres:
        import hive_data

        await hive_data.close(drain_timeout=5.0)
        _postgres = False
        return
    connection, _sqlite = _sqlite, None
    if connection is not None:
        await asyncio.to_thread(connection.close)


def sqlite_connection() -> sqlite3.Connection | None:
    """The SQLite connection this store is using, or None on Postgres.

    For the length of the migration only, and for one caller: the test suite
    sets up rows directly rather than through a route. It is the same
    connection the operations use, so a test cannot set up state the store
    would not see. It goes when the SQLite half goes.
    """
    return _sqlite


def _connection() -> sqlite3.Connection:
    if _sqlite is None:
        raise RuntimeError(
            "hive-tools' store was never opened. The application's lifespan "
            "opens it; a caller reaching a database before that is a route "
            "running outside the app.")
    return _sqlite


async def _one(sql: str, args: tuple[Any, ...] = ()) -> Row | None:
    def run() -> Row | None:
        row = _connection().execute(sql, args).fetchone()
        return dict(row) if row is not None else None

    return await asyncio.to_thread(run)


async def _all(sql: str, args: tuple[Any, ...] = ()) -> list[Row]:
    def run() -> list[Row]:
        return [dict(r) for r in _connection().execute(sql, args).fetchall()]

    return await asyncio.to_thread(run)


async def _write(sql: str, args: tuple[Any, ...] = ()) -> int:
    def run() -> int:
        connection = _connection()
        cursor = connection.execute(sql, args)
        connection.commit()
        return cursor.rowcount

    return await asyncio.to_thread(run)


# -------------------------------------------------------------------- users

async def count_users() -> int:
    if _postgres:
        from hive_data import tools

        return len(await tools.users())
    row = await _one("SELECT COUNT(*) AS cnt FROM users")
    return int(row["cnt"]) if row else 0


async def user_by_username(username: str) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.user(username)
    return await _one(
        "SELECT id, username, password_hash, is_admin, created_at, "
        "disabled_at FROM users WHERE username = ?", (username,))


async def user_by_id(user_id: int) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.user_by_id(user_id)
    return await _one(
        "SELECT id, username, password_hash, is_admin, created_at, "
        "disabled_at FROM users WHERE id = ?", (user_id,))


async def create_user(username: str, password_hash: str, is_admin: int,
                      created_at: int) -> Row | None:
    if _postgres:
        from hive_data import tools

        await tools.create_user(username, password_hash, is_admin, created_at)
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

        await tools.create_token(name, token_hash, created_at)
        return
    await _write(
        "INSERT INTO api_tokens (name, token_hash, created_at) "
        "VALUES (?, ?, ?)", (name, token_hash, created_at))


async def live_token(token_hash: str) -> Row | None:
    """The token a digest belongs to, while it is still valid."""
    if _postgres:
        from hive_data import tools

        return await tools.live_token(token_hash)
    return await _one(
        "SELECT id, name FROM api_tokens "
        "WHERE token_hash = ? AND revoked_at IS NULL", (token_hash,))


async def record_token_use(token_id: int, at: int) -> None:
    if _postgres:
        from hive_data import tools

        await tools.record_token_use(token_id, at)
        return
    await _write("UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
                 (at, token_id))


async def tokens() -> list[Row]:
    """Every token, newest first, and never the digests."""
    if _postgres:
        from hive_data import tools

        return await tools.tokens()
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

        return int((await tools.revoke_token(token_id, at))["rows"])
    return await _write(
        "UPDATE api_tokens SET revoked_at = ? "
        "WHERE id = ? AND revoked_at IS NULL", (at, token_id))


# ----------------------------------------------------------------- settings

async def settings() -> list[Row]:
    if _postgres:
        from hive_data import tools

        return await tools.hitl_settings()
    return await _all(
        "SELECT tool_name, mode, sudo_timeout_minutes "
        "FROM tool_hitl_settings ORDER BY tool_name")


async def setting(tool_name: str) -> Row | None:
    if _postgres:
        from hive_data import tools

        return await tools.hitl_setting(tool_name)
    return await _one(
        "SELECT tool_name, mode, sudo_timeout_minutes "
        "FROM tool_hitl_settings WHERE tool_name = ?", (tool_name,))


async def set_setting(tool_name: str, mode: str,
                      sudo_timeout_minutes: int) -> None:
    """The operator's own edit: replace whatever is there."""
    if _postgres:
        from hive_data import tools

        await tools.set_hitl_setting(tool_name, mode, sudo_timeout_minutes)
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
            tool_name, mode, sudo_timeout_minutes))["rows"])
    return await _write(
        "INSERT OR IGNORE INTO tool_hitl_settings (tool_name, mode, "
        "sudo_timeout_minutes) VALUES (?, ?, ?)",
        (tool_name, mode, sudo_timeout_minutes))


# ---------------------------------------------------------------- approvals

async def create_request(request_id: str, tool_name: str, caller_mind: str,
                         params_json: str, created_at: int, expires_at: int,
                         tool_method: str | None, tool_path: str | None,
                         tool_body: str | None) -> None:
    if _postgres:
        from hive_data import tools

        await tools.create_request(
            request_id, tool_name, caller_mind, params_json, created_at,
            expires_at, tool_method=tool_method, tool_path=tool_path,
            tool_body=tool_body)
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

        return await tools.request(request_id)
    return await _one(
        "SELECT id, tool_name, caller_mind, status, created_at, expires_at, "
        "decided_at FROM hitl_requests WHERE id = ?", (request_id,))


async def recent_requests(limit: int = QUEUE_LIMIT) -> list[Row]:
    """The queue the approvals page renders: newest first, capped.

    Without the captured bodies. On the live data the newest fifty rows
    carry megabytes of them, some holding credentials, and the page shows
    none of it.
    """
    if _postgres:
        from hive_data import tools

        return await tools.recent_requests(limit=limit)
    return await _all(
        "SELECT id, tool_name, caller_mind, status, created_at, expires_at, "
        "decided_at FROM hitl_requests ORDER BY created_at DESC LIMIT ?",
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

        written = await tools.decide_request(request_id, status, decided_at,
                                             now)
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
        connection.commit()
        return dict(row) if row is not None else None

    return await asyncio.to_thread(run)


# ------------------------------------------------------------------- grants

async def active_grant(tool_name: str, caller_mind: str,
                       now: int) -> Row | None:
    """A sudo grant still live at `now`, for this mind and this tool."""
    if _postgres:
        from hive_data import tools

        return await tools.active_grant(tool_name, caller_mind, now)
    return await _one(
        "SELECT id, tool_name, caller_mind, granted_at, expires_at "
        "FROM sudo_grants WHERE tool_name = ? AND caller_mind = ? "
        "AND expires_at > ?", (tool_name, caller_mind, now))


async def grant_sudo(tool_name: str, caller_mind: str, granted_at: int,
                     expires_at: int) -> None:
    if _postgres:
        from hive_data import tools

        await tools.grant_sudo(tool_name, caller_mind, granted_at,
                               expires_at)
        return
    await _write(
        "INSERT INTO sudo_grants (tool_name, caller_mind, granted_at, "
        "expires_at) VALUES (?, ?, ?, ?)",
        (tool_name, caller_mind, granted_at, expires_at))
