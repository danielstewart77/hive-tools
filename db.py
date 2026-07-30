"""Database initialization and connection helpers for HiveTools."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DB_PATH


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Create all tables and return a connection.

    Uses CREATE TABLE IF NOT EXISTS so it is safe to call repeatedly.
    """
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            disabled_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS api_tokens (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL,
            last_used_at INTEGER,
            revoked_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS tool_hitl_settings (
            tool_name TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'off',
            sudo_timeout_minutes INTEGER NOT NULL DEFAULT 15
        );

        CREATE TABLE IF NOT EXISTS hitl_requests (
            id TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            caller_mind TEXT NOT NULL,
            params_json TEXT NOT NULL DEFAULT '{}',
            tool_method TEXT,
            tool_path TEXT,
            tool_body TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            decided_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS sudo_grants (
            id INTEGER PRIMARY KEY,
            tool_name TEXT NOT NULL,
            caller_mind TEXT NOT NULL,
            granted_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL
        );
        """
    )
    # Migrate existing DBs — add columns added after initial schema
    for col, defn in [("tool_method", "TEXT"), ("tool_path", "TEXT"), ("tool_body", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE hitl_requests ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass  # Column already exists

    return conn


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Return a connection to an initialized database."""
    return init_db(db_path)


# Default HITL settings: tool_name -> mode
_DEFAULT_HITL_SETTINGS: list[tuple[str, str, int]] = [
    # Calendar read tools -- off
    ("calendar_list_events", "off", 15),
    ("calendar_get_event", "off", 15),
    ("calendar_list", "off", 15),
    ("calendar_check_availability", "off", 15),
    # Calendar write tools -- on
    ("calendar_create_event", "on", 15),
    ("calendar_quick_add", "on", 15),
    ("calendar_update_event", "on", 15),
    ("calendar_delete_event", "on", 15),
    ("calendar_invite", "on", 15),
    # Gmail read tools -- off
    ("gmail_read_emails", "off", 15),
    ("gmail_get_email", "off", 15),
    ("gmail_list_labels", "off", 15),
    # Gmail write tools -- on
    ("gmail_send_email", "on", 15),
    ("gmail_reply_email", "on", 15),
    ("gmail_delete_email", "on", 15),
    ("gmail_move_email", "on", 15),
    # LinkedIn -- on
    ("linkedin_post", "on", 15),
    # Docker mutating -- on
    ("docker_compose_up", "on", 15),
    ("docker_compose_restart", "on", 15),
    ("docker_compose_down", "on", 15),
    # Docker read-only -- off
    ("docker_compose_logs", "off", 15),
    ("docker_compose_status", "off", 15),
    ("docker_list_containers", "off", 15),
    ("docker_list_networks", "off", 15),
    # Browser -- on for actions that introduce external content or change page state
    ("browser_navigate", "on", 15),
    ("browser_click", "on", 15),
    ("browser_type", "on", 15),
    ("browser_web_search", "on", 15),
    # Ollama generic LLM endpoints — read-only side effects, no approval needed
    ("ollama_structured", "off", 15),
    ("ollama_chat", "off", 15),
    # Schema mutation — always HITL-gated. Adding new node/edge types is a
    # durable schema decision and should never auto-execute.
    ("schema_new_type", "on", 15),
    # Training pipeline — off, because the only caller is the console and a
    # console click is already a human act taken behind a login. Gating them
    # would page the same person who just pressed the button, and the panel
    # would sit on a pending response with nothing to show.
    ("training_curate", "off", 15),
    ("training_export", "off", 15),
    ("training_train", "off", 15),
]


def seed_default_hitl_settings(db: sqlite3.Connection) -> None:
    """Insert default HITL settings using INSERT OR IGNORE (does not overwrite)."""
    for tool_name, mode, timeout in _DEFAULT_HITL_SETTINGS:
        db.execute(
            "INSERT OR IGNORE INTO tool_hitl_settings (tool_name, mode, sudo_timeout_minutes) "
            "VALUES (?, ?, ?)",
            (tool_name, mode, timeout),
        )
    db.commit()
