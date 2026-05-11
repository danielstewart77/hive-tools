"""Tests for default HITL settings seeding."""

from __future__ import annotations

import sqlite3


def test_seed_creates_default_settings(db_conn: sqlite3.Connection) -> None:
    from db import seed_default_hitl_settings

    seed_default_hitl_settings(db_conn)

    rows = db_conn.execute("SELECT tool_name, mode FROM tool_hitl_settings").fetchall()
    tool_names = {row["tool_name"] for row in rows}

    # Should have entries for all expected tools
    assert "calendar_list_events" in tool_names
    assert "gmail_send_email" in tool_names
    assert "linkedin_post" in tool_names
    assert "docker_compose_up" in tool_names


def test_seed_idempotent_does_not_overwrite(db_conn: sqlite3.Connection) -> None:
    from db import seed_default_hitl_settings

    seed_default_hitl_settings(db_conn)

    # Manually change a setting
    db_conn.execute(
        "UPDATE tool_hitl_settings SET mode = 'sudo' WHERE tool_name = 'gmail_send_email'"
    )
    db_conn.commit()

    # Seed again
    seed_default_hitl_settings(db_conn)

    row = db_conn.execute(
        "SELECT mode FROM tool_hitl_settings WHERE tool_name = 'gmail_send_email'"
    ).fetchone()
    # Should NOT be overwritten
    assert row["mode"] == "sudo"


def test_default_modes_match_spec(db_conn: sqlite3.Connection) -> None:
    from db import seed_default_hitl_settings

    seed_default_hitl_settings(db_conn)

    def get_mode(tool_name: str) -> str:
        row = db_conn.execute(
            "SELECT mode FROM tool_hitl_settings WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()
        return row["mode"] if row else "missing"

    # Read tools should be off
    assert get_mode("calendar_list_events") == "off"
    assert get_mode("calendar_get_event") == "off"
    assert get_mode("gmail_read_emails") == "off"
    assert get_mode("gmail_get_email") == "off"

    # Write tools should be on
    assert get_mode("gmail_send_email") == "on"
    assert get_mode("calendar_create_event") == "on"
    assert get_mode("linkedin_post") == "on"
    assert get_mode("docker_compose_up") == "on"
