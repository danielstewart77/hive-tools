"""Gmail FastAPI router -- migrated from hive_mind_mcp."""

from __future__ import annotations

import base64
import html
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from hitl import hitl_gate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/gmail", tags=["gmail"])

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ---- Service helper ----


def _get_gmail_service() -> Any:
    """Build and return an authenticated Gmail API service, or None."""
    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        log.error("Google API libraries not installed")
        return None

    token_paths = [
        Path("credentials/token.json"),
        Path.home() / ".gmail-mcp" / "token.json",
    ]

    creds = None
    for path in token_paths:
        if path.exists():
            creds = Credentials.from_authorized_user_file(str(path), SCOPES)
            break

    if creds is None:
        log.error("No Gmail token found")
        return None

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            for path in token_paths:
                if path.exists():
                    try:
                        path.write_text(creds.to_json())
                    except OSError:
                        log.warning("Could not persist refreshed token to %s", path)
                    break
        except RefreshError:
            log.error("Gmail refresh token expired")
            return None

    return build("gmail", "v1", credentials=creds)


# ---- MIME body extraction ----


def _html_to_text(h: str) -> str:
    """Convert HTML to readable plain text."""
    h = re.sub(r"<style[^>]*>.*?</style>", "", h, flags=re.DOTALL | re.IGNORECASE)
    h = re.sub(r"<script[^>]*>.*?</script>", "", h, flags=re.DOTALL | re.IGNORECASE)
    h = re.sub(
        r'<a\s[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r"\2 (\1)",
        h,
        flags=re.DOTALL | re.IGNORECASE,
    )
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.IGNORECASE)
    h = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", h, flags=re.IGNORECASE)
    h = re.sub(r"<[^>]+>", "", h)
    h = html.unescape(h)
    h = re.sub(r"[ \t]+", " ", h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()


def _decode_body(data: str, charset: str = "utf-8") -> str:
    raw = base64.urlsafe_b64decode(data)
    return raw.decode(charset, errors="replace")


def _extract_body(payload: dict[str, Any]) -> tuple[str, str]:
    """Walk MIME tree and extract body text."""
    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])
    charset = "utf-8"
    for param in payload.get("headers", []):
        if param.get("name", "").lower() == "content-type":
            m = re.search(r"charset=([^\s;]+)", param.get("value", ""), re.IGNORECASE)
            if m:
                charset = m.group(1).strip("\"'")

    body_data = payload.get("body", {}).get("data")
    if body_data and mime_type == "text/plain":
        return _decode_body(body_data, charset), "text/plain"
    if body_data and mime_type == "text/html":
        return _html_to_text(_decode_body(body_data, charset)), "text/html_converted"

    plain_result = None
    html_result = None
    for part in parts:
        text, btype = _extract_body(part)
        if text:
            if btype == "text/plain" and plain_result is None:
                plain_result = (text, btype)
            elif btype == "text/html_converted" and html_result is None:
                html_result = (text, btype)

    if plain_result:
        return plain_result
    if html_result:
        return html_result

    return ("", "none")


def _extract_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract attachment metadata from MIME tree."""
    attachments: list[dict[str, Any]] = []
    parts = payload.get("parts", [])
    for part in parts:
        filename = part.get("filename")
        if filename:
            attachments.append({
                "filename": filename,
                "mime_type": part.get("mimeType", "unknown"),
                "size": part.get("body", {}).get("size", 0),
            })
        attachments.extend(_extract_attachments(part))
    return attachments


def _get_header(headers: list[dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


# ---- Request models for write endpoints ----


class GmailSendBody(BaseModel):
    to: str
    subject: str
    body: str
    cc: str = ""


class GmailReplyBody(BaseModel):
    message_id: str
    body: str


class GmailTrashBody(BaseModel):
    message_id: str


class GmailMoveBody(BaseModel):
    message_id: str
    add_labels: str = ""
    remove_labels: str = ""


# ---- Read endpoints (default HITL: off) ----


@router.get("/messages")
async def read_emails(
    query: str = Query(...),
    max_results: int = 10,
    caller: str = Depends(hitl_gate("gmail_read_emails")),
) -> Any:
    """Search and fetch emails."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    max_results = max(1, min(max_results, 250))
    total_estimated = 0

    try:
        message_ids: list[str] = []
        page_token = None
        while len(message_ids) < max_results:
            page_size = min(100, max_results - len(message_ids))
            result = service.users().messages().list(
                userId="me", q=query, maxResults=page_size, pageToken=page_token
            ).execute()
            total_estimated = result.get("resultSizeEstimate", 0)
            for msg in result.get("messages", []):
                message_ids.append(msg["id"])
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        messages = []
        for msg_id in message_ids:
            msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
            headers = msg.get("payload", {}).get("headers", [])
            body, body_type = _extract_body(msg.get("payload", {}))
            messages.append({
                "id": msg["id"],
                "thread_id": msg.get("threadId", ""),
                "from": _get_header(headers, "From"),
                "to": _get_header(headers, "To"),
                "subject": _get_header(headers, "Subject"),
                "date": _get_header(headers, "Date"),
                "snippet": msg.get("snippet", ""),
                "body": body,
                "body_type": body_type,
                "labels": msg.get("labelIds", []),
            })

        return {"messages": messages, "returned": len(messages), "total_estimated": total_estimated}
    except Exception as e:
        log.exception("read_emails failed")
        return {"error": str(e)}


@router.get("/messages/{message_id}")
async def get_email(
    message_id: str,
    caller: str = Depends(hitl_gate("gmail_get_email")),
) -> Any:
    """Fetch a single email by ID."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = msg.get("payload", {}).get("headers", [])
        body, body_type = _extract_body(msg.get("payload", {}))
        attachments = _extract_attachments(msg.get("payload", {}))

        return {
            "id": msg["id"],
            "thread_id": msg.get("threadId", ""),
            "from": _get_header(headers, "From"),
            "to": _get_header(headers, "To"),
            "cc": _get_header(headers, "Cc"),
            "subject": _get_header(headers, "Subject"),
            "date": _get_header(headers, "Date"),
            "snippet": msg.get("snippet", ""),
            "body": body,
            "body_type": body_type,
            "labels": msg.get("labelIds", []),
            "attachments": attachments,
        }
    except Exception as e:
        log.exception("get_email failed")
        return {"error": str(e)}


@router.get("/labels")
async def list_labels(
    caller: str = Depends(hitl_gate("gmail_list_labels")),
) -> Any:
    """List all Gmail labels."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        result = service.users().labels().list(userId="me").execute()
        return [
            {"id": lbl["id"], "name": lbl["name"], "type": lbl.get("type", "user")}
            for lbl in result.get("labels", [])
        ]
    except Exception as e:
        log.exception("list_labels failed")
        return {"error": str(e)}


# ---- Write endpoints (default HITL: on) ----


@router.post("/send")
async def send_email(
    body: GmailSendBody,
    caller: str = Depends(hitl_gate("gmail_send_email")),
) -> Any:
    """Send an email via Gmail."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        from email.mime.text import MIMEText

        msg = MIMEText(body.body)
        msg["to"] = body.to
        msg["subject"] = body.subject
        if body.cc:
            msg["cc"] = body.cc

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        result = service.users().messages().send(userId="me", body={"raw": raw}).execute()

        return {
            "status": "sent",
            "message_id": result["id"],
            "thread_id": result.get("threadId", ""),
        }
    except Exception as e:
        log.exception("send_email failed")
        return {"error": str(e)}


@router.post("/reply")
async def reply_to_email(
    body: GmailReplyBody,
    caller: str = Depends(hitl_gate("gmail_reply_email")),
) -> Any:
    """Reply to an email thread."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        original = service.users().messages().get(
            userId="me", id=body.message_id, format="metadata",
            metadataHeaders=["Subject", "From", "To", "Message-ID"],
        ).execute()

        headers = original.get("payload", {}).get("headers", [])
        orig_subject = _get_header(headers, "Subject")
        orig_from = _get_header(headers, "From")
        orig_msg_id = _get_header(headers, "Message-ID")
        thread_id = original.get("threadId", "")

        subject = orig_subject if orig_subject.lower().startswith("re:") else f"Re: {orig_subject}"

        from email.mime.text import MIMEText

        msg = MIMEText(body.body)
        msg["to"] = orig_from
        msg["subject"] = subject
        if orig_msg_id:
            msg["In-Reply-To"] = orig_msg_id
            msg["References"] = orig_msg_id

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        result = service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id}
        ).execute()

        return {
            "status": "sent",
            "message_id": result["id"],
            "thread_id": result.get("threadId", ""),
        }
    except Exception as e:
        log.exception("reply_to_email failed")
        return {"error": str(e)}


@router.post("/trash")
async def trash_email(
    body: GmailTrashBody,
    caller: str = Depends(hitl_gate("gmail_delete_email")),
) -> Any:
    """Move a Gmail message to Trash."""
    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        service.users().messages().trash(userId="me", id=body.message_id).execute()
        return {"status": "trashed", "message_id": body.message_id}
    except Exception as e:
        log.exception("delete_email failed")
        return {"error": str(e)}


@router.post("/move")
async def move_email(
    body: GmailMoveBody,
    caller: str = Depends(hitl_gate("gmail_move_email")),
) -> Any:
    """Modify Gmail labels on a message."""
    add = [lbl.strip() for lbl in body.add_labels.split(",") if lbl.strip()] if body.add_labels else []
    remove = [lbl.strip() for lbl in body.remove_labels.split(",") if lbl.strip()] if body.remove_labels else []

    if not add and not remove:
        return {"error": "Provide at least one of add_labels or remove_labels."}

    service = _get_gmail_service()
    if service is None:
        return {"error": "Gmail not authenticated"}

    try:
        result = service.users().messages().modify(
            userId="me",
            id=body.message_id,
            body={"addLabelIds": add, "removeLabelIds": remove},
        ).execute()
        return {
            "status": "updated",
            "message_id": body.message_id,
            "labels": result.get("labelIds", []),
        }
    except Exception as e:
        log.exception("move_email failed")
        return {"error": str(e)}
