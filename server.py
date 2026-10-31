"""HiveTools FastAPI application — main entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import asyncio

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from auth_api import require_api_token
from auth_session import (
    create_user,
    get_current_user_from_request,
    set_session_cookie,
    verify_user_credentials,
)
import store
from config import BASE_DIR, DB_PATH
from hitl import (
    HITLPendingException,
    UncapturableCall,
    UnknownAction,
    _replay_request_async,
    get_hitl_status,
    respond_to_hitl,
    sync_tool_registry,
)
from hive_logging import configure_logging, install_fastapi_logging, log_event
from schemas import HITLRespondRequest
from tools.docker_ops import BlockedProjectError

log = configure_logging("hive-tools")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _already_seeded(exc: BaseException) -> bool:
    """Whether a failed admin seed failed because the admin already exists.

    Matched on the class names rather than imported, because which of them
    can be raised depends on which store answered - `hive_data.Conflict` on
    Postgres, `sqlite3.IntegrityError` on the file - and importing
    `hive_data` here would make the Postgres package a hard dependency of
    the SQLite path.
    """
    for cause in (exc, exc.__cause__, exc.__context__):
        if cause is None:
            continue
        if type(cause).__name__ in ("Conflict", "IntegrityError",
                                    "UniqueViolationError"):
            return True
    return False


def create_app(db_path: str | None = None) -> FastAPI:
    """Application factory."""
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Open the store, seed what a fresh install needs, then run.

        The store is opened here rather than while the app is being built,
        because opening a connection pool is async and because a module that
        connected at import would do it again in every test that imported
        it. Everything that writes at startup therefore also happens here:
        the bootstrap admin, the shipped approval modes, and the gated tools
        the routers registered as they were imported.

        Then: pick up a training run this process was watching when it died.
        The watcher holding the restore obligation lives in this process's
        event loop, so a redeploy mid-run would otherwise leave the voice
        stack stopped and Ollama evicted with nobody left to put them
        back. The journal on disk is what survives; this reads it.
        """
        await store.open(db_path or DB_PATH)
        _app.state.db = store.sqlite_connection()
        log.info("hive-tools store: %s",
                 "postgres" if store.on_postgres() else "sqlite")

        if await store.count_users() == 0:
            import os

            bootstrap_password = os.getenv("HT_ADMIN_PASSWORD", "changeme")
            try:
                await create_user("admin", bootstrap_password, is_admin=True)
            except Exception as exc:
                # A read, a check and a write. The database is shared now, so
                # an overlapping redeploy or a second process on this schema
                # can seed between the two - and a uniqueness conflict raised
                # here is a crash loop rather than a boot, over a user that
                # exists. Anything else still raises.
                if not _already_seeded(exc):
                    raise
                log.info("Admin user was already seeded by another process")
            else:
                log.info("Admin user seeded from HT_ADMIN_PASSWORD (env var "
                         "no longer needed after this boot)")

        from db import seed_default_hitl_settings

        await seed_default_hitl_settings()

        # Every tool router has been imported by now, so `hitl_gate()`'s
        # side effects have populated `hitl._REGISTERED_TOOLS`. Seeding is
        # insert-if-absent: a mode the operator turned off stays off across
        # a restart.
        await sync_tool_registry()

        try:
            from tools import training_gpu

            from tools.training import close_ledger_run

            outcome = await training_gpu.reconcile(
                notify=training_gpu.notify_owner, close_run=close_ledger_run
            )
            if outcome.get("action") != "nothing-to-do":
                log.info("training.gpu.reconcile outcome=%s", outcome)
        except Exception as exc:  # noqa: BLE001 - startup must not be fatal
            log.warning("training.gpu.reconcile_failed error=%s", exc)
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(
        title="HiveTools",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    install_fastapi_logging(app, log, "hive-tools")

    # ---- Health ----
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---- Login ----
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "login.html")

    @app.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Any:
        user = await verify_user_credentials(username, password)
        if not user:
            log_event(log, "tools.login.rejected", level=logging.WARNING, username=username)
            return JSONResponse(
                {"detail": "Invalid credentials"}, status_code=401
            )
        response = RedirectResponse(url="/", status_code=303)
        set_session_cookie(response, user)
        log_event(log, "tools.login.succeeded", username=username)
        return response

    # ---- HITL polling endpoints ----
    @app.get("/hitl/{request_id}")
    async def hitl_status(
        request_id: str,
        caller: str = Depends(require_api_token),
    ) -> Any:
        status = await get_hitl_status(request_id)
        if status is None:
            return JSONResponse({"detail": "Not found"}, status_code=404)

        s = status["status"]
        if s == "pending":
            return JSONResponse(status, status_code=202)
        if s == "approved":
            return JSONResponse(status, status_code=200)
        if s == "denied":
            return JSONResponse(status, status_code=403)
        if s == "expired":
            return JSONResponse(status, status_code=408)

        return JSONResponse(status, status_code=200)

    @app.post("/hitl/{request_id}/respond")
    async def hitl_respond(
        request_id: str,
        body: HITLRespondRequest,
        caller: str = Depends(require_api_token),
    ) -> Any:
        try:
            result = await respond_to_hitl(request_id, body.action)
        except UnknownAction as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if result is None:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if "error" in result and result["error"] == "already_resolved":
            return JSONResponse(result, status_code=409)
        replay = result.pop("_replay", None)
        log_event(
            log, "hitl.responded", request_id=request_id, action=body.action,
            caller=caller, replay_scheduled=bool(replay),
        )
        if replay:
            asyncio.create_task(_replay_request_async(**replay))
        return JSONResponse(result, status_code=200)

    # ---- Dashboard (requires session auth) ----
    @app.get("/")
    async def dashboard(request: Request) -> Any:
        user = await get_current_user_from_request(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        return templates.TemplateResponse(
            request, "dashboard.html", {"user": user}
        )

    # Exception handlers
    @app.exception_handler(HITLPendingException)
    async def hitl_pending_handler(request: Request, exc: HITLPendingException) -> JSONResponse:
        return JSONResponse(
            {"hitl_request_id": exc.request_id, "status": "pending"},
            status_code=202,
        )

    @app.exception_handler(UncapturableCall)
    async def uncapturable_call_handler(
        request: Request, exc: UncapturableCall,
    ) -> JSONResponse:
        """A gated call whose body cannot be stored as the call it was.

        400 and not 500: the fault is in what arrived, and the sentence says
        what about it. Before this it was an unhandled `ValueRefused` from
        the database driver - an opaque 500, with no approval queued and no
        notification sent, so the gated call simply failed.
        """
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(BlockedProjectError)
    async def blocked_project_handler(
        request: Request, exc: BlockedProjectError,
    ) -> JSONResponse:
        """A mutating operation on a project that may not be mutated.

        403, because the caller authenticated fine and is not allowed to do
        this - and not 500, which is what it answered while the handler's
        own import sat inside `except ImportError: pass` against a class
        that did not exist. Not 200 either, which is what that handler said:
        a refusal reported as a success is a caller that believes its
        `compose up` ran.

        Registered directly rather than behind a swallowed import. The
        router includes below are optional and belong in a `try`; an
        exception class this application's own code raises is not.
        """
        return JSONResponse({"error": str(exc)}, status_code=403)

    # Include tool routers
    try:
        from tools.calendar import router as calendar_router

        app.include_router(calendar_router)
    except ImportError:
        pass

    try:
        from tools.gmail import router as gmail_router

        app.include_router(gmail_router)
    except ImportError:
        pass

    try:
        from tools.linkedin import router as linkedin_router

        app.include_router(linkedin_router)
    except ImportError:
        pass

    try:
        from tools.docker_ops import router as docker_router

        app.include_router(docker_router)
    except ImportError:
        pass

    try:
        from tools.browser import router as browser_router

        app.include_router(browser_router)
    except ImportError:
        pass

    try:
        from tools.ollama import router as ollama_router

        app.include_router(ollama_router)
    except ImportError:
        pass

    try:
        from tools.sms import router as sms_router

        app.include_router(sms_router)
    except ImportError:
        pass

    try:
        from tools.training import router as training_router

        app.include_router(training_router)
    except ImportError:
        pass

    try:
        from tools.schema import router as schema_router

        app.include_router(schema_router)
    except ImportError:
        pass

    # Include UI router
    try:
        from ui import router as ui_router

        app.include_router(ui_router)
    except ImportError:
        pass

    return app


# Module-level app for `python server.py` and uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    log.info("Starting HiveTools on 0.0.0.0:9421")
    uvicorn.run(app, host="0.0.0.0", port=9421, log_config=None, access_log=False)
