# hive-tools

The **body** of a [hive-mind](https://github.com/danielstewart77/hive-mind)
deployment: one FastAPI service that gives AI minds hands in the outside world.
Where the mind reasons, hive-tools acts — reading mail, booking calendar events,
driving a browser, sending SMS, managing containers, and running fine-tunes.

Every capability is a router under `tools/`, mounted only if its dependencies
import cleanly, so a missing Google credential disables the Gmail routes rather
than taking the service down.

## Capabilities

| Router | What it does |
|---|---|
| `gmail` | Read, search, and send mail via the Gmail API |
| `calendar` | Read and create Google Calendar events |
| `browser` | Fetch and extract page content |
| `linkedin` | Post and read via the LinkedIn API |
| `docker_ops` | Inspect and manage containers on the host daemon |
| `ollama` | Chat, structured output, and model listing against Ollama |
| `sms` | Send SMS through a self-hosted android-sms-gateway |
| `training` | Curate captured turns and launch LoRA fine-tune runs |
| `schema` | Serve the tool schemas minds use for discovery |

## Human-in-the-loop

Outward-facing actions are gated. `hitl_gate("<tool_name>")` intercepts a call
and raises `HITLPendingException` until a human approves it; approval prompts go
to the operator's Telegram. New tool names auto-register into
`tool_hitl_settings` at startup (`sync_tool_registry` in `hitl.py`) defaulting to
**always prompt** — a newly added tool is gated by default, never open by
default.

## Running it

```bash
cp .env.example .env        # then fill in real values — see the warning below
docker compose up -d --build
```

The service listens on `9421`. FastAPI's `/docs`, `/redoc`, and `/openapi.json`
are disabled deliberately: an exposed OpenAPI spec is a machine-readable map of
every route and schema, and this service's routes reach real mail and real
phones.

### Security notes

- **Set `HT_SECRET_KEY` and `HT_ADMIN_PASSWORD` before first boot.** Both have
  insecure fallbacks in code (`change-this-in-production`, `changeme`), and the
  first boot seeds an `admin` user from the latter. Leaving them unset gives you
  a deployment whose admin credential is published in this README.
- `credentials/` and `data/` are ignored wholesale, not by file extension —
  OAuth tokens, service-account JSON, and the SQLite store never enter git.
- The Docker socket is mounted for the `docker_ops` router. That is root-
  equivalent access to the host daemon. Run this service only on a host you
  trust it with, and keep `9421` off the public internet.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest        # zero failures is the bar
```

Tests land with code changes.

## Adding a tool

1. Write `tools/<name>.py` exposing an `APIRouter`.
2. Mount it in `server.py` behind a guarded import, matching the existing
   pattern.
3. Gate any outward-facing action with `hitl_gate("<tool_name>")`.
4. Add tests under `tests/api/`.

## License

MIT — see [LICENSE](LICENSE).
