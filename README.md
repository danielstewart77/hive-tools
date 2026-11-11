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
default. That seed is insert-if-absent: it runs for every gated tool on
every start, so a replacing write would re-arm every gate the operator had
turned off on each deploy.

A decision is one guarded statement, so two people reaching the queue at the
same moment produce one decision and the loser is told which of three
reasons it lost — no such request, already settled, or expired while the
page was open. Those have different remedies, which is why they are not
folded together. An expired request is drawn as expired: nothing ever writes
`expired` to a row, the expiry is a rule applied when the row is read, and
`effective_status` is the one place it lives.

The decision and the sudo grant it may earn are one unit of work. Committed
separately they leave an end state nothing can recover from — the request
says approved, the call it authorised was never replayed, and the decision
is guarded on `pending` so it can never be approved again.

An approval replays the captured call with the internal credential, and what
the replay did is stored on the request: whether it succeeded, its HTTP
status, and the body the tool answered with. A status outside 2xx, a replay
that never got an answer, and a 2xx whose body carries `error` (a tool that
caught its own failure) are all stored as failed. `GET /hitl/{id}` returns it
as `outcome` on an approved request — `null` until the replay has landed —
and hive-tools then sends the asking mind one broker message through
hive-comms (`POST /broker/messages`, from and to that mind's id, on a fresh
conversation id, `metadata.request_type` `hitl_result`), using the
`COMMS_URL` and `COMMS_BEARER_TOKEN` service credential. A message that
cannot be delivered is logged; the stored outcome stands.

A gated call whose body holds a NUL byte is refused with a 400 rather than
queued. A NUL is valid UTF-8, so `errors="replace"` leaves it alone, and
Postgres refuses it outright. Refused rather than escaped,
because the whole purpose of capturing the body is that an approval replays
the call that was approved, and a body stored with the NUL taken out is a
different call.

## Where the data lives

The `tools` schema of the hive's Postgres, through `hive_data`, at
`HIVE_DATABASE_URL`. It is required: unset, the service refuses to start —
before the first-boot admin seed below can read an empty store and seed it.
The SQLite file this service used before the cutover is retired and nothing
reads it. Nothing outside `store.py` opens a database or writes SQL.

## Running it

```bash
cp .env.example .env        # then fill in real values — see the warning below
docker compose up -d --build
```

The image takes `UID`, `GID` and `DOCKER_GID` build args (defaults 1000, 1000,
983) for the app user and the group that owns the host's Docker socket; pass
the host's own values (`--build-arg DOCKER_GID=$(getent group docker | cut -d: -f3)`)
anywhere else. No GPU is needed: only the training router's VRAM arbitration
calls `nvidia-smi`, and without it the service starts and serves, reporting no
card.

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
  OAuth tokens and service-account JSON never enter git.
- The Docker socket is mounted for the `docker_ops` router. That is root-
  equivalent access to the host daemon. Run this service only on a host you
  trust it with, and keep `9421` off the public internet.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
HIVE_DATA_TEST_ADMIN_DSN=postgresql://... .venv/bin/python -m pytest   # zero failures is the bar
```

The suite runs on a scratch Postgres it creates, claims and drops on the
server `HIVE_DATA_TEST_ADMIN_DSN` names, never on the live database.

Tests land with code changes.

## Adding a tool

1. Write `tools/<name>.py` exposing an `APIRouter`.
2. Mount it in `server.py` behind a guarded import, matching the existing
   pattern.
3. Gate any outward-facing action with `hitl_gate("<tool_name>")`.
4. Add tests under `tests/api/`.

## License

MIT — see [LICENSE](LICENSE).
