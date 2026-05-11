# State — REQ-038 Ollama classification endpoint

Plan: `/home/daniel/Storage/hive-tools/docs/PLAN-OLLAMA-CLASSIFY.md`

## Tools detected
- pytest (.venv/bin/pytest, configured via pyproject.toml [tool.pytest.ini_options])
- mypy (.venv/bin/mypy)
- ruff (.venv/bin/ruff)

## Tasks
- [x] Add `("ollama_classify", "off", 15)` to `_DEFAULT_HITL_SETTINGS` in db.py
- [x] Create tools/ollama.py with /ollama/classify route + httpx async client
- [x] Wire ollama_router into server.py:create_app
- [x] tests/api/test_ollama.py — 6 tests cover auth, hitl-off, hitl-on, non-JSON fallback, 502, default seed

## Validation results
- pytest tests/api/test_ollama.py: **6/6 PASS**
- pytest full suite: 105 passed / 8 pre-existing failures (docker_ops, e2e, hitl_notify) — not introduced by this batch
- ruff (modified files): **clean**
- mypy tools/ollama.py: **clean** (no issues)

## Validation retry counters
- pytest: error-count: 0
- mypy: error-count: 0
- ruff: error-count: 0

## Follow-up (out of scope, post-PASS)

- **Container rebuild**: hive-tools runs in Docker via `/home/daniel/Storage/Dev/hive_mind/docker-compose.yml`. The route is not live until:
  ```
  docker compose -f /home/daniel/Storage/Dev/hive_mind/docker-compose.yml build hive-tools
  docker compose -f /home/daniel/Storage/Dev/hive_mind/docker-compose.yml up -d hive-tools
  ```
  Defer to user.

RESULT: PASS
