# AH-MemoryHub backend

Python 3.12+, FastAPI, Pydantic v2. The default mode is a deterministic in-memory AH core; Neo4j is an optional projection.

```powershell
uv sync --extra test
uv run pytest -q
uv run uvicorn app.main:app --reload
```

Demo flow:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/demo/seed
curl -X POST http://127.0.0.1:8000/api/v1/queries -H 'content-type: application/json' -d '{"question":"почему остановился агрегат?","max_ticks":8}'
curl http://127.0.0.1:8000/api/v1/memory/stats
curl -X POST http://127.0.0.1:8000/api/v1/dsl/query -H 'content-type: application/json' -d '{"expression":"findRoles(role=LOCATION, value=насосном зале)"}'
```

`POST /api/v1/memory/export` returns the complete `AH=<S,C,P,H,L>` dump, source manifest, and SHA-256 checksum. `M4`/`M5` are explicitly reported unavailable until external model providers are configured.

