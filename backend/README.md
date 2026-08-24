# AH-MemoryHub backend

Python 3.12+, FastAPI, Pydantic v2. The default mode is a deterministic in-memory AH core; Neo4j is an optional projection.

```powershell
uv sync --extra test
uv run pytest -q
uv run python -m app.conformance
uv run uvicorn app.main:app --reload --env-file .env
```

`python -m app.conformance` is the Junior gate: it validates all seven `e`
payload variants (including first-class `s*` and `m*`), the exact eight-fact
manual fixture from section 7, figure 14, an IS-A hierarchy, an H/FOLLOW
episode, and lossless export/import. The same report is available at
`GET /api/v1/conformance/junior`.

Copy `.env.example` to `.env` before launch. The parser is provider-neutral:

```env
AH_PARSER_PROVIDER=openai_compatible
AH_LLM_BASE_URL=https://your-provider.example/v1
AH_LLM_MODEL=your-model-id
AH_LLM_API_KEY=your-key
AH_INGESTION_INITIAL_WEIGHT=1.0
```

The local `.env` is configured for OpenRouter Ox Alpha. Paste an OpenRouter key
into `AH_LLM_API_KEY`; `json_object` is used because this model does not enforce
JSON Schema. Server-side Pydantic and AH compiler validation remain mandatory.
The Ingestion model selector can explicitly choose `stealth/ox-alpha` or
`~deepseek/deepseek-v4-flash-latest`; both overrides reuse the server-side key.
Ox Alpha uses JSON mode, while DeepSeek requests strict JSON Schema output.
`deepseek/deepseek-v4-flash-0731:nitro` is the throughput-routed preset. Parser
completions are capped by `AH_LLM_MAX_TOKENS` (4096 by default), and DeepSeek
reasoning is disabled for extraction to prevent billed runaway output.

Any Chat Completions compatible service can be used: Kimi, OpenRouter, OpenAI,
or a local server. The model only proposes typed facts and exact quotes. AH Core
anchors every quote back to the original document, validates roles/templates,
and assigns UIDs. `auto` falls back to `rule-based-offline` and reports that fact
in the preview response; `openai_compatible` returns an explicit parser error.
Parser confidence is retained in Source Evidence and never becomes activation
weight. Repeated extraction is cached by document hash, model, and prompt
version. `POST /api/v1/evaluations/ingestion/rabbit` runs the live configured
provider against the eight-fact monograph fixture; six matches are required.

The external parser contract is Perception IR v3: exact Mentions, backward
coreference, atomic facts, and explicit `ATOM/AND/OR/VERY` terms. The preview
groups facts by source span and shows canonical Role Bindings; the original
quote remains immutable evidence.

Queries remain local by default. When a user explicitly selects an external
answer model, only the question, activated typed AH facts, and their exact
source quotes are sent. The structured answer is capped by
`AH_ANSWER_MAX_TOKENS` (2048), every sentence must cite a known `[E#]`, and any
provider or citation failure falls back to the deterministic grounded answer.

Demo flow:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/demo/seed
curl -X POST http://127.0.0.1:8000/api/v1/queries -H 'content-type: application/json' -d '{"question":"почему остановился агрегат?","max_ticks":8}'
curl http://127.0.0.1:8000/api/v1/memory/stats
curl -X POST http://127.0.0.1:8000/api/v1/dsl/query -H 'content-type: application/json' -d '{"expression":"findRoles(role=LOCATION, value=насосном зале)"}'
```

`POST /api/v1/memory/export` returns the complete `AH=<S,C,P,H,L>` dump, source manifest, and SHA-256 checksum. `M4`/`M5` are explicitly reported unavailable until external model providers are configured.
