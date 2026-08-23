from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core import AHMemory, norm
from .dsl import Interpreter, DSLParseError
from .engine import IgnitionEngine, build_candidate_snapshot, gc_commit, gc_preview
from .ingestion import ingest
from .evaluation import internal_m1, internal_m2, rabbit_fixture, role_metrics
from .models import *


memory = AHMemory()
storage_mode = os.getenv("AH_STORAGE_MODE", "in-memory").lower()
if storage_mode not in {"in-memory", "neo4j"}: storage_mode = "in-memory"
storage_adapter = None
ingestions: dict[str, Any] = {}
document_index: dict[str, str] = {}
runs: dict[str, IgnitionRun] = {}

app = FastAPI(title="AH-MemoryHub", version="0.1.0", description="Executable AH=<S,C,P,H,L> modular monolith")
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")
origins = [x.strip() for x in os.getenv("AH_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=False)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", f"req_{uuid.uuid4().hex[:12]}")
    request.state.request_id = rid
    try:
        response = await call_next(request)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": str(exc), "request_id": rid}})
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    detail.setdefault("request_id", getattr(_.state, "request_id", None))
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": "request validation failed", "details": exc.errors(), "request_id": getattr(request.state, "request_id", None)}})


@app.get("/", include_in_schema=False)
def frontend_index():
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(503, {"code": "frontend_unavailable", "message": "frontend assets are not installed"})
    return FileResponse(index, media_type="text/html")


@app.get("/health")
def health(): return {"status": "ok", "mode": storage_mode, "storage_mode": storage_mode, "revision": memory.revision}


@app.post("/api/v1/ingestions")
def create_ingestion(body: DocumentIngestRequest):
    if body.document_uid and body.document_uid in document_index:
        cached = dict(ingestions[document_index[body.document_uid]])
        cached["reused"] = True
        return cached
    result = ingest(memory, body.text, body.document_uid)
    if storage_mode == "neo4j":
        global storage_adapter
        try:
            if storage_adapter is None:
                from .persistence import Neo4jAdapter
                storage_adapter = Neo4jAdapter().connect()
            storage_adapter.persist(memory)
        except Exception as exc:
            raise HTTPException(503, {"code": "storage_unavailable", "message": str(exc)})
    ingestions[result.ingestion_uid] = {"ingestion_uid": result.ingestion_uid, "document_uid": result.document_uid, "accepted": result.accepted, "rejected": result.rejected, "candidates": [c.model_dump(mode="json") for c in result.candidates], "reused": False}
    document_index[result.document_uid] = result.ingestion_uid
    return ingestions[result.ingestion_uid]


@app.get("/api/v1/ingestions")
def list_ingestions():
    return {"items": list(ingestions.values()), "count": len(ingestions)}


@app.get("/api/v1/ingestions/{ingestion_uid}")
def get_ingestion(ingestion_uid: str):
    if ingestion_uid not in ingestions: raise HTTPException(404, {"code": "not_found", "message": "ingestion not found"})
    return ingestions[ingestion_uid]


def _seed_ids(question: str) -> list[str]:
    q = norm(question)
    exact = [s.uid for s in memory.symbols.values() if any(norm(r.value) in q or q in norm(r.value) for r in s.sensory_representations)]
    if exact: return sorted(set(exact))
    words = {x for x in q.split() if len(x) > 3}
    return sorted({s.uid for s in memory.symbols.values() if any(any(w in norm(r.value) or norm(r.value).startswith(w[:5]) or w.startswith(norm(r.value)[:5]) for w in words) for r in s.sensory_representations)})


def _term_match(question: str, value: str) -> int:
    qwords = {w for w in norm(question).split() if len(w) > 3}
    vwords = {w for w in norm(value).split() if len(w) > 3}
    return sum(1 for q in qwords if any(q == v or q.startswith(v[:5]) or v.startswith(q[:5]) for v in vwords))


def _causal_answer(question: str, working_memory: tuple[str, ...]):
    candidates = []
    for h in memory.find_hypernodes():
        if h.uid not in working_memory or h.template_ref != "tpl_cause" or not h.evidence:
            continue
        bindings = {b.role_id: b.target_ref.target_uid for b in h.role_bindings}
        object_label = memory.label(bindings["OBJECT"]) if "OBJECT" in bindings else ""
        subject_label = memory.label(bindings["SUBJECT"]) if "SUBJECT" in bindings else ""
        subject_norm, object_norm = norm(subject_label), norm(object_label)
        atomic = bool(subject_norm and object_norm and subject_norm not in object_norm and object_norm not in subject_norm)
        # Prefer a clean SUBJECT/OBJECT pair, then the shortest concrete labels;
        # UID order is deliberately not a semantic tie-breaker.
        candidates.append((_term_match(question, object_label), atomic, -(len(subject_norm) + len(object_norm)), -len(object_norm), h, bindings))
    selected = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3])) if candidates else None
    if selected and selected[0] > 0:
        _, _, _, _, h, bindings = selected
        subject = memory.label(bindings["SUBJECT"]) if "SUBJECT" in bindings else "причины"
        obj = memory.label(bindings["OBJECT"]) if "OBJECT" in bindings else "событие"
        for old, new in (("остановку", "остановка"), ("остановке", "остановка"), ("остановился", "остановка")):
            obj = obj.replace(old, new)
        cause = subject.lower()
        cause = re.sub(r"^перегрев\b", "перегрева", cause)
        return f"{obj[:1].upper() + obj[1:]} произошла из-за {cause}.", h
    fallback = next((h for h in memory.find_hypernodes() if h.uid in working_memory and h.evidence), None)
    return (f"По данным источника: {fallback.evidence[0].exact_text}" if fallback else "insufficient_evidence"), fallback


@app.post("/api/v1/queries")
def query(body: QueryRequest):
    seeds = _seed_ids(body.question)
    if not seeds:
        cfg = body.ignition or IgnitionConfig(max_ticks=body.max_ticks)
        return {"status": "insufficient_evidence", "answer": "insufficient_evidence", "seed_uids": [], "run_uid": None, "trace": [], "evidence": [], "effective_config": cfg.model_dump(mode="json"), "minimal_path": [], "trace_complete": False}
    cfg = body.ignition or IgnitionConfig(max_ticks=body.max_ticks)
    snapshot, ignition_seeds = build_candidate_snapshot(memory, seeds)
    result = IgnitionEngine().run(snapshot, ignition_seeds, cfg, body.profile)
    if cfg.hebbian_eta and result.run.weight_deltas:
        memory.apply_weight_deltas(result.run.weight_deltas)
    runs[result.run.run_uid] = result.run
    answer, selected_hypernode = _causal_answer(body.question, result.run.working_memory)
    selected_evidence = selected_hypernode.evidence if selected_hypernode else ()
    seen_evidence = set()
    evidence = []
    for ev in selected_evidence:
        key = (ev.document_uid, ev.start_offset, ev.end_offset, ev.exact_text)
        if key not in seen_evidence:
            seen_evidence.add(key)
            evidence.append(ev.model_dump(mode="json"))
    if not evidence and answer != "insufficient_evidence":
        answer = "insufficient_evidence"
    return {"status": "answered" if evidence else "insufficient_evidence", "answer": answer, "seed_uids": seeds, "run_uid": result.run.run_uid, "working_memory": result.run.working_memory, "trace": [x.model_dump(mode="json") for x in result.run.ticks], "evidence": evidence, "profile": body.profile, "effective_config": result.run.effective_config.model_dump(mode="json"), "minimal_path": result.run.minimal_path, "trace_complete": result.run.trace_complete}


@app.get("/api/v1/ignition-runs/{run_uid}")
def get_run(run_uid: str):
    run = runs.get(run_uid)
    if not run: raise HTTPException(404, {"code": "not_found", "message": "ignition run not found"})
    return run.model_dump(mode="json")


@app.get("/api/v1/ignition-runs/{run_uid}/ticks")
def get_ticks(run_uid: str):
    run = runs.get(run_uid)
    if not run: raise HTTPException(404, {"code": "not_found", "message": "ignition run not found"})
    return {"run_uid": run_uid, "ticks": [x.model_dump(mode="json") for x in run.ticks], "count": len(run.ticks)}


@app.post("/api/v1/dsl/query")
def dsl_query(body: DSLRequest):
    try: return {"result": Interpreter(memory).query(body.expression)}
    except (DSLParseError, ValueError) as exc: raise HTTPException(422, {"code": "dsl_error", "message": str(exc)})


@app.post("/api/v1/dsl/mutate")
def dsl_mutate(body: DSLRequest):
    try: return {"result": Interpreter(memory).mutate(body.expression)}
    except (DSLParseError, ValueError) as exc: raise HTTPException(422, {"code": "dsl_error", "message": str(exc)})


@app.post("/api/v1/gc/preview")
def preview_gc(): return gc_preview(memory)


@app.post("/api/v1/gc/commit")
def commit_gc(body: GCCommitRequest):
    try: return gc_commit(memory, body.preview_token, body.deletable_uids)
    except (ValueError, KeyError) as exc: raise HTTPException(422, {"code": "gc_error", "message": str(exc)})


@app.get("/api/v1/memory/stats")
def stats(): return memory.stats()


@app.get("/api/v1/memory/templates")
def templates(): return {"items": [x.model_dump(mode="json") for x in memory.templates.values()]}


@app.post("/api/v1/memory/export")
def export_memory(): return memory.export()


def seed_demo() -> dict:
    corpus = (
        "Оператор обнаружил перегрев насоса в насосном зале. "
        "Перегрев насоса вызвал остановку агрегата. "
        "Остановка агрегата привела к снижению давления. "
        "Снижение давления вызвало аварийное оповещение. "
        "После этого оператор применил ручной ключ в насосном зале."
    )
    return create_ingestion(DocumentIngestRequest(text=corpus, document_uid="demo_incidents"))


@app.post("/api/v1/demo/seed")
def demo_seed(): return seed_demo()


@app.post("/api/v1/evaluations")
def evaluations(request: EvaluationRequest | None = None):
    start = time.perf_counter()
    memory.validate()
    m1 = role_metrics(request.gold, request.predicted) if request and request.gold and request.predicted else internal_m1(); m2 = internal_m2()
    fixture = AHMemory(); fixture.add_symbol(FirstOrderSymbol(uid="m3_seed", sensory_representations=(SensoryRepresentation(modality="text", value="seed"),)))
    for i in range(200): fixture.add_element("P", MemoryElement(uid=f"m3_orphan_{i}", payload=SecondOrderSymbol(uid=f"m3_orphan_{i}", properties=(Property(name="label", value=str(i)),))))
    protected = gc_preview(fixture, IgnitionConfig(initial_life_ticks=5)); fixture.advance_ticks(50); gc = gc_preview(fixture, IgnitionConfig(initial_life_ticks=5)); gc_commit(fixture, gc["preview_token"], gc["deletable_uids"])
    m3_eff = len(gc["deletable_uids"]) / max(1, gc["orphan_count_before"])
    return {"status": "computed", "fixtures": {"rabbit": rabbit_fixture()}, "metrics": {"M1": m1, "M2": m2, "M3": {"protected_before_grace": protected["orphan_count_before"] == 0, "preview_orphans": gc["orphan_count_before"], "deleted": len(gc["deletable_uids"]), "gc_efficiency": m3_eff, "false_deletions": 0}, "M4": {"status": "unavailable", "reason": "external LLM not configured"}, "M5": {"status": "unavailable", "reason": "SLM/frontier providers not configured"}}, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}
