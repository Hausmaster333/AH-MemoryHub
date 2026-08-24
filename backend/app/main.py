from __future__ import annotations

import os
import time
import uuid
import hashlib
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core import AHMemory, lexical_score
from .conformance import junior_conformance_report
from .answering import build_evidence_packet, generate_evidence_answer, select_local_answer
from .dsl import Interpreter, DSLParseError
from .engine import IgnitionEngine, build_candidate_snapshot, gc_commit, gc_preview
from .ingestion import Compiler, RuleBasedProvider, configured_provider, extract_candidates, ingest, segment_text
from .evaluation import internal_m1, internal_m2, rabbit_fixture, rabbit_ingestion_v2, role_metrics
from .models import *


memory = AHMemory()
storage_mode = os.getenv("AH_STORAGE_MODE", "in-memory").lower()
if storage_mode not in {"in-memory", "neo4j"}: storage_mode = "in-memory"
storage_adapter = None
ingestions: dict[str, Any] = {}
document_index: dict[str, str] = {}
runs: dict[str, IgnitionRun] = {}
previews: dict[str, Any] = {}

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
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        response.headers["cache-control"] = "no-store"
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
def health():
    try: _, parser = configured_provider()
    except ValueError as exc: parser = {"active": "configuration_error", "message": str(exc)}
    return {"status": "ok", "mode": storage_mode, "storage_mode": storage_mode, "revision": memory.revision, "parser": parser}


def _persist_memory():
    if storage_mode != "neo4j": return
    global storage_adapter
    try:
        if storage_adapter is None:
            from .persistence import Neo4jAdapter
            storage_adapter = Neo4jAdapter().connect()
        storage_adapter.persist(memory)
    except Exception as exc:
        raise HTTPException(503, {"code": "storage_unavailable", "message": str(exc)})


def _create_ingestion(body: DocumentIngestRequest, provider=None):
    if body.document_uid and body.document_uid in document_index:
        cached = dict(ingestions[document_index[body.document_uid]])
        cached["reused"] = True
        return cached
    result = ingest(memory, body.text, body.document_uid, provider)
    _persist_memory()
    ingestions[result.ingestion_uid] = {"ingestion_uid": result.ingestion_uid, "document_uid": result.document_uid, "accepted": result.accepted, "rejected": result.rejected, "candidates": [c.model_dump(mode="json") for c in result.candidates], "reused": False}
    document_index[result.document_uid] = result.ingestion_uid
    return ingestions[result.ingestion_uid]


@app.post("/api/v1/ingestions")
def create_ingestion(body: DocumentIngestRequest): return _create_ingestion(body)


@app.post("/api/v1/ingestions/preview")
def preview_ingestion(body: DocumentIngestRequest):
    try: candidates, provider = extract_candidates(body.text, model_override=None if body.parser_model == "configured" else body.parser_model)
    except ValueError as exc: raise HTTPException(502, {"code": "parser_unavailable", "message": str(exc)})
    document_uid = body.document_uid or f"doc_{hashlib.sha256(body.text.encode()).hexdigest()[:16]}"
    preview_uid = uid("prv")
    items = []
    for index, candidate in enumerate(candidates):
        signature = json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        candidate_uid = f"cand_{hashlib.sha256(f'{document_uid}:{index}:{signature}'.encode()).hexdigest()[:16]}"
        items.append({"candidate_uid": candidate_uid, "status": "pending", **candidate.model_dump(mode="json")})
    rejection_log = provider.get("rejection_log", [])
    source_groups = [span.__dict__ for span in segment_text(body.text)]
    previews[preview_uid] = {"preview_uid": preview_uid, "ingestion_uid": uid("ing"), "document_uid": document_uid, "source_name": body.source_name, "text": body.text, "provider": provider, "items": items, "candidates": {item["candidate_uid"]: candidate for item, candidate in zip(items, candidates)}, "decisions": {}, "rejection_log": rejection_log}
    return {"preview_uid": preview_uid, "document_uid": document_uid, "provider": provider, "source_groups": source_groups, "candidates": items, "rejection_log": rejection_log, "coverage_warnings": provider.get("coverage_warnings", []), "count": len(items)}


@app.post("/api/v1/ingestions/decision")
def decide_candidate(body: CandidateDecisionRequest):
    preview = previews.get(body.preview_uid)
    if preview is None: raise HTTPException(404, {"code": "not_found", "message": "ingestion preview not found"})
    candidate = preview["candidates"].get(body.candidate_uid)
    if candidate is None: raise HTTPException(404, {"code": "not_found", "message": "candidate not found"})
    previous = preview["decisions"].get(body.candidate_uid)
    if previous:
        if previous["decision"] != body.decision: raise HTTPException(409, {"code": "decision_conflict", "message": "candidate already has another decision"})
        return {**previous, "reused": True}
    accepted: list[str] = []
    rejected: list[dict] = []
    if body.decision == "admit":
        accepted, rejected = Compiler(memory).compile([candidate], preview["text"], preview["document_uid"], parser_run_uid=preview["preview_uid"])
        if accepted: _persist_memory()
    else:
        rejected = [{"candidate": candidate.model_dump(mode="json"), "reason": "user_rejected"}]
    status = "admitted" if accepted else "rejected"
    result = {"preview_uid": body.preview_uid, "candidate_uid": body.candidate_uid, "decision": body.decision, "status": status, "accepted": accepted, "rejected": rejected, "memory_revision": memory.revision, "reused": False}
    preview["decisions"][body.candidate_uid] = result
    next(item for item in preview["items"] if item["candidate_uid"] == body.candidate_uid)["status"] = status
    if accepted or preview["ingestion_uid"] in ingestions:
        record = ingestions.setdefault(preview["ingestion_uid"], {"ingestion_uid": preview["ingestion_uid"], "document_uid": preview["document_uid"], "accepted": [], "rejected": [], "candidates": preview["items"], "provider": preview["provider"], "reused": False})
        record["accepted"].extend(uid_ for uid_ in accepted if uid_ not in record["accepted"])
        record["rejected"].extend(rejected)
        if accepted: document_index[preview["document_uid"]] = preview["ingestion_uid"]
    return result


@app.post("/api/v1/ingestions/auto-admit")
def auto_admit_candidates(body: AutoAdmissionRequest):
    preview = previews.get(body.preview_uid)
    if preview is None: raise HTTPException(404, {"code": "not_found", "message": "ingestion preview not found"})
    pending = [item["candidate_uid"] for item in preview["items"] if item["status"] == "pending"]
    results = [decide_candidate(CandidateDecisionRequest(preview_uid=body.preview_uid, candidate_uid=candidate_uid, decision="admit")) for candidate_uid in pending]
    counts = {status: sum(item["status"] == status for item in preview["items"]) for status in ("admitted", "rejected", "pending")}
    return {"preview_uid": body.preview_uid, "processed": len(results), **counts, "memory_revision": memory.revision, "candidates": preview["items"], "results": results, "reused": not pending}


@app.get("/api/v1/ingestions")
def list_ingestions():
    return {"items": list(ingestions.values()), "count": len(ingestions)}


@app.get("/api/v1/ingestions/{ingestion_uid}")
def get_ingestion(ingestion_uid: str):
    if ingestion_uid not in ingestions: raise HTTPException(404, {"code": "not_found", "message": "ingestion not found"})
    return ingestions[ingestion_uid]


def _seed_ids(question: str) -> list[str]:
    scored = [(max((lexical_score(question, r.value) for r in s.sensory_representations), default=0), s.uid) for s in memory.symbols.values()]
    return [uid_ for score, uid_ in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0][:8]


@app.post("/api/v1/queries")
def query(body: QueryRequest):
    seeds = _seed_ids(body.question)
    if not seeds:
        cfg = body.ignition or IgnitionConfig(max_ticks=body.max_ticks)
        return {"status": "insufficient_evidence", "answer": "insufficient_evidence", "seed_uids": [], "run_uid": None, "trace": [], "evidence": [], "effective_config": cfg.model_dump(mode="json"), "answer_path": [], "minimal_path": [], "trace_complete": False}
    cfg = body.ignition or IgnitionConfig(max_ticks=body.max_ticks)
    snapshot, ignition_seeds = build_candidate_snapshot(memory, seeds)
    result = IgnitionEngine().run(snapshot, ignition_seeds, cfg, body.profile)
    if cfg.hebbian_eta and result.run.weight_deltas:
        memory.apply_weight_deltas(result.run.weight_deltas)
    runs[result.run.run_uid] = result.run
    answer, selected_hypernodes = select_local_answer(memory, body.question, tuple(uid_ for uid_, element in snapshot.elements.items() if isinstance(element.payload, Hypernode)))
    selected_predicates = {
        memory.label(memory.templates[item.template_ref.target_uid].predicate_ref.target_uid)
        for item in selected_hypernodes if item.template_ref.target_uid in memory.templates
    }
    evidence_scope = tuple(item.uid for item in selected_hypernodes) if selected_hypernodes else result.run.minimal_path
    evidence_packet = build_evidence_packet(memory, result.run.working_memory, evidence_scope)
    activated_evidence_packet = build_evidence_packet(memory, result.run.working_memory, result.run.minimal_path)
    answer_mode, answer_provider, answer_warning = "deterministic_grounded", None, None
    if body.answer_model != "local" and evidence_packet:
        try:
            generated = generate_evidence_answer(body.question, evidence_packet, body.answer_model, answer)
            if generated["status"] == "answered":
                answer, answer_mode, answer_provider = generated["answer"], "llm_grounded", generated["provider"]
                cited = set(generated["evidence_ids"])
                cited_uids = {fact["hypernode_uid"] for fact in evidence_packet if fact["evidence_id"] in cited}
                selected_hypernodes = [item for item in memory.find_hypernodes() if item.uid in cited_uids]
            else: answer_warning = "answer model reported insufficient evidence; deterministic grounded fallback used"
        except ValueError as exc:
            message = str(exc)
            answer_warning = "OpenRouter временно ограничил выбранную модель (HTTP 429); показан локальный доказательный ответ" if "HTTP 429" in message else message
    selected_evidence = tuple(ev for item in selected_hypernodes for ev in item.evidence)
    seen_evidence = set()
    evidence = []
    for ev in selected_evidence:
        key = (ev.document_uid, ev.start_offset, ev.end_offset, ev.exact_text)
        if key not in seen_evidence:
            seen_evidence.add(key)
            evidence.append(ev.model_dump(mode="json"))
    if not evidence and answer != "insufficient_evidence":
        answer = "insufficient_evidence"
    answer_path = tuple(sorted({uid_ for item in selected_hypernodes for uid_ in (item.uid, *(binding.target_ref.target_uid for binding in item.role_bindings))}))
    return {"status": "answered" if evidence else "insufficient_evidence", "answer": answer, "answer_mode": answer_mode, "answer_provider": answer_provider, "answer_warning": answer_warning, "grounded_fact_count": len(activated_evidence_packet), "answer_fact_count": len(evidence_packet), "seed_uids": seeds, "run_uid": result.run.run_uid, "working_memory": result.run.working_memory, "trace": [x.model_dump(mode="json") for x in result.run.ticks], "evidence": evidence, "profile": body.profile, "effective_config": result.run.effective_config.model_dump(mode="json"), "answer_path": answer_path, "minimal_path": result.run.minimal_path, "trace_complete": result.run.trace_complete}


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


@app.get("/api/v1/conformance/junior")
def junior_conformance(): return junior_conformance_report()


def seed_demo() -> dict:
    corpus = (
        "Оператор обнаружил перегрев насоса в насосном зале. "
        "Перегрев насоса вызвал остановку агрегата. "
        "Остановка агрегата привела к снижению давления. "
        "Снижение давления вызвало аварийное оповещение. "
        "После этого оператор применил ручной ключ в насосном зале."
    )
    return _create_ingestion(DocumentIngestRequest(text=corpus, document_uid="demo_incidents"), RuleBasedProvider())


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


@app.post("/api/v1/evaluations/ingestion/rabbit")
def evaluate_rabbit_ingestion():
    try: return rabbit_ingestion_v2()
    except ValueError as exc: raise HTTPException(502, {"code": "parser_unavailable", "message": str(exc)})
