"""Paired local-model pilot. Saves raw answers for semantic review; does not invent official M4/M5 scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from fastapi.testclient import TestClient
from app.core import AHMemory
from app.ingestion import Compiler, OpenAICompatibleProvider
from app.models import CandidateFact
from app.answering import build_evidence_packet, parse_evidence_response, evidence_answer_request
from app.benchmark_recording import save_json
import app.main as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ah-records", type=Path, required=True)
    parser.add_argument("--rag-contexts", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--responses", type=Path, help="reuse raw responses from a previous run only when requests match exactly")
    parser.add_argument("--offline", action="store_true", help="revalidate an existing completed run without model calls")
    args = parser.parse_args()
    if args.out.exists() and not args.resume: parser.error("output exists; use --resume")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    retrieval = json.loads(args.rag_contexts.read_text(encoding="utf-8"))
    manifest = json.loads((args.ah_records / "run.json").read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"model": args.model, "questions": digest(args.questions), "rag": digest(args.rag_contexts), "predictions": {doc["file"]: digest(args.ah_records / doc["file"]) for doc in manifest["documents"]}, "code": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *sorted((ROOT / "backend/app").glob("*.py"))]}}
    if retrieval["questions_sha256"] != identity["questions"]: parser.error("retrieval question set changed")
    if retrieval.get("corpus_sha256") != manifest.get("corpus_sha256"): parser.error("retrieval corpus differs from extraction corpus")
    if [row["id"] for row in retrieval["cases"]] != [row["id"] for row in questions["cases"]]: parser.error("retrieval order mismatch")
    result = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"scope": "local paired AH vs vector RAG pilot; citation validity is not semantic entailment; semantic review and commercial arm pending", "identity": identity, "status": "running", "cases": []}
    imported = {}
    if args.responses:
        previous = json.loads(args.responses.read_text(encoding="utf-8"))
        imported = {(row["id"], row["arm"]): row for row in previous["cases"] if "response" in row}
        result["response_import"] = {"path": str(args.responses.resolve()), "sha256": digest(args.responses), "identity": previous["identity"]}
    if result["identity"] != identity: parser.error("resume inputs/code changed")
    memory, sources = AHMemory(), {}
    result["ingestions"] = []
    for doc in manifest["documents"]:
        stored = json.loads((args.ah_records / doc["file"]).read_text(encoding="utf-8"))
        if stored["status"] != "completed" or stored["settings"]["model"] != args.model: parser.error("incomplete or different-model predictions")
        if hashlib.sha256(stored["text"].encode()).hexdigest() != stored["source_sha256"]: parser.error("source recording changed")
        sources[doc["id"]] = stored["text"]
        accepted, rejected = Compiler(memory).compile([CandidateFact.model_validate(row) for row in stored["predictions"]], stored["text"], doc["id"])
        result["ingestions"].append({"document_id": doc["id"], "accepted": len(accepted), "rejected": rejected})
    if {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources.items()} != questions["source_sha256"]:
        parser.error("AH and question corpus differ")
    if "memory" in result:
        memory = AHMemory.from_export(result["memory"])
    else:
        result["memory"] = memory.export()
    provider = OpenAICompatibleProvider("http://127.0.0.1:11434/v1", args.model, timeout_seconds=180)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior = {(row["id"], row["arm"]): row for row in result["cases"]}
    with patch.multiple(api, memory=memory, storage_mode="in-memory", runs={}):
        client = TestClient(api.app)
        for case, rag in zip(questions["cases"], retrieval["cases"]):
            response = client.post("/api/v1/queries", json={"question": case["question"], "profile": "directed_v1", "answer_model": "local"})
            response.raise_for_status()
            trace = response.json()
            ah = build_evidence_packet(memory, tuple(trace["working_memory"]), tuple(trace["answer_path"])) if trace["status"] == "answered" else []
            # Stable evidence labels/text exclude random graph UIDs from the model request and resume comparison.
            ah = sorted(ah, key=lambda fact: (fact["predicate"], json.dumps(fact["bindings"], sort_keys=True), fact["source_quotes"]))
            packets = {"ah": [{"evidence_id": f"E{i + 1}", "text": json.dumps({k: f[k] for k in ("predicate", "bindings", "source_quotes")}, ensure_ascii=False, sort_keys=True)} for i, f in enumerate(ah)], "rag": []}
            for i, chunk in enumerate(rag["contexts"]):
                if sources[chunk["document_id"]][chunk["start"]:chunk["end"]] != chunk["text"]: parser.error("RAG context is not source anchored")
                packets["rag"].append({"evidence_id": f"E{i + 1}", "text": chunk["text"]})
            for arm, facts in packets.items():
                payload = evidence_answer_request(provider, case["question"], facts)
                payload["max_tokens"] = 768
                row = prior.get((case["id"], arm))
                if row and row["request"] != payload: parser.error("resume request changed")
                if row is None:
                    row = {"id": case["id"], "arm": arm, "request": payload, "ah_trace_complete": trace["trace_complete"] if arm == "ah" else None}
                    if arm == "ah":
                        row["ah_trace"] = {key: trace.get(key, []) for key in ("seed_uids", "answer_path", "working_memory")}
                        row["ah_trace"]["trace_complete"] = trace["trace_complete"]
                        row["ah_facts"] = [{**fact, "evidence_id": f"E{i + 1}"} for i, fact in enumerate(ah)]
                    result["cases"].append(row)
                if "response" not in row and (case["id"], arm) in imported:
                    stored = imported[(case["id"], arm)]
                    if stored["request"] != payload: parser.error("imported request changed")
                    row.update(response=stored["response"], elapsed_seconds=stored["elapsed_seconds"], imported_response=True)
                if "response" not in row:
                    if args.offline: parser.error("offline run lacks a recorded response")
                    save_json(args.out, result)
                    start = time.perf_counter()
                    try:
                        row["response"] = provider._request_json(payload)
                    except ValueError as exc:
                        row["provider_error"] = str(exc)
                        row["answer_status"] = "provider_error"
                        save_json(args.out, result)
                        raise
                    row.pop("provider_error", None)
                    row["elapsed_seconds"] = time.perf_counter() - start
                    save_json(args.out, result)
                try:
                    row["validated"] = parse_evidence_response(row["response"], facts)
                    row["answer_status"] = row["validated"]["status"]
                    row.pop("validation_error", None)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    row["validation_error"] = str(exc)
                    row["answer_status"] = "invalid_response"
                    row.pop("validated", None)
                save_json(args.out, result)
            print(f"{case['id']}: both arms recorded", flush=True)
    result["status"] = "completed"
    save_json(args.out, result)


if __name__ == "__main__": main()
