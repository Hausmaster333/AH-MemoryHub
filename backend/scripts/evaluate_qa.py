"""Small offline end-to-end QA regression benchmark, not the official M2 score."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from app.core import AHMemory
from app.engine import evidence_trace
from app.ingestion import RuleBasedProvider
from app.ingestion import Compiler
from app.models import CandidateFact
import app.main as api


def check_response(case, data, memory, sources):
    if case.get("answer") is None and not case.get("quotes"):
        return {"refusal": data["status"] == "insufficient_evidence", "no_evidence": not data["evidence"] and not data["answer_path"]}
    run = api.runs.get(data.get("run_uid"))
    goals = tuple(uid for uid in data["answer_path"] if uid not in memory.links)
    proof = evidence_trace(run, goals, data["seed_uids"]) if run else ()
    return {"answer": data["status"] == "answered" and (data["answer"] == case["answer"] if "answer" in case else all(part.casefold() in data["answer"].casefold() for part in case["answer_contains"])),
            "proof": bool(proof) and data["trace_complete"] and set(goals) <= set(data["working_memory"]) and set(data["answer_path"]) <= set(proof),
            "quotes": bool(data["evidence"]) and all(ev["document_uid"] in sources and sources[ev["document_uid"]][ev["start_offset"]:ev["end_offset"]] == ev["exact_text"] for ev in data["evidence"]),
            "expected_quotes": set(case.get("quotes", [])) <= {ev["exact_text"] for ev in data["evidence"]},
            "source_scope": not case.get("document_id") or all(ev["document_uid"] == case["document_id"] for ev in data["evidence"])}


def evaluate_recorded(corpus: dict, records: Path) -> dict:
    manifest = json.loads((records / "run.json").read_text(encoding="utf-8"))
    memory, sources, ingestions, hashes = AHMemory(), {}, [], {}
    for document in manifest["documents"]:
        path = (records / document["file"]).resolve()
        if not path.is_relative_to(records.resolve()): raise ValueError("record path escapes directory")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("status") != "completed" or hashlib.sha256(stored["text"].encode()).hexdigest() != stored["source_sha256"]:
            raise ValueError("incomplete or changed source recording")
        sources[document["id"]] = stored["text"]
        hashes[document["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted, rejected = Compiler(memory).compile([CandidateFact.model_validate(row) for row in stored["predictions"]], stored["text"], document["id"])
        ingestions.append({"document_id": document["id"], "accepted": len(accepted), "rejected": rejected})
        print(f"compiled {document['id']}: {len(accepted)}", flush=True)
    for case in corpus["cases"]:
        if case.get("document_id") and corpus["source_sha256"][case["document_id"]] != hashlib.sha256(sources[case["document_id"]].encode()).hexdigest():
            raise ValueError("QA corpus source mismatch")
        if any(quote not in sources[case["document_id"]] for quote in case.get("quotes", [])): raise ValueError("expected quote not in source")
    results = []
    with patch.multiple(api, memory=memory, storage_mode="in-memory", runs={}), patch("app.ingestion.OpenAICompatibleProvider._request_json", side_effect=AssertionError("offline benchmark forbids model calls")):
        client = TestClient(api.app)
        for case in corpus["cases"]:
            response = client.post("/api/v1/queries", json={"question": case["question"], "profile": "directed_v1", "answer_model": "local"})
            response.raise_for_status()
            data = response.json()
            checks = check_response(case, data, memory, sources)
            results.append({"id": case["id"], "passed": all(checks.values()), "checks": checks, "response": data})
    return {"scope": "QA on one shared memory compiled from recorded model predictions; no fresh extraction or model answer", "provider": "recorded_predictions", "profile": "directed_v1", "records_sha256": hashes, "prediction_code_sha256": manifest.get("code_sha256"), "ingestions": ingestions, "total": len(results), "passed": sum(row["passed"] for row in results), "cases": results}


def evaluate(corpus: dict) -> dict:
    results = []
    for case in corpus["cases"]:
        memory = AHMemory()
        provider = RuleBasedProvider()
        with patch.multiple(api, memory=memory, storage_mode="in-memory", runs={}, ingestions={}, document_index={}), patch("app.ingestion.configured_provider", return_value=(provider, {"configured": "rule", "active": "rule", "model_id": provider.model_id})), patch("app.ingestion.OpenAICompatibleProvider._request_json", side_effect=AssertionError("offline benchmark forbids model calls")):
            client = TestClient(api.app)
            ingestion = client.post("/api/v1/ingestions", json={"text": case["source"]})
            ingestion.raise_for_status()
            response = client.post("/api/v1/queries", json={"question": case["question"], "profile": "directed_v1", "answer_model": "local"})
            response.raise_for_status()
            data = response.json()
            checks = check_response(case, data, memory, {ingestion.json()["document_uid"]: case["source"]})
            results.append({"id": case["id"], "passed": all(checks.values()), "checks": checks, "response": data, "ingestion": ingestion.json()})
    return {"scope": "end-to-end regression corpus; not official M2 or held-out accuracy", "provider": "rule", "profile": "directed_v1", "total": len(results), "passed": sum(row["passed"] for row in results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "docs/testing/qa-regression.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=Path, help="compile recorded predictions into one shared memory without model calls")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    report = evaluate_recorded(corpus, args.records) if args.records else evaluate(corpus)
    report["corpus_sha256"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    report["code_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "backend/app").glob("*.py"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{report['total']}")
    for case in report["cases"]:
        if not case["passed"]: print(case["id"], case["checks"], ascii(case["response"]["answer"]))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
