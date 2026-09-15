from __future__ import annotations

import argparse
import json
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / "backend" / ".env")

from app.evaluation import CORPUS_PATH, GOLD_PATH, corpus_m2, corpus_role_metrics, internal_m3, load_gold
from app.ingestion import RuleBasedProvider, OpenAICompatibleProvider, extract_candidates
from app.benchmark_recording import document_predictions, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned AH-MemoryHub M1/M2 benchmark")
    parser.add_argument("--provider", choices=("rule", "nitro", "local"), default="rule")
    parser.add_argument("--model", help="installed local Ollama model; required for --provider local")
    parser.add_argument("--perception-format", choices=("ir", "roles"), default="ir", help="local extraction response schema")
    parser.add_argument("--execute", action="store_true", help="allow paid OpenRouter calls")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--matcher", choices=("legacy_v1", "morph_v1"), default="legacy_v1")
    parser.add_argument("--records", type=Path, help="new directory for per-document responses and predictions")
    parser.add_argument("--corpus", type=Path, help="custom source corpus; requires --record-only")
    parser.add_argument("--record-only", action="store_true", help="capture extraction without claiming M1/M2/M3 scores")
    parser.add_argument("--replay", type=Path, help="replay previous model records, without network")
    args = parser.parse_args()
    if args.corpus and not args.record_only: parser.error("custom corpus requires --record-only; the standard gold does not apply")
    if args.replay:
        previous = json.loads((args.replay / "run.json").read_text(encoding="utf-8"))
        args.provider = previous["provider"]
        if args.provider not in ("nitro", "local"): parser.error("replay requires recorded model responses")
    if args.provider == "local" and not args.model and not args.replay: parser.error("--provider local requires --model")
    if args.provider == "nitro" and not args.execute and not args.replay: parser.error("Nitro sends source text to OpenRouter; add --execute")
    records = args.records or (args.out.with_suffix(".records") if args.out else ROOT / "docs/testing/results" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ"))
    records.mkdir(parents=True, exist_ok=False)
    corpus_path = args.corpus or CORPUS_PATH
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    gold = corpus if args.record_only else load_gold()
    sources = {document["id"]: document["text"] for document in corpus["documents"]}
    if len(sources) != len(corpus["documents"]) or not sources: parser.error("corpus must have unique document ids and be nonempty")
    predictions, providers = {}, {}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "provider": args.provider, "mode": "replay" if args.replay else "fresh", "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [*sorted((ROOT / "backend/app").glob("*.py")), Path(__file__)]}, "documents": []}
    save_json(records / "run.json", manifest)
    manifest["corpus_sha256"] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    manifest["gold_sha256"] = None if args.record_only else hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    manifest["replayed_from"] = str(args.replay.resolve()) if args.replay else None
    save_json(records / "run.json", manifest)
    previous = json.loads((args.replay / "run.json").read_text(encoding="utf-8")) if args.replay else None
    if previous and [item["id"] for item in previous["documents"]] != [item["id"] for item in gold["documents"]]:
        parser.error("replay document set/order does not match benchmark")
    for index, document in enumerate(gold["documents"]):
        record_path = records / f"document-{index:03}.json"
        if args.provider == "local" and not args.replay:
            provider = OpenAICompatibleProvider("http://127.0.0.1:11434/v1", args.model, timeout_seconds=180, response_format="json_schema", perception_format=args.perception_format)
            with patch("app.ingestion.configured_provider", return_value=(provider, {"configured": "local", "active": "openai_compatible", "model_id": provider.model_id, "fallback": False})):
                candidates, meta = document_predictions(sources[document["id"]], record_path, args.model)
        elif args.provider == "nitro" or args.replay:
            recorded = json.loads((args.replay / record_path.name).read_text(encoding="utf-8")) if args.replay else None
            model = recorded["settings"]["model"] if recorded else "deepseek/deepseek-v4-flash-0731:nitro"
            candidates, meta = document_predictions(sources[document["id"]], record_path, model, recorded)
        else:
            candidates, meta = extract_candidates(sources[document["id"]], RuleBasedProvider())
            save_json(record_path, {"text": sources[document["id"]], "predictions": [candidate.model_dump(mode="json") for candidate in candidates], "metadata": meta, "status": "completed", "provider": "rule"})
        predictions[document["id"]] = candidates
        providers[document["id"]] = {key: meta.get(key) for key in ("active", "model_id", "prompt_version", "cached", "warnings", "coverage_warnings")}
        manifest["documents"].append({"id": document["id"], "file": record_path.name})
        save_json(records / "run.json", manifest)
        print(f"{index + 1}/{len(gold['documents'])}: {document['id']} ({len(candidates)} facts)", file=sys.stderr, flush=True)
    report = {"provider": args.provider, "providers": providers}
    if args.record_only:
        report.update(scope="extraction recording only; no M1/M2/M3 measurement", facts=sum(len(rows) for rows in predictions.values()))
    else:
        report.update(M1=corpus_role_metrics(predictions, matcher=args.matcher), M2=corpus_m2(), M3=internal_m3())
    report["recording"] = {"directory": str(records.resolve()), "mode": manifest["mode"]}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(payload)


if __name__ == "__main__": main()
