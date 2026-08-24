from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / "backend" / ".env")

from app.evaluation import CORPUS_PATH, GOLD_PATH, corpus_m2, corpus_role_metrics, internal_m3, load_gold
from app.ingestion import RuleBasedProvider, extract_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned AH-MemoryHub M1/M2 benchmark")
    parser.add_argument("--provider", choices=("rule", "nitro"), default="rule")
    parser.add_argument("--execute", action="store_true", help="allow paid OpenRouter calls")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.provider == "nitro" and not args.execute: parser.error("Nitro sends source text to OpenRouter; add --execute")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    gold = load_gold()
    sources = {document["id"]: document["text"] for document in corpus["documents"]}
    predictions, providers = {}, {}
    for document in gold["documents"]:
        if args.provider == "nitro":
            candidates, meta = extract_candidates(sources[document["id"]], model_override="deepseek/deepseek-v4-flash-0731:nitro")
        else:
            candidates, meta = extract_candidates(sources[document["id"]], RuleBasedProvider())
        predictions[document["id"]] = candidates
        providers[document["id"]] = {key: meta.get(key) for key in ("active", "model_id", "prompt_version", "cached", "warnings", "coverage_warnings")}
    report = {"provider": args.provider, "providers": providers, "M1": corpus_role_metrics(predictions), "M2": corpus_m2(), "M3": internal_m3()}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(payload)


if __name__ == "__main__": main()
