"""Preview the heterogeneous corpus repeatedly and report latency/error totals."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CORPUS_PATH = Path(__file__).resolve().parents[2] / "docs" / "testing" / "heterogeneous-corpus.json"
EXTENSIONS = {
    "maintenance-report": ".txt", "shift-log": ".log", "incident-ticket": ".json",
    "sensor-diary": ".csv", "dispatcher-report": ".txt", "curator-note": ".md",
    "operations-log": ".log", "expedition-journal": ".md",
    "production-summary": ".html", "technical-rehearsal-note": ".txt",
}


def build_jobs(documents: list[dict], repeat: int) -> list[dict]:
    jobs = []
    for cycle in range(repeat):
        for document in documents:
            suffix = "" if cycle == 0 else f" Идентификатор нагрузочного пакета: {cycle + 1}."
            jobs.append({
                "id": f"{document['id']}-{cycle + 1}",
                "source_name": document["id"] + EXTENSIONS.get(document["source_format"], ".txt"),
                "text": document["text"] + suffix,
            })
    return jobs


def send_preview(base_url: str, model: str, timeout: float, job: dict) -> dict:
    body = json.dumps({"text": job["text"], "source_name": job["source_name"], "parser_model": model}, ensure_ascii=False).encode("utf-8")
    request = Request(f"{base_url.rstrip('/')}/api/v1/ingestions/preview", data=body, headers={"content-type": "application/json; charset=utf-8"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return {"id": job["id"], "ok": True, "seconds": time.perf_counter() - started, "candidates": payload.get("count", 0), "coverage_warnings": len(payload.get("coverage_warnings", []))}
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"id": job["id"], "ok": False, "seconds": time.perf_counter() - started, "error": str(exc)}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash-0731:nitro")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--execute", action="store_true", help="Actually send paid/external parser requests")
    args = parser.parse_args()
    if args.repeat < 1 or args.concurrency < 1:
        parser.error("--repeat and --concurrency must be positive")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    jobs = build_jobs(corpus["documents"], args.repeat)
    plan = {"documents": len(jobs), "characters": sum(len(job["text"]) for job in jobs), "model": args.model, "concurrency": args.concurrency, "execute": args.execute}
    if not args.execute:
        print(json.dumps({"status": "dry-run", **plan}, ensure_ascii=False, indent=2))
        return
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send_preview, args.base_url, args.model, args.timeout, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    durations = [result["seconds"] for result in results]
    print(json.dumps({
        "status": "complete", **plan,
        "succeeded": sum(result["ok"] for result in results),
        "failed": sum(not result["ok"] for result in results),
        "candidates": sum(result.get("candidates", 0) for result in results),
        "coverage_warnings": sum(result.get("coverage_warnings", 0) for result in results),
        "latency_seconds": {"p50": round(percentile(durations, .5), 3), "p95": round(percentile(durations, .95), 3), "max": round(max(durations, default=0), 3)},
        "errors": [result for result in results if not result["ok"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
