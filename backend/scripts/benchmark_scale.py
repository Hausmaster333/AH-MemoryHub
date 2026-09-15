"""Offline synthetic scalability probe, not the official semantic benchmark corpus."""
import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core import AHMemory
from app.engine import IgnitionEngine, gc_preview, gc_commit
from app.ingestion import RuleBasedProvider, ingest
from app.models import IgnitionConfig


def synthetic_corpus(min_words=15000):
    documents, words = [], 0
    while words < min_words:
        n = len(documents) + 1
        cause = (
            f"Подтверждённое журналом смены повреждение уплотнения насосного контура НК-{n} "
            "после длительной непрерывной эксплуатации оборудования при повышенной нагрузке "
            "с зафиксированным отклонением температуры рабочей жидкости от установленного "
            "регламентом диапазона допустимых значений несмотря на выполненный персоналом "
            "визуальный осмотр доступных соединений перед началом очередного производственного цикла"
        )
        effect = (
            f"автоматическое отключение резервного агрегата РА-{n} по сигналу независимой "
            "системы контроля состояния оборудования с регистрацией времени события "
            "дежурным специалистом эксплуатационной службы согласно действующему порядку "
            "технического обслуживания после подтверждения показаний измерительных приборов "
            "и проверки исправности соединительных кабелей без дополнительного вмешательства "
            "оператора центрального пульта управления технологической установкой"
        )
        text = f"{cause} вызвало {effect}."
        documents.append({"id": f"synthetic_{n}", "text": text})
        words += len(re.findall(r"\S+", text))
    return {"synthetic": True, "purpose": "size/latency only, not semantic generalization", "words": words, "documents": documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    corpus = synthetic_corpus()
    (args.out / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    memory, accepted, rejected = AHMemory(), 0, []
    start = time.perf_counter()
    for index, document in enumerate(corpus["documents"]):
        result = ingest(memory, document["text"], document["id"], RuleBasedProvider())
        accepted += len(result.accepted)
        rejected.extend(result.rejected)
        if (index + 1) % 20 == 0:
            print(f"{index+1}/{len(corpus['documents'])}: {len(memory.elements)} elements, {time.perf_counter()-start:.1f}s", flush=True)
    ingestion_seconds = time.perf_counter() - start
    start = time.perf_counter(); snapshot = memory.snapshot(); snapshot_ms = (time.perf_counter() - start) * 1000
    engine = IgnitionEngine()
    seeds = list(snapshot.symbols)  # Dense excitation, not an isolated unused template symbol.
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        result = engine.run(snapshot, seeds, IgnitionConfig(max_ticks=1, rhythm_hz=0), "directed_v1")
        samples.append((time.perf_counter() - start) * 1000)
    live_before = set(memory.symbols) | set(memory.elements)
    result = engine.run(snapshot, seeds, IgnitionConfig(max_ticks=50, rhythm_hz=0), "directed_v1")
    memory.advance_ticks(result.run.elapsed_ticks)
    preview = gc_preview(memory)
    collected = gc_commit(memory, preview["preview_token"])
    memory.validate()
    counts = {section: len(items) for section,items in memory.sections.items()}
    counts.update(S=len(memory.symbols), L=len(memory.links), CPH=len(memory.elements))
    report = {"synthetic": True, "official_acceptance": False, "words": corpus["words"], "documents":len(corpus["documents"]), "accepted":accepted, "rejected":len(rejected), "counts_after_gc":counts, "ingestion_seconds":ingestion_seconds, "snapshot_ms":snapshot_ms, "dense_single_tick_ms":samples, "max_tick_ms":max(samples), "gc_deleted":collected["deleted_count"], "lost_live":sorted(live_before-(set(memory.symbols)|set(memory.elements))), "python":sys.version, "platform":platform.platform()}
    report["size_latency_checks"] = corpus["words"] >= 15000 and counts["CPH"] >= 1000 and counts["S"] >= 150 and max(samples) <= 500 and not report["lost_live"]
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "memory.json").write_text(json.dumps(memory.export(), ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
