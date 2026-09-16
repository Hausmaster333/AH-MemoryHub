from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import main as api
from app.core import AHMemory, lexical_key
from app.ingestion import RuleBasedProvider, ingest
from app.models import QueryRequest
from app.models import Hypernode


CHAINS = (
    ("server", ("короткое замыкание", "отключение сервера", "остановка обработки", "задержка отчёта", "нарушение регламента", "уведомление дежурного", "ручная проверка")),
    ("water", ("засор фильтра", "падение расхода воды", "перегрев насоса", "аварийный сигнал", "остановка линии", "переключение клапана", "восстановление подачи")),
    ("warehouse", ("обрыв кабеля", "потеря связи со сканером", "ошибка учёта палеты", "блокировка конвейера", "задержка отгрузки", "вызов техника", "возобновление сортировки")),
    ("greenhouse", ("отказ вентилятора", "рост температуры", "увядание растений", "срабатывание тревоги", "запуск резервного контура", "охлаждение секции", "возврат в штатный режим")),
)
DEPTHS = ((1, 2, 3, 4, 5), (1, 2, 3, 5, 6), (1, 2, 4, 5, 6), (1, 3, 4, 5, 6))


def reset_api_memory() -> None:
    api.memory = AHMemory()
    api.ingestions.clear(); api.document_index.clear(); api.runs.clear(); api.previews.clear()


def source(nodes: tuple[str, ...]) -> str:
    return " ".join(f"{nodes[index].capitalize()} вызвало {nodes[index + 1]}." for index in range(len(nodes) - 1))


def contains_phrase(answer: str, expected: str) -> bool:
    return set(lexical_key(expected)) <= set(lexical_key(answer))


def role_chain_complete(memory: AHMemory, hypernodes: list[str], nodes: tuple[str, ...], depth: int) -> bool:
    if len(hypernodes) != depth: return False
    for index, uid in enumerate(hypernodes):
        item = memory.elements[uid].payload
        if not isinstance(item, Hypernode): return False
        roles = {binding.role_id: binding.target_ref.target_uid for binding in item.role_bindings}
        if lexical_key(memory.label(roles.get("SUBJECT", ""))) != lexical_key(nodes[depth - index - 1]): return False
        if lexical_key(memory.label(roles.get("OBJECT", ""))) != lexical_key(nodes[depth - index]): return False
        if not item.evidence or not item.evidence[0].exact_text: return False
    return True


def main() -> None:
    cases = []
    for (domain, nodes), depths in zip(CHAINS, DEPTHS):
        reset_api_memory()
        text = source(nodes)
        admitted = ingest(api.memory, text, f"m2-e2e-{domain}", RuleBasedProvider())
        for depth in depths:
            question = f"Какова первопричина события «{nodes[depth]}»?"
            response = api.query(QueryRequest(question=question, max_ticks=24, answer_model="local"))
            answer_path = list(response["answer_path"])
            hypernodes = [uid for uid in answer_path if uid in api.memory.elements and isinstance(api.memory.elements[uid].payload, Hypernode)]
            semantic_chain = role_chain_complete(api.memory, hypernodes, nodes, depth)
            activated_uids = {uid for event in response["trace"] for uid in (event.get("link_or_hypernode_uid"), event.get("target_uid")) if uid}
            trace_complete = bool(
                response["trace_complete"]
                and semantic_chain
                and set(hypernodes) <= activated_uids
                and set(answer_path) <= set(response["minimal_path"])
            )
            answer_correct = response["status"] == "answered" and contains_phrase(response["answer"], nodes[0])
            cases.append({
                "id": f"{domain}-{depth}", "depth": depth, "question": question,
                "expected_answer": nodes[0], "actual_answer": response["answer"],
                "answer_correct": answer_correct, "trace_complete": trace_complete,
                "semantic_role_chain_complete": semantic_chain,
                "answer_path": answer_path, "minimal_path": list(response["minimal_path"]),
                "evidence_count": len(response["evidence"]), "admitted_hypernodes": len(admitted.accepted),
            })
    passed = sum(case["answer_correct"] and case["trace_complete"] for case in cases)
    explain_score = sum(case["answer_correct"] * (case["depth"] / 6) * case["trace_complete"] for case in cases) / len(cases)
    report = {
        "benchmark": "m2-internal-e2e-v2", "official_acceptance": False,
        "pipeline": "rule-parser -> admission -> AH hypernodes -> /api/v1/queries logic -> ignition -> local evidence answer",
        "trace_scope": "semantic role chain and selected UID activation dependencies; does not claim ignition physically traverses each semantic edge in sequence",
        "total": len(cases), "passed": passed, "trace_pass_rate": passed / len(cases),
        "explain_score": explain_score, "max_depth": max((case["depth"] for case in cases if case["trace_complete"]), default=0),
        "cases": cases,
    }
    out = ROOT / "docs" / "testing" / "results" / "m2-internal-e2e-v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    if passed != len(cases): raise SystemExit(1)


if __name__ == "__main__": main()
