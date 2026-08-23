from __future__ import annotations

from .core import AHMemory, norm
from .conformance import RABBIT_SOURCE, junior_conformance_report
from .engine import IgnitionEngine
from .ingestion import Provider, RuleBasedProvider, extract_candidates
from .models import *

def rabbit_fixture() -> dict:
    report = junior_conformance_report()
    return {"facts": 8, "accepted": 8, "conformant": report["status"] == "conformant", "report": report}


def rabbit_ingestion_v2(provider: Provider | None = None) -> dict:
    candidates, provider_meta = extract_candidates(RABBIT_SOURCE, provider)
    gold = (
        (({"IS-A"}, {"SUBJECT": ("зая",), "OBJECT": ("звер",)}),),
        (({"LIVE", "LOCATED_AT"}, {"SUBJECT": ("зая",), "LOCATION": ("луг", "лес")}),),
        (({"HAS"}, {"SUBJECT": ("зая",), "OBJECT": ("лап",)}),),
        (({"RUN"}, {"SUBJECT": ("зая",), "HOW-TO": ("быстр",)}),),
        (({"HAS"}, {"SUBJECT": ("зая",), "OBJECT": ("уш", "ух")}), ({"HAS_STATE"}, {"SUBJECT": ("уш", "ух"), "STATE": ("длин",)})),
        (({"HAS"}, {"SUBJECT": ("зая",), "OBJECT": ("хвост",)}), ({"HAS_STATE"}, {"SUBJECT": ("хвост",), "STATE": ("круг", "пушист")})),
        (({"HAS"}, {"SUBJECT": ("зая",), "OBJECT": ("шерст",), "TIME": ("лет",)}), ({"HAS_STATE"}, {"SUBJECT": ("шерст",), "STATE": ("корич",), "TIME": ("лет",)})),
        (({"HAS"}, {"SUBJECT": ("зая",), "OBJECT": ("шерст",), "TIME": ("зим",)}), ({"HAS_STATE"}, {"SUBJECT": ("шерст",), "STATE": ("бел",), "TIME": ("зим",)})),
    )
    matched: list[int] = []
    for index, alternatives in enumerate(gold, 1):
        if any(
            candidate.predicate in predicates
            and all(any(token in {binding.role_id: norm(binding.value) for binding in candidate.bindings}.get(role, "") for token in tokens) for role, tokens in roles.items())
            for predicates, roles in alternatives for candidate in candidates
        ): matched.append(index)
    score = len(matched)
    return {"status": "pass" if score >= 6 else "fail", "score": score, "total": 8, "matched_facts": matched, "candidate_count": len(candidates), "provider": provider_meta}


def role_metrics(gold: list[dict], predicted: list[dict]) -> dict:
    roles = {"SUBJECT", "OBJECT", "LOCATION"}; result = {}
    for role in roles:
        g = {(norm(x.get("predicate", "")), norm(x.get("value", ""))) for row in gold for x in row.get("bindings", []) if x.get("role") == role}
        p = {(norm(x.get("predicate", "")), norm(x.get("value", ""))) for row in predicted for x in row.get("bindings", []) if x.get("role") == role}
        tp = len(g & p); precision = tp / len(p) if p else 0.0; recall = tp / len(g) if g else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[role] = {"precision": precision, "recall": recall, "f1": f1, "gold": len(g), "predicted": len(p)}
    result["weighted_f1"] = (2 * result["SUBJECT"]["f1"] + 2 * result["OBJECT"]["f1"] + result["LOCATION"]["f1"]) / 5
    return result


def internal_m1() -> dict:
    text = "Перегрев насоса вызвал остановку агрегата в насосном зале."
    gold = [{"predicate": "CAUSE", "bindings": [{"role": "SUBJECT", "value": "Перегрев насоса"}, {"role": "OBJECT", "value": "остановку агрегата"}, {"role": "LOCATION", "value": "насосном зале"}]}]
    predicted = [{"predicate": c.predicate, "bindings": [{"role": b.role_id, "value": b.value} for b in c.bindings]} for c in RuleBasedProvider().extract(text)]
    return role_metrics(gold, predicted)


def internal_m2() -> dict:
    m = AHMemory()
    for i in range(4): m.add_symbol(FirstOrderSymbol(uid=f"m2_s{i}", sensory_representations=(SensoryRepresentation(modality="text", value=f"m2-{i}"),)))
    for i in range(3): m.add_link(AssociativeLink(uid=f"m2_l{i}", type_id="CAUSE", weight=1, source_ref=SReference(reference_uid=f"m2_a{i}", target_uid=f"m2_s{i}"), target_ref=SReference(reference_uid=f"m2_b{i}", target_uid=f"m2_s{i+1}")))
    run = IgnitionEngine().run(m.snapshot(), ["m2_s0"], IgnitionConfig(max_ticks=6))
    gold_path = {"m2_s0", "m2_l0", "m2_s1", "m2_l1", "m2_s2", "m2_l2", "m2_s3"}
    traced = {x for t in run.run.ticks for x in (t.source_uid, t.target_uid, t.link_or_hypernode_uid) if x}
    complete = gold_path <= traced
    return {"correct": complete, "depth": 3, "trace_complete": complete, "explain_score": 1.0 if complete else 0.0}
