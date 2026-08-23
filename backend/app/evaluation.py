from __future__ import annotations

from .core import AHMemory, norm
from .engine import IgnitionEngine
from .ingestion import RuleBasedProvider
from .models import *

RABBIT_FACTS = (
    "Кролик является животным.", "Животное имеет тело.", "Кролик находится в саду.",
    "Кролик ест морковь.", "Морковь является овощем.", "Кролик быстро бежит.",
    "После еды кролик отдыхает.", "Сад находится у дома.",
)

def rabbit_fixture() -> dict:
    accepted = sum(1 for text in RABBIT_FACTS if RuleBasedProvider().extract(text))
    return {"facts": len(RABBIT_FACTS), "accepted": accepted, "conformant": accepted >= 6}


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
