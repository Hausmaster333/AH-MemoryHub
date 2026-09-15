from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from .core import AHMemory, lexical_key, norm, _lemma, _morph_analyzer
from .conformance import RABBIT_SOURCE, junior_conformance_report
from .engine import IgnitionEngine, evidence_trace, gc_commit, gc_preview
from .ingestion import Provider, RuleBasedProvider, extract_candidates, segment_text
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
        g = {(norm(row.get("predicate", "")), norm(x.get("value", ""))) for row in gold for x in row.get("bindings", []) if x.get("role") == role}
        p = {(norm(row.get("predicate", "")), norm(x.get("value", ""))) for row in predicted for x in row.get("bindings", []) if x.get("role") == role}
        tp = len(g & p); precision = tp / len(p) if p else 0.0; recall = tp / len(g) if g else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[role] = {"precision": precision, "recall": recall, "f1": f1, "gold": len(g), "predicted": len(p)}
    literal = 2 * result["SUBJECT"]["f1"] + 2 * result["OBJECT"]["f1"] + result["LOCATION"]["f1"]
    result["weighted_f1"] = literal
    result["normalized_weighted_f1"] = literal / 5
    return result


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "docs" / "testing" / "heterogeneous-corpus.json"
GOLD_PATH = ROOT / "docs" / "testing" / "gold-v3.json"
RESULT_PATH = ROOT / "docs" / "testing" / "results" / "m1-m2-m3-nitro-v3.json"


def latest_corpus_report() -> dict | None:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else None


def load_gold(path: Path = GOLD_PATH) -> dict:
    gold = json.loads(path.read_text(encoding="utf-8"))
    if "base" not in gold: return gold
    base = json.loads((path.parent / gold["base"]).read_text(encoding="utf-8"))
    for document in base["documents"]:
        for extra in gold.get("additional_facts", {}).get(document["id"], []):
            existing = next((fact for fact in document["facts"] if fact["predicate"] == extra["predicate"] and fact["quote"] == extra["quote"]), None)
            if existing:
                for role, values in extra["bindings"].items():
                    existing["bindings"].setdefault(role, [])
                    existing["bindings"][role].extend(value for value in values if value not in existing["bindings"][role])
            else:
                document["facts"].append(extra)
    base.update({key: value for key, value in gold.items() if key not in {"base", "additional_facts"}})
    return base


def _value_matches(left: str, right: str) -> bool:
    a, b = set(lexical_key(left)), set(lexical_key(right))
    identifiers = lambda tokens: {token for token in tokens if any(char.isdigit() for char in token)}
    return identifiers(a) == identifiers(b) and bool(a and b) and len(a & b) / min(len(a), len(b)) >= .8


def value_matches_morph(left: str, right: str) -> bool:
    def terms(value):
        tokens = re.findall(r"[\w]+(?:[-‑–][\w]+)*", norm(value))
        # Gold location values sometimes omit the leading locative preposition.
        if tokens and tokens[0] in {"в", "во", "на"}: tokens = tokens[1:]
        return Counter(_lemma(token) for token in tokens)
    a, b = terms(left), terms(right)
    return bool(a) and a == b


def _sentence_index(text: str, offset: int) -> int:
    return next(span.index for span in segment_text(text) if span.start <= offset < span.end)


def _candidate_values(candidate: CandidateFact, binding: CandidateBinding) -> tuple[str, ...]:
    if binding.value:
        functional = re.fullmatch(r"(?:AND|OR)\((.+)\)", binding.value, re.I)
        return tuple(item.strip() for item in functional.group(1).split(",")) if functional else (binding.value,)
    mentions = {mention.mention_id: mention for mention in candidate.mentions}
    return tuple((mentions[item].canonical_label or mentions[item].observed_text) for item in binding.term.mention_ids if item in mentions) if binding.term else ()


def corpus_role_metrics(predictions: dict[str, list[CandidateFact]], corpus_path: Path = CORPUS_PATH, gold_path: Path = GOLD_PATH, matcher: str = "legacy_v1") -> dict:
    if matcher not in {"legacy_v1", "morph_v1"}: raise ValueError("unknown M1 matcher")
    matches = value_matches_morph if matcher == "morph_v1" else _value_matches
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    gold = load_gold(gold_path)
    sources = {document["id"]: document["text"] for document in corpus["documents"]}
    expected: dict[str, list[tuple[str, int, str, str]]] = {}
    actual: dict[str, list[tuple[str, int, str, str]]] = {}
    labelled: dict[str, set[int]] = {}
    for document in gold["documents"]:
        text = sources[document["id"]]
        for fact in document["facts"]:
            offset = text.find(fact["quote"])
            if offset < 0: raise ValueError(f"gold quote is not in source: {fact['quote']}")
            sentence = _sentence_index(text, offset)
            labelled.setdefault(document["id"], set()).add(sentence)
            for role, values in fact["bindings"].items():
                expected.setdefault(role, []).extend((document["id"], sentence, fact["predicate"], value) for value in values)
    for document_id, candidates in predictions.items():
        text = sources[document_id]
        for candidate in candidates:
            sentence = _sentence_index(text, candidate.source_start)
            if sentence not in labelled.get(document_id, set()): continue
            for binding in candidate.bindings:
                actual.setdefault(binding.role_id, []).extend((document_id, sentence, candidate.predicate, value) for value in _candidate_values(candidate, binding))
    roles = sorted(set(expected) | set(actual) | {"SUBJECT", "OBJECT", "LOCATION"})
    result, totals, errors = {}, {"tp": 0, "gold": 0, "predicted": 0}, []
    for role in roles:
        remaining = list(actual.get(role, [])); tp = 0
        for document_id, sentence, predicate, value in expected.get(role, []):
            match = next((i for i, item in enumerate(remaining) if item[:3] == (document_id, sentence, predicate) and matches(value, item[3])), None)
            if match is not None:
                tp += 1; remaining.pop(match)
            else:
                errors.append({"type": "false_negative", "document_id": document_id, "sentence_index": sentence, "predicate": predicate, "role": role, "value": value})
        errors.extend({"type": "false_positive", "document_id": item[0], "sentence_index": item[1], "predicate": item[2], "role": role, "value": item[3]} for item in remaining)
        g, p = len(expected.get(role, [])), len(actual.get(role, []))
        precision, recall = (tp / p if p else 0.0), (tp / g if g else 0.0)
        result[role] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "gold": g, "predicted": p}
        totals["tp"] += tp; totals["gold"] += g; totals["predicted"] += p
    weights = {role: (2 if role in {"SUBJECT", "OBJECT"} else 1) for role in roles}
    literal_all_roles = sum(result[role]["f1"] * weights[role] for role in roles)
    result["weighted_f1"] = literal_all_roles
    result["normalized_weighted_f1"] = literal_all_roles / sum(weights.values())
    required_roles = ("SUBJECT", "OBJECT", "LOCATION")
    literal_required = sum(result[role]["f1"] * (2 if role in {"SUBJECT", "OBJECT"} else 1) for role in required_roles)
    result["required_roles_weighted_f1"] = literal_required
    result["normalized_required_roles_weighted_f1"] = literal_required / 5
    precision = totals["tp"] / totals["predicted"] if totals["predicted"] else 0.0
    recall = totals["tp"] / totals["gold"] if totals["gold"] else 0.0
    result["micro"] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, **totals}
    document_scores = {}
    for document in gold["documents"]:
        document_id = document["id"]
        g = sum(len(values) for fact in document["facts"] for values in fact["bindings"].values())
        fn = sum(error["type"] == "false_negative" and error["document_id"] == document_id for error in errors)
        fp = sum(error["type"] == "false_positive" and error["document_id"] == document_id for error in errors)
        tp, p = g - fn, g - fn + fp
        document_precision, document_recall = (tp / p if p else 0.0), (tp / g if g else 0.0)
        document_scores[document_id] = {"precision": document_precision, "recall": document_recall, "f1": 2 * document_precision * document_recall / (document_precision + document_recall) if document_precision + document_recall else 0.0, "tp": tp, "gold": g, "predicted": p}
    document_f1 = [score["f1"] for score in document_scores.values()]
    result["documents"] = document_scores
    result["document_macro_f1"] = sum(document_f1) / len(document_f1)
    result["document_min_f1"] = min(document_f1)
    result["document_max_f1"] = max(document_f1)
    result["benchmark"] = {"schema": gold["schema"], "documents": len(gold["documents"]), "source_sentences": sum(len(segment_text(sources[doc["id"]])) for doc in gold["documents"]), "labelled_facts": sum(len(doc["facts"]) for doc in gold["documents"]), "role_assignments": sum(len(values) for doc in gold["documents"] for fact in doc["facts"] for values in fact["bindings"].values())}
    result["errors"] = errors
    result["matcher"] = matcher
    return result


def corpus_m2(gold_path: Path = GOLD_PATH) -> dict:
    gold = load_gold(gold_path)
    benchmark = gold["trace_benchmark"]
    required_total = gold.get("required_trace_questions", len(benchmark))
    cases = []
    for chain in benchmark:
        memory = AHMemory()
        prefix = f"m2_{chain['id']}"
        for index, label in enumerate(chain["nodes"]): memory.add_symbol(FirstOrderSymbol(uid=f"{prefix}_s{index}", sensory_representations=(SensoryRepresentation(modality="text", value=label),)))
        for index, link_type in enumerate(chain["link_types"]): memory.add_link(AssociativeLink(uid=f"{prefix}_l{index}", type_id=link_type, weight=1, source_ref=SReference(reference_uid=f"{prefix}_a{index}", target_uid=f"{prefix}_s{index}"), target_ref=SReference(reference_uid=f"{prefix}_b{index}", target_uid=f"{prefix}_s{index + 1}")))
        for question in chain["questions"]:
            depth = question["depth"]
            run = IgnitionEngine().run(memory.snapshot(), [f"{prefix}_s0"], IgnitionConfig(max_ticks=depth + 4))
            gold_path = {f"{prefix}_s{index}" for index in range(depth + 1)} | {f"{prefix}_l{index}" for index in range(depth)}
            target = f"{prefix}_s{depth}"
            traced = set(evidence_trace(run.run, (target,), [f"{prefix}_s0"])) if target in run.run.working_memory else set()
            actual_answer = memory.label(target) if target in traced else "insufficient_evidence"
            cases.append({"id": f"{chain['id']}-{depth}", "depth": depth, "question": question["question"], "expected_answer": question["expected_answer"], "actual_answer": actual_answer, "answer_correct": norm(actual_answer) == norm(question["expected_answer"]), "trace_complete": gold_path <= traced, "gold_path": sorted(gold_path)})
    passed = sum(case["answer_correct"] and case["trace_complete"] for case in cases)
    weighted_sum = sum(case["answer_correct"] * (case["depth"] / 6) * case["trace_complete"] for case in cases)
    folds = [cases[index:index + 20] for index in range(0, len(cases), 20)]
    fold_scores = [sum(case["answer_correct"] * (case["depth"] / 6) * case["trace_complete"] for case in fold) / len(fold) for fold in folds]
    return {"scope": "activation fixture with supplied target; not end-to-end question answering", "trace_method": "seed_dependency_closure", "correct": passed == len(cases) and len(cases) >= required_total, "passed": passed, "total": len(cases), "required_total": required_total, "coverage": min(1.0, len(cases) / required_total), "trace_pass_rate": passed / len(cases), "max_depth": max((case["depth"] for case in cases if case["trace_complete"]), default=0), "explain_score": weighted_sum / len(cases), "available_cases_score": weighted_sum / len(cases), "fold_size": 20, "fold_count": len(folds), "fold_scores": fold_scores, "provisional": len(cases) < required_total, "cases": cases}


def internal_m1() -> dict:
    text = "Перегрев насоса вызвал остановку агрегата в насосном зале."
    gold = [{"predicate": "CAUSE", "bindings": [{"role": "SUBJECT", "value": "Перегрев насоса"}, {"role": "OBJECT", "value": "остановку агрегата"}, {"role": "LOCATION", "value": "насосном зале"}]}]
    predicted = [{"predicate": c.predicate, "bindings": [{"role": b.role_id, "value": b.value} for b in c.bindings]} for c in RuleBasedProvider().extract(text)]
    return role_metrics(gold, predicted)


def internal_m2() -> dict:
    return corpus_m2()


def _m3_case(orphan_count: int, live_count: int, topology: str) -> dict:
    started = time.perf_counter()
    memory = AHMemory()
    memory.add_symbol(FirstOrderSymbol(uid="m3_seed", sensory_representations=(SensoryRepresentation(modality="text", value="connected root"),)))
    sections = {"C": {}, "P": {}, "H": {}}
    links = {}
    live_uids = {f"m3_live_{index}" for index in range(live_count)}
    for index in range(live_count):
        live_uid = f"m3_live_{index}"
        sections["P"][live_uid] = MemoryElement(uid=live_uid, payload=SecondOrderSymbol(uid=live_uid, properties=(Property(name="label", value=f"live {index}"),)))
        source_uid = "m3_seed" if topology == "star" or index == 0 else f"m3_live_{index - 1}"
        source_ref = SReference(reference_uid=f"m3_live_source_{index}", target_uid=source_uid) if source_uid == "m3_seed" else MReference(reference_uid=f"m3_live_source_{index}", target_uid=source_uid)
        links[f"m3_live_link_{index}"] = AssociativeLink(uid=f"m3_live_link_{index}", type_id="LIVE", weight=1, source_ref=source_ref, target_ref=MReference(reference_uid=f"m3_live_target_{index}", target_uid=live_uid))
    for index in range(orphan_count):
        orphan_uid = f"m3_orphan_{index}"
        sections["P"][orphan_uid] = MemoryElement(uid=orphan_uid, payload=SecondOrderSymbol(uid=orphan_uid, properties=(Property(name="label", value=f"orphan {index}"),)))
        if topology == "chain" and index:
            links[f"m3_orphan_link_{index}"] = AssociativeLink(uid=f"m3_orphan_link_{index}", type_id="ORPHAN_CHAIN", weight=1, source_ref=MReference(reference_uid=f"m3_orphan_source_{index}", target_uid=f"m3_orphan_{index - 1}"), target_ref=MReference(reference_uid=f"m3_orphan_target_{index}", target_uid=orphan_uid))
    # Benchmark setup is one atomic validated revision; per-item commits would measure fixture construction, not GC.
    memory._commit(dict(memory.symbols), sections, links, {})
    grace_preview = gc_preview(memory, IgnitionConfig(initial_life_ticks=5))
    memory.advance_ticks(50)
    before = gc_preview(memory, IgnitionConfig(initial_life_ticks=5))
    orphan_before = before["orphan_count_before"]
    gc_commit(memory, before["preview_token"], before["deletable_uids"])
    after = gc_preview(memory, IgnitionConfig(initial_life_ticks=5))
    orphan_after = after["orphan_count_before"]
    false_deletions = len(live_uids - set(memory.elements))
    efficiency = 1 - orphan_after / orphan_before if orphan_before else 0.0
    return {"topology": topology, "protected_before_grace": grace_preview["orphan_count_before"] == 0, "ticks": 50, "orphan_nodes_before_gc": orphan_before, "orphan_nodes_after_gc": orphan_after, "deleted": len(before["deletable_uids"]), "gc_efficiency": efficiency, "live_nodes_before": len(live_uids), "live_nodes_after": len(live_uids & set(memory.elements)), "false_deletions": false_deletions, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}


def _m3_zero_weight_case() -> dict:
    memory = AHMemory()
    memory.add_symbol(FirstOrderSymbol(uid="m3_zero_seed", sensory_representations=(SensoryRepresentation(modality="text", value="zero-weight root"),)))
    memory.add_symbol(FirstOrderSymbol(uid="m3_zero_target", sensory_representations=(SensoryRepresentation(modality="text", value="zero-weight target"),)))
    memory.add_symbol(FirstOrderSymbol(uid="m3_zero_predicate", sensory_representations=(SensoryRepresentation(modality="text", value="ZERO"),)))
    memory.add_template(ControlTemplate(uid="m3_zero_template", predicate_ref=SReference(reference_uid="m3_zero_predicate_ref", target_uid="m3_zero_predicate"), ordered_roles=(Role(role_id="SUBJECT"), Role(role_id="OBJECT"))))
    memory.add_element("P", MemoryElement(uid="m3_zero_hypernode", payload=Hypernode(uid="m3_zero_hypernode", weight=0, template_ref=ElementReference(reference_uid="m3_zero_template_ref", target_uid="m3_zero_template"), role_bindings=(RoleBinding(role_id="SUBJECT", target_ref=SReference(reference_uid="m3_zero_subject", target_uid="m3_zero_seed")), RoleBinding(role_id="OBJECT", target_ref=SReference(reference_uid="m3_zero_object", target_uid="m3_zero_target"))))))
    memory.advance_ticks(50)
    preview = gc_preview(memory, IgnitionConfig(initial_life_ticks=5))
    return {"deletable_uids": preview["deletable_uids"], "passed": {"m3_zero_hypernode"} <= set(preview["deletable_uids"])}


def internal_m3() -> dict:
    cases = [_m3_case(size, max(25, size // 5), topology) for size in (100, 200, 250, 500, 1000) for topology in ("star", "chain")]
    orphan_before = sum(case["orphan_nodes_before_gc"] for case in cases)
    orphan_after = sum(case["orphan_nodes_after_gc"] for case in cases)
    live_before = sum(case["live_nodes_before"] for case in cases)
    live_after = sum(case["live_nodes_after"] for case in cases)
    efficiencies = [case["gc_efficiency"] for case in cases]
    return {
        "case_count": len(cases),
        "topologies": sorted({case["topology"] for case in cases}),
        "protected_before_grace": all(case["protected_before_grace"] for case in cases),
        "ticks": 50,
        "orphan_nodes_before_gc": orphan_before,
        "orphan_nodes_after_gc": orphan_after,
        "deleted": sum(case["deleted"] for case in cases),
        "gc_efficiency": 1 - orphan_after / orphan_before if orphan_before else 0.0,
        "mean_gc_efficiency": sum(efficiencies) / len(efficiencies),
        "min_gc_efficiency": min(efficiencies),
        "live_nodes_before": live_before,
        "live_nodes_after": live_after,
        "false_deletions": sum(case["false_deletions"] for case in cases),
        "total_examined_nodes": orphan_before + live_before,
        "elapsed_ms": round(sum(case["elapsed_ms"] for case in cases), 3),
        "official_200_cases": [case for case in cases if case["orphan_nodes_before_gc"] == 200],
        "zero_weight_conformance": _m3_zero_weight_case(),
        "cases": cases,
    }
