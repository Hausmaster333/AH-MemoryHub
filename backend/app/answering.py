from __future__ import annotations

import json
import os
import re

from .core import AHMemory, lexical_score, norm
from .ingestion import OpenAICompatibleProvider, configured_provider


def build_evidence_packet(memory: AHMemory, working_memory: tuple[str, ...], evidence_path: tuple[str, ...] = ()) -> list[dict]:
    facts = []
    eligible = set(evidence_path) or set(working_memory)
    for hypernode in sorted(memory.find_hypernodes(), key=lambda item: item.uid):
        if hypernode.uid not in eligible or not hypernode.evidence: continue
        template = memory.templates.get(hypernode.template_ref.target_uid)
        facts.append({
            "evidence_id": f"E{len(facts) + 1}", "hypernode_uid": hypernode.uid,
            "predicate": memory.label(template.predicate_ref.target_uid) if template else hypernode.template_ref.target_uid,
            "bindings": {binding.role_id: memory.label(binding.target_ref.target_uid) for binding in hypernode.role_bindings},
            "source_quotes": list(dict.fromkeys(evidence.exact_text for evidence in hypernode.evidence)),
        })
    return facts


def select_local_answer(memory: AHMemory, question: str, candidate_uids: tuple[str, ...]) -> tuple[str, list]:
    eligible = set(candidate_uids)
    facts = [item for item in memory.find_hypernodes() if item.uid in eligible and item.evidence]
    query = norm(question)
    intent = (
        "temporal" if re.search(r"до как|как долго|до какого момента|срок", query) else
        "location" if re.search(r"\bгде\b|мест|наход|установ", query) else
        "responsibility" if re.search(r"кто отвеча|какая.*команд.*отвеча", query) else
        "ownership" if re.search(r"кому принадлеж|чей|владел", query) else
        "purpose" if re.search(r"для чего|назначени|используется", query) else
        "tool" if re.search(r"\bчем\b|каким.*инструмент", query) else
        "follow" if re.search(r"что.*после|следу", query) else
        "cause_result" if re.search(r"что (?:вызвал|вызвала|вызвало)|к чему прив", query) else
        "cause_reason" if re.search(r"почему|причин|из-за", query) else
        "state" if re.search(r"свойств|состояни|характерист|каки[ей]", query) else "generic"
    )
    specs = {
        "temporal": ({"HAS_STATE"}, "SUBJECT"), "location": ({"LOCATED_AT", "LIVE"}, "SUBJECT"),
        "responsibility": ({"HAS"}, "OBJECT"),
        "ownership": ({"HAS"}, "OBJECT"), "purpose": ({"PURPOSE"}, "SUBJECT"),
        "tool": ({"USES_TOOL", "ACTION"}, "SUBJECT"), "follow": ({"FOLLOW"}, "SUBJECT"),
        "cause_result": ({"CAUSE"}, "SUBJECT"), "cause_reason": ({"CAUSE"}, "OBJECT"),
        "state": ({"HAS_STATE", "HAS"}, "SUBJECT"),
    }
    predicates, focus_role = specs.get(intent, (None, None))
    all_evidence = [ev for item in memory.find_hypernodes() for ev in item.evidence]
    scored = []
    for item in facts:
        template = memory.templates.get(item.template_ref.target_uid)
        predicate = memory.label(template.predicate_ref.target_uid) if template else item.template_ref.target_uid
        if predicates and predicate not in predicates: continue
        bindings = {binding.role_id: memory.label(binding.target_ref.target_uid) for binding in item.role_bindings}
        if intent == "temporal" and not bindings.get("TIME"): continue
        focus = bindings.get(focus_role, "") if focus_role else ""
        evidence = item.evidence[0]
        nearby = " ".join(ev.exact_text for ev in all_evidence if ev.document_uid == evidence.document_uid and abs(ev.start_offset - evidence.start_offset) <= 100)
        focus_score = lexical_score(question, focus) if focus else 0
        score = 4 * focus_score + lexical_score(question, evidence.exact_text) + lexical_score(question, nearby) if focus else max((lexical_score(question, value) for value in bindings.values()), default=0)
        scored.append((score, focus_score, item, bindings))
    best = max(((focus_score if intent == "state" else score) for score, focus_score, _, _ in scored), default=0)
    selected = [(item, bindings) for score, focus_score, item, bindings in scored if best > 0 and (focus_score if intent == "state" else score) == best]
    if not selected: return "insufficient_evidence", []
    if intent == "cause_reason" and "первоприч" in query:
        immediate = max(scored, key=lambda row: (row[1], row[0]))
        if immediate[1] <= 0: return "insufficient_evidence", []
        item, bindings = immediate[2], immediate[3]
        chain, visited = [item], {item.uid}
        target, current = bindings.get("OBJECT", "событие"), bindings.get("SUBJECT", "")
        while current:
            previous = [row for row in scored if row[2].uid not in visited and lexical_score(current, row[3].get("OBJECT", "")) > 0]
            if not previous: break
            _, _, item, bindings = max(previous, key=lambda row: lexical_score(current, row[3].get("OBJECT", "")))
            chain.append(item); visited.add(item.uid); current = bindings.get("SUBJECT", "")
        return f"Первопричина события «{target}» — {current}.", chain
    if intent == "state":
        subject = selected[0][1].get("SUBJECT", "Объект")
        values = []
        for _, bindings in selected:
            value = bindings.get("STATE") or bindings.get("OBJECT", "состояние")
            if bindings.get("TIME"): value += f" ({bindings['TIME']})"
            values.append(value)
        return f"{subject}: {'; '.join(dict.fromkeys(values))}.", [item for item, _ in selected]
    if intent == "location":
        subject = selected[0][1].get("SUBJECT", "Объект")
        values = [bindings.get("LOCATION", "неизвестное место") for _, bindings in selected]
        return f"{subject}: {'; '.join(dict.fromkeys(values))}.", [item for item, _ in selected]
    if intent == "ownership":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    if intent == "responsibility":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    if intent == "purpose":
        _, bindings = selected[0]
        return f"{bindings.get('SUBJECT', 'Объект')}: {bindings.get('PURPOSE', 'назначение не указано')}.", [selected[0][0]]
    if intent == "tool":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    if intent == "follow":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    if intent == "temporal":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    item, _ = selected[0]
    return item.evidence[0].exact_text, [item]


def validate_evidence_answer(decoded: dict, facts: list[dict]) -> dict:
    status, answer = decoded.get("status"), re.sub(r"^(?:\[E\d+\]\s*)+", "", str(decoded.get("answer", "")).strip())
    evidence_ids = list(dict.fromkeys(str(value) for value in decoded.get("evidence_ids", [])))
    if status == "insufficient_evidence":
        if answer or evidence_ids: raise ValueError("insufficient answer contains claims")
        return {"status": status, "answer": "", "evidence_ids": []}
    allowed, citations = {fact["evidence_id"] for fact in facts}, re.findall(r"\[(E\d+)\]", answer)
    if status != "answered" or not answer or not evidence_ids or set(evidence_ids) - allowed or set(citations) != set(evidence_ids):
        raise ValueError("answer contains missing or unknown evidence citations")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
    if any(not re.search(r"\[E\d+\][.!?]?$", sentence) for sentence in sentences): raise ValueError("every answer sentence must end with a citation")
    return {"status": "answered", "answer": answer, "evidence_ids": evidence_ids}


def generate_evidence_answer(question: str, facts: list[dict], model_override: str, grounded_draft: str = "") -> dict:
    provider, meta = configured_provider(None if model_override == "configured" else model_override)
    if not isinstance(provider, OpenAICompatibleProvider): raise ValueError("external answer provider is not configured")
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
            "answer": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        },
        "required": ["status", "answer", "evidence_ids"],
    }
    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": "Ответь только по AH_FACTS, без внешних знаний. RoleBindings — уже проверенные канонические факты. Непустой GROUNDED_DRAFT означает, что релевантные факты уже найдены: ОБЯЗАТЕЛЬНО верни status=answered, переформулируй черновик и укажи использованные evidence_ids. Не добавляй новых утверждений. Для вопроса о свойствах используй OBJECT из HAS и STATE из HAS_STATE с совпадающим SUBJECT; перечисли все подтверждённые значения. В идентификаторах считай варианты только с дефисом или без него эквивалентными (например, ДТ4 = ДТ-4). Каждое предложение закончи ссылкой вида [E1] перед точкой. insufficient_evidence допустим только при пустом GROUNDED_DRAFT. Ответь кратко на языке вопроса."},
            {"role": "user", "content": json.dumps({"question": question, "GROUNDED_DRAFT": grounded_draft, "AH_FACTS": facts}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "evidence_bound_answer", "strict": True, "schema": schema}} if provider.response_format == "json_schema" else {"type": "json_object"},
        "temperature": 0, "max_tokens": int(os.getenv("AH_ANSWER_MAX_TOKENS", "2048")),
    }
    if "deepseek" in provider.model.lower(): payload["reasoning"] = {"effort": "none", "exclude": True}
    response = provider._request_json(payload)
    try:
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, list): content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        decoded = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.I))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("answer model returned invalid structured JSON") from exc
    try:
        validated = validate_evidence_answer(decoded, facts)
    except ValueError:
        evidence_ids = list(dict.fromkeys(str(value) for value in decoded.get("evidence_ids", [])))
        allowed = {fact["evidence_id"] for fact in facts}
        if decoded.get("status") != "answered" or not grounded_draft or not evidence_ids or set(evidence_ids) - allowed: raise
        answer = f"{grounded_draft.rstrip(' .!?')} {''.join(f'[{value}]' for value in evidence_ids)}."
        validated = validate_evidence_answer({"status": "answered", "answer": answer, "evidence_ids": evidence_ids}, facts)
    return {**validated, "provider": meta}
