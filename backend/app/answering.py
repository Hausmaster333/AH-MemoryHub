from __future__ import annotations

import json
import os
import re

from .core import AHMemory, lexical_score, norm, _morph_analyzer, _lemma
from .ingestion import OpenAICompatibleProvider, configured_provider


class AnswerProviderError(ValueError):
    """The answer provider could not be configured or contacted."""


class AnswerResponseError(ValueError):
    """The provider returned an invalid or unsupported answer."""


def build_evidence_packet(memory: AHMemory, working_memory: tuple[str, ...], evidence_path: tuple[str, ...] = ()) -> list[dict]:
    facts = []
    eligible = set(working_memory) & set(evidence_path) if evidence_path else set(working_memory)
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
    # Negative questions need explicit negative facts; absence is not negation.
    if re.search(r"\bне\b", query): return "insufficient_evidence", []
    classification = re.match(r"^(?:какой|какая|какое|какие)\s+(.+?)\s+(?:явля(?:ется|ются)|относится\s+к)\s+(.+?)[?.!]*$", query)
    class_phrase = classification[2] if classification else ""
    scope = re.search(r"\s+(?:на|в|у)\s+(.+)$", class_phrase) if classification else None
    if scope: class_phrase = class_phrase[:scope.start()]
    intent = (
        "classification" if classification else
        "temporal" if re.search(r"до как|до чего|как долго|до какого момента|срок|\bкогда\b|\bво сколько\b|\bв какое время\b", query) else
        "location" if re.search(r"\bгде\b|мест|наход|установ", query) else
        "responsibility" if re.search(r"кто отвеча|какая.*команд.*отвеча", query) else
        "ownership" if re.search(r"\bкому\s+принадлеж|\b(?:чей|чья|чьё|чье|чьи)\b|\bвладел|\bсобственник", query) else
        "actor" if re.search(r"^кто\b", query) else
        "purpose" if re.search(r"для чего|назначени|используется", query) else
        "tool" if re.search(r"\bчем\b|каким.*инструмент|каки[ех]\s+инструмент|\b(?:инструмент|прибор)\w*\b.*\b(?:использова|примени)\w*", query) else
        "follow" if re.search(r"что.*после|следу", query) else
        "cause_reason" if re.search(r"что вызвало\b", query) else
        "cause_result" if re.search(r"что (?:вызвал|вызвала)\b|к чему прив", query) else
        "cause_reason" if re.search(r"почему|причин|из-за", query) else
        "state" if re.search(r"свойств|состояни|характерист|каки[ей]", query) else "generic"
    )
    specs = {
        "classification": ({"IS-A"}, "SUBJECT"),
        "temporal": ({"HAS_STATE", "ACTION"}, "SUBJECT"), "location": ({"LOCATED_AT", "LIVE"}, "SUBJECT"),
        "responsibility": ({"HAS"}, "OBJECT"), "actor": ({"ACTION"}, "OBJECT"),
        "ownership": ({"HAS"}, "OBJECT"), "purpose": ({"PURPOSE"}, "SUBJECT"),
        "tool": ({"USES_TOOL", "ACTION"}, "SUBJECT"), "follow": ({"FOLLOW"}, "SUBJECT"),
        "cause_result": ({"CAUSE"}, "SUBJECT"), "cause_reason": ({"CAUSE"}, "OBJECT"),
        "state": ({"HAS_STATE", "HAS"}, "SUBJECT"),
    }
    predicates, focus_role = specs.get(intent, (None, None))
    # ponytail: explicit inverse location questions; other constructions retain the existing path.
    location_subject = re.match(r"^(?:какой|какая|какое|какие)\s+(.+?)\s+(?:установлен[аоы]?|находится|находятся|расположен[аоы]?)\s+(?:в|на|у)\s+", query) if intent == "location" else None
    location_inverse = intent == "location" and (location_subject is not None or re.match(r"^(?:что|кто)\s+(?:находится|находятся|установлен[аоы]?|расположен[аоы]?)\s+(?:в|на|у)\s+", query))
    if location_inverse: focus_role = "LOCATION"
    target = re.search(r"(?:что вызвал[ао]?|к чему прив[её]л\w*|(?:перво)?причин[аы](?:\s+событи[ея])?)\s+[«\"]?(.+?)[»\"]?[?.!]*$", query) if intent in {"cause_reason", "cause_result"} else None
    if intent == "cause_reason" and target is None:
        target = re.search(r"почему\s+\w+(?:лся|лась|лось|лись)\s+(.+?)[?.!]*$", query)
    # ponytail: explicit causal question forms only; broader questions still use lexical ranking.
    target_words = re.findall(r"[\w-]+", target.group(1)) if target else []
    target_words = [word for word in target_words if len(word) > 3 or any(char.isdigit() for char in word)]
    tool_subject = re.search(r"(?:использовал\w*|пользовал\w*|применил\w*)\s+(.+?)[?.!]*$", query) if intent == "tool" else None
    short_names = [word for word in re.findall(r"\w+", re.split(r"\b(?:для|при)\b", tool_subject[1], maxsplit=1)[0]) if len(word) <= 3] if tool_subject else []
    state_query = re.search(r"оста\w*\s+(.+?)[?.!]*$", query) if intent == "temporal" else None
    identifiers = {word.replace("-", "") for word in re.findall(r"[\w-]+", query) if any(char.isdigit() for char in word)}
    time_verbs = [word for word in re.findall(r"[а-яё]+", query) if _morph_analyzer().parse(word)[0].tag.POS in {"VERB", "INFN"}] if intent in {"temporal", "actor"} else []
    tool_event = re.search(r"\b(?:для|при)\s+(.+?)(?=\s+(?:использова|примени)\w*\b|[?.!]|$)", query) if intent == "tool" else None
    tool_event_terms = [word for word in re.findall(r"[\w-]+", tool_event[1]) if any(char.isdigit() for char in word) or _morph_analyzer().parse(word)[0].tag.POS == "NOUN"] if tool_event else []
    def focus_value(bindings):
        if intent == "temporal" and "OBJECT" in bindings:
            return bindings.get("SUBJECT", "") + " " + bindings["OBJECT"]
        return bindings.get(focus_role, "") if focus_role else ""
    all_evidence = [ev for item in memory.find_hypernodes() for ev in item.evidence]
    scored = []
    for item in facts:
        template = memory.templates.get(item.template_ref.target_uid)
        predicate = memory.label(template.predicate_ref.target_uid) if template else item.template_ref.target_uid
        if predicates and predicate not in predicates: continue
        bindings = {binding.role_id: memory.label(binding.target_ref.target_uid) for binding in item.role_bindings}
        if classification:
            subject_terms = [word for word in re.findall(r"[\w-]+", classification[1]) if len(word) > 3 or any(char.isdigit() for char in word)]
            class_terms = [word for word in re.findall(r"[\w-]+", class_phrase) if len(word) > 3 or any(char.isdigit() for char in word)]
            if not subject_terms or not class_terms or any(lexical_score(word, bindings.get("SUBJECT", "")) == 0 for word in subject_terms) or any(lexical_score(word, bindings.get("OBJECT", "")) == 0 for word in class_terms): continue
            if scope:
                scope_words = " ".join(re.findall(r"[\w-]+", scope[1]))
                grounded_words = " ".join(re.findall(r"[\w-]+", norm(" ".join(bindings.values()) + " " + item.evidence[0].exact_text)))
                if not scope_words or f" {scope_words} " not in f" {grounded_words} ": continue
        if intent == "temporal" and not bindings.get("TIME"): continue
        if intent == "temporal" and predicate == "HAS_STATE" and time_verbs and not state_query:
            event_words = {parsed.normal_form for word in re.findall(r"[а-яё]+", norm(bindings.get("STATE", "") + " " + item.evidence[0].exact_text))
                           for parsed in _morph_analyzer().parse(word) if parsed.tag.POS in {"VERB", "INFN", "PRTF", "PRTS"}}
            if any(_lemma(verb) not in event_words for verb in time_verbs): continue
        if intent == "tool" and not bindings.get("TOOL"): continue
        if tool_event_terms and predicate == "ACTION" and any(lexical_score(word, bindings.get("OBJECT", "") + " " + bindings.get("PURPOSE", "")) == 0 for word in tool_event_terms): continue
        if location_subject and any(lexical_score(word, bindings.get("SUBJECT", "")) == 0 for word in re.findall(r"\w+", location_subject[1])): continue
        if short_names and not set(short_names) <= set(re.findall(r"\w+", norm(bindings.get("SUBJECT", "")))): continue
        if state_query and any(lexical_score(word, bindings.get("STATE", "")) == 0 for word in re.findall(r"\w+", state_query[1]) if len(word) > 3): continue
        if intent in {"temporal", "actor"} and predicate == "ACTION" and (not time_verbs or any(lexical_score(verb, bindings.get("OBJECT", "")) == 0 for verb in time_verbs)): continue
        focus = focus_value(bindings)
        evidence = item.evidence[0]
        nearby = " ".join(ev.exact_text for ev in all_evidence if ev.document_uid == evidence.document_uid and abs(ev.start_offset - evidence.start_offset) <= 100)
        focus_score = lexical_score(question, focus) if focus else 0
        score = 4 * focus_score + lexical_score(question, evidence.exact_text) + lexical_score(question, nearby) if focus else max((lexical_score(question, value) for value in bindings.values()), default=0)
        scored.append((score, focus_score, item, bindings))
    scope_support = {}
    eligible_scored = []
    for row in scored:
        _, focus_score, item, bindings = row
        if focus_role and focus_score <= 0: continue
        if any(lexical_score(word, focus_value(bindings)) == 0 for word in target_words): continue
        focus_ids = {word.replace("-", "") for word in re.findall(r"[\w-]+", norm(focus_value(bindings) if focus_role else " ".join(bindings.values()))) if any(char.isdigit() for char in word)}
        missing_ids = identifiers - focus_ids
        if missing_ids:
            if intent != "temporal": continue
            linked = []
            for link in memory.links.values():
                if link.type_id != "IN_SCOPE" or not link.uid.startswith("l_scope_") or link.source_ref.target_uid != item.uid: continue
                scope_node = memory.get_hypernode(link.target_ref.target_uid)
                if not scope_node or scope_node.uid not in eligible or not scope_node.evidence: continue
                scope_bindings = {binding.role_id: memory.label(binding.target_ref.target_uid) for binding in scope_node.role_bindings}
                scope_value = scope_bindings.get("OBJECT", "")
                scope_ids = {word.replace("-", "") for word in re.findall(r"[\w-]+", norm(scope_value)) if any(char.isdigit() for char in word)}
                scope_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", norm(scope_value)) if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
                query_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", query) if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
                event_nouns = query_nouns - scope_nouns
                if missing_ids <= scope_ids and scope_nouns & query_nouns and event_nouns and all(
                    lexical_score(word, focus_value(bindings)) > 0 for word in event_nouns
                ):
                    linked.append(scope_node)
            if len(linked) != 1: continue
            scope_support[item.uid] = linked[0]
        eligible_scored.append(row)
    if intent == "classification":
        subjects = {bindings.get("SUBJECT", "") for _, _, _, bindings in eligible_scored}
        if len(subjects) != 1: return "insufficient_evidence", []
    best = max(((focus_score if intent in {"state", "cause_reason", "cause_result"} else score) for score, focus_score, _, _ in eligible_scored), default=0)
    selected = [(item, bindings) for score, focus_score, item, bindings in eligible_scored if best > 0 and (focus_score if intent in {"state", "cause_reason", "cause_result"} else score) == best]
    if not selected: return "insufficient_evidence", []
    if intent in {"cause_reason", "cause_result"}:
        # Repeated evidence for the same cause/effect is not another answer or another root.
        answer_role = "SUBJECT" if intent == "cause_reason" else "OBJECT"
        representatives = {}
        for item, bindings in selected:
            answer_uid = next(binding.target_ref.target_uid for binding in item.role_bindings if binding.role_id == answer_role)
            previous = representatives.get(answer_uid)
            if previous is None or len(item.evidence[0].exact_text) < len(previous[0].evidence[0].exact_text): representatives[answer_uid] = (item, bindings)
        selected = list(representatives.values())
    if intent == "cause_reason" and "первоприч" in query:
        if len(selected) != 1: return "insufficient_evidence", []
        item, bindings = selected[0]
        chain, visited = [item], {item.uid}
        target, current = bindings.get("OBJECT", "событие"), bindings.get("SUBJECT", "")
        while current:
            subject_uid = next(binding.target_ref.target_uid for binding in item.role_bindings if binding.role_id == "SUBJECT")
            previous = [row for row in scored if any(binding.role_id == "OBJECT" and binding.target_ref.target_uid == subject_uid for binding in row[2].role_bindings)]
            if not previous: break
            by_cause = {}
            for row in sorted(previous, key=lambda row: row[0], reverse=True):
                cause_uid = next(binding.target_ref.target_uid for binding in row[2].role_bindings if binding.role_id == "SUBJECT")
                by_cause.setdefault(cause_uid, row)
            previous = list(by_cause.values())
            if len(previous) != 1 or previous[0][2].uid in visited:
                return "insufficient_evidence", []
            _, _, item, bindings = previous[0]
            chain.append(item); visited.add(item.uid); current = bindings.get("SUBJECT", "")
        return f"Первопричина события «{target}» — {current}.", chain
    if intent in {"cause_reason", "cause_result"}:
        selected.sort(key=lambda row: (row[0].evidence[0].document_uid, row[0].evidence[0].start_offset, row[0].uid))
        return " ".join(dict.fromkeys(item.evidence[0].exact_text for item, _ in selected)), [item for item, _ in selected]
    if intent == "state":
        subject = selected[0][1].get("SUBJECT", "Объект")
        values = []
        for _, bindings in selected:
            value = bindings.get("STATE") or bindings.get("OBJECT", "состояние")
            if bindings.get("TIME"): value += f" ({bindings['TIME']})"
            values.append(value)
        return f"{subject}: {'; '.join(dict.fromkeys(values))}.", [item for item, _ in selected]
    if intent == "classification":
        return selected[0][0].evidence[0].exact_text, [selected[0][0]]
    if intent == "location":
        if location_inverse:
            return " ".join(dict.fromkeys(f"{bindings['SUBJECT']}: {bindings['LOCATION']}." for _, bindings in selected)), [item for item, _ in selected]
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
        item = selected[0][0]
        support = scope_support.get(item.uid)
        return item.evidence[0].exact_text, [item, support] if support else [item]
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


def render_evidence_claims(decoded: dict, facts: list[dict]) -> dict:
    """Render explicitly attributed claims; legacy inline answers still use the strict validator."""
    if not isinstance(decoded, dict): raise ValueError("answer must be a JSON object")
    if "claims" not in decoded: return validate_evidence_answer(decoded, facts)
    if not isinstance(decoded["claims"], list): raise ValueError("claims must be an array")
    sentences, references = [], []
    for claim in decoded["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not isinstance(claim.get("evidence_ids"), list):
            raise ValueError("invalid claim structure")
        text = claim["text"].strip()
        refs = claim["evidence_ids"]
        if not text or len(re.split(r"(?<=[.!?])\s+", text)) != 1 or re.search(r"\[E\d+\]", text):
            raise ValueError("each claim must contain one sentence without inline citations")
        if not refs or any(not isinstance(ref, str) for ref in refs): raise ValueError("claim lacks evidence ids")
        sentences.append(f"{text.rstrip('.!?')} {''.join(f'[{ref}]' for ref in dict.fromkeys(refs))}.")
        references.extend(refs)
    return validate_evidence_answer({"status": decoded.get("status"), "answer": " ".join(sentences), "evidence_ids": references}, facts)


def evidence_answer_request(provider: OpenAICompatibleProvider, question: str, facts: list[dict], grounded_draft: str = "") -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
            "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "text": {"type": "string", "minLength": 1},
                "evidence_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "uniqueItems": True},
            }, "required": ["text", "evidence_ids"]}},
        },
        "required": ["status", "claims"],
    }
    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": "Ответь кратко на языке вопроса только по EVIDENCE, без внешних знаний. Не путай идентификаторы объектов. Верни status=answered и claims: отдельное утверждение в каждом объекте, text — одно предложение БЕЗ ссылок, evidence_ids — идентификаторы источников этого утверждения. Ссылки добавит приложение. Например: {\"status\":\"answered\",\"claims\":[{\"text\":\"Причина — перегрев\",\"evidence_ids\":[\"E1\"]}]}. Если доказательств недостаточно, верни status=insufficient_evidence и claims=[]. Непустой GROUNDED_DRAFT — уже найденный ответ: переформулируй его без добавления новых утверждений и укажи соответствующие источники. Для свойств перечисли все подтверждённые значения для нужного объекта."},
            {"role": "user", "content": json.dumps({"question": question, "GROUNDED_DRAFT": grounded_draft, "EVIDENCE": facts}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "evidence_bound_answer", "strict": True, "schema": schema}} if provider.response_format == "json_schema" else {"type": "json_object"},
        "temperature": 0, "max_tokens": int(os.getenv("AH_ANSWER_MAX_TOKENS", "2048")),
    }
    if "deepseek" in provider.model.lower(): payload["reasoning"] = {"effort": "none", "exclude": True}
    return payload


def parse_evidence_response(response: dict, facts: list[dict]) -> dict:
    """Apply the same structured-answer validation to API and recorded benchmark responses."""
    try:
        if response["choices"][0].get("finish_reason") == "length": raise ValueError("answer model output was truncated")
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, list): content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        decoded = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.I))
    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise AnswerResponseError("answer model returned invalid structured JSON") from exc
    except ValueError as exc:
        raise AnswerResponseError(str(exc)) from exc
    try:
        return render_evidence_claims(decoded, facts)
    except (ValueError, TypeError, AttributeError) as exc:
        raise AnswerResponseError(str(exc)) from exc


def generate_evidence_answer(question: str, facts: list[dict], model_override: str, grounded_draft: str = "") -> dict:
    try:
        provider, meta = configured_provider(None if model_override == "configured" else model_override)
        if not isinstance(provider, OpenAICompatibleProvider): raise ValueError("external answer provider is not configured")
        payload = evidence_answer_request(provider, question, facts, grounded_draft)
        response = provider._request_json(payload)
    except ValueError as exc:
        raise AnswerProviderError(str(exc)) from exc
    validated = parse_evidence_response(response, facts)
    return {**validated, "provider": meta}
