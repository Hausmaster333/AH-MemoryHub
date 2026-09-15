from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import memory_locked, AHMemory, ROLE_IDS, lexical_key, norm, _morph_analyzer, _lemma
from .models import *


TEMPLATE_ROLES = {
    "CAUSE": ("CAUSE", "SUBJECT", "OBJECT", "TIME", "LOCATION"),
    "FOLLOW": ("SUBJECT", "OBJECT", "TIME", "LOCATION"),
    "IS-A": ("SUBJECT", "OBJECT"),
    "LOCATED_AT": ("SUBJECT", "LOCATION", "TIME"),
    "USES_TOOL": ("SUBJECT", "TOOL", "OBJECT", "LOCATION"),
    "HAS_STATE": ("SUBJECT", "STATE", "TIME"),
    "OBSERVED": ("SUBJECT", "OBJECT", "LOCATION", "TIME"),
    "RESULT": ("SUBJECT", "RESULT", "CAUSE"),
    "HAS": ("SUBJECT", "OBJECT", "TIME"),
    "RUN": ("SUBJECT", "HOW-TO"),
    "LIVE": ("SUBJECT", "LOCATION"),
    "ACTION": ("SUBJECT", "OBJECT", "TOOL", "LOCATION", "TIME", "PURPOSE", "HOW-TO"),
    "PURPOSE": ("SUBJECT", "PURPOSE"),
    "IN_SCOPE": ("SUBJECT", "OBJECT"),
}

TEMPLATE_REQUIRED = {
    "CAUSE": ("SUBJECT", "OBJECT"), "FOLLOW": ("SUBJECT", "OBJECT"),
    "IS-A": ("SUBJECT", "OBJECT"), "LOCATED_AT": ("SUBJECT", "LOCATION"),
    "USES_TOOL": ("SUBJECT", "TOOL"), "HAS_STATE": ("SUBJECT", "STATE"),
    "OBSERVED": ("SUBJECT", "OBJECT"), "RESULT": ("SUBJECT", "RESULT"),
    "HAS": ("SUBJECT", "OBJECT"), "RUN": ("SUBJECT", "HOW-TO"),
    "LIVE": ("SUBJECT", "LOCATION"),
    "ACTION": ("SUBJECT", "OBJECT"), "PURPOSE": ("SUBJECT", "PURPOSE"),
    "IN_SCOPE": ("SUBJECT", "OBJECT"),
}
PROMPT_VERSION = "perception-ir-v3.23"  # The model request is unchanged; IN_SCOPE is deterministic only.
MODEL_TEMPLATE_ROLES = {name: roles for name, roles in TEMPLATE_ROLES.items() if name != "IN_SCOPE"}
_CANDIDATE_CACHE: dict[str, tuple[tuple[CandidateFact, ...], tuple[dict, ...]]] = {}
ROLE_CANONICALIZATION = {
    ("CAUSE", "RESULT"): "OBJECT",
    ("CAUSE", "CAUSE"): "SUBJECT",
    ("RUN", "AGENT"): "SUBJECT",
    ("RUN", "STATE"): "HOW-TO",
    ("RUN", "OBJECT"): "HOW-TO",
    ("HAS", "PART"): "OBJECT",
    ("HAS_STATE", "VALUE"): "STATE",
    ("IS-A", "VALUE"): "OBJECT",
    ("LOCATED_AT", "DESTINATION"): "LOCATION",
    ("LIVE", "DESTINATION"): "LOCATION",
}
PARSER_MODEL_PRESETS = {
    "stealth/ox-alpha": "json_object",
    "~deepseek/deepseek-v4-flash-latest": "json_schema",
    "deepseek/deepseek-v4-flash-0731:nitro": "json_schema",
}


@dataclass(frozen=True)
class TextSpan:
    uid: str
    index: int
    start: int
    end: int
    text: str
    context_before: str = ""
    context_after: str = ""


def segment_text(text: str) -> list[TextSpan]:
    """Deterministic sentence/newline spans with offsets into the untouched document."""
    bounds: list[tuple[int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if start is None and not char.isspace(): start = index
        boundary = char == "\n" or (char in ".!?" and (index + 1 == len(text) or text[index + 1].isspace()))
        if start is not None and boundary:
            end = index if char == "\n" else index + 1
            while end > start and text[end - 1].isspace(): end -= 1
            if end > start: bounds.append((start, end))
            start = None
    if start is not None:
        end = len(text)
        while end > start and text[end - 1].isspace(): end -= 1
        if end > start: bounds.append((start, end))
    raw = [(start, end, text[start:end]) for start, end in bounds]
    return [TextSpan(
        uid=f"span_{hashlib.sha256(f'{start}:{end}:{value}'.encode()).hexdigest()[:16]}",
        index=index, start=start, end=end, text=value,
        context_before=raw[index - 1][2] if index else "",
        context_after=raw[index + 1][2] if index + 1 < len(raw) else "",
    ) for index, (start, end, value) in enumerate(raw)]


def _span_for_range(spans: list[TextSpan], start: int, end: int) -> TextSpan | None:
    return next((span for span in spans if span.start <= start and end <= span.end), None)


class Provider:
    model_id = "provider"

    def extract(self, text: str) -> list[CandidateFact]:
        raise NotImplementedError


def _locate_quote(text: str, quote: str, cursor: int = 0) -> tuple[int, int] | None:
    start = text.find(quote, cursor)
    if start < 0: start = text.find(quote)
    if start >= 0: return start, start + len(quote)
    compact = lambda value: " ".join(re.findall(r"\w+", norm(value)))
    wanted = compact(quote)
    ranked = sorted(((SequenceMatcher(None, wanted, compact(span.text)).ratio(), span) for span in segment_text(text)), key=lambda item: item[0], reverse=True)
    if ranked and ranked[0][0] >= .92 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .03):
        return ranked[0][1].start, ranked[0][1].end
    return None


def _locate_mention(text: str, observed: str, canonical: str | None = None) -> tuple[int, int] | None:
    match = re.search(re.escape(observed), text, re.I)
    if match: return match.start(), match.end()
    wanted = re.findall(r"[\w-]+", norm(canonical or observed))
    words = list(re.finditer(r"[\w-]+", text))
    matches = []
    for index in range(len(words) - len(wanted) + 1):
        start, end = words[index].start(), words[index + len(wanted) - 1].end()
        actual = re.findall(r"[\w-]+", norm(text[start:end]))
        if all(left == right or (len(left) >= 4 and len(right) >= 4 and left[:3] == right[:3]) for left, right in zip(wanted, actual)): matches.append((start, end))
    return matches[0] if len(matches) == 1 else None


def _passive_cause(text: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+?)\s+(?:был[аои]?\s+)?вызван[аоы]?\s+(?:не\s+[^,.;]+,\s*а\s+)?([^,.;]+)[.]?", text, re.I)
    if not match or re.match(r"не\b", match[2], re.I): return None
    return match[2].strip(), match[1].strip()


class RuleBasedProvider(Provider):
    model_id = "rule-based-offline"

    def extract(self, text: str) -> list[CandidateFact]:
        candidates = []
        for span in segment_text(text):
            s = span.text.strip(" .!?")
            low = norm(s)
            predicate = "OBSERVED"
            explicit_follow = re.match(r"после того как\s+(.+?)[,;]\s*(.+)$", s, re.I)
            if _passive_cause(s) or re.search(r"(?:вызвал\w*|прив\w*\s+к|из-за|потому что|обусловил\w*)", low): predicate = "CAUSE"
            elif any(x in low for x in ("использовал", "применил", "инструмент", "ключ")): predicate = "USES_TOOL"
            elif explicit_follow: predicate = "FOLLOW"
            elif any(x in low for x in ("это", "является", "относится к")): predicate = "IS-A"
            elif any(x in low for x in ("находился", "находится", "в цехе", "в помещении")): predicate = "LOCATED_AT"
            elif any(x in low for x in ("состояние", "перешёл", "перешел")): predicate = "HAS_STATE"
            # Keep a deterministic, conservative extraction: explicit clauses become roles.
            if predicate == "CAUSE":
                separator = r"(?:потому что|из-за|вызвал\w*|прив\w*\s+к|обусловил\w*)"
            elif predicate == "USES_TOOL":
                separator = r"(?:применил\w*|использовал\w*)"
            elif predicate == "FOLLOW":
                separator = r"(?:после того как|после|затем|далее|следом|сначала)"
            else:
                separator = r"(?:обнаружил\w*|наблюдал\w*)"
            parts = list(explicit_follow.groups()) if predicate == "FOLLOW" and explicit_follow else re.split(r"\s+" + separator + r"\s+", s, maxsplit=1, flags=re.I)
            subj = parts[0].strip(" ,:")
            obj = parts[-1].strip(" ,:") if len(parts) > 1 else s
            if predicate == "CAUSE" and _passive_cause(s): subj, obj = _passive_cause(s)
            subj = re.sub(r"^(?:после этого|затем|далее|сначала)\s+", "", subj, flags=re.I).strip()
            bindings = [CandidateBinding(role_id="SUBJECT", value=subj), CandidateBinding(role_id="OBJECT", value=obj)]
            loc = re.search(r"(?<!\w)(?:в|на)\s+([А-Яа-яЁёA-Za-z0-9 _-]+?)(?:,|\s+(?:и|после|затем)|$)", s)
            if loc: bindings.append(CandidateBinding(role_id="LOCATION", value=loc.group(1).strip()))
            tm = re.search(r"\b(\d{4}-\d\d-\d\d|\d{1,2}:\d\d|ночью|утром|вечером)\b", s, re.I)
            if tm: bindings.append(CandidateBinding(role_id="TIME", value=tm.group(1)))
            tool = re.search(r"((?:[А-Яа-яЁёA-Za-z0-9_-]+\s+)?(?:ключ\w*|инструмент\w*))", s, re.I)
            if tool and predicate == "USES_TOOL": bindings.append(CandidateBinding(role_id="TOOL", value=tool.group(1).strip()))
            candidates.append(CandidateFact(
                predicate=predicate, bindings=tuple(bindings), source_start=span.start,
                source_end=span.end, exact_text=span.text, confidence=0.78,
                model_id=self.model_id, span_uid=span.uid, sentence_index=span.index,
                context_before=span.context_before, context_after=span.context_after,
            ))
        tools = {candidate.span_uid: candidate for candidate in _tool_use_candidates(text)}
        return [tools.get(candidate.span_uid, candidate) if candidate.predicate == "USES_TOOL" else candidate for candidate in candidates] + _temporal_state_candidates(text)


class OpenAICompatibleProvider(Provider):
    """Structured text perception for any Chat Completions compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_seconds: float = 45.0, response_format: str = "json_schema", perception_format: str = "ir"):
        if not base_url.strip(): raise ValueError("AH_LLM_BASE_URL is required")
        if not model.strip(): raise ValueError("AH_LLM_MODEL is required")
        if response_format not in {"json_schema", "json_object"}: raise ValueError("AH_LLM_RESPONSE_FORMAT must be json_schema or json_object")
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"): self.endpoint += "/chat/completions"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format
        if perception_format not in {"ir", "roles"}: raise ValueError("perception_format must be ir or roles")
        self.perception_format = perception_format
        self.model_id = self.model
        self.last_warnings: list[str] = []

    def _request_json(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method="POST")
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode())
            except HTTPError as exc:
                detail = exc.read(2048).decode(errors="replace")
                transient = exc.code in {408, 500, 502, 503, 504} or (exc.code == 429 and any(marker in detail.lower() for marker in ("temporarily", "rate-limit", "rate_limit")))
                if transient and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                raise ValueError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise ValueError(f"LLM endpoint failed: {exc}") from exc
        raise ValueError("LLM endpoint failed after retries")

    def _completion(self, text: str) -> dict:
        roles = sorted(ROLE_IDS)
        example = {
            "mentions": [
                {"id": "m1", "span_index": 0, "text": "насос", "type": "entity", "canonical_label": "насос", "coref_to": None},
                {"id": "m2", "span_index": 0, "text": "цехе A", "type": "location", "canonical_label": "цех A", "coref_to": None},
                {"id": "m3", "span_index": 0, "text": "цехе B", "type": "location", "canonical_label": "цех B", "coref_to": None},
            ],
            "facts": [{
                "span_index": 0,
                "predicate": "LOCATED_AT",
                "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["m1"]}}, {"role_id": "LOCATION", "term": {"operator": "OR", "mention_ids": ["m2", "m3"]}}],
                "quote": "точная непрерывная подстрока исходного текста",
                "confidence": 0.9,
                "unresolved_entities": False,
                "section_hint": "P",
                "section_confidence": 0.85,
                "section_reason": "устойчивый факт о конкретном объекте",
            }]
        }
        output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mentions": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "span_index": {"type": "integer", "minimum": 0},
                            "text": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": ["entity", "event", "state", "time", "location", "property"]},
                            "canonical_label": {"type": ["string", "null"]},
                            "coref_to": {"type": ["string", "null"]},
                        },
                        "required": ["id", "span_index", "text", "type", "canonical_label", "coref_to"],
                    },
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "span_index": {"type": "integer", "minimum": 0},
                            "predicate": {"type": "string", "enum": sorted(MODEL_TEMPLATE_ROLES)},
                            "bindings": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "role_id": {"type": "string", "enum": roles},
                                        "term": {
                                            "type": "object", "additionalProperties": False,
                                            "properties": {
                                                "operator": {"type": "string", "enum": ["ATOM", "AND", "OR", "VERY"]},
                                                "mention_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                                            },
                                            "required": ["operator", "mention_ids"],
                                        },
                                    },
                                    "required": ["role_id", "term"],
                                },
                            },
                            "quote": {"type": "string", "minLength": 1},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "unresolved_entities": {"type": "boolean"},
                            "section_hint": {"type": "string", "enum": ["C", "P", "H"]},
                            "section_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "section_reason": {"type": "string"},
                        },
                        "required": ["span_index", "predicate", "bindings", "quote", "confidence", "unresolved_entities", "section_hint", "section_confidence", "section_reason"],
                    },
                }
            },
            "required": ["mentions", "facts"],
        }
        system = (
            "Ты модуль восприятия AH-памяти. Верни Perception IR, не отвечай на документ и не добавляй внешние знания. "
            f"Допустимые предикаты: {', '.join(MODEL_TEMPLATE_ROLES)}. Допустимые роли: {', '.join(roles)}. "
            "Каждому mention и fact назначь span_index из SPANS. Сначала перечисли все упоминания: text — точная подстрока указанного SPAN, canonical_label — краткая нормальная форма, coref_to — id более раннего однозначного упоминания. "
            "Каждый facts[] содержит ровно один атомарный предикат. Не пропускай декларативные предложения: каждое должно дать хотя бы один факт либо unresolved_entities=true. "
            "Разложи владение и свойство на HAS(owner, part) и HAS_STATE(part, state). Для «X принадлежит Y» верни HAS(Y, X); "
            "действие и его причину — на отдельный факт действия и CAUSE. Несколько однородных свойств — отдельные HAS_STATE. "
            "AND/OR/VERY кодируй только через term.operator; не склеивай операнды строкой. quote равен полному text выбранного SPAN; mention.text копируй из него дословно. "
            "FOLLOW допустим только при двух явно названных событиях; маркер «после этого» сам по себе не факт FOLLOW. "
            "Не разрешай неоднозначную анафору: поставь unresolved_entities=true. В CAUSE причина — SUBJECT, следствие — OBJECT; в HAS часть — OBJECT. "
            "Произошедшее действие представь как ACTION: SUBJECT — исполнитель, OBJECT — полная заземлённая фраза действия без исполнителя; TOOL, LOCATION, TIME, PURPOSE и HOW-TO добавляй только при явном основании. "
            "RUN допустим только для поведения и образа действия, например RUN(заяц, быстро). OBSERVED допустим только при явно названном наблюдателе и наблюдаемом объекте. "
            "USES_TOOL допустим только когда SUBJECT применяет настоящий инструмент. Конструкция «X используется для Y» описывает назначение: верни PURPOSE с SUBJECT=X и PURPOSE=Y. "
            "В FOLLOW SUBJECT — предшествующее событие, OBJECT — более позднее событие целиком. Например: «После восстановления давления сигнал отключился» → FOLLOW(восстановление давления, отключение сигнала). "
            "Временную границу состояния не теряй: «остался остановлен до завершения проверки» → HAS_STATE с TIME=«до завершения проверки». "
            "Для каждого факта предложи section_hint: C — общее правило или знание о классе объектов; P — устойчивое знание данного агента о конкретном именованном объекте; H — произошедшее событие, действие или временное состояние. "
            "Один документ может содержать факты разных секций. section_confidence оценивает только классификацию секции, section_reason кратко указывает основание. "
            "Пример общего разложения: «У двигателя два датчика: красный и синий» → HAS(двигатель, датчик), HAS_STATE(датчик, красный), HAS_STATE(датчик, синий). "
            f"Верни только JSON вида {json.dumps(example, ensure_ascii=False)}"
        )
        spans = segment_text(text)
        if self.perception_format == "roles":
            # ponytail: scalar role values reuse the existing compiler; explicit logical terms require IR.
            output_schema["properties"].pop("mentions")
            output_schema["required"] = ["facts"]
            binding_schema = output_schema["properties"]["facts"]["items"]["properties"]["bindings"]["items"]
            binding_schema["properties"].pop("term")
            binding_schema["properties"]["value"] = {"type": "string", "minLength": 1}
            binding_schema["required"] = ["role_id", "value"]
            system = (
                "Извлеки из документа атомарные факты без внешних знаний. Верни JSON facts по схеме. "
                f"Предикаты и разрешённые роли: {json.dumps(MODEL_TEMPLATE_ROLES, ensure_ascii=False)}. "
                "Каждый факт: span_index — index предложения из SPANS, quote — полный text этого SPAN. "
                "bindings содержат role_id и value — точную фразу из цитаты; не используй ссылки на упоминания. "
                "Сохраняй идентификаторы объектов, время, инструменты. Не пропускай предложения. "
                "CAUSE: SUBJECT причина, OBJECT следствие. HAS: SUBJECT владелец, OBJECT часть. "
                "USES_TOOL: SUBJECT исполнитель, TOOL инструмент. HAS_STATE: SUBJECT объект, STATE состояние, TIME временная граница если указана. "
                "FOLLOW: SUBJECT предыдущее событие, OBJECT последующее. ACTION: SUBJECT исполнитель, OBJECT действие. "
                "Не утверждай отрицательные, условные и предположительные события. Не разрешай неоднозначные местоимения: unresolved_entities=true. "
                "section_hint: C общие знания о классе, P устойчивые свойства именованного объекта, H события и временные состояния."
            )
        span_index = [{"index": span.index, "start": span.start, "end": span.end, "text": span.text} for span in spans]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"PROMPT_VERSION={PROMPT_VERSION}\nSPANS={json.dumps(span_index, ensure_ascii=False)}\nDOCUMENT:\n{text}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "ah_fact_candidates", "strict": True, "schema": output_schema}} if self.response_format == "json_schema" else {"type": "json_object"},
            "temperature": 0,
            "max_tokens": int(os.getenv("AH_LLM_MAX_TOKENS", "8192")),
        }
        if "deepseek" in self.model.lower(): payload["reasoning"] = {"effort": "none", "exclude": True}
        while True:
            try:
                return self._request_json(payload)
            except ValueError as exc:
                message = str(exc).lower()
                if "temperature" in message and "temperature" in payload:
                    self.last_warnings.append("provider rejected temperature=0; retried without temperature")
                    payload.pop("temperature")
                    continue
                response_format = payload.get("response_format")
                if "response_format" not in message and "http 400" not in message: raise
                if response_format and response_format.get("type") == "json_schema":
                    self.last_warnings.append("provider rejected JSON Schema; retried with JSON mode")
                    payload["response_format"] = {"type": "json_object"}
                    continue
                if response_format:
                    self.last_warnings.append("provider rejected JSON mode; retried with prompt-only JSON")
                    payload.pop("response_format", None)
                    continue
                raise

    def _extract_single(self, text: str, source_offset: int, document_spans: list[SourceSpan]) -> list[CandidateFact]:
        response = self._completion(text)
        local_spans = segment_text(text)
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("LLM structured output was truncated at AH_LLM_MAX_TOKENS")
            content = choice["message"]["content"]
            if isinstance(content, list): content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            content = str(content).strip()
            if content.startswith("```"): content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            decoded = json.loads(content)
            raw_facts = decoded.get("facts", [])
            raw_mentions = decoded.get("mentions", [])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM response does not contain valid structured facts JSON") from exc
        aliases = {"СУБЪЕКТ": "SUBJECT", "ОБЪЕКТ": "OBJECT", "ЛОКАЦИЯ": "LOCATION", "ВРЕМЯ": "TIME", "ПРИЧИНА": "CAUSE", "ЦЕЛЬ": "PURPOSE", "ИНСТРУМЕНТ": "TOOL", "INSTRUMENT": "TOOL", "РЕЗУЛЬТАТ": "RESULT"}
        mentions: list[CandidateMention] = []
        mention_cursor = 0
        mention_type_aliases = {
            "part": "entity", "object": "entity", "organization": "entity", "organisation": "entity",
            "company": "entity", "person": "entity", "device": "entity", "action": "event",
            "attribute": "property", "color": "state", "place": "location",
        }
        for index, item in enumerate(raw_mentions):
            try:
                observed = str(item["text"]).strip()
                canonical = str(item["canonical_label"]).strip() if item.get("canonical_label") else None
                mention_span = int(item["span_index"]) if "span_index" in item else None
                if mention_span is not None and not 0 <= mention_span < len(local_spans): raise ValueError("mention span_index is outside SPANS")
                search_start = local_spans[mention_span].start if mention_span is not None else 0
                search_end = local_spans[mention_span].end if mention_span is not None else len(text)
                start = text.find(observed, max(search_start, mention_cursor if mention_span is None else search_start), search_end)
                if start < 0:
                    located = _locate_mention(text[search_start:search_end], observed, canonical)
                    if located:
                        start = search_start + located[0]
                        observed = text[start:search_start + located[1]]
                if start < 0: raise ValueError("mention is not an exact substring of the source")
                mention_cursor = start + len(observed)
                mention_type = str(item.get("type", "entity")).lower().strip()
                mentions.append(CandidateMention(
                    mention_id=str(item["id"]), observed_text=observed, source_start=source_offset + start,
                    source_end=source_offset + start + len(observed), mention_type=mention_type_aliases.get(mention_type, mention_type),
                    canonical_label=canonical,
                    coref_to=str(item["coref_to"]) if item.get("coref_to") else None,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.last_warnings.append(f"mention {index + 1} rejected: {exc}")
        mention_ids = {mention.mention_id for mention in mentions}
        candidates: list[CandidateFact] = []
        cursor = 0
        for index, item in enumerate(raw_facts):
            try:
                predicate = str(item["predicate"]).upper().strip()
                if predicate not in TEMPLATE_ROLES: raise ValueError(f"unknown predicate {predicate}")
                supplied_quote = str(item["quote"]).strip()
                fact_span = int(item["span_index"]) if "span_index" in item else None
                if fact_span is not None:
                    if not 0 <= fact_span < len(local_spans): raise ValueError("fact span_index is outside SPANS")
                    selected_span = local_spans[fact_span]
                    start, end, quote = selected_span.start, selected_span.end, selected_span.text
                    if _locate_quote(selected_span.text, supplied_quote) is None:
                        self.last_warnings.append(f"fact {index + 1}: quote replaced by declared source span")
                else:
                    located = _locate_quote(text, supplied_quote, cursor)
                    if located is None: raise ValueError("quote is not an exact or conservatively aligned source span")
                    start, end = located
                    quote = text[start:end]
                cursor = end
                bindings = []
                for raw_binding in item.get("bindings", []):
                    role = aliases.get(str(raw_binding.get("role_id", "")).upper().strip(), str(raw_binding.get("role_id", "")).upper().strip())
                    if "term" in raw_binding:
                        raw_term = dict(raw_binding["term"])
                        if raw_term.get("operator") in {"ATOM", "VERY"} and len(raw_term.get("mention_ids", ())) > 1:
                            raw_term["operator"] = "OR" if re.search(r"\b(?:или|либо)\b", quote, re.I) else "AND"
                            self.last_warnings.append(f"fact {index + 1}: repaired multi-operand scalar term")
                        term = CandidateTerm.model_validate(raw_term)
                        if any(mention_id not in mention_ids for mention_id in term.mention_ids): raise ValueError("term references unknown mention")
                        bindings.append(CandidateBinding(role_id=role, term=term))
                    else:
                        bindings.append(CandidateBinding(role_id=role, value=str(raw_binding.get("value", "")).strip()))
                if not bindings: raise ValueError("fact has no role bindings")
                absolute_start = source_offset + start
                span = _span_for_range(document_spans, absolute_start, source_offset + end)
                candidates.append(CandidateFact(
                    predicate=predicate, bindings=tuple(bindings), source_start=absolute_start,
                    source_end=source_offset + end, exact_text=quote,
                    confidence=float(item.get("confidence", .5)),
                    unresolved_entities=bool(item.get("unresolved_entities", False)),
                    model_id=self.model_id, span_uid=span.uid if span else None,
                    sentence_index=span.index if span else 0,
                    context_before=span.context_before if span else "",
                    context_after=span.context_after if span else "",
                    mentions=tuple(mentions), group_uid=span.uid if span else None,
                    section_hint=str(item.get("section_hint", "")).upper() or None,
                    section_confidence=float(item.get("section_confidence", .5)),
                    section_reason=str(item.get("section_reason", "")).strip(),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.last_warnings.append(f"fact {index + 1} rejected: {exc}")
        return candidates

    def extract(self, text: str) -> list[CandidateFact]:
        self.last_warnings = []
        spans = segment_text(text)
        if not spans: return []
        def extract_range(start, end):
            left, right = spans[start].start, spans[end - 1].end
            for attempt in range(2):
                try:
                    return self._extract_single(text[left:right], left, spans)
                except ValueError as exc:
                    if attempt == 0:
                        self.last_warnings.append(f"structured output failed; retried once: {exc}")
                        continue
                    if "structured output was truncated" not in str(exc) or end - start <= 1:
                        raise
                    middle = (start + end) // 2
                    self.last_warnings.append(f"truncated span range {start}:{end}; split at {middle}")
                    return extract_range(start, middle) + extract_range(middle, end)
        if len(spans) <= 8:
            return extract_range(0, len(spans))
        candidates = []
        start, chunk_count = 0, 0
        while start < len(spans):
            end = min(start + 6, len(spans))
            chunk_count += 1
            candidates.extend(extract_range(start, end))
            if end == len(spans): break
            start = end - 1
        self.last_warnings.append(f"document parsed in {chunk_count} overlapping chunks")
        return candidates


_ANAPHORA = {"он", "она", "оно", "они", "его", "её", "ее", "их", "это", "этот", "эта", "эти"}


def _canonical_mention_label(mention: CandidateMention) -> str:
    observed = mention.observed_text.strip()
    canonical = (mention.canonical_label or observed).strip()
    # ponytail: case/spacing normalization only; reviewed aliases can enable semantic renaming later.
    return canonical if norm(canonical) == norm(observed) else observed


def _resolved_mention_label(mention_id: str, mentions: dict[str, CandidateMention], trail: tuple[str, ...] = ()) -> str:
    if mention_id in trail: raise ValueError("coreference_cycle|coreference graph must be acyclic")
    mention = mentions.get(mention_id)
    if mention is None: raise ValueError(f"unknown_mention|unknown mention: {mention_id}")
    if mention.coref_to:
        target = mentions.get(mention.coref_to)
        if target is None: raise ValueError(f"unknown_coreference|unknown coreference target: {mention.coref_to}")
        if target.source_start > mention.source_start: raise ValueError("forward_coreference|coreference must point to an earlier mention")
        return _resolved_mention_label(mention.coref_to, mentions, trail + (mention_id,))
    label = _canonical_mention_label(mention)
    short = set(lexical_key(label))
    if short and len(short) <= 2:
        candidates = []
        for earlier in mentions.values():
            if earlier.source_start >= mention.source_start: continue
            earlier_label = _canonical_mention_label(earlier)
            words = re.findall(r"[\w-]+", earlier.observed_text)
            named = any(char.isdigit() for char in earlier.observed_text) or any(word[:1].isupper() for word in words[1:])
            if named and short < set(lexical_key(earlier_label)): candidates.append(earlier_label)
        if len(set(candidates)) == 1: return candidates[0]
    return label


def _term_label(term: CandidateTerm, mentions: dict[str, CandidateMention]) -> str:
    values = [_resolved_mention_label(mention_id, mentions) for mention_id in term.mention_ids]
    return values[0] if term.operator == "ATOM" else f"{term.operator}({', '.join(values)})"


def _candidate_signature(candidate: CandidateFact, include_section: bool = False) -> tuple:
    signature = candidate.predicate, tuple(sorted((binding.role_id, norm(binding.value or "")) for binding in candidate.bindings))
    return (candidate.section_hint or "C", *signature) if include_section else signature


def _semantic_duplicate(left: CandidateFact, right: CandidateFact) -> bool:
    if left.predicate != right.predicate or (left.group_uid or left.span_uid) != (right.group_uid or right.span_uid): return False
    a, b = ({binding.role_id: binding.value or "" for binding in candidate.bindings} for candidate in (left, right))
    return a.keys() == b.keys() and all(_grounded_value(a[role], b[role]) and _grounded_value(b[role], a[role]) for role in a)


def _collapse_semantic_duplicates(candidates: list[CandidateFact]) -> tuple[list[CandidateFact], list[dict]]:
    kept, rejected = [], []
    for candidate in candidates:
        if any(_semantic_duplicate(candidate, prior) for prior in kept):
            rejected.append({"candidate": _candidate_log(candidate), "reason": "semantic_duplicate", "message": "an equivalent fact from the same source group was already kept"})
        else:
            kept.append(candidate)
    return kept, rejected


def _candidate_log(candidate: CandidateFact) -> dict:
    return {
        "predicate": candidate.predicate,
        "bindings": [binding.model_dump(mode="json") for binding in candidate.bindings],
        "source_start": candidate.source_start, "source_end": candidate.source_end,
        "exact_text": candidate.exact_text, "confidence": candidate.confidence,
        "model_id": candidate.model_id, "group_uid": candidate.group_uid or candidate.span_uid,
        "section_hint": candidate.section_hint, "section_confidence": candidate.section_confidence,
        "section_reason": candidate.section_reason,
    }


def _grounded_value(value: str, quote: str) -> bool:
    """Require every meaningful value token to have a conservative source match."""
    ignored = {"and", "or", "very", "состояние", "статус"}
    left = [token for token in re.findall(r"[\w-]+", norm(value)) if (len(token) > 2 or any(char.isdigit() for char in token)) and token not in ignored]
    right = [token for token in re.findall(r"[\w-]+", norm(quote)) if len(token) > 2 or any(char.isdigit() for char in token)]
    similar = lambda a, b: a == b or (not any(char.isdigit() for char in a + b) and ((min(len(a), len(b)) >= 3 and a[:3] == b[:3]) or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= .66)))
    return bool(left) and all(any(similar(a, b) for b in right) for a in left)


_GENERAL_KNOWLEDGE = re.compile(r"\b(?:обычно|как правило|может|могут|относится к|является видом|используется для)\b", re.I)
_EVENT_OCCURRENCE = re.compile(r"\b(?:произош|случил|вызвал|вызвала|вызвало|прив[её]л|привела|остановил|применил|добавил|отключил|восстановил|повысил|снизил|поместил|откатил)\w*\b", re.I)
_CONCRETE_OBJECT = re.compile(r"(?:\b[А-ЯA-ZА-ЯЁ]{1,8}[-‑–]?\d+[\w.-]*\b|[«\"][^»\"]+[»\"]|\b(?:оператор|инженер|куратор|астроном|шеф)\s+[А-ЯЁ][а-яё]+)")
_TRANSIENT_STATE = re.compile(r"\b(?:горяч|перегрет|остановлен|поврежд|приостановлен|заблокирован|охлажд[её]н|отключ[её]н|низк\w*\s+достоверност)\w*\b", re.I)
_RUN_BEHAVIOUR = re.compile(r"\b(?:беж|бега|движ|лет|плыв|полз|скач|работа)\w*\b", re.I)
_OBSERVATION = re.compile(r"\b(?:наблюда|зафиксир|обнаруж|зарегистрир|увид|замет)\w*\b", re.I)
_EFFECT_VERB_PATTERN = r"(?:восстановил|повысил|снизил|уменьшил|сократил|изменил|смягчил|улучшил)(?:а|о|и)?\b"
_CAUSE_MARKER = re.compile(rf"\b(?:вызвал|вызвала|вызвало|вызвали|прив[её]л\w*\s+к|из-за|поэтому|обусловил\w*|{_EFFECT_VERB_PATTERN})\b", re.I)
def _finite_action(text: str):
    """Locate a finite verb, not a noun which happens to share its ending."""
    for token in re.finditer(r"\b[А-Яа-яЁё-]+\b", text):
        parsed = _morph_analyzer().parse(token.group())[0]
        if parsed.tag.POS == "VERB":
            return re.compile(r".+", re.S).match(text, token.start())
    return None


def route_candidates(candidates: list[CandidateFact]) -> list[CandidateFact]:
    """Assign C/P/H without ever treating S as a fact destination."""
    group_hints: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.section_hint:
            group_hints.setdefault(candidate.group_uid or candidate.span_uid or "", []).append(candidate.section_hint)
    routed = []
    for candidate in candidates:
        quote = candidate.exact_text
        grounded_text = " ".join([quote, *(binding.value or "" for binding in candidate.bindings)])
        roles = {binding.role_id for binding in candidate.bindings}
        inherited = group_hints.get(candidate.group_uid or candidate.span_uid or "", [])
        proposed = candidate.section_hint or (max(set(inherited), key=inherited.count) if inherited else None)
        if candidate.predicate in {"ACTION", "FOLLOW", "OBSERVED", "IN_SCOPE"} or "TIME" in roles or _EVENT_OCCURRENCE.search(quote):
            section, confidence, reason = "H", .98, "событие, действие или временно ограниченный факт"
        elif candidate.predicate == "HAS_STATE" and _TRANSIENT_STATE.search(quote):
            section, confidence, reason = "H", .92, "временное состояние конкретного объекта"
        elif _GENERAL_KNOWLEDGE.search(quote) and not _CONCRETE_OBJECT.search(grounded_text):
            section, confidence, reason = "C", .95, "обобщение или знание о классе объектов"
        elif _CONCRETE_OBJECT.search(grounded_text) and proposed != "H":
            section, confidence, reason = "P", max(.85, candidate.section_confidence), candidate.section_reason or "устойчивый факт о конкретном именованном объекте"
        elif proposed:
            section, confidence, reason = proposed, candidate.section_confidence, candidate.section_reason or "классификация модуля восприятия"
        else:
            section, confidence, reason = "C", .7, "консервативное общее знание по умолчанию"
        routed.append(candidate.model_copy(update={"section_hint": section, "section_confidence": confidence, "section_reason": reason}))
    return routed


def _coverage_warnings(text: str, candidates: list[CandidateFact]) -> list[dict]:
    covered = {candidate.group_uid or candidate.span_uid for candidate in candidates}
    return [
        {"group_uid": span.uid, "source_start": span.start, "source_end": span.end, "exact_text": span.text,
         "reason": "uncovered_source_span", "message": "parser returned no reviewable fact for this source span"}
        for span in segment_text(text) if span.uid not in covered
    ]


def _split_explicit_follow(value: str) -> tuple[str, str] | None:
    """Return (earlier, later) only for an explicit `После <event> <event>` clause."""
    stripped = value.strip(" .!?")
    body = re.sub(r"^После\s+", "", stripped, flags=re.I)
    if body == stripped or re.match(r"этого\b", body, re.I): return None
    words = list(re.finditer(r"[А-Яа-яЁёA-Za-z0-9-]+", body))
    for verb_index, match in enumerate(words):
        if not re.search(r"(?:л|ла|ло|ли|лся|лась|лось|лись)$", match.group(0), re.I): continue
        verb = match.group(0).casefold()
        if verb.startswith("произош"):
            split = verb_index
        elif words[verb_index - 1].group(0)[:1].isupper():
            title = words[verb_index - 2].group(0).casefold() if verb_index >= 2 else ""
            split = verb_index - (2 if title in {"оператор", "инженер", "куратор", "астроном", "шеф", "техник"} else 1)
        else:
            split = verb_index - (2 if verb_index >= 4 else 1)
        if split <= 0: continue
        earlier = body[:words[split].start()].strip(" ,")
        later = body[words[split].start():].strip(" ,")
        if earlier and later: return earlier, later
    return None


def _explicit_follow_candidates(text: str) -> list[CandidateFact]:
    """Conservative Russian `После <event> <later event>` recovery when the LLM omits FOLLOW."""
    recovered = []
    for span in segment_text(text):
        events = _split_explicit_follow(span.text)
        if not events: continue
        earlier, later = events
        recovered.append(CandidateFact(
            predicate="FOLLOW", bindings=(CandidateBinding(role_id="SUBJECT", value=earlier), CandidateBinding(role_id="OBJECT", value=later)),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.8,
            model_id="deterministic-follow", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))
    return recovered


def _named_scope_declarations(text: str) -> list[tuple[CandidateFact, int, int]]:
    """Only a source-anchored declaration about a named section can scope its events."""
    spans = segment_text(text)
    headings = [span for span in spans if len(span.text) <= 120 and not re.search(r"[.!?]$", span.text)
                and text[span.end:span.end + 2] == "\n\n"]
    declarations = []
    for index, heading in enumerate(headings):
        region_end = headings[index + 1].start if index + 1 < len(headings) else len(text)
        title_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", norm(heading.text))
                       if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
        if not title_nouns: continue
        for span in spans:
            if not heading.end < span.start < region_end: continue
            match = re.fullmatch(r"\s*(.+?)\s+относится\s+к\s+([^.!?]+)[.]?\s*", span.text, re.I)
            if not match or re.search(r"\b(?:не|если|возможно|предположительно|планируется)\b", span.text, re.I): continue
            subject, scope = match.group(1).strip(), match.group(2).strip()
            subject_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", norm(subject))
                             if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
            ids = [word for word in re.findall(r"[\w-]+", scope) if any(c.isalpha() for c in word) and any(c.isdigit() for c in word)]
            if not subject_nouns or not subject_nouns <= title_nouns or len(ids) != 1: continue
            candidate = CandidateFact(
                predicate="IN_SCOPE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject),
                                                 CandidateBinding(role_id="OBJECT", value=scope)),
                source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
                model_id="deterministic-explicit-scope", span_uid=span.uid, sentence_index=span.index,
                context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
                section_hint="H",
            )
            declarations.append((candidate, heading.end, region_end))
    return declarations


def _coordinated_property_candidates(text: str) -> list[CandidateFact]:
    recovered = []
    for span in segment_text(text):
        match = re.match(r"(.+?)\s+имеет\s+(.+?)\s+и\s+находится\s+в\s+(.+?)[.]?$", span.text, re.I)
        facts: tuple[tuple[str, str, str], ...] = ()
        if match:
            subject, obj, state = (value.strip(" .") for value in match.groups())
            facts = (("HAS", "OBJECT", obj), ("HAS_STATE", "STATE", state))
        elif not re.search(r"[,;]", span.text):
            states = re.match(r"(.+?)\s+[—-]?\s*([А-Яа-яЁё-]+)\s+и\s+([А-Яа-яЁё-]+)[.]?$", span.text, re.I)
            if states:
                subject, first, second = (value.strip(" .—-") for value in states.groups())
                # ponytail: nominal adjective clauses only; action objects are not object states.
                morph = _morph_analyzer()
                if all(morph.parse(value)[0].tag.POS in {"ADJF", "ADJS", "PRTF", "PRTS"} for value in (first, second)) and not any(morph.parse(word)[0].tag.POS in {"VERB", "INFN", "GRND"} for word in re.findall(r"[А-Яа-яЁё-]+", subject)):
                    facts = (("HAS_STATE", "STATE", first), ("HAS_STATE", "STATE", second))
        if not facts: continue
        for predicate, role, value in facts:
            recovered.append(CandidateFact(
                predicate=predicate, bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id=role, value=value)),
                source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.85,
                model_id="deterministic-coordination", span_uid=span.uid, sentence_index=span.index,
                context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
            ))
    return recovered


def _tool_use_candidates(text: str) -> list[CandidateFact]:
    """Recover an explicit actor/tool pair without maintaining a tool lexicon."""
    recovered = []
    for span in segment_text(text):
        match = re.search(r"^(.+?)\s+(?:использовал\w*|применил\w*)\s+(.+?)(?=\s+для\s+|\s+чтобы\s+|[.]?$)", span.text, re.I)
        if not match: continue
        subject, tool = (value.strip(" .") for value in match.groups())
        subject = re.sub(r"^(?:затем|далее|после этого)\s+", "", subject, flags=re.I)
        recovered.append(CandidateFact(
            predicate="USES_TOOL", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="TOOL", value=tool)),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
            model_id="deterministic-tool", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))
    return recovered


def _effect_candidates(text: str) -> list[CandidateFact]:
    """Recover explicit cause/effect verbs, including coordinated effects."""
    recovered = []
    for span in segment_text(text):
        match = re.match(rf"^(.+?)\s+({_EFFECT_VERB_PATTERN})\s+(.+?)(?:\s+и\s+({_EFFECT_VERB_PATTERN})\s+(.+?))?[.]?$", span.text, re.I)
        if not match: continue
        subject, first_verb, first_object, second_verb, second_object = match.groups()
        effects = [(first_verb, first_object), *(([(second_verb, second_object)] if second_verb and second_object else []))]
        for verb, obj in effects:
            recovered.append(CandidateFact(
                predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject.strip()), CandidateBinding(role_id="OBJECT", value=f"{verb} {obj}".strip())),
                source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.88,
                model_id="deterministic-effect", span_uid=span.uid, sentence_index=span.index,
                context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
            ))
    return recovered


def _stable_fact_candidates(text: str) -> list[CandidateFact]:
    """Recover explicit stable location, ownership, containment, and purpose clauses."""
    recovered: list[CandidateFact] = []

    def add(span: TextSpan, predicate: str, bindings: tuple[tuple[str, str], ...]):
        recovered.append(CandidateFact(
            predicate=predicate, bindings=tuple(CandidateBinding(role_id=role, value=value.strip(" .")) for role, value in bindings),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
            model_id="deterministic-stable", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))

    for span in segment_text(text):
        value = span.text.strip(" .")
        located = re.match(r"(.+?)\s+(?:установлен\w*|расположен\w*|размещ[её]н\w*|находится|хранится|наблюдается)\s+(?:в|на)\s+(.+)$", value, re.I)
        if located and not re.match(r"(?:статус|состояни)\w*\b", located.group(2), re.I):
            subject, location = located.groups()
            operands = [item.strip().removeprefix("в ").removeprefix("на ") for item in re.split(r"\s+или\s+", location, flags=re.I)]
            add(span, "LOCATED_AT", (("SUBJECT", subject), ("LOCATION", f"OR({', '.join(operands)})" if len(operands) > 1 else operands[0])))
        belongs = re.match(r"(.+?)\s+принадлежит\s+(.+?)(?=\s+и\s+(?:используется|содержит|включает|имеет|находится)\b|$)", value, re.I)
        if belongs: add(span, "HAS", (("SUBJECT", belongs.group(2)), ("OBJECT", belongs.group(1))))
        owner = re.match(r"Владельцем\s+(.+?)\s+является\s+(.+)$", value, re.I)
        if owner: add(span, "HAS", (("SUBJECT", owner.group(2)), ("OBJECT", owner.group(1))))
        containment = re.match(r"У\s+(.+?)\s+есть\s+(.+)$", value, re.I)
        if containment: add(span, "HAS", (("SUBJECT", containment.group(1)), ("OBJECT", containment.group(2))))
        responsibility = re.match(r"За\s+(.+?)\s+отвечает\s+(.+)$", value, re.I)
        if responsibility: add(span, "HAS", (("SUBJECT", responsibility.group(2)), ("OBJECT", responsibility.group(1))))
        purpose = re.match(r"(.+?)(?:\s+принадлежит\s+.+?)?\s+(?:и\s+)?(?:используется|предназначен\w*)\s+для\s+(.+)$", value, re.I)
        if purpose: add(span, "PURPOSE", (("SUBJECT", purpose.group(1)), ("PURPOSE", purpose.group(2))))
    return recovered


def _is_nominal_classification(subject: str, category: str) -> bool:
    """A clock/value or a finite event clause cannot be a taxonomic class."""
    if re.fullmatch(r"[\d\s:.,/%+−-]+", subject.strip()) or re.match(r"^[+−-]?\d", category.strip()):
        return False
    return not any(_morph_analyzer().parse(word)[0].tag.POS in {"VERB", "INFN", "PRTS"}
                   for word in re.findall(r"[а-яё]+", category.casefold()))


def _declarative_recovery_candidates(text: str) -> list[CandidateFact]:
    """Recover common Russian definition/property constructions from exact spans."""
    recovered: list[CandidateFact] = []
    last_subject = ""

    def add(span: TextSpan, predicate: str, bindings: tuple[tuple[str, str], ...]):
        recovered.append(CandidateFact(
            predicate=predicate, bindings=tuple(CandidateBinding(role_id=role, value=value) for role, value in bindings),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.84,
            model_id="deterministic-declarative", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))

    for span in segment_text(text):
        dash = re.search(r"\s[—-]\s", span.text)
        definition = re.match(r"(.+?)\s+[—-]\s+(.+?)(?:,\s+котор\w*\b|[.]?$)", span.text, re.I) if dash and "," not in span.text[:dash.start()] else None
        if definition:
            subject, value = (part.strip(" .") for part in definition.groups())
            # A labelled scalar is a property, not IS-A. A rejected alternative stays in the source quote.
            scalar = re.split(r",\s+а\s+не\s+", value, maxsplit=1, flags=re.I)[0]
            if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d|[+−-]?\d+(?:[.,]\d+)?\s*(?:%|°[CС]|бар|МПа|мм|см|кг|минут[аы]?|секунд[аы]?)", scalar, re.I) and not re.fullmatch(r"[\d\s:.,/%+−-]+", subject) and not re.search(r"\b(?:не|если|возможно|предположительно)\b", subject, re.I):
                add(span, "HAS_STATE", (("SUBJECT", subject), ("STATE", scalar)))
        if definition and _is_nominal_classification(definition.group(1), definition.group(2)):
            last_subject = definition.group(1).strip(" .")
            add(span, "IS-A", (("SUBJECT", last_subject), ("OBJECT", definition.group(2).strip(" ."))))
        if last_subject:
            habitat = re.search(r"\b(?:обитает|жив[её]т)\s+(?:в|на)\s+(.+?)[.]?$", span.text, re.I)
            if habitat:
                operands = [item.strip().removeprefix("в ").removeprefix("на ") for item in re.split(r"\s+или\s+", habitat.group(1), flags=re.I)]
                add(span, "LIVE", (("SUBJECT", last_subject), ("LOCATION", f"OR({', '.join(operands)})" if len(operands) > 1 else operands[0])))
            possession = re.match(r"У\s+(?:него|не[её]|них)\s+(.+?)(?:,|[.]?$)", span.text, re.I)
            if possession:
                add(span, "HAS", (("SUBJECT", last_subject), ("OBJECT", possession.group(1).strip(" ."))))
            behaviour = re.search(r"\b(бега\w*|беж\w*|движ\w*|лет\w*|плыв\w*)\s+(?:он|она|они)\s+(.+?)[.]?$", span.text, re.I)
            if behaviour:
                add(span, "RUN", (("SUBJECT", last_subject), ("HOW-TO", behaviour.group(2).strip(" ."))))
            parts = re.match(r"(\w+)\s+\w+\s+([^,]+),\s+а\s+(.+?)\s+[—-]\s+(.+?)[.]?$", span.text, re.I)
            time_words = r"(?:зимой|летом|весной|осенью|утром|дн[её]м|вечером|ночью)"
            if parts and not re.match(time_words + "$", parts.group(1), re.I) and not re.match(time_words + "$", parts.group(3), re.I):
                first_part, first_state, second_part, second_state = (value.strip(" .") for value in parts.groups())
                add(span, "HAS", (("SUBJECT", last_subject), ("OBJECT", f"{first_state} {first_part.lower()}")))
                add(span, "HAS", (("SUBJECT", last_subject), ("OBJECT", f"{second_state} {second_part}")))
            seasonal = re.match(r"(\w+)\s+(.+?)\s+([^,]+),\s+а\s+(\w+)\s+[—-]\s+(.+?)[.]?$", span.text, re.I)
            if seasonal and re.match(time_words + "$", seasonal.group(1), re.I):
                first_time, item, first_state, second_time, second_state = (value.strip(" .") for value in seasonal.groups())
                add(span, "HAS", (("SUBJECT", last_subject), ("OBJECT", f"{item} {first_state}"), ("TIME", first_time)))
                add(span, "HAS", (("SUBJECT", last_subject), ("OBJECT", f"{item} {second_state}"), ("TIME", second_time)))
    return recovered


# ponytail: explicit event/time nouns only; dates/durations stay with the model until parsed unambiguously.
_EVENT_DEADLINE = r"до\s+(?:(?:начала|завершения|окончания|получения|подтверждения|восстановления|прибытия|возвращения|устранения)\s+[^,.;]+|(?:следующ\w+|ближайш\w+|очередн\w+)\s+(?:окна|сеанса|смены|этапа|проверки|запуска|утра|вечера|дня|ночи)\b[^,.;]*)"


def _temporal_state_candidates(text: str) -> list[CandidateFact]:
    recovered = []
    for span in segment_text(text):
        completed = re.fullmatch(r"В\s+((?:[01]?\d|2[0-3]):[0-5]\d)\s+([^,;.!?]+)[.]?", span.text, re.I)
        if completed:
            clock, clause = completed.groups()
            clause = clause.strip()
            verb = re.match(r"[А-Яа-яЁё-]+\b", clause)
            parsed_verb = _morph_analyzer().parse(verb.group())[0] if verb else None
            event = clause[verb.end():].strip() if verb else ""
            finite = _finite_action(clause)
            if (parsed_verb and finite and finite.start() == 0
                    and parsed_verb.tag.tense == "past"
                    and parsed_verb.normal_form in {"завершиться", "закончиться"}
                    and event and not _finite_action(event)
                    and any(_morph_analyzer().parse(word)[0].tag.POS == "NOUN" for word in re.findall(r"[А-Яа-яЁё-]+", event))
                    and len(re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", span.text)) == 1
                    and not re.search(r"\b(?:не|ни|если|должн\w*|планир\w*|предположительно|вероятно|возможно|якобы)\b", span.text, re.I)):
                recovered.append(CandidateFact(
                    predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=event),
                                                     CandidateBinding(role_id="STATE", value=verb.group()),
                                                     CandidateBinding(role_id="TIME", value=clock)),
                    source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
                    model_id="deterministic-temporal-completion", span_uid=span.uid, sentence_index=span.index,
                    context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
                ))
        stamped = re.match(r"^((?:[01]?\d|2[0-3]):[0-5]\d)\s*[-–—]\s*(.+?)\s+был[аои]?\s+(временно\s+)?([а-яё-]+)\b", span.text, re.I)
        if stamped:
            clock, subject, modifier, state = stamped.groups()
            tail = span.text[stamped.end():]
            if ((_TRANSIENT_STATE.fullmatch(state) or any(parse.tag.POS == "PRTS" and parse.normal_form == "снять" for parse in _morph_analyzer().parse(state)))
                    and not re.search(r"\b(?:не|должен|должна|должно|должны|предположительно|вероятно|возможно)\b", span.text, re.I)
                    and re.fullmatch(r"(?:\s+для\s+[а-яё\s-]+)?[.]?", tail, re.I)
                    and not _finite_action(tail)
                    and len(re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", span.text)) == 1
                    and any(_morph_analyzer().parse(word)[0].tag.POS == "NOUN" or any(char.isdigit() for char in word)
                            for word in re.findall(r"[\w-]+", subject))):
                recovered.append(CandidateFact(
                    predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject),
                                                     CandidateBinding(role_id="STATE", value=(modifier or "") + state),
                                                     CandidateBinding(role_id="TIME", value=clock)),
                    source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
                    model_id="deterministic-temporal-state", span_uid=span.uid, sentence_index=span.index,
                    context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
                ))
        match = re.fullmatch(rf"(.+?)\s+оста(?:лся|лась|лось|лись|ётся|ется|ются|вался|валась|валось|вались)\s+(.+?)\s+({_EVENT_DEADLINE})[.]?", span.text, re.I)
        if not match: continue
        subject, state, boundary = (value.strip(" .") for value in match.groups())
        recovered.append(CandidateFact(
            predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="STATE", value=state), CandidateBinding(role_id="TIME", value=boundary)),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
            model_id="deterministic-temporal-state", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))
    return recovered


_ATOMIC_ROLES = {
    "HAS": {"OBJECT"}, "HAS_STATE": {"STATE"}, "USES_TOOL": {"TOOL"},
    "ACTION": {"TOOL"}, "CAUSE": {"OBJECT"}, "RESULT": {"RESULT"},
}


def _expand_atomic_candidate(candidate: CandidateFact) -> list[CandidateFact]:
    roles = _ATOMIC_ROLES.get(candidate.predicate, set())
    for role in roles:
        matching = [binding for binding in candidate.bindings if binding.role_id == role]
        if len(matching) > 1:
            common = [binding for binding in candidate.bindings if binding.role_id != role]
            return [item for binding in matching for item in _expand_atomic_candidate(candidate.model_copy(update={"bindings": tuple([*common, binding])}))]
    for index, binding in enumerate(candidate.bindings):
        if binding.role_id in roles and binding.term and binding.term.operator == "AND":
            expanded = []
            for mention_id in binding.term.mention_ids:
                bindings = list(candidate.bindings)
                bindings[index] = binding.model_copy(update={"term": CandidateTerm(operator="ATOM", mention_ids=(mention_id,))})
                expanded.extend(_expand_atomic_candidate(candidate.model_copy(update={"bindings": tuple(bindings)})))
            return expanded
    return [candidate]


def _explicit_event_bindings(bindings: list[CandidateBinding], subject: str, obj: str) -> list[CandidateBinding]:
    """Repair reversed/missing event roles without discarding anchored roles and terms."""
    result = [binding for binding in bindings if binding.role_id not in {"SUBJECT", "OBJECT"}]
    for role, segment in (("SUBJECT", subject), ("OBJECT", obj)):
        existing = next((binding for binding in bindings if binding.role_id == role), None)
        result.append(existing if existing and norm(existing.value or "") and norm(existing.value) in norm(segment)
                      else CandidateBinding(role_id=role, value=segment, observed=segment))
    return result


def canonicalize_candidates(text: str, candidates: list[CandidateFact]) -> tuple[list[CandidateFact], list[dict]]:
    """Resolve safe local anaphora, validate semantics, and collapse exact fact duplicates."""
    spans = segment_text(text)
    accepted: list[CandidateFact] = []
    by_signature: dict[tuple, int] = {}
    rejected: list[dict] = []
    last_by_role: dict[str, str] = {}

    def reject(candidate: CandidateFact, code: str, message: str):
        rejected.append({"candidate": _candidate_log(candidate), "reason": code, "message": message})

    atomic = [item for candidate in candidates for item in _expand_atomic_candidate(candidate)]
    for raw in sorted(atomic, key=lambda item: (item.source_start, item.source_end, item.predicate)):
        try:
            if raw.source_end > len(text) or text[raw.source_start:raw.source_end] != raw.exact_text:
                raise ValueError("source_span_mismatch|source offsets do not reproduce exact_text")
            predicate = raw.predicate.upper().strip()
            roles = TEMPLATE_ROLES.get(predicate)
            if roles is None: raise ValueError(f"unknown_template|unknown template: {predicate}")
            span = _span_for_range(spans, raw.source_start, raw.source_end)
            canonical_start, canonical_end, canonical_text = raw.source_start, raw.source_end, raw.exact_text
            mention_map: dict[str, CandidateMention] = {}
            for mention in raw.mentions:
                if mention.mention_id in mention_map: raise ValueError(f"duplicate_mention|duplicate mention id: {mention.mention_id}")
                if mention.source_end > len(text) or text[mention.source_start:mention.source_end] != mention.observed_text:
                    raise ValueError(f"mention_span_mismatch|mention {mention.mention_id} is not anchored to source")
                mention_map[mention.mention_id] = mention
            for mention in raw.mentions: _resolved_mention_label(mention.mention_id, mention_map)
            normalized_bindings: list[CandidateBinding] = []
            seen_roles: set[str] = set()
            unresolved = False
            for binding in raw.bindings:
                observed_role = binding.role_id.upper().strip()
                role = ROLE_CANONICALIZATION.get((predicate, observed_role), observed_role)
                value = _term_label(binding.term, mention_map) if binding.term else " ".join((binding.value or "").strip().split())
                if role not in roles: raise ValueError(f"role_not_allowed|role {role} is not allowed for {predicate}")
                if role in seen_roles: raise ValueError(f"duplicate_role|role {role} occurs more than once")
                if not value: raise ValueError(f"empty_binding|role {role} has an empty value")
                observed = binding.observed
                if norm(value) in _ANAPHORA:
                    antecedent = last_by_role.get(role) or (last_by_role.get("SUBJECT") if role != "SUBJECT" else None)
                    if antecedent:
                        observed, value = value, antecedent
                    else:
                        unresolved = True
                normalized_bindings.append(CandidateBinding(role_id=role, value=value, observed=observed, term=binding.term))
                seen_roles.add(role)
            if predicate == "IS-A":
                values = {binding.role_id: binding.value for binding in normalized_bindings}
                if not _is_nominal_classification(values.get("SUBJECT", ""), values.get("OBJECT", "")):
                    raise ValueError("invalid_classification|a time, numeric value or event clause is not a taxonomic class")
            used_for = re.search(r"\b(?:используется|предназначен\w*)\s+для\s+(.+)$", span.text if span else raw.exact_text, re.I)
            if predicate in {"USES_TOOL", "RUN", "ACTION", "PURPOSE"} and used_for:
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                purpose = used_for.group(1).strip(" .")
                if not subject or not purpose: raise ValueError("invalid_function|used-for construction requires a subject and purpose")
                predicate, roles = "PURPOSE", TEMPLATE_ROLES["PURPOSE"]
                normalized_bindings = [subject, CandidateBinding(role_id="PURPOSE", value=purpose, observed=purpose)]
                seen_roles = {"SUBJECT", "PURPOSE"}
                if span:
                    canonical_start = span.start + used_for.start()
                    canonical_end = span.end
                    canonical_text = text[canonical_start:canonical_end]
            if predicate == "RUN" and not _RUN_BEHAVIOUR.search(raw.exact_text):
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                action = re.search(r"\b[А-Яа-яЁёA-Za-z-]+(?:л|ла|ло|ли|лся|лась|лось|лись)\b.*", raw.exact_text, re.I)
                if not subject or not action: raise ValueError("invalid_run|RUN requires an explicit behaviour or manner")
                predicate, roles = "ACTION", TEMPLATE_ROLES["ACTION"]
                action_value = action.group(0).strip(" .")
                normalized_bindings = [subject, CandidateBinding(role_id="OBJECT", value=action_value, observed=action_value)]
                seen_roles = {"SUBJECT", "OBJECT"}
            if predicate == "ACTION":
                responsibility = re.search(r"\bза\s+(.+?)\s+отвечает\s+(.+?)(?:[.,;]|$)", raw.exact_text, re.I)
                if responsibility:
                    predicate, roles = "HAS", TEMPLATE_ROLES["HAS"]
                    normalized_bindings = [CandidateBinding(role_id="SUBJECT", value=responsibility.group(2).strip(), observed=responsibility.group(2).strip()), CandidateBinding(role_id="OBJECT", value=responsibility.group(1).strip(), observed=responsibility.group(1).strip())]
                    seen_roles = {"SUBJECT", "OBJECT"}
                else:
                    subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                    action = _finite_action(raw.exact_text)
                    if not subject or not action: raise ValueError("invalid_action|ACTION requires an explicit finite action")
                    optional = [binding for binding in normalized_bindings if binding.role_id not in {"SUBJECT", "OBJECT"}]
                    action_value = action.group(0).strip(" .")
                    model_action = next((binding.value for binding in normalized_bindings if binding.role_id == "OBJECT"), "")
                    model_verb = _finite_action(model_action)
                    if model_action and model_verb and model_verb.start() == 0 and norm(model_action) in norm(raw.exact_text):
                        action_value = model_action
                    normalized_bindings = [subject, CandidateBinding(role_id="OBJECT", value=action_value, observed=action_value), *optional]
                    seen_roles = {binding.role_id for binding in normalized_bindings}
            if predicate == "HAS":
                responsibility = re.search(r"\bза\s+(.+?)\s+отвечает\s+(.+?)(?:[.,;]|$)", raw.exact_text, re.I)
                if responsibility:
                    normalized_bindings = [CandidateBinding(role_id="SUBJECT", value=responsibility.group(2).strip()), CandidateBinding(role_id="OBJECT", value=responsibility.group(1).strip())]
                    seen_roles = {"SUBJECT", "OBJECT"}
                # Keep the model's grounded object boundary: a sentence may enumerate several properties.
            if predicate == "CAUSE":
                if re.search(r"\b(?:если|возможно|вероятно|предположительно)\b", raw.exact_text, re.I):
                    raise ValueError("unasserted_cause|a conditional or uncertain cause is not an asserted event")
                passive_cause = _passive_cause(raw.exact_text)
                if (not _CAUSE_MARKER.search(raw.exact_text) and not passive_cause) or re.search(rf"\bне\s+(?:вызва\w*|прив[её]л\w*|из-за|обуслов\w*|{_EFFECT_VERB_PATTERN})\b", raw.exact_text, re.I):
                    raise ValueError("invalid_cause|CAUSE requires an explicit causal construction")
                if re.search(r"\bвызван[аоы]?\b", raw.exact_text, re.I) and passive_cause is None:
                    raise ValueError("invalid_cause|passive cause must be an explicit positive clause")
                if passive_cause:
                    normalized_bindings = _explicit_event_bindings(normalized_bindings, *passive_cause)
                    seen_roles = {binding.role_id for binding in normalized_bindings}
                explicit_cause = re.match(r"(.+?)\s+(?:вызвал\w*|прив[её]л\w*\s+к)\s+(.+?)[.]?$", raw.exact_text, re.I)
                if explicit_cause:
                    cause, effect = (value.strip(" .") for value in explicit_cause.groups())
                    spatial = re.search(r"\s+((?:внутри|снаружи)\s+[^,.;]+|(?:на|у)\s+(?:входе|выходе)\s+[^,.;]+)$", effect, re.I)
                    location = next((binding for binding in normalized_bindings if binding.role_id == "LOCATION"), None)
                    if spatial and (location is None or norm(location.value or "") == norm(spatial.group(1))):
                        if location is None:
                            normalized_bindings.append(CandidateBinding(role_id="LOCATION", value=spatial.group(1), observed=spatial.group(1)))
                        effect = effect[:spatial.start()].strip()
                    normalized_bindings = _explicit_event_bindings(normalized_bindings, cause, effect)
                    seen_roles = {binding.role_id for binding in normalized_bindings}
            if predicate == "HAS_STATE":
                explicit_owner = re.match(r"(.+?)\s+имеет\s+(.+?)[.]?$", raw.exact_text, re.I)
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                state = next((binding for binding in normalized_bindings if binding.role_id == "STATE"), None)
                if explicit_owner and subject and state and norm(subject.value) == norm(state.value) and norm(state.value) in norm(explicit_owner.group(2)):
                    normalized_bindings = [CandidateBinding(role_id="SUBJECT", value=explicit_owner.group(1), observed=explicit_owner.group(1)) if binding is subject else binding for binding in normalized_bindings]
            if predicate == "HAS_STATE" and re.search(r"\bимеет\b.+\bи\s+находится\b", raw.exact_text, re.I):
                explicit_owner = re.match(r"(.+?)\s+имеет\b", raw.exact_text, re.I)
                if explicit_owner:
                    normalized_bindings = [CandidateBinding(role_id="SUBJECT", value=explicit_owner.group(1), observed=explicit_owner.group(1)) if binding.role_id == "SUBJECT" else binding for binding in normalized_bindings]
            if predicate == "HAS_STATE" and re.match(r"\s*(?:находится|оста[её]тся)\b", raw.exact_text, re.I) and last_by_role.get("SUBJECT"):
                normalized_bindings = [binding.model_copy(update={"value": last_by_role["SUBJECT"], "term": None}) if binding.role_id == "SUBJECT" else binding for binding in normalized_bindings]
            if predicate == "FOLLOW" and re.match(r"\s*после\b", raw.exact_text, re.I):
                explicit_events = _split_explicit_follow(raw.exact_text)
                if explicit_events:
                    earlier, later = explicit_events
                    normalized_bindings = _explicit_event_bindings(normalized_bindings, earlier, later)
                    seen_roles = {binding.role_id for binding in normalized_bindings}
                else:
                    raise ValueError("implicit_follow|FOLLOW requires two explicitly named events")
            elif predicate == "FOLLOW":
                def mention_start(binding: CandidateBinding) -> int | None:
                    starts = [mention_map[mention_id].source_start for mention_id in binding.term.mention_ids if mention_id in mention_map] if binding.term else []
                    return min(starts) if starts else None
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                obj = next((binding for binding in normalized_bindings if binding.role_id == "OBJECT"), None)
                subject_start, object_start = mention_start(subject) if subject else None, mention_start(obj) if obj else None
                if subject and obj and subject_start is not None and object_start is not None:
                    if subject_start > object_start:
                        subject, obj = obj.model_copy(update={"role_id": "SUBJECT"}), subject.model_copy(update={"role_id": "OBJECT"})
                    normalized_bindings = [binding for binding in normalized_bindings if binding.role_id not in {"SUBJECT", "OBJECT"}]
                    normalized_bindings.extend((subject, obj))
            if "TIME" in roles and "TIME" not in seen_roles:
                time_match = re.search(rf"\b{_EVENT_DEADLINE}(?=[.;]?$)", raw.exact_text, re.I)
                if time_match is None and predicate == "ACTION":
                    # Only an unambiguous clock in this action's own phrase; never borrow a nearby event's time.
                    action = next((binding.value for binding in normalized_bindings if binding.role_id == "OBJECT"), "")
                    clocks = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", raw.exact_text)
                    if len(clocks) == 1:
                        time_match = re.search(r"\bв\s+(?:[01]?\d|2[0-3]):[0-5]\d\b", action, re.I)
                if time_match:
                    value = time_match.group(0).strip()
                    normalized_bindings.append(CandidateBinding(role_id="TIME", value=value, observed=value))
                    seen_roles.add("TIME")
            missing = set(TEMPLATE_REQUIRED[predicate]) - seen_roles
            if missing: raise ValueError(f"required_role_missing|required roles missing: {', '.join(sorted(missing))}")
            if unresolved or (raw.unresolved_entities and not any(binding.observed for binding in normalized_bindings)):
                raise ValueError("unresolved_anaphora|candidate contains an unresolved entity reference")
            values = {binding.role_id: norm(binding.value) for binding in normalized_bindings}
            for binding in normalized_bindings:
                if binding.role_id != "STATE": continue
                if binding.term and any(mention_map[mention_id].source_start >= raw.source_start and mention_map[mention_id].source_end <= raw.source_end for mention_id in binding.term.mention_ids if mention_id in mention_map): continue
                state_tokens = [token for token in re.findall(r"\w+", norm(binding.value)) if len(token) > 3 and token not in {"состояние", "статус", "state", "status"}]
                quote_tokens = re.findall(r"\w+", norm(raw.exact_text))
                if state_tokens and not all(any(left == right or (len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]) for right in quote_tokens) for left in state_tokens):
                    raise ValueError("ungrounded_state|STATE is not fully supported by exact_text")
            if values.get("SUBJECT") and values.get("SUBJECT") == values.get("OBJECT"):
                raise ValueError("self_relation|SUBJECT and OBJECT must be distinct")
            if predicate == "OBSERVED" and not _OBSERVATION.search(raw.exact_text):
                raise ValueError("invalid_observation|OBSERVED requires an explicit observer and observation act")
            if predicate in {"USES_TOOL", "ACTION"} and re.search(r"\bне\s+(?:использова\w*|примени\w*)\b", raw.exact_text, re.I):
                raise ValueError("negated_action|a negated tool use cannot be a positive action")
            if predicate == "RUN":
                manner = next(binding.value for binding in normalized_bindings if binding.role_id == "HOW-TO")
                if not _grounded_value(manner, raw.exact_text):
                    raise ValueError("invalid_manner|RUN.HOW-TO must be a source-grounded manner")
            for binding in normalized_bindings:
                if binding.role_id not in {"SUBJECT", "OBJECT", "TOOL", "LOCATION", "TIME", "PURPOSE", "HOW-TO", "STATE"}: continue
                if binding.role_id in {"SUBJECT", "OBJECT"} and norm(binding.value) == norm(raw.exact_text):
                    raise ValueError("sentence_copy_role|a semantic role cannot be the entire source sentence")
                if binding.term:
                    anchored = any(mention_map[mention_id].source_start >= raw.source_start and mention_map[mention_id].source_end <= raw.source_end for mention_id in binding.term.mention_ids if mention_id in mention_map)
                    if not anchored and not _grounded_value(binding.value, raw.exact_text): raise ValueError(f"ungrounded_role|role {binding.role_id} mention is outside exact_text")
                    continue
                if binding.observed: continue
                if binding.role_id == "SUBJECT" and norm(binding.value) == norm(last_by_role.get("SUBJECT", "")): continue
                if not _grounded_value(binding.value, raw.exact_text):
                    raise ValueError(f"ungrounded_role|role {binding.role_id} is not grounded in exact_text")
            if predicate == "FOLLOW" and re.match(r"\s*(?:после этого|затем|далее|следом)\b", raw.exact_text, re.I):
                quote = norm(raw.exact_text)
                anchored = all(any(token in quote for token in norm(binding.value).split() if len(token) > 2) for binding in normalized_bindings if binding.role_id in {"SUBJECT", "OBJECT"})
                if not anchored: raise ValueError("implicit_follow|a temporal marker alone does not identify two explicit events")
            candidate = raw.model_copy(update={
                "predicate": predicate, "bindings": tuple(normalized_bindings),
                "source_start": canonical_start, "source_end": canonical_end, "exact_text": canonical_text,
                "unresolved_entities": False, "span_uid": span.uid if span else raw.span_uid,
                "sentence_index": span.index if span else raw.sentence_index,
                "context_before": span.context_before if span else raw.context_before,
                "context_after": span.context_after if span else raw.context_after,
            })
            signature = _candidate_signature(candidate)
            duplicate_index = by_signature.get(signature)
            if duplicate_index is not None:
                prior = accepted[duplicate_index]
                if candidate.confidence > prior.confidence:
                    reject(prior, "superseded_duplicate", "an overlapping duplicate with higher parser confidence was kept")
                    accepted[duplicate_index] = candidate
                else:
                    reject(candidate, "duplicate_fact", "the same canonical fact was already proposed")
                continue
            by_signature[signature] = len(accepted)
            accepted.append(candidate)
            for binding in candidate.bindings:
                if binding.role_id in {"SUBJECT", "OBJECT", "AGENT", "PATIENT"}: last_by_role[binding.role_id] = binding.value
        except ValueError as exc:
            code, _, message = str(exc).partition("|")
            reject(raw, code, message or code)
    return accepted, rejected


def configured_provider(model_override: str | None = None) -> tuple[Provider, dict]:
    mode = os.getenv("AH_PARSER_PROVIDER", "auto").strip().lower()
    base_url = os.getenv("AH_LLM_BASE_URL", "").strip()
    model = model_override or os.getenv("AH_LLM_MODEL", "").strip()
    api_key = os.getenv("AH_LLM_API_KEY", "").strip()
    if model_override not in {None, *PARSER_MODEL_PRESETS}: raise ValueError(f"unsupported parser model: {model_override}")
    if model_override and not base_url: base_url = "https://openrouter.ai/api/v1"
    use_llm = bool(model_override) or mode in {"openai", "openai_compatible"} or (mode == "auto" and bool(model) and bool(base_url or api_key))
    if mode not in {"auto", "rule", "openai", "openai_compatible"}: raise ValueError(f"unknown AH_PARSER_PROVIDER: {mode}")
    if not use_llm:
        provider = RuleBasedProvider()
        return provider, {"configured": mode, "active": "rule", "model_id": provider.model_id, "fallback": False}
    response_format = PARSER_MODEL_PRESETS.get(model_override, os.getenv("AH_LLM_RESPONSE_FORMAT", "json_schema").strip().lower())
    provider = OpenAICompatibleProvider(base_url or "https://api.openai.com/v1", model, api_key, float(os.getenv("AH_LLM_TIMEOUT_SECONDS", "45")), response_format, os.getenv("AH_LLM_PERCEPTION_FORMAT", "ir"))
    return provider, {"configured": "ui_override" if model_override else mode, "active": "openai_compatible", "model_id": provider.model_id, "fallback": False}


def _resolve_temporal_subjects(text: str, candidates: list[CandidateFact]) -> list[CandidateFact]:
    """Conservative paragraph-local discourse inference, recorded as anchored coreference."""
    spans = segment_text(text)
    result = []
    identifier = r"[A-ZА-ЯЁ][A-ZА-ЯЁ0-9]*-\d[\w-]*"
    for candidate in candidates:
        result.append(candidate)
        if candidate.predicate != "HAS_STATE": continue
        reference = re.match(r"(.+?)\s+оста(?:лся|лась|лось|лись|ётся|ется|ются|вался|валась|валось|вались)\b", candidate.exact_text, re.I)
        if not reference or any(char.isdigit() for char in reference[1]): continue
        words = reference[1].split()
        if len(words) > 2 or (len(words) == 2 and norm(words[0]) not in {"этот", "эта", "это", "данный", "данная", "данное", "основной", "основная", "основное"}): continue
        head = norm(words[-1])
        paragraph_start = text.rfind("\n\n", 0, candidate.source_start) + 2
        paragraph_start = max(0, paragraph_start if paragraph_start > 1 else 0)
        earlier = text[paragraph_start:candidate.source_start]
        # ponytail: explicit numbered introductions and one referent only; a discourse parser can expand coverage later.
        ids = {norm(m[1]) for m in re.finditer(rf"\b{re.escape(head)}\s+({identifier})\b", earlier, re.I)}
        if len(ids) != 1 or re.search(rf"\b(?:друг\w*|втор\w*|резервн\w*|кажд\w*|люб\w*|несколько)\s+{re.escape(head)}\b", earlier, re.I): continue
        anchors = []
        for span in spans:
            if span.start < paragraph_start or span.end > candidate.source_start: continue
            intro = re.match(rf"((?:[А-Яа-яЁё]+\s+){{1,3}}({identifier}))\s+(?:находится|расположен\w*|размещ[её]н\w*|установлен\w*|имеет)\b", span.text)
            if intro and norm(intro[1].split()[0]) in {"если", "каждый", "любой", "всякий", "не", "ни"}: continue
            if intro and norm(intro[1].split()[-2]) == head and norm(intro[2]) in ids:
                anchors.append((span, intro))
        if not anchors: continue
        span, intro = anchors[0]
        subject = next((binding for binding in candidate.bindings if binding.role_id == "SUBJECT"), None)
        if subject is None or any(char.isdigit() for char in subject.value or ""): continue
        if subject.term and subject.term.operator != "ATOM": continue
        anchor_id, reference_id = f"docref_{span.start}", f"docref_{candidate.source_start}"
        if {anchor_id, reference_id} & {m.mention_id for m in candidate.mentions}: continue
        anchor = CandidateMention(mention_id=anchor_id, observed_text=intro[1], source_start=span.start, source_end=span.start + len(intro[1]))
        mention = CandidateMention(mention_id=reference_id, observed_text=reference[1], source_start=candidate.source_start, source_end=candidate.source_start + len(reference[1]), coref_to=anchor_id)
        binding = subject.model_copy(update={"value": anchor.observed_text, "observed": mention.observed_text, "term": CandidateTerm(mention_ids=(reference_id,))})
        result[-1] = candidate.model_copy(update={"bindings": tuple(binding if item is subject else item for item in candidate.bindings), "mentions": (*candidate.mentions, anchor, mention)})
    return result


def extract_candidates(text: str, provider: Provider | None = None, model_override: str | None = None) -> tuple[list[CandidateFact], dict]:
    if provider is not None:
        raw = provider.extract(text)
        candidates, rejected = canonicalize_candidates(text, raw)
        scoped, scope_rejected = canonicalize_candidates(text, [candidate for candidate, _, _ in _named_scope_declarations(text)])
        signatures = {_candidate_signature(candidate) for candidate in candidates}
        candidates.extend(candidate for candidate in scoped if _candidate_signature(candidate) not in signatures)
        rejected.extend(scope_rejected)
        candidates, semantic_duplicates = _collapse_semantic_duplicates(_resolve_temporal_subjects(text, candidates))
        rejected.extend(semantic_duplicates)
        candidates = route_candidates(candidates)
        return candidates, {"configured": "explicit", "active": provider.__class__.__name__, "model_id": getattr(provider, "model_id", provider.__class__.__name__), "fallback": False, "warnings": getattr(provider, "last_warnings", []), "rejection_log": rejected, "coverage_warnings": _coverage_warnings(text, candidates), "prompt_version": PROMPT_VERSION, "cached": False}
    selected, meta = configured_provider(model_override) if model_override else configured_provider()
    cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}:{getattr(selected, 'perception_format', 'ir')}".encode()).hexdigest()
    cached = _CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        candidates = list(cached[0])
        return candidates, {**meta, "warnings": [], "rejection_log": list(cached[1]), "coverage_warnings": _coverage_warnings(text, candidates), "prompt_version": PROMPT_VERSION, "cached": True}
    try:
        raw = selected.extract(text)
    except ValueError as exc:
        if meta["configured"] != "auto": raise
        selected = RuleBasedProvider(); raw = selected.extract(text)
        meta.update({"active": "rule", "model_id": selected.model_id, "fallback": True, "warning": str(exc)})
        cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}:ir".encode()).hexdigest()
    candidates, rejected = canonicalize_candidates(text, raw)
    uncovered = {item["group_uid"] for item in _coverage_warnings(text, candidates)}
    if not isinstance(selected, RuleBasedProvider):
        fallback, fallback_rejected = canonicalize_candidates(text, RuleBasedProvider().extract(text))
        signatures = {_candidate_signature(candidate) for candidate in candidates}
        represented_follow_groups = {candidate.group_uid or candidate.span_uid for candidate in candidates if candidate.predicate == "FOLLOW"}
        candidates.extend(candidate for candidate in fallback if ((candidate.group_uid or candidate.span_uid) in uncovered or (candidate.predicate == "FOLLOW" and (candidate.group_uid or candidate.span_uid) not in represented_follow_groups)) and _candidate_signature(candidate) not in signatures)
        rejected.extend(item for item in fallback_rejected if item.get("candidate", {}).get("group_uid") in uncovered)
        recovered_follow, recovered_rejected = canonicalize_candidates(text, _explicit_follow_candidates(text))
        represented_follow_groups = {candidate.group_uid or candidate.span_uid for candidate in candidates if candidate.predicate == "FOLLOW"}
        candidates.extend(candidate for candidate in recovered_follow if (candidate.group_uid or candidate.span_uid) not in represented_follow_groups)
        rejected.extend(recovered_rejected)
        recovered_properties, property_rejected = canonicalize_candidates(text, _coordinated_property_candidates(text))
        def property_key(candidate: CandidateFact):
            values = tuple(sorted((binding.role_id, lexical_key(binding.value or "")) for binding in candidate.bindings if binding.role_id != "SUBJECT"))
            return candidate.group_uid or candidate.span_uid, candidate.predicate, values
        represented = {property_key(candidate) for candidate in candidates}
        candidates.extend(candidate for candidate in recovered_properties if property_key(candidate) not in represented)
        rejected.extend(property_rejected)
        recovered_tools, tool_rejected = canonicalize_candidates(text, _tool_use_candidates(text))
        represented_tool_groups = {candidate.group_uid or candidate.span_uid for candidate in candidates if candidate.predicate == "USES_TOOL"}
        candidates.extend(candidate for candidate in recovered_tools if (candidate.group_uid or candidate.span_uid) not in represented_tool_groups)
        rejected.extend(tool_rejected)
        recovered_effects, effect_rejected = canonicalize_candidates(text, _effect_candidates(text))
        signatures = {_candidate_signature(candidate) for candidate in candidates}
        candidates.extend(candidate for candidate in recovered_effects if _candidate_signature(candidate) not in signatures)
        rejected.extend(effect_rejected)
        recovered_stable, stable_rejected = canonicalize_candidates(text, _stable_fact_candidates(text))
        signatures = {_candidate_signature(candidate) for candidate in candidates}
        candidates.extend(candidate for candidate in recovered_stable if _candidate_signature(candidate) not in signatures)
        rejected.extend(stable_rejected)
        recovered_declarative, declarative_rejected = canonicalize_candidates(text, _declarative_recovery_candidates(text))
        signatures = {_candidate_signature(candidate) for candidate in candidates}
        candidates.extend(candidate for candidate in recovered_declarative if _candidate_signature(candidate) not in signatures)
        rejected.extend(declarative_rejected)
        recovered_temporal, temporal_rejected = canonicalize_candidates(text, _temporal_state_candidates(text))
        represented_temporal_groups = {candidate.group_uid or candidate.span_uid for candidate in candidates if candidate.predicate == "HAS_STATE" and any(binding.role_id == "TIME" for binding in candidate.bindings)}
        candidates.extend(candidate for candidate in recovered_temporal if (candidate.group_uid or candidate.span_uid) not in represented_temporal_groups)
        rejected.extend(temporal_rejected)
    recovered_scopes, scope_rejected = canonicalize_candidates(text, [candidate for candidate, _, _ in _named_scope_declarations(text)])
    signatures = {_candidate_signature(candidate) for candidate in candidates}
    candidates.extend(candidate for candidate in recovered_scopes if _candidate_signature(candidate) not in signatures)
    rejected.extend(scope_rejected)
    candidates, semantic_duplicates = _collapse_semantic_duplicates(_resolve_temporal_subjects(text, candidates))
    rejected.extend(semantic_duplicates)
    candidates = route_candidates(candidates)
    if len(_CANDIDATE_CACHE) >= 128: _CANDIDATE_CACHE.pop(next(iter(_CANDIDATE_CACHE)))
    _CANDIDATE_CACHE[cache_key] = (tuple(candidates), tuple(rejected))
    meta["warnings"] = getattr(selected, "last_warnings", [])
    meta.update({"rejection_log": rejected, "coverage_warnings": _coverage_warnings(text, candidates), "prompt_version": PROMPT_VERSION, "cached": False})
    return candidates, meta


@dataclass
class IngestionResult:
    ingestion_uid: str
    document_uid: str
    accepted: list[str]
    rejected: list[dict]
    candidates: list[CandidateFact]


class Compiler:
    def __init__(self, memory: AHMemory): self.memory = memory

    def ensure_templates(self):
        for name, roles in TEMPLATE_ROLES.items():
            uid_ = f"tpl_{name.lower()}"
            if uid_ not in self.memory.templates:
                predicate_uid = f"pred_{name.lower().replace('-', '_')}"
                if predicate_uid not in self.memory.symbols:
                    self.memory.add_symbol(FirstOrderSymbol(uid=predicate_uid, sensory_representations=(SensoryRepresentation(modality="text", value=name),)))
                predicate_ref_uid = f"sr_{predicate_uid}"
                predicate_ref = SReference(reference_uid=predicate_ref_uid, target_uid=predicate_uid)
                if predicate_ref_uid not in self.memory.elements:
                    self.memory.add_element("C", MemoryElement(uid=predicate_ref_uid, payload=predicate_ref))
                required = set(TEMPLATE_REQUIRED[name])
                self.memory.add_template(ControlTemplate(uid=uid_, predicate_ref=predicate_ref, ordered_roles=tuple(Role(role_id=r, required=r in required) for r in roles)))

    def _symbol(self, label: str) -> str:
        label = label.strip()
        existing = self.memory.find_abstract_symbols(label)
        if existing: return existing[0].uid
        key = lexical_key(label)
        lexical_match = next((symbol for symbol in self.memory.symbols.values() if key and any(lexical_key(representation.value) == key for representation in symbol.sensory_representations)), None)
        if lexical_match: return lexical_match.uid
        sid = uid("s")
        self.memory.add_symbol(FirstOrderSymbol(uid=sid, sensory_representations=(SensoryRepresentation(modality="text", value=label), SensoryRepresentation(modality="normalized", value=norm(label)))))
        return sid

    def _grounded_symbol(self, label: str) -> str:
        """Return the grounded S actant while ensuring its addressable m, s*, and m* exist."""
        symbol_uid = self._symbol(label)
        concepts = self.memory.find_symbols(label)
        if not concepts:
            grounded_target = next((link.target_ref.target_uid for link in self.memory.links.values() if link.type_id == "GROUNDS" and link.source_ref.target_uid == symbol_uid), None)
            existing = self.memory.get_symbol(grounded_target) if grounded_target else None
            if existing: concepts = [existing]
        concept_uid = concepts[0].uid if concepts else uid("m")
        if not concepts:
            self.memory.add_element("C", MemoryElement(uid=concept_uid, payload=SecondOrderSymbol(uid=concept_uid, properties=(Property(name="label", value=label),))))
        grounded = any(link.type_id == "GROUNDS" and link.source_ref.target_uid == symbol_uid and link.target_ref.target_uid == concept_uid for link in self.memory.links.values())
        if not grounded:
            s_reference = SReference(reference_uid=uid("sr"), target_uid=symbol_uid)
            m_reference = MReference(reference_uid=uid("mr"), target_uid=concept_uid)
            self.memory.add_element("C", MemoryElement(uid=s_reference.reference_uid, payload=s_reference))
            self.memory.add_element("C", MemoryElement(uid=m_reference.reference_uid, payload=m_reference))
            self.memory.add_link(AssociativeLink(uid=uid("l"), type_id="GROUNDS", weight=1, source_ref=s_reference, target_ref=m_reference))
            self.memory.add_link(AssociativeLink(uid=uid("l"), type_id="EVOKES", weight=1, source_ref=m_reference, target_ref=s_reference))
        return symbol_uid

    def _reference_label(self, reference: Reference) -> str:
        element = self.memory.elements.get(reference.target_uid)
        if element and isinstance(element.payload, FunctionalSymbol):
            return f"{element.payload.function_id}({', '.join(self._reference_label(operand) for operand in element.payload.ordered_operands)})"
        return self.memory.label(reference.target_uid)

    def _term_reference(self, term: CandidateTerm, mentions: dict[str, CandidateMention]) -> Reference:
        operands = tuple(SReference(reference_uid=uid("sr"), target_uid=self._grounded_symbol(_resolved_mention_label(mention_id, mentions))) for mention_id in term.mention_ids)
        if term.operator == "ATOM": return operands[0]
        signature = term.operator, tuple(norm(self._reference_label(operand)) for operand in operands)
        for element in self.memory.elements.values():
            payload = element.payload
            if isinstance(payload, FunctionalSymbol) and (payload.function_id, tuple(norm(self._reference_label(operand)) for operand in payload.ordered_operands)) == signature:
                return ElementReference(reference_uid=uid("er"), target_uid=payload.uid)
        functional = FunctionalSymbol(uid=uid("fn"), function_id=term.operator, ordered_operands=operands)
        self.memory.add_element("C", MemoryElement(uid=functional.uid, payload=functional))
        return ElementReference(reference_uid=uid("er"), target_uid=functional.uid)

    def _attach_history_fact(self, hypernode_uid: str, document_uid: str):
        episode_uid = f"episode_{hashlib.sha256(document_uid.encode()).hexdigest()[:16]}"
        existing = self.memory.get_list(episode_uid)
        members = list(existing.ordered_members) if existing else []
        if not any(ref.target_uid == hypernode_uid for ref in members):
            members.append(ElementReference(reference_uid=uid("er"), target_uid=hypernode_uid))
        members.sort(key=lambda ref: self.memory.get_hypernode(ref.target_uid).evidence[0].start_offset)
        episode = MemoryList(
            uid=episode_uid, list_type="Episode", ordered_members=tuple(members),
            properties=(Property(name="document_uid", value=document_uid),),
        )
        if existing:
            self.memory.edit_element(episode_uid, MemoryElement(uid=episode_uid, payload=episode))
        else:
            self.memory.add_element("H", MemoryElement(uid=episode_uid, payload=episode))
        self._sync_episode_follow(episode)

    def _sync_episode_follow(self, episode: MemoryList):
        members = [self.memory.get_hypernode(ref.target_uid) for ref in episode.ordered_members]
        existing = {(link.source_ref.target_uid, link.target_ref.target_uid) for link in self.memory.links.values() if link.type_id == "FOLLOW"}
        for earlier, later in zip(members, members[1:]):
            if not earlier or not later or (earlier.uid, later.uid) in existing or not self._follow_supported(earlier, later): continue
            self.memory.add_link(AssociativeLink(
                uid=uid("l"), type_id="FOLLOW", weight=1,
                source_ref=ElementReference(reference_uid=uid("er"), target_uid=earlier.uid),
                target_ref=ElementReference(reference_uid=uid("er"), target_uid=later.uid),
            ))
            existing.add((earlier.uid, later.uid))

    def _follow_supported(self, earlier: Hypernode, later: Hypernode) -> bool:
        earlier_ev, later_ev = earlier.evidence[0], later.evidence[0]
        if earlier_ev.start_offset >= later_ev.start_offset: return False
        if re.match(r"\s*(?:после|затем|далее|следом)\b", later_ev.exact_text, re.I): return True
        if later.template_ref.target_uid == "tpl_follow": return True
        if earlier.template_ref.target_uid == later.template_ref.target_uid == "tpl_cause":
            def role_value(node: Hypernode, role: str) -> str:
                binding = next((item for item in node.role_bindings if item.role_id == role), None)
                return norm(self._reference_label(binding.target_ref)) if binding else ""
            return bool(role_value(earlier, "OBJECT") and role_value(earlier, "OBJECT") == role_value(later, "SUBJECT"))
        return False

    def _sync_scope_links(self, text: str, document_uid: str) -> bool:
        """Link an event to one explicit scope declaration in its named source section."""
        old = {link_uid for link_uid, link in self.memory.links.items()
               if link.type_id == "IN_SCOPE" and link_uid.startswith("l_scope_")
               and (node := self.memory.get_hypernode(link.source_ref.target_uid))
               and any(ev.document_uid == document_uid for ev in node.evidence)}
        if old:
            links = {uid_: link for uid_, link in self.memory.links.items() if uid_ not in old}
            self.memory._commit(dict(self.memory.symbols), {name: dict(items) for name, items in self.memory.sections.items()}, links, dict(self.memory.templates))
        changed = bool(old)
        declarations = _named_scope_declarations(text)
        regions = {(start, end) for _, start, end in declarations}
        for start, end in regions:
            stated = [candidate for candidate, left, right in declarations if (left, right) == (start, end)]
            if len(stated) != 1: continue  # Competing scope declarations make the section ambiguous.
            assertion = stated[0]
            scope_nodes = [node for node in self.memory.find_hypernodes("tpl_in_scope")
                           if any(ev.document_uid == document_uid and ev.content_hash == hashlib.sha256(text.encode()).hexdigest()
                                  and ev.start_offset == assertion.source_start
                                  and ev.end_offset == assertion.source_end for ev in node.evidence)]
            if len(scope_nodes) != 1: continue
            scope = scope_nodes[0]
            for section in ("H",):
                for element in self.memory.sections[section].values():
                    if not isinstance(element.payload, Hypernode): continue
                    event = element.payload
                    ev = event.evidence[0] if event.evidence else None
                    if not ev or ev.document_uid != document_uid or ev.content_hash != hashlib.sha256(text.encode()).hexdigest() or not assertion.source_end < ev.start_offset < end: continue
                    if event.template_ref.target_uid not in {"tpl_has_state", "tpl_action", "tpl_observed"}: continue
                    if not any(binding.role_id == "TIME" for binding in event.role_bindings): continue
                    if re.search(r"[«»\"]|\b(?:ранее|прежде|историческ\w*|стар\w*|прежн\w*|прошл\w*)\b", ev.exact_text, re.I): continue
                    if text[:ev.start_offset].count("«") > text[:ev.start_offset].count("»"): continue
                    scope_value = next(self.memory.label(binding.target_ref.target_uid) for binding in scope.role_bindings if binding.role_id == "OBJECT")
                    scope_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", norm(scope_value)) if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
                    event_nouns = {_lemma(word) for word in re.findall(r"[а-яё]+", norm(ev.exact_text)) if _morph_analyzer().parse(word)[0].tag.POS == "NOUN"}
                    scope_ids = {word.casefold().replace("-", "") for word in re.findall(r"[\w-]+", scope_value) if any(c.isalpha() for c in word) and any(c.isdigit() for c in word)}
                    event_ids = {word.casefold().replace("-", "") for word in re.findall(r"[\w-]+", ev.exact_text) if any(c.isalpha() for c in word) and any(c.isdigit() for c in word)}
                    if scope_nouns & event_nouns and event_ids - scope_ids: continue
                    self.memory.add_link(AssociativeLink(
                        uid=uid("l_scope"), type_id="IN_SCOPE", weight=1,
                        source_ref=ElementReference(reference_uid=uid("er"), target_uid=event.uid),
                        target_ref=ElementReference(reference_uid=uid("er"), target_uid=scope.uid),
                    ))
                    changed = True
        return changed

    def _compile_one(self, candidate: CandidateFact, text: str, document_uid: str, parser_run_uid: str) -> str:
        self.ensure_templates()
        c = candidate
        if c.source_end > len(text) or text[c.source_start:c.source_end] != c.exact_text:
            raise ValueError("source_span_mismatch|source span does not match original text")
        roles = TEMPLATE_ROLES.get(c.predicate)
        if roles is None: raise ValueError(f"unknown_template|unknown template: {c.predicate}")
        if c.predicate == "IN_SCOPE" and not any(
            c.source_start == expected.source_start and c.source_end == expected.source_end
            and _candidate_signature(c) == _candidate_signature(expected)
            for expected, _, _ in _named_scope_declarations(text)
        ): raise ValueError("unsupported_scope|scope assertion must match a named source section")
        target_section = c.section_hint or "C"
        signature = _candidate_signature(c, include_section=True)
        for existing in self.memory.find_hypernodes(f"tpl_{c.predicate.lower()}"):
            existing_section = next(section for section, elements in self.memory.sections.items() if existing.uid in elements)
            existing_signature = (existing_section, c.predicate, tuple(sorted((binding.role_id, norm(self._reference_label(binding.target_ref))) for binding in existing.role_bindings)))
            if signature == existing_signature: raise ValueError("duplicate_memory_fact|the canonical fact already exists in AH Memory")
        bindings = []
        mentions = {mention.mention_id: mention for mention in c.mentions}
        for binding in c.bindings:
            if binding.role_id not in roles: raise ValueError(f"role_not_allowed|role {binding.role_id} is not allowed for {c.predicate}")
            target_ref = self._term_reference(binding.term, mentions) if binding.term else SReference(reference_uid=uid("sr"), target_uid=self._grounded_symbol(binding.value or ""))
            bindings.append(RoleBinding(role_id=binding.role_id, target_ref=target_ref))
        ev = SourceEvidence(
            document_uid=document_uid, chunk_uid=c.span_uid or f"chunk_{document_uid}",
            start_offset=c.source_start, end_offset=c.source_end, exact_text=c.exact_text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(), parser_run_uid=parser_run_uid,
            model_id=c.model_id, parser_confidence=c.confidence,
        )
        evidence = [ev]
        for binding in c.bindings:
            for mention_id in binding.term.mention_ids if binding.term else ():
                visited = set()
                while mention_id:
                    if mention_id in visited or mention_id not in mentions: raise ValueError("invalid_coreference|missing or cyclic mention")
                    visited.add(mention_id)
                    mention = mentions[mention_id]
                    if mention.source_end > len(text) or text[mention.source_start:mention.source_end] != mention.observed_text: raise ValueError("mention_span_mismatch|coreference is not source anchored")
                    if mention.coref_to:
                        target = mentions.get(mention.coref_to)
                        if target is None or target.source_end > mention.source_start: raise ValueError("invalid_coreference|antecedent must precede reference")
                        support = _span_for_range(segment_text(text), target.source_start, target.source_end)
                        if support and not any(item.start_offset == support.start and item.end_offset == support.end for item in evidence):
                            evidence.append(ev.model_copy(update={"chunk_uid": support.uid, "start_offset": support.start, "end_offset": support.end, "exact_text": support.text}))
                    mention_id = mention.coref_to
        activation_weight = float(os.getenv("AH_INGESTION_INITIAL_WEIGHT", "1.0"))
        if not 0 <= activation_weight <= 1: raise ValueError("invalid_activation_weight|AH_INGESTION_INITIAL_WEIGHT must be in [0,1]")
        template_uid = f"tpl_{c.predicate.lower()}"
        hypernode = Hypernode(
            uid=uid("h"), weight=activation_weight,
            template_ref=ElementReference(reference_uid=uid("er"), target_uid=template_uid),
            role_bindings=tuple(bindings), evidence=tuple(evidence), created_tick=self.memory.current_tick,
            origin="ingestion",
        )
        self.memory.add_element(target_section, MemoryElement(uid=hypernode.uid, payload=hypernode))
        # Retrieval associations activate a stored fact; they do not infer its truth from an actant.
        actants = {binding.target_ref.target_uid: binding.target_ref for binding in bindings}
        for reference in actants.values():
            self.memory.add_link(AssociativeLink(uid=uid("l"), type_id="RECALLS", weight=activation_weight / len(actants), source_ref=reference, target_ref=ElementReference(reference_uid=uid("er"), target_uid=hypernode.uid)))
        by_role = {binding.role_id: binding.target_ref for binding in bindings}
        if c.predicate in {"CAUSE", "FOLLOW", "IS-A"}:
            self.memory.add_link(AssociativeLink(uid=uid("l"), type_id=c.predicate, weight=activation_weight, source_ref=by_role["SUBJECT"], target_ref=by_role["OBJECT"]))
        if target_section == "H": self._attach_history_fact(hypernode.uid, document_uid)
        return hypernode.uid

    @memory_locked
    def compile(self, candidates: list[CandidateFact], text: str, document_uid: str, parser_run_uid: str | None = None) -> tuple[list[str], list[dict]]:
        accepted, rejected = [], []
        parser_run_uid = parser_run_uid or uid("run")
        working = AHMemory.from_export(self.memory.export())
        for candidate in candidates:
            staged = AHMemory.from_export(working.export())
            try:
                accepted_uid = Compiler(staged)._compile_one(candidate, text, document_uid, parser_run_uid)
                staged.validate(); working = staged; accepted.append(accepted_uid)
            except Exception as exc:
                code, _, message = str(exc).partition("|")
                rejected.append({"candidate": _candidate_log(candidate), "reason": code, "message": message or code})
        scoped_changed = Compiler(working)._sync_scope_links(text, document_uid)
        if accepted or scoped_changed:
            working.validate()
            self.memory._commit(dict(working.symbols), {section: dict(values) for section, values in working.sections.items()}, dict(working.links), dict(working.templates))
        return accepted, rejected


def ingest(memory: AHMemory, text: str, document_uid: str | None = None, provider: Provider | None = None) -> IngestionResult:
    document_uid = document_uid or f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    candidates, meta = extract_candidates(text, provider)
    accepted, rejected = Compiler(memory).compile(candidates, text, document_uid)
    return IngestionResult(uid("ing"), document_uid, accepted, list(meta.get("rejection_log", [])) + rejected, candidates)
