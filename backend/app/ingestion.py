from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import AHMemory, ROLE_IDS, lexical_key, norm
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
}

TEMPLATE_REQUIRED = {
    "CAUSE": ("SUBJECT", "OBJECT"), "FOLLOW": ("SUBJECT", "OBJECT"),
    "IS-A": ("SUBJECT", "OBJECT"), "LOCATED_AT": ("SUBJECT", "LOCATION"),
    "USES_TOOL": ("SUBJECT", "TOOL"), "HAS_STATE": ("SUBJECT", "STATE"),
    "OBSERVED": ("SUBJECT", "OBJECT"), "RESULT": ("SUBJECT", "RESULT"),
    "HAS": ("SUBJECT", "OBJECT"), "RUN": ("SUBJECT", "HOW-TO"),
    "LIVE": ("SUBJECT", "LOCATION"),
}
PROMPT_VERSION = "perception-ir-v3.16"
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


class RuleBasedProvider(Provider):
    model_id = "rule-based-offline"

    def extract(self, text: str) -> list[CandidateFact]:
        candidates = []
        for span in segment_text(text):
            s = span.text.strip(" .!?")
            low = norm(s)
            predicate = "OBSERVED"
            explicit_follow = re.match(r"после того как\s+(.+?)[,;]\s*(.+)$", s, re.I)
            if re.search(r"(?:вызвал\w*|прив\w*\s+к|из-за|потому что|обусловил\w*)", low): predicate = "CAUSE"
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
        return candidates


class OpenAICompatibleProvider(Provider):
    """Structured text perception for any Chat Completions compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_seconds: float = 45.0, response_format: str = "json_schema"):
        if not base_url.strip(): raise ValueError("AH_LLM_BASE_URL is required")
        if not model.strip(): raise ValueError("AH_LLM_MODEL is required")
        if response_format not in {"json_schema", "json_object"}: raise ValueError("AH_LLM_RESPONSE_FORMAT must be json_schema or json_object")
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"): self.endpoint += "/chat/completions"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format
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
                {"id": "m1", "text": "насос", "type": "entity", "canonical_label": "насос", "coref_to": None},
                {"id": "m2", "text": "цехе A", "type": "location", "canonical_label": "цех A", "coref_to": None},
                {"id": "m3", "text": "цехе B", "type": "location", "canonical_label": "цех B", "coref_to": None},
            ],
            "facts": [{
                "predicate": "LOCATED_AT",
                "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["m1"]}}, {"role_id": "LOCATION", "term": {"operator": "OR", "mention_ids": ["m2", "m3"]}}],
                "quote": "точная непрерывная подстрока исходного текста",
                "confidence": 0.9,
                "unresolved_entities": False,
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
                            "text": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": ["entity", "event", "state", "time", "location", "property"]},
                            "canonical_label": {"type": ["string", "null"]},
                            "coref_to": {"type": ["string", "null"]},
                        },
                        "required": ["id", "text", "type", "canonical_label", "coref_to"],
                    },
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "predicate": {"type": "string", "enum": sorted(TEMPLATE_ROLES)},
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
                        },
                        "required": ["predicate", "bindings", "quote", "confidence", "unresolved_entities"],
                    },
                }
            },
            "required": ["mentions", "facts"],
        }
        system = (
            "Ты модуль восприятия AH-памяти. Верни Perception IR, не отвечай на документ и не добавляй внешние знания. "
            f"Допустимые предикаты: {', '.join(TEMPLATE_ROLES)}. Допустимые роли: {', '.join(roles)}. "
            "Сначала перечисли все упоминания: text — точная подстрока DOCUMENT, canonical_label — краткая нормальная форма, coref_to — id более раннего однозначного упоминания. "
            "Каждый facts[] содержит ровно один атомарный предикат. Не пропускай декларативные предложения: каждое должно дать хотя бы один факт либо unresolved_entities=true. "
            "Разложи владение и свойство на HAS(owner, part) и HAS_STATE(part, state). Для «X принадлежит Y» верни HAS(Y, X); "
            "действие и его причину — на отдельный факт действия и CAUSE. Несколько однородных свойств — отдельные HAS_STATE. "
            "AND/OR/VERY кодируй только через term.operator; не склеивай операнды строкой. quote — точная непрерывная подстрока DOCUMENT. "
            "FOLLOW допустим только при двух явно названных событиях; маркер «после этого» сам по себе не факт FOLLOW. "
            "Не разрешай неоднозначную анафору: поставь unresolved_entities=true. В CAUSE следствие — OBJECT, в RUN носитель функции — SUBJECT, в HAS часть — OBJECT. "
            "USES_TOOL допустим только когда SUBJECT применяет настоящий инструмент. Конструкция «X используется для Y» описывает функцию: верни RUN(X, Y), а не USES_TOOL. "
            "В FOLLOW SUBJECT — более позднее событие целиком, включая действие, OBJECT — предшествующее событие. Например: «После восстановления давления сигнал отключился» → FOLLOW(отключение сигнала, восстановление давления). "
            "Временную границу состояния не теряй: «остался остановлен до завершения проверки» → HAS_STATE с TIME=«до завершения проверки». "
            "Пример общего разложения: «У двигателя два датчика: красный и синий» → HAS(двигатель, датчик), HAS_STATE(датчик, красный), HAS_STATE(датчик, синий). "
            f"Верни только JSON вида {json.dumps(example, ensure_ascii=False)}"
        )
        spans = segment_text(text)
        span_index = [{"index": span.index, "start": span.start, "end": span.end, "text": span.text} for span in spans]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"PROMPT_VERSION={PROMPT_VERSION}\nSPANS={json.dumps(span_index, ensure_ascii=False)}\nDOCUMENT:\n{text}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "ah_fact_candidates", "strict": True, "schema": output_schema}} if self.response_format == "json_schema" else {"type": "json_object"},
            "temperature": 0,
            "max_tokens": int(os.getenv("AH_LLM_MAX_TOKENS", "4096")),
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

    def extract(self, text: str) -> list[CandidateFact]:
        self.last_warnings = []
        spans = segment_text(text)
        response = self._completion(text)
        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list): content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            content = str(content).strip()
            if content.startswith("```"): content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            decoded = json.loads(content)
            raw_facts = decoded.get("facts", [])
            raw_mentions = decoded.get("mentions", [])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM response does not contain valid structured facts JSON") from exc
        aliases = {"СУБЪЕКТ": "SUBJECT", "ОБЪЕКТ": "OBJECT", "ЛОКАЦИЯ": "LOCATION", "ВРЕМЯ": "TIME", "ПРИЧИНА": "CAUSE", "ИНСТРУМЕНТ": "TOOL", "INSTRUMENT": "TOOL", "РЕЗУЛЬТАТ": "RESULT"}
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
                start = text.find(observed, mention_cursor)
                if start < 0: start = text.find(observed)
                if start < 0: raise ValueError("mention is not an exact substring of the source")
                mention_cursor = start + len(observed)
                mention_type = str(item.get("type", "entity")).lower().strip()
                mentions.append(CandidateMention(
                    mention_id=str(item["id"]), observed_text=observed, source_start=start,
                    source_end=start + len(observed), mention_type=mention_type_aliases.get(mention_type, mention_type),
                    canonical_label=str(item["canonical_label"]).strip() if item.get("canonical_label") else None,
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
                quote = str(item["quote"]).strip()
                start = text.find(quote, cursor)
                if start < 0: start = text.find(quote)
                if start < 0: raise ValueError("quote is not an exact substring of the source")
                cursor = start + len(quote)
                bindings = []
                for raw_binding in item.get("bindings", []):
                    role = aliases.get(str(raw_binding.get("role_id", "")).upper().strip(), str(raw_binding.get("role_id", "")).upper().strip())
                    if "term" in raw_binding:
                        term = CandidateTerm.model_validate(raw_binding["term"])
                        if any(mention_id not in mention_ids for mention_id in term.mention_ids): raise ValueError("term references unknown mention")
                        bindings.append(CandidateBinding(role_id=role, term=term))
                    else:
                        bindings.append(CandidateBinding(role_id=role, value=str(raw_binding.get("value", "")).strip()))
                if not bindings: raise ValueError("fact has no role bindings")
                span = _span_for_range(spans, start, start + len(quote))
                candidates.append(CandidateFact(
                    predicate=predicate, bindings=tuple(bindings), source_start=start,
                    source_end=start + len(quote), exact_text=quote,
                    confidence=float(item.get("confidence", .5)),
                    unresolved_entities=bool(item.get("unresolved_entities", False)),
                    model_id=self.model_id, span_uid=span.uid if span else None,
                    sentence_index=span.index if span else 0,
                    context_before=span.context_before if span else "",
                    context_after=span.context_after if span else "",
                    mentions=tuple(mentions), group_uid=span.uid if span else None,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.last_warnings.append(f"fact {index + 1} rejected: {exc}")
        return candidates


_ANAPHORA = {"он", "она", "оно", "они", "его", "её", "ее", "их", "это", "этот", "эта", "эти"}


def _resolved_mention_label(mention_id: str, mentions: dict[str, CandidateMention], trail: tuple[str, ...] = ()) -> str:
    if mention_id in trail: raise ValueError("coreference_cycle|coreference graph must be acyclic")
    mention = mentions.get(mention_id)
    if mention is None: raise ValueError(f"unknown_mention|unknown mention: {mention_id}")
    if mention.coref_to:
        target = mentions.get(mention.coref_to)
        if target is None: raise ValueError(f"unknown_coreference|unknown coreference target: {mention.coref_to}")
        if target.source_start > mention.source_start: raise ValueError("forward_coreference|coreference must point to an earlier mention")
        return _resolved_mention_label(mention.coref_to, mentions, trail + (mention_id,))
    return (mention.canonical_label or mention.observed_text).strip()


def _term_label(term: CandidateTerm, mentions: dict[str, CandidateMention]) -> str:
    values = [_resolved_mention_label(mention_id, mentions) for mention_id in term.mention_ids]
    return values[0] if term.operator == "ATOM" else f"{term.operator}({', '.join(values)})"


def _candidate_signature(candidate: CandidateFact) -> tuple:
    return candidate.predicate, tuple(sorted((binding.role_id, norm(binding.value or "")) for binding in candidate.bindings))


def _candidate_log(candidate: CandidateFact) -> dict:
    return {
        "predicate": candidate.predicate,
        "bindings": [binding.model_dump(mode="json") for binding in candidate.bindings],
        "source_start": candidate.source_start, "source_end": candidate.source_end,
        "exact_text": candidate.exact_text, "confidence": candidate.confidence,
        "model_id": candidate.model_id, "group_uid": candidate.group_uid or candidate.span_uid,
    }


def _coverage_warnings(text: str, candidates: list[CandidateFact]) -> list[dict]:
    covered = {candidate.group_uid or candidate.span_uid for candidate in candidates}
    return [
        {"group_uid": span.uid, "source_start": span.start, "source_end": span.end, "exact_text": span.text,
         "reason": "uncovered_source_span", "message": "parser returned no reviewable fact for this source span"}
        for span in segment_text(text) if span.uid not in covered
    ]


def _explicit_follow_candidates(text: str) -> list[CandidateFact]:
    """Conservative Russian `После <event> <later event>` recovery when the LLM omits FOLLOW."""
    recovered = []
    for span in segment_text(text):
        body = re.sub(r"^После\s+", "", span.text.strip(" .!?"), flags=re.I)
        if body == span.text.strip(" .!?") or re.match(r"этого\b", body, re.I): continue
        words = list(re.finditer(r"[А-Яа-яЁёA-Za-z0-9-]+", body))
        split = None
        for verb_index, match in enumerate(words):
            if not re.search(r"(?:л|ла|ло|ли|лся|лась|лось|лись)$", match.group(0), re.I): continue
            verb = match.group(0).casefold()
            candidate_split = verb_index if verb.startswith("произош") else verb_index - (2 if verb_index >= 4 or (verb_index >= 2 and words[verb_index - 1].group(0)[:1].isupper()) else 1)
            if candidate_split > 0:
                split = candidate_split
                break
        if split is None: continue
        prior = body[:words[split].start()].strip(" ,")
        later = body[words[split].start():].strip(" ,")
        if not prior or not later: continue
        recovered.append(CandidateFact(
            predicate="FOLLOW", bindings=(CandidateBinding(role_id="SUBJECT", value=later), CandidateBinding(role_id="OBJECT", value=prior)),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.8,
            model_id="deterministic-follow", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))
    return recovered


def _coordinated_property_candidates(text: str) -> list[CandidateFact]:
    recovered = []
    for span in segment_text(text):
        match = re.match(r"(.+?)\s+имеет\s+(.+?)\s+и\s+находится\s+в\s+(.+?)[.]?$", span.text, re.I)
        if not match: continue
        subject, obj, state = (value.strip(" .") for value in match.groups())
        for predicate, role, value in (("HAS", "OBJECT", obj), ("HAS_STATE", "STATE", state)):
            recovered.append(CandidateFact(
                predicate=predicate, bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id=role, value=value)),
                source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.85,
                model_id="deterministic-coordination", span_uid=span.uid, sentence_index=span.index,
                context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
            ))
    return recovered


def _temporal_state_candidates(text: str) -> list[CandidateFact]:
    recovered = []
    for span in segment_text(text):
        match = re.match(r"(.+?)\s+оста\w*\s+(.+?)\s+(до\s+.+?)[.]?$", span.text, re.I)
        if not match: continue
        subject, state, boundary = (value.strip(" .") for value in match.groups())
        recovered.append(CandidateFact(
            predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="STATE", value=state), CandidateBinding(role_id="TIME", value=boundary)),
            source_start=span.start, source_end=span.end, exact_text=span.text, confidence=.9,
            model_id="deterministic-temporal-state", span_uid=span.uid, sentence_index=span.index,
            context_before=span.context_before, context_after=span.context_after, group_uid=span.uid,
        ))
    return recovered


def canonicalize_candidates(text: str, candidates: list[CandidateFact]) -> tuple[list[CandidateFact], list[dict]]:
    """Resolve safe local anaphora, validate semantics, and collapse exact fact duplicates."""
    spans = segment_text(text)
    accepted: list[CandidateFact] = []
    by_signature: dict[tuple, int] = {}
    rejected: list[dict] = []
    last_by_role: dict[str, str] = {}

    def reject(candidate: CandidateFact, code: str, message: str):
        rejected.append({"candidate": _candidate_log(candidate), "reason": code, "message": message})

    for raw in sorted(candidates, key=lambda item: (item.source_start, item.source_end, item.predicate)):
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
            used_for = re.search(r"\bиспользуется\s+для\s+(.+)$", span.text if span else raw.exact_text, re.I)
            if predicate in {"USES_TOOL", "RUN"} and used_for:
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                purpose = used_for.group(1).strip(" .")
                if not subject or not purpose: raise ValueError("invalid_function|used-for construction requires a subject and purpose")
                predicate, roles = "RUN", TEMPLATE_ROLES["RUN"]
                normalized_bindings = [subject, CandidateBinding(role_id="HOW-TO", value=purpose, observed=purpose)]
                seen_roles = {"SUBJECT", "HOW-TO"}
                if span:
                    canonical_start = span.start + used_for.start()
                    canonical_end = span.end
                    canonical_text = text[canonical_start:canonical_end]
            if predicate == "HAS":
                responsibility = re.search(r"\bза\s+(.+?)\s+отвечает\s+(.+?)(?:[.,;]|$)", raw.exact_text, re.I)
                if responsibility:
                    normalized_bindings = [CandidateBinding(role_id="SUBJECT", value=responsibility.group(2).strip()), CandidateBinding(role_id="OBJECT", value=responsibility.group(1).strip())]
                    seen_roles = {"SUBJECT", "OBJECT"}
                described_object = re.search(r"\bимеет\s+(.+?)(?:\s+и\s+(?:находится|является|оста[её]тся)\b|[.,;]|$)", raw.exact_text, re.I)
                obj = next((binding for binding in normalized_bindings if binding.role_id == "OBJECT"), None)
                if described_object and obj:
                    value = described_object.group(1).strip()
                    if norm(obj.value) in norm(value) and norm(obj.value) != norm(value):
                        normalized_bindings = [binding.model_copy(update={"value": value, "observed": value, "term": None}) if binding is obj else binding for binding in normalized_bindings]
            if predicate == "HAS_STATE" and re.search(r"\bимеет\b.+\bи\s+находится\b", raw.exact_text, re.I) and last_by_role.get("SUBJECT"):
                normalized_bindings = [binding.model_copy(update={"value": last_by_role["SUBJECT"], "term": None}) if binding.role_id == "SUBJECT" else binding for binding in normalized_bindings]
            if predicate == "HAS_STATE" and re.match(r"\s*(?:находится|оста[её]тся)\b", raw.exact_text, re.I) and last_by_role.get("SUBJECT"):
                normalized_bindings = [binding.model_copy(update={"value": last_by_role["SUBJECT"], "term": None}) if binding.role_id == "SUBJECT" else binding for binding in normalized_bindings]
            if predicate == "FOLLOW" and re.match(r"\s*после\b", raw.exact_text, re.I):
                def mention_start(binding: CandidateBinding) -> int | None:
                    starts = [mention_map[mention_id].source_start for mention_id in binding.term.mention_ids if mention_id in mention_map] if binding.term else []
                    return min(starts) if starts else None
                subject = next((binding for binding in normalized_bindings if binding.role_id == "SUBJECT"), None)
                obj = next((binding for binding in normalized_bindings if binding.role_id == "OBJECT"), None)
                subject_start, object_start = mention_start(subject) if subject else None, mention_start(obj) if obj else None
                if subject and obj and subject_start is not None and object_start is not None:
                    if subject_start < object_start:
                        subject, obj = obj.model_copy(update={"role_id": "SUBJECT"}), subject.model_copy(update={"role_id": "OBJECT"})
                        subject_start = object_start
                    later_event = text[subject_start:raw.source_end].strip(" .")
                    normalized_bindings = [binding for binding in normalized_bindings if binding.role_id not in {"SUBJECT", "OBJECT"}]
                    normalized_bindings.extend((subject.model_copy(update={"value": later_event, "observed": later_event, "term": None}), obj))
            if "TIME" in roles:
                time_match = re.search(r"\bдо\s+[^,.;]+(?=[.;]?$)", raw.exact_text, re.I)
                if time_match:
                    value = time_match.group(0).strip()
                    normalized_bindings = [binding for binding in normalized_bindings if binding.role_id != "TIME"]
                    normalized_bindings.append(CandidateBinding(role_id="TIME", value=value, observed=value))
                    seen_roles.add("TIME")
            missing = set(TEMPLATE_REQUIRED[predicate]) - seen_roles
            if missing: raise ValueError(f"required_role_missing|required roles missing: {', '.join(sorted(missing))}")
            if unresolved or (raw.unresolved_entities and not any(binding.observed for binding in normalized_bindings)):
                raise ValueError("unresolved_anaphora|candidate contains an unresolved entity reference")
            values = {binding.role_id: norm(binding.value) for binding in normalized_bindings}
            for binding in normalized_bindings:
                if binding.role_id != "STATE": continue
                state_tokens = [token for token in re.findall(r"\w+", norm(binding.value)) if len(token) > 3 and token not in {"состояние", "статус", "state", "status"}]
                quote_tokens = re.findall(r"\w+", norm(raw.exact_text))
                if state_tokens and not all(any(left == right or (len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]) for right in quote_tokens) for left in state_tokens):
                    raise ValueError("ungrounded_state|STATE is not fully supported by exact_text")
            if predicate in {"CAUSE", "FOLLOW", "IS-A"} and values.get("SUBJECT") == values.get("OBJECT"):
                raise ValueError("self_relation|SUBJECT and OBJECT must be distinct")
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
    provider = OpenAICompatibleProvider(base_url or "https://api.openai.com/v1", model, api_key, float(os.getenv("AH_LLM_TIMEOUT_SECONDS", "45")), response_format)
    return provider, {"configured": "ui_override" if model_override else mode, "active": "openai_compatible", "model_id": provider.model_id, "fallback": False}


def extract_candidates(text: str, provider: Provider | None = None, model_override: str | None = None) -> tuple[list[CandidateFact], dict]:
    if provider is not None:
        raw = provider.extract(text)
        candidates, rejected = canonicalize_candidates(text, raw)
        return candidates, {"configured": "explicit", "active": provider.__class__.__name__, "model_id": getattr(provider, "model_id", provider.__class__.__name__), "fallback": False, "warnings": getattr(provider, "last_warnings", []), "rejection_log": rejected, "coverage_warnings": _coverage_warnings(text, candidates), "prompt_version": PROMPT_VERSION, "cached": False}
    selected, meta = configured_provider(model_override) if model_override else configured_provider()
    cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}".encode()).hexdigest()
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
        cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}".encode()).hexdigest()
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
        represented = {(candidate.group_uid or candidate.span_uid, candidate.predicate, norm(next((binding.value for binding in candidate.bindings if binding.role_id == "SUBJECT"), ""))) for candidate in candidates}
        candidates.extend(candidate for candidate in recovered_properties if ((candidate.group_uid or candidate.span_uid), candidate.predicate, norm(next((binding.value for binding in candidate.bindings if binding.role_id == "SUBJECT"), ""))) not in represented)
        rejected.extend(property_rejected)
        recovered_temporal, temporal_rejected = canonicalize_candidates(text, _temporal_state_candidates(text))
        represented_temporal_groups = {candidate.group_uid or candidate.span_uid for candidate in candidates if candidate.predicate == "HAS_STATE" and any(binding.role_id == "TIME" for binding in candidate.bindings)}
        candidates.extend(candidate for candidate in recovered_temporal if (candidate.group_uid or candidate.span_uid) not in represented_temporal_groups)
        rejected.extend(temporal_rejected)
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

    def _compile_one(self, candidate: CandidateFact, text: str, document_uid: str, parser_run_uid: str) -> str:
        self.ensure_templates()
        c = candidate
        if c.source_end > len(text) or text[c.source_start:c.source_end] != c.exact_text:
            raise ValueError("source_span_mismatch|source span does not match original text")
        roles = TEMPLATE_ROLES.get(c.predicate)
        if roles is None: raise ValueError(f"unknown_template|unknown template: {c.predicate}")
        signature = _candidate_signature(c)
        for existing in self.memory.find_hypernodes(f"tpl_{c.predicate.lower()}"):
            existing_signature = (c.predicate, tuple(sorted((binding.role_id, norm(self._reference_label(binding.target_ref))) for binding in existing.role_bindings)))
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
        activation_weight = float(os.getenv("AH_INGESTION_INITIAL_WEIGHT", "1.0"))
        if not 0 <= activation_weight <= 1: raise ValueError("invalid_activation_weight|AH_INGESTION_INITIAL_WEIGHT must be in [0,1]")
        template_uid = f"tpl_{c.predicate.lower()}"
        hypernode = Hypernode(
            uid=uid("h"), weight=activation_weight,
            template_ref=ElementReference(reference_uid=uid("er"), target_uid=template_uid),
            role_bindings=tuple(bindings), evidence=(ev,), created_tick=self.memory.current_tick,
            origin="ingestion",
        )
        section = "H" if c.predicate in {"FOLLOW", "OBSERVED"} else "C"
        self.memory.add_element(section, MemoryElement(uid=hypernode.uid, payload=hypernode))
        by_role = {binding.role_id: binding.target_ref for binding in bindings}
        if c.predicate in {"CAUSE", "FOLLOW", "IS-A"}:
            self.memory.add_link(AssociativeLink(uid=uid("l"), type_id=c.predicate, weight=activation_weight, source_ref=by_role["SUBJECT"], target_ref=by_role["OBJECT"]))
        return hypernode.uid

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
        if accepted:
            self.memory._commit(dict(working.symbols), {section: dict(values) for section, values in working.sections.items()}, dict(working.links), dict(working.templates))
        return accepted, rejected


def ingest(memory: AHMemory, text: str, document_uid: str | None = None, provider: Provider | None = None) -> IngestionResult:
    document_uid = document_uid or f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    candidates, meta = extract_candidates(text, provider)
    accepted, rejected = Compiler(memory).compile(candidates, text, document_uid)
    return IngestionResult(uid("ing"), document_uid, accepted, list(meta.get("rejection_log", [])) + rejected, candidates)
