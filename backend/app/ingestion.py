from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import AHMemory, ROLE_IDS, norm
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
PROMPT_VERSION = "ingestion-v2.2"
_CANDIDATE_CACHE: dict[str, tuple[tuple[CandidateFact, ...], tuple[dict, ...]]] = {}
ROLE_CANONICALIZATION = {
    ("CAUSE", "RESULT"): "OBJECT",
    ("RUN", "AGENT"): "SUBJECT",
    ("RUN", "STATE"): "HOW-TO",
    ("HAS", "PART"): "OBJECT",
    ("IS-A", "VALUE"): "OBJECT",
    ("LOCATED_AT", "DESTINATION"): "LOCATION",
    ("LIVE", "DESTINATION"): "LOCATION",
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
            "facts": [{
                "predicate": "CAUSE",
                "bindings": [{"role_id": "SUBJECT", "value": "перегрев насоса"}, {"role_id": "OBJECT", "value": "остановка агрегата"}],
                "quote": "точная непрерывная подстрока исходного текста",
                "confidence": 0.9,
                "unresolved_entities": False,
            }]
        }
        output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
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
                                        "value": {"type": "string", "minLength": 1},
                                    },
                                    "required": ["role_id", "value"],
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
            "required": ["facts"],
        }
        system = (
            "Ты модуль восприятия AH-памяти. Извлекай только явно выраженные факты; не отвечай на текст и не добавляй знания. "
            f"Допустимые предикаты: {', '.join(TEMPLATE_ROLES)}. Допустимые роли: {', '.join(roles)}. "
            "Каждый facts[] — один атомарный факт. Заполняй обязательные роли по их смыслу, не объединяй два предиката в один факт. "
            "quote обязана быть точной непрерывной подстрокой DOCUMENT. Не создавай FOLLOW только из слов «после этого» или «затем»: "
            "FOLLOW допустим лишь когда в тексте явно названы оба события. Местоимение помечай unresolved_entities=true, если антецедент неоднозначен. "
            "Примеры: «Перегрев вызвал остановку» → CAUSE(SUBJECT=перегрев, OBJECT=остановка); "
            "«Насос находится в зале» → LOCATED_AT(SUBJECT=насос, LOCATION=зал); "
            "В CAUSE следствие всегда OBJECT, в RUN действующий объект всегда SUBJECT, в HAS часть всегда OBJECT. "
            "«После этого оператор применил ключ» → USES_TOOL, не FOLLOW. "
            f"Верни только JSON вида {json.dumps(example, ensure_ascii=False)}"
        )
        spans = segment_text(text)
        span_index = [{"index": span.index, "start": span.start, "end": span.end, "text": span.text} for span in spans]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"PROMPT_VERSION={PROMPT_VERSION}\nSPANS={json.dumps(span_index, ensure_ascii=False)}\nDOCUMENT:\n{text}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "ah_fact_candidates", "strict": True, "schema": output_schema}} if self.response_format == "json_schema" else {"type": "json_object"},
            "temperature": 0,
        }
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
            raw_facts = json.loads(content).get("facts", [])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM response does not contain valid structured facts JSON") from exc
        aliases = {"СУБЪЕКТ": "SUBJECT", "ОБЪЕКТ": "OBJECT", "ЛОКАЦИЯ": "LOCATION", "ВРЕМЯ": "TIME", "ПРИЧИНА": "CAUSE", "ИНСТРУМЕНТ": "TOOL", "INSTRUMENT": "TOOL", "РЕЗУЛЬТАТ": "RESULT"}
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
                    value = str(raw_binding.get("value", "")).strip()
                    if role not in TEMPLATE_ROLES[predicate]: raise ValueError(f"role {role} is not allowed for {predicate}")
                    bindings.append(CandidateBinding(role_id=role, value=value))
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
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.last_warnings.append(f"fact {index + 1} rejected: {exc}")
        return candidates


_ANAPHORA = {"он", "она", "оно", "они", "его", "её", "ее", "их", "это", "этот", "эта", "эти"}


def _candidate_signature(candidate: CandidateFact) -> tuple:
    return candidate.predicate, tuple(sorted((binding.role_id, norm(binding.value)) for binding in candidate.bindings))


def canonicalize_candidates(text: str, candidates: list[CandidateFact]) -> tuple[list[CandidateFact], list[dict]]:
    """Resolve safe local anaphora, validate semantics, and collapse exact fact duplicates."""
    spans = segment_text(text)
    accepted: list[CandidateFact] = []
    by_signature: dict[tuple, int] = {}
    rejected: list[dict] = []
    last_by_role: dict[str, str] = {}

    def reject(candidate: CandidateFact, code: str, message: str):
        rejected.append({"candidate": candidate.model_dump(mode="json"), "reason": code, "message": message})

    for raw in sorted(candidates, key=lambda item: (item.source_start, item.source_end, item.predicate)):
        try:
            if raw.source_end > len(text) or text[raw.source_start:raw.source_end] != raw.exact_text:
                raise ValueError("source_span_mismatch|source offsets do not reproduce exact_text")
            predicate = raw.predicate.upper().strip()
            roles = TEMPLATE_ROLES.get(predicate)
            if roles is None: raise ValueError(f"unknown_template|unknown template: {predicate}")
            span = _span_for_range(spans, raw.source_start, raw.source_end)
            normalized_bindings: list[CandidateBinding] = []
            seen_roles: set[str] = set()
            unresolved = False
            for binding in raw.bindings:
                observed_role = binding.role_id.upper().strip()
                role = ROLE_CANONICALIZATION.get((predicate, observed_role), observed_role)
                value = " ".join(binding.value.strip().split())
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
                normalized_bindings.append(CandidateBinding(role_id=role, value=value, observed=observed))
                seen_roles.add(role)
            missing = set(TEMPLATE_REQUIRED[predicate]) - seen_roles
            if missing: raise ValueError(f"required_role_missing|required roles missing: {', '.join(sorted(missing))}")
            if unresolved or (raw.unresolved_entities and not any(binding.observed for binding in normalized_bindings)):
                raise ValueError("unresolved_anaphora|candidate contains an unresolved entity reference")
            values = {binding.role_id: norm(binding.value) for binding in normalized_bindings}
            if predicate in {"CAUSE", "FOLLOW", "IS-A"} and values.get("SUBJECT") == values.get("OBJECT"):
                raise ValueError("self_relation|SUBJECT and OBJECT must be distinct")
            if predicate == "FOLLOW" and re.match(r"\s*(?:после этого|затем|далее|следом)\b", raw.exact_text, re.I):
                quote = norm(raw.exact_text)
                anchored = all(any(token in quote for token in norm(binding.value).split() if len(token) > 2) for binding in normalized_bindings if binding.role_id in {"SUBJECT", "OBJECT"})
                if not anchored: raise ValueError("implicit_follow|a temporal marker alone does not identify two explicit events")
            candidate = raw.model_copy(update={
                "predicate": predicate, "bindings": tuple(normalized_bindings),
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


def configured_provider() -> tuple[Provider, dict]:
    mode = os.getenv("AH_PARSER_PROVIDER", "auto").strip().lower()
    base_url = os.getenv("AH_LLM_BASE_URL", "").strip()
    model = os.getenv("AH_LLM_MODEL", "").strip()
    api_key = os.getenv("AH_LLM_API_KEY", "").strip()
    use_llm = mode in {"openai", "openai_compatible"} or (mode == "auto" and bool(model) and bool(base_url or api_key))
    if mode not in {"auto", "rule", "openai", "openai_compatible"}: raise ValueError(f"unknown AH_PARSER_PROVIDER: {mode}")
    if not use_llm:
        provider = RuleBasedProvider()
        return provider, {"configured": mode, "active": "rule", "model_id": provider.model_id, "fallback": False}
    provider = OpenAICompatibleProvider(base_url or "https://api.openai.com/v1", model, api_key, float(os.getenv("AH_LLM_TIMEOUT_SECONDS", "45")), os.getenv("AH_LLM_RESPONSE_FORMAT", "json_schema").strip().lower())
    return provider, {"configured": mode, "active": "openai_compatible", "model_id": provider.model_id, "fallback": False}


def extract_candidates(text: str, provider: Provider | None = None) -> tuple[list[CandidateFact], dict]:
    if provider is not None:
        raw = provider.extract(text)
        candidates, rejected = canonicalize_candidates(text, raw)
        return candidates, {"configured": "explicit", "active": provider.__class__.__name__, "model_id": getattr(provider, "model_id", provider.__class__.__name__), "fallback": False, "warnings": getattr(provider, "last_warnings", []), "rejection_log": rejected, "prompt_version": PROMPT_VERSION, "cached": False}
    selected, meta = configured_provider()
    cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}".encode()).hexdigest()
    cached = _CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached[0]), {**meta, "warnings": [], "rejection_log": list(cached[1]), "prompt_version": PROMPT_VERSION, "cached": True}
    try:
        raw = selected.extract(text)
    except ValueError as exc:
        if meta["configured"] != "auto": raise
        selected = RuleBasedProvider(); raw = selected.extract(text)
        meta.update({"active": "rule", "model_id": selected.model_id, "fallback": True, "warning": str(exc)})
        cache_key = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:{selected.model_id}:{PROMPT_VERSION}".encode()).hexdigest()
    candidates, rejected = canonicalize_candidates(text, raw)
    if len(_CANDIDATE_CACHE) >= 128: _CANDIDATE_CACHE.pop(next(iter(_CANDIDATE_CACHE)))
    _CANDIDATE_CACHE[cache_key] = (tuple(candidates), tuple(rejected))
    meta["warnings"] = getattr(selected, "last_warnings", [])
    meta.update({"rejection_log": rejected, "prompt_version": PROMPT_VERSION, "cached": False})
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
        sid = uid("s")
        self.memory.add_symbol(FirstOrderSymbol(uid=sid, sensory_representations=(SensoryRepresentation(modality="text", value=label), SensoryRepresentation(modality="normalized", value=norm(label)))))
        return sid

    def _grounded_symbol(self, label: str) -> str:
        """Return the grounded S actant while ensuring its addressable m, s*, and m* exist."""
        symbol_uid = self._symbol(label)
        concepts = self.memory.find_symbols(label)
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

    def _compile_one(self, candidate: CandidateFact, text: str, document_uid: str, parser_run_uid: str) -> str:
        self.ensure_templates()
        c = candidate
        if c.source_end > len(text) or text[c.source_start:c.source_end] != c.exact_text:
            raise ValueError("source_span_mismatch|source span does not match original text")
        roles = TEMPLATE_ROLES.get(c.predicate)
        if roles is None: raise ValueError(f"unknown_template|unknown template: {c.predicate}")
        signature = _candidate_signature(c)
        for existing in self.memory.find_hypernodes(f"tpl_{c.predicate.lower()}"):
            existing_signature = (c.predicate, tuple(sorted((binding.role_id, norm(self.memory.label(binding.target_ref.target_uid))) for binding in existing.role_bindings)))
            if signature == existing_signature: raise ValueError("duplicate_memory_fact|the canonical fact already exists in AH Memory")
        bindings = []
        for binding in c.bindings:
            if binding.role_id not in roles: raise ValueError(f"role_not_allowed|role {binding.role_id} is not allowed for {c.predicate}")
            bindings.append(RoleBinding(role_id=binding.role_id, target_ref=SReference(reference_uid=uid("sr"), target_uid=self._grounded_symbol(binding.value))))
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
                rejected.append({"candidate": candidate.model_dump(mode="json"), "reason": code, "message": message or code})
        if accepted:
            self.memory._commit(dict(working.symbols), {section: dict(values) for section, values in working.sections.items()}, dict(working.links), dict(working.templates))
        return accepted, rejected


def ingest(memory: AHMemory, text: str, document_uid: str | None = None, provider: Provider | None = None) -> IngestionResult:
    document_uid = document_uid or f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    candidates, meta = extract_candidates(text, provider)
    accepted, rejected = Compiler(memory).compile(candidates, text, document_uid)
    return IngestionResult(uid("ing"), document_uid, accepted, list(meta.get("rejection_log", [])) + rejected, candidates)
