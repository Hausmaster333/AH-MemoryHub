from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

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
}


class Provider:
    model_id = "provider"

    def extract(self, text: str) -> list[CandidateFact]:
        raise NotImplementedError


class RuleBasedProvider(Provider):
    model_id = "rule-based-demo"

    def extract(self, text: str) -> list[CandidateFact]:
        candidates = []
        offset = 0
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
            if not sentence.strip(): continue
            start = text.find(sentence, offset)
            end = start + len(sentence)
            offset = end
            s = sentence.strip(" .")
            low = norm(s)
            predicate = "OBSERVED"
            if re.search(r"(?:вызвал\w*|прив\w*\s+к|из-за|потому что|обусловил\w*)", low): predicate = "CAUSE"
            elif any(x in low for x in ("использовал", "применил", "инструмент", "ключ")): predicate = "USES_TOOL"
            elif any(x in low for x in ("после", "затем", "далее", "следом", "сначала")): predicate = "FOLLOW"
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
            parts = re.split(r"\s+" + separator + r"\s+", s, maxsplit=1, flags=re.I)
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
            candidates.append(CandidateFact(predicate=predicate, bindings=tuple(bindings), source_start=max(0, start), source_end=max(0, end), exact_text=s, confidence=0.78, model_id=self.model_id))
        return candidates


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
                self.memory.add_template(ControlTemplate(uid=uid_, predicate_ref=name, ordered_roles=tuple(Role(role_id=r, required=False) for r in roles)))

    def _symbol(self, label: str) -> str:
        label = label.strip()
        existing = self.memory.find_abstract_symbols(label)
        if existing: return existing[0].uid
        sid = uid("s")
        self.memory.add_symbol(FirstOrderSymbol(uid=sid, sensory_representations=(SensoryRepresentation(modality="text", value=label), SensoryRepresentation(modality="normalized", value=norm(label)))))
        return sid

    def compile(self, candidates: list[CandidateFact], text: str, document_uid: str, parser_run_uid: str | None = None) -> tuple[list[str], list[dict]]:
        self.ensure_templates()
        accepted, rejected = [], []
        parser_run_uid = parser_run_uid or uid("run")
        for c in candidates:
            before = (self.memory.symbols.copy(), {s: vals.copy() for s, vals in self.memory.sections.items()}, self.memory.links.copy(), self.memory.templates.copy(), self.memory.revision)
            try:
                if c.source_end > len(text) or text[c.source_start:c.source_end].strip(" .") != c.exact_text.strip(" ."):
                    raise ValueError("source span does not match original text")
                roles = TEMPLATE_ROLES.get(c.predicate)
                if roles is None: raise ValueError(f"unknown template: {c.predicate}")
                bindings = []
                for b in c.bindings:
                    if b.role_id not in roles: raise ValueError(f"role {b.role_id} is not allowed for {c.predicate}")
                    bindings.append(RoleBinding(role_id=b.role_id, target_ref=SReference(reference_uid=uid("sr"), target_uid=self._symbol(b.value))))
                ev = SourceEvidence(document_uid=document_uid, chunk_uid=f"chunk_{document_uid}", start_offset=c.source_start, end_offset=c.source_end, exact_text=c.exact_text, content_hash=hashlib.sha256(text.encode()).hexdigest(), parser_run_uid=parser_run_uid, model_id=c.model_id, parser_confidence=c.confidence)
                h = Hypernode(uid=uid("h"), weight=max(0.01, c.confidence), template_ref=f"tpl_{c.predicate.lower()}", role_bindings=tuple(bindings), evidence=(ev,), created_tick=self.memory.current_tick, origin="ingestion")
                section = "H" if c.predicate in {"FOLLOW", "OBSERVED"} else "C"
                self.memory.add_element(section, MemoryElement(uid=h.uid, payload=h))
                accepted.append(h.uid)
                # Explicit semantic links are deterministic and remain ordinary directed links.
                by_role = {b.role_id: b.target_ref for b in bindings}
                if c.predicate in {"CAUSE", "FOLLOW", "IS-A"} and "SUBJECT" in by_role and "OBJECT" in by_role and by_role["SUBJECT"].target_uid != by_role["OBJECT"].target_uid:
                    self.memory.add_link(AssociativeLink(uid=uid("l"), type_id=c.predicate, weight=h.weight, source_ref=by_role["SUBJECT"], target_ref=by_role["OBJECT"]))
            except Exception as exc:
                self.memory.symbols, self.memory.sections, self.memory.links, self.memory.templates, self.memory.revision = before
                rejected.append({"candidate": c.model_dump(mode="json"), "reason": str(exc)})
        return accepted, rejected


def ingest(memory: AHMemory, text: str, document_uid: str | None = None, provider: Provider | None = None) -> IngestionResult:
    document_uid = document_uid or f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    candidates = (provider or RuleBasedProvider()).extract(text)
    accepted, rejected = Compiler(memory).compile(candidates, text, document_uid)
    return IngestionResult(uid("ing"), document_uid, accepted, rejected, candidates)
