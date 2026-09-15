from __future__ import annotations

from copy import deepcopy
from functools import wraps, lru_cache
from threading import RLock
from dataclasses import dataclass
from typing import Iterable
import hashlib
import json
import re

from .models import *


ROLE_IDS = (
    "SUBJECT", "OBJECT", "LOCATION", "TIME", "CAUSE", "TOOL", "RESULT", "AGENT",
    "PATIENT", "STATE", "SOURCE", "DESTINATION", "CONDITION", "VALUE", "PART",
    "HOW-TO", "AUXILLIARY", "RECIPIENT", "ABSENTEE", "DURATION", "PURPOSE", "MATERIAL", "AMOUNT",
)


def norm(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def lexical_key(text: str) -> tuple[str, ...]:
    """Conservative wordform key used for grounding, never for claim validation."""
    return tuple(token[:5] if len(token) > 5 and not any(char.isdigit() for char in token) else token for token in re.findall(r"[\w-]+", norm(text)))


@lru_cache(maxsize=1)
def _morph_analyzer():
    from pymorphy3 import MorphAnalyzer
    return MorphAnalyzer()


@lru_cache(maxsize=8192)
def _lemma(token: str) -> str:
    if len(token) == 1 or not re.fullmatch(r"[а-яё]+", token): return token
    parsed = _morph_analyzer().parse(token)
    return parsed[0].normal_form if parsed[0].is_known else token


def lexical_score(query: str, value: str) -> int:
    """Token overlap with conservative morphology and compact equipment identifiers."""
    def tokens(text: str) -> set[str]:
        canonical = re.sub(r"(?<=\w)-(?=\d)", "", norm(text))
        return {_lemma(token) if len(token) <= 4 else token for token in re.findall(r"\w+", canonical) if len(token) > 3 or any(char.isdigit() for char in token)}
    query_tokens, value_tokens = tokens(query), tokens(value)
    return sum(1 for left in query_tokens if any(left == right or (not any(char.isdigit() for char in left + right) and len(left) >= 5 and len(right) >= 5 and (left.startswith(right[:5]) or right.startswith(left[:5]))) for right in value_tokens))


class InvariantError(ValueError):
    pass


@dataclass
class MemorySnapshot:
    revision: int
    current_tick: int
    symbols: dict[str, FirstOrderSymbol]
    elements: dict[str, MemoryElement]
    links: dict[str, AssociativeLink]
    templates: dict[str, ControlTemplate]


def memory_locked(function):
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        memory = getattr(owner, "memory", owner)
        with memory._lock:
            return function(owner, *args, **kwargs)
    return wrapped


class AHMemory:
    """The sole mutation boundary for AH=<S,C,P,H,L>."""

    def __init__(self):
        # ponytail: one lock per aggregate; shard by agent if concurrent throughput becomes necessary.
        self._lock = RLock()
        self.created_ticks: dict[str, int] = {}
        self.symbols: dict[str, FirstOrderSymbol] = {}
        self.sections: dict[str, dict[str, MemoryElement]] = {"C": {}, "P": {}, "H": {}}
        self.links: dict[str, AssociativeLink] = {}
        self.templates: dict[str, ControlTemplate] = {}
        self.revision = 0
        self.current_tick = 0
        self._gc_previews: dict[str, dict] = {}

    @property
    @memory_locked
    def elements(self) -> dict[str, MemoryElement]:
        return {k: v for section in self.sections.values() for k, v in section.items()}

    @memory_locked
    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self.revision, self.current_tick, deepcopy(self.symbols), deepcopy(self.elements), deepcopy(self.links), deepcopy(self.templates))

    @memory_locked
    def advance_ticks(self, ticks: int = 1) -> int:
        if ticks < 0: raise ValueError("ticks must be non-negative")
        self.current_tick += ticks
        return self.current_tick

    @memory_locked
    def _commit(self, symbols, sections, links, templates):
        old = (self.symbols, self.sections, self.links, self.templates, self.revision)
        self.symbols, self.sections, self.links, self.templates = symbols, sections, links, templates
        try:
            self.validate()
        except Exception:
            self.symbols, self.sections, self.links, self.templates, self.revision = old
            raise
        live = set(self.symbols) | set(self.elements)
        self.created_ticks = {key: self.created_ticks.get(key, self.current_tick) for key in live}
        self.revision += 1

    @memory_locked
    def add_symbol(self, symbol: FirstOrderSymbol) -> FirstOrderSymbol:
        symbols, sections, links, templates = dict(self.symbols), {name: dict(values) for name, values in self.sections.items()}, dict(self.links), dict(self.templates)
        if symbol.uid in symbols or symbol.uid in self.elements or symbol.uid in links:
            raise InvariantError(f"duplicate uid: {symbol.uid}")
        symbols[symbol.uid] = symbol
        self._commit(symbols, sections, links, templates)
        return symbol

    @memory_locked
    def edit_symbol(self, uid_: str, sensory_representations: Iterable[SensoryRepresentation]) -> FirstOrderSymbol:
        if uid_ not in self.symbols:
            raise KeyError(uid_)
        symbol = FirstOrderSymbol(uid=uid_, sensory_representations=tuple(sensory_representations))
        symbols, sections, links, templates = dict(self.symbols), {name: dict(values) for name, values in self.sections.items()}, dict(self.links), dict(self.templates)
        symbols[uid_] = symbol
        self._commit(symbols, sections, links, templates)
        return symbol

    @memory_locked
    def add_template(self, template: ControlTemplate) -> ControlTemplate:
        if template.uid in self.templates or template.uid in self.elements:
            raise InvariantError(f"duplicate uid: {template.uid}")
        if len({r.role_id for r in template.ordered_roles}) != len(template.ordered_roles):
            raise InvariantError("duplicate template role")
        templates = dict(self.templates)
        templates[template.uid] = template
        # Templates are addressable memory elements in C while kept in a fast registry.
        sections = {name: dict(values) for name, values in self.sections.items()}
        sections["C"][template.uid] = MemoryElement(uid=template.uid, payload=template)
        self._commit(dict(self.symbols), sections, dict(self.links), templates)
        return template

    @memory_locked
    def add_element(self, section: str, element: MemoryElement) -> MemoryElement:
        if section not in self.sections:
            raise InvariantError("section must be C, P, or H")
        if element.uid in self.elements or element.uid in self.symbols or element.uid in self.links:
            raise InvariantError(f"duplicate uid: {element.uid}")
        sections = {name: dict(values) for name, values in self.sections.items()}
        sections[section][element.uid] = element
        templates = dict(self.templates)
        if isinstance(element.payload, ControlTemplate):
            if section != "C": raise InvariantError("templates belong in C")
            templates[element.uid] = element.payload
        self._commit(dict(self.symbols), sections, dict(self.links), templates)
        return element

    @memory_locked
    def edit_element(self, uid_: str, element: MemoryElement) -> MemoryElement:
        if uid_ != element.uid:
            raise InvariantError("identity is immutable")
        section = next((s for s, vals in self.sections.items() if uid_ in vals), None)
        if section is None:
            raise KeyError(uid_)
        sections = {name: dict(values) for name, values in self.sections.items()}
        sections[section][uid_] = element
        templates = dict(self.templates)
        if uid_ in templates:
            if not isinstance(element.payload, ControlTemplate): raise InvariantError("template kind is immutable")
            templates[uid_] = element.payload
        elif isinstance(element.payload, ControlTemplate):
            raise InvariantError("cannot change element into template")
        self._commit(dict(self.symbols), sections, dict(self.links), templates)
        return element

    @memory_locked
    def add_link(self, link: AssociativeLink) -> AssociativeLink:
        if link.uid in self.links or link.uid in self.elements or link.uid in self.symbols:
            raise InvariantError(f"duplicate uid: {link.uid}")
        links = dict(self.links)
        links[link.uid] = link
        self._commit(dict(self.symbols), {name: dict(values) for name, values in self.sections.items()}, links, dict(self.templates))
        return link

    @memory_locked
    def edit_link(self, uid_: str, link: AssociativeLink) -> AssociativeLink:
        if uid_ != link.uid or uid_ not in self.links:
            raise InvariantError("link identity is immutable")
        links = dict(self.links)
        links[uid_] = link
        self._commit(dict(self.symbols), {name: dict(values) for name, values in self.sections.items()}, links, dict(self.templates))
        return link

    @memory_locked
    def add_property(self, owner_uid: str, prop: Property, meta: bool = False):
        element = self.elements.get(owner_uid)
        if not element or not hasattr(element.payload, "properties"): raise KeyError(owner_uid)
        payload = element.payload
        field = "meta_properties" if meta else "properties"
        current = getattr(payload, field)
        if any(x.name == prop.name for x in current): raise InvariantError("duplicate property")
        return self.edit_element(owner_uid, element.model_copy(update={"payload": payload.model_copy(update={field: current + (prop,)})}))

    @memory_locked
    def edit_property(self, owner_uid: str, name: str, prop: Property, meta: bool = False):
        element = self.elements.get(owner_uid)
        if not element or not hasattr(element.payload, "properties"): raise KeyError(owner_uid)
        payload = element.payload; field = "meta_properties" if meta else "properties"; current = getattr(payload, field)
        if not any(x.name == name for x in current): raise KeyError(name)
        values = tuple(prop if x.name == name else x for x in current)
        return self.edit_element(owner_uid, element.model_copy(update={"payload": payload.model_copy(update={field: values})}))

    @memory_locked
    def apply_weight_deltas(self, deltas: dict[str, float]) -> None:
        """Apply one post-run Hebbian batch; snapshots are never mutated during ticks."""
        sections = {name: dict(values) for name, values in self.sections.items()}; links = dict(self.links)
        for uid_, delta in deltas.items():
            if uid_ in links:
                link = links[uid_]; links[uid_] = link.model_copy(update={"weight": max(0.0, min(1.0, link.weight + delta))})
            else:
                for section in sections.values():
                    element = section.get(uid_)
                    if element and isinstance(element.payload, Hypernode):
                        section[uid_] = element.model_copy(update={"payload": element.payload.model_copy(update={"weight": max(0.0, min(1.0, element.payload.weight + delta))})})
        self._commit(dict(self.symbols), sections, links, dict(self.templates))

    def _ref_exists(self, ref: Reference, elements: dict[str, MemoryElement] | None = None) -> bool:
        if ref.kind == "S":
            return ref.target_uid in self.symbols
        target = (self.elements if elements is None else elements).get(ref.target_uid)
        if ref.kind == "M":
            return target is not None and isinstance(target.payload, SecondOrderSymbol)
        return target is not None

    def _reference_records(self):
        """Yield every stored first-class reference, preserving its identity."""
        for element in self.elements.values():
            payload = element.payload
            if isinstance(payload, (SReference, MReference)):
                yield payload
            elif isinstance(payload, MemoryList):
                yield from payload.ordered_members
            elif isinstance(payload, FunctionalSymbol):
                yield from payload.ordered_operands
            elif isinstance(payload, ControlTemplate):
                yield payload.predicate_ref
            elif isinstance(payload, Hypernode):
                yield payload.template_ref
                yield from (binding.target_ref for binding in payload.role_bindings)
        for link in self.links.values():
            yield link.source_ref
            yield link.target_ref

    @memory_locked
    def validate(self) -> None:
        elements = self.elements
        all_uids = list(self.symbols) + list(elements) + list(self.links)
        if len(all_uids) != len(set(all_uids)):
            raise InvariantError("UIDs must be globally unique")
        if set(self.templates) - set(elements):
            raise InvariantError("template registry contains dangling template")
        references = list(self._reference_records())
        reference_index: dict[str, Reference] = {}
        for ref in references:
            prior = reference_index.get(ref.reference_uid)
            if prior is not None and (prior.kind != ref.kind or prior.target_uid != ref.target_uid):
                raise InvariantError(f"reference UID has conflicting identity: {ref.reference_uid}")
            reference_index[ref.reference_uid] = ref
        external_reference_uids = []
        for reference_uid, ref in reference_index.items():
            stored = elements.get(reference_uid)
            if stored is None:
                external_reference_uids.append(reference_uid)
            elif not isinstance(stored.payload, (SReference, MReference)) or stored.payload != ref:
                raise InvariantError(f"reference UID conflicts with addressable element: {reference_uid}")
        identity_uids = list(all_uids) + external_reference_uids
        if len(identity_uids) != len(set(identity_uids)):
            raise InvariantError("reference and addressable UIDs must be globally unique")
        for section, values in self.sections.items():
            if section not in {"C", "P", "H"}:
                raise InvariantError("invalid section")
            for e in values.values():
                p = e.payload
                if isinstance(p, (SReference, MReference)) and not self._ref_exists(p, elements):
                    raise InvariantError("dangling reference element")
                if isinstance(p, ControlTemplate) and p.uid not in self.templates:
                    raise InvariantError("unregistered template")
                if isinstance(p, ControlTemplate) and (self.templates[p.uid] != p or len({r.role_id for r in p.ordered_roles}) != len(p.ordered_roles)):
                    raise InvariantError("inconsistent template or duplicate role")
                if isinstance(p, ControlTemplate) and not self._ref_exists(p.predicate_ref, elements):
                    raise InvariantError("template predicate must reference S")
                if isinstance(p, (SecondOrderSymbol, MemoryList, Hypernode)):
                    props = p.properties + p.meta_properties
                    if len({x.name for x in props}) != len(props):
                        raise InvariantError("property names must be unique")
                if isinstance(p, MemoryList):
                    for ref in p.ordered_members:
                        if not self._ref_exists(ref, elements): raise InvariantError("dangling list reference")
                if isinstance(p, FunctionalSymbol):
                    for ref in p.ordered_operands:
                        if not self._ref_exists(ref, elements): raise InvariantError("dangling operand reference")
                if isinstance(p, Hypernode):
                    template_element = elements.get(p.template_ref.target_uid)
                    template = self.templates.get(p.template_ref.target_uid)
                    if not template_element or not isinstance(template_element.payload, ControlTemplate):
                        raise InvariantError("template reference must target ControlTemplate")
                    if not template: raise InvariantError("dangling template reference")
                    declared = {r.role_id: r for r in template.ordered_roles}
                    seen = set()
                    for b in p.role_bindings:
                        if b.role_id not in declared: raise InvariantError(f"role not declared: {b.role_id}")
                        if b.role_id in seen and not declared[b.role_id].multiplicity: raise InvariantError("duplicate role binding")
                        if not self._ref_exists(b.target_ref, elements): raise InvariantError("dangling role reference")
                        seen.add(b.role_id)
                    for role in template.ordered_roles:
                        if role.required and role.role_id not in seen: raise InvariantError(f"required role missing: {role.role_id}")
                    if p.origin == "ingestion" and not p.evidence: raise InvariantError("admitted hypernodes require source evidence")
        for link in self.links.values():
            if not self._ref_exists(link.source_ref, elements) or not self._ref_exists(link.target_ref, elements):
                raise InvariantError("dangling link reference")
        self._validate_dags("IS-A")
        self._validate_dags("FOLLOW")

    def _validate_dags(self, type_id: str):
        graph: dict[str, set[str]] = {}
        for link in self.links.values():
            if link.type_id == type_id and link.weight > 0:
                graph.setdefault(link.source_ref.target_uid, set()).add(link.target_ref.target_uid)
        visiting, done = set(), set()
        def visit(node: str):
            if node in visiting: raise InvariantError(f"{type_id} must be acyclic")
            if node in done: return
            visiting.add(node)
            for child in graph.get(node, ()): visit(child)
            visiting.remove(node); done.add(node)
        for node in graph: visit(node)

    # Normative retrieval operations
    @memory_locked
    def get_abstract_symbol(self, uid_: str): return self.symbols.get(uid_)
    @memory_locked
    def find_abstract_symbols(self, primary_symbol: str):
        q = norm(primary_symbol)
        return [s for s in self.symbols.values() if any(norm(r.value) == q for r in s.sensory_representations)]
    @memory_locked
    def reference_index(self) -> dict[str, Reference]: return {r.reference_uid: r for r in self._reference_records()}
    @memory_locked
    def get_s_reference(self, reference_uid: str):
        ref = self.reference_index().get(reference_uid)
        return ref if ref is not None and ref.kind == "S" else None
    @memory_locked
    def find_s_references(self, target_uid: str): return [r for r in self.reference_index().values() if r.kind == "S" and r.target_uid == target_uid]
    @memory_locked
    def get_m_reference(self, reference_uid: str):
        ref = self.reference_index().get(reference_uid)
        return ref if ref is not None and ref.kind == "M" else None
    @memory_locked
    def find_m_references(self, target_uid: str): return [r for r in self.reference_index().values() if r.kind == "M" and r.target_uid == target_uid]
    @memory_locked
    def get_symbol(self, uid_: str):
        element = self.elements.get(uid_)
        return element.payload if element and isinstance(element.payload, SecondOrderSymbol) else None
    @memory_locked
    def find_symbols(self, primary_symbol: str):
        q = norm(primary_symbol)
        return [e.payload for e in self.elements.values() if isinstance(e.payload, SecondOrderSymbol) and any(norm(str(p.value)) == q for p in e.payload.properties)]
    @memory_locked
    def get_list(self, uid_: str):
        element = self.elements.get(uid_)
        return element.payload if element and isinstance(element.payload, MemoryList) else None
    @memory_locked
    def find_lists(self, element_uid: str | None = None, list_type: str | None = None):
        return [e.payload for e in self.elements.values() if isinstance(e.payload, MemoryList) and (element_uid is None or any(r.target_uid == element_uid for r in e.payload.ordered_members)) and (list_type is None or e.payload.list_type == list_type)]
    @memory_locked
    def get_template(self, uid_: str): return self.templates.get(uid_)
    @memory_locked
    def get_hypernode(self, uid_: str):
        e = self.elements.get(uid_); return e.payload if e and isinstance(e.payload, Hypernode) else None
    @memory_locked
    def find_hypernodes(self, query: str | None = None):
        q = norm(query or "")
        out = []
        for e in self.elements.values():
            if isinstance(e.payload, Hypernode) and (not q or e.payload.template_ref.target_uid == query or any(r.target_uid == query or norm(self.label(r.target_uid)) == q for r in [b.target_ref for b in e.payload.role_bindings])):
                out.append(e.payload)
        return out
    @memory_locked
    def find_roles(self, role: str, value: str):
        q = norm(value); out = []
        for h in self.find_hypernodes():
            for b in h.role_bindings:
                if b.role_id == role and (b.target_ref.target_uid == value or norm(self.label(b.target_ref.target_uid)) == q): out.append(h); break
        return out
    @memory_locked
    def get_link(self, uid_: str): return self.links.get(uid_)
    @memory_locked
    def find_links(self, addressable_uid: str): return [l for l in self.links.values() if l.source_ref.target_uid == addressable_uid or l.target_ref.target_uid == addressable_uid]

    @memory_locked
    def label(self, target_uid: str) -> str:
        s = self.symbols.get(target_uid)
        if s: return s.sensory_representations[0].value
        e = self.elements.get(target_uid)
        if e and isinstance(e.payload, (SReference, MReference)):
            return self.label(e.payload.target_uid)
        if e and isinstance(e.payload, SecondOrderSymbol):
            for p in e.payload.properties:
                if p.name == "label": return str(p.value)
        if e and isinstance(e.payload, FunctionalSymbol):
            separator = " или " if e.payload.function_id == "OR" else " и " if e.payload.function_id == "AND" else ", "
            return separator.join(self.label(ref.target_uid) for ref in e.payload.ordered_operands)
        if e and isinstance(e.payload, MemoryList):
            return ", ".join(self.label(ref.target_uid) for ref in e.payload.ordered_members)
        return target_uid

    @memory_locked
    def stats(self):
        return {"symbols": len(self.symbols), "common": len(self.sections["C"]), "private": len(self.sections["P"]), "history": len(self.sections["H"]), "links": len(self.links), "templates": len(self.templates), "references": len(self.reference_index()), "revision": self.revision, "current_tick": self.current_tick}

    # Names from the normative operation table are kept as thin aliases.
    addAbstractSymbol = add_symbol
    editAbstractSymbol = edit_symbol
    addElement = add_element
    editElement = edit_element
    addLink = add_link
    editLink = edit_link
    addProperty = add_property
    editProperty = edit_property
    getAbstractSymbol = get_abstract_symbol
    findAbstractSymbols = find_abstract_symbols
    getSReference = get_s_reference
    findSReferences = find_s_references
    getMReference = get_m_reference
    findMReferences = find_m_references
    getSymbol = get_symbol
    findSymbols = find_symbols
    getList = get_list
    findLists = find_lists
    getTemplate = get_template
    getHypernode = get_hypernode
    findHypernodes = find_hypernodes
    findRoles = find_roles
    getLink = get_link
    findLinks = find_links

    @memory_locked
    def export(self) -> dict:
        data = {"schema": "AH-2026-executable-v1", "revision": self.revision, "current_tick": self.current_tick, "created_ticks": dict(self.created_ticks), "S": [x.model_dump(mode="json") for x in self.symbols.values()], "C": [x.model_dump(mode="json") for x in self.sections["C"].values()], "P": [x.model_dump(mode="json") for x in self.sections["P"].values()], "H": [x.model_dump(mode="json") for x in self.sections["H"].values()], "L": [x.model_dump(mode="json") for x in self.links.values()]}
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
        return {"dump": data, "sha256": hashlib.sha256(raw).hexdigest(), "source_manifest": [ev.model_dump(mode="json") for h in self.find_hypernodes() for ev in h.evidence]}

    @classmethod
    def from_export(cls, exported: dict) -> "AHMemory":
        data = exported.get("dump", exported)
        m = cls()
        m.symbols = {x["uid"]: FirstOrderSymbol.model_validate(x) for x in data.get("S", [])}
        for section in ("C", "P", "H"):
            m.sections[section] = {x["uid"]: MemoryElement.model_validate(x) for x in data.get(section, [])}
        m.templates = {uid_: e.payload for uid_, e in m.sections["C"].items() if isinstance(e.payload, ControlTemplate)}
        m.links = {x["uid"]: AssociativeLink.model_validate(x) for x in data.get("L", [])}
        m.revision = int(data.get("revision", 0))
        m.current_tick = int(data.get("current_tick", 0))
        live = set(m.symbols) | set(m.elements)
        m.created_ticks = {key: int(data.get("created_ticks", {}).get(key, m.current_tick)) for key in live}
        if any(tick < 0 or tick > m.current_tick for tick in m.created_ticks.values()):
            raise InvariantError("invalid creation tick")
        m.validate()
        return m
