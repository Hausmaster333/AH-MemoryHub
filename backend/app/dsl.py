from __future__ import annotations

import re
import json
from dataclasses import dataclass
from .core import AHMemory, memory_locked
from .models import AssociativeLink, SReference, ElementReference, MemoryElement, SecondOrderSymbol, MemoryList, FirstOrderSymbol, Property, uid


class DSLParseError(ValueError): pass


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, str]


def _split_top(text: str, sep: str) -> list[str]:
    out, start, stack, quote, escaped = [], 0, [], None, False
    for i, ch in enumerate(text):
        if quote is not None:
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == quote: quote = None
        elif ch in "\"'": quote = ch
        elif ch in "([{": stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[ch]:
                raise DSLParseError("mismatched delimiter")
        elif ch == sep and not stack: out.append(text[start:i].strip()); start = i + 1
    if quote is not None: raise DSLParseError("unterminated quoted value")
    if stack: raise DSLParseError("unclosed delimiter")
    out.append(text[start:].strip())
    return out


def parse_call(text: str) -> Call:
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*", text)
    if not m: raise DSLParseError("expected function call")
    args: dict[str, str] = {}
    body = m.group(2) or ""
    for item in _split_top(body, ",") if body else []:
        if "=" not in item: raise DSLParseError("DSL only accepts named arguments")
        k, v = item.split("=", 1); k = k.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k): raise DSLParseError("invalid argument name")
        if k in args: raise DSLParseError("duplicate argument name")
        args[k] = v.strip().strip("\"'")
    return Call(m.group(1), args)


def parse(text: str) -> list[Call]:
    if len(text) > 4000: raise DSLParseError("DSL expression too long")
    calls = []
    for part in _split_top(text, "|"):
        if part.startswith("intersect(") and part.endswith(")"):
            inner = part[len("intersect("):-1]
            calls.append(Call("intersect", {"expr": inner}))
        else: calls.append(parse_call(part))
    return calls


class Interpreter:
    READS = {
        "getAbstractSymbol", "findAbstractSymbols", "getSReference", "findSReferences",
        "getMReference", "findMReferences", "getSymbol", "findSymbols", "getList",
        "findLists", "getTemplate", "getHypernode", "findHypernodes", "findRoles",
        "getLink", "findLinks", "follow",
    }

    def __init__(self, memory: AHMemory): self.memory = memory

    @memory_locked
    def query(self, expression: str):
        values = self._query(expression)
        return [x.model_dump(mode="json") if hasattr(x, "model_dump") else x for x in values]

    def _query(self, expression: str):
        result = None
        for call in parse(expression):
            if call.name == "getAbstractSymbol": result = self.memory.get_abstract_symbol(call.args.get("uid", ""))
            elif call.name == "findAbstractSymbols": result = self.memory.find_abstract_symbols(call.args.get("value", ""))
            elif call.name == "getSReference": result = self.memory.get_s_reference(call.args.get("uid", ""))
            elif call.name == "findSReferences": result = self.memory.find_s_references(call.args.get("target", call.args.get("uid", "")))
            elif call.name == "getMReference": result = self.memory.get_m_reference(call.args.get("uid", ""))
            elif call.name == "findMReferences": result = self.memory.find_m_references(call.args.get("target", call.args.get("uid", "")))
            elif call.name == "getSymbol": result = self.memory.get_symbol(call.args.get("uid", ""))
            elif call.name == "findRoles": result = self.memory.find_roles(call.args.get("role", ""), call.args.get("value", ""))
            elif call.name == "findLists": result = self.memory.find_lists(call.args.get("element"), call.args.get("type"))
            elif call.name == "getList": result = self.memory.get_list(call.args.get("uid", ""))
            elif call.name == "findHypernodes": result = self.memory.find_hypernodes(call.args.get("query"))
            elif call.name == "getHypernode": result = self.memory.get_hypernode(call.args.get("uid", ""))
            elif call.name == "findLinks": result = self.memory.find_links(call.args.get("uid", ""))
            elif call.name == "getLink": result = self.memory.get_link(call.args.get("uid", ""))
            elif call.name == "findSymbols": result = self.memory.find_symbols(call.args.get("value", ""))
            elif call.name == "getTemplate": result = self.memory.get_template(call.args.get("uid", ""))
            elif call.name == "intersect":
                nested = self._query(call.args["expr"])
                left = result if isinstance(result, (list, tuple, set)) else ([] if result is None else [result])
                right_ids = {x if isinstance(x, str) else x.uid for x in nested}
                right_ids.update(ref.target_uid for x in nested if isinstance(x, MemoryList) for ref in x.ordered_members)
                result = [x for x in left if (x if isinstance(x, str) else x.uid) in right_ids]
            elif call.name == "follow":
                depth = min(int(call.args.get("depth", "1")), 20)
                current = {x if isinstance(x, str) else x.uid for x in (result or [])}; seen = set(current)
                for _ in range(depth):
                    nxt = {l.target_ref.target_uid for l in self.memory.links.values() if l.type_id == "FOLLOW" and l.source_ref.target_uid in current}
                    current = nxt - seen; seen |= nxt
                result = sorted(seen)
            else: raise DSLParseError(f"unknown read operation: {call.name}")
        values = result if isinstance(result, (list, tuple, set)) else ([] if result is None else [result])
        return list(values)

    @memory_locked
    def mutate(self, expression: str):
        calls = parse(expression)
        if len(calls) != 1: raise DSLParseError("one mutation per expression")
        a = calls[0].args
        name = calls[0].name
        if name in {"addAbstractSymbol", "editAbstractSymbol", "addElement", "editElement", "addProperty", "editProperty", "addLink"} and "data" in a:
            try:
                data = json.loads(a["data"])
            except json.JSONDecodeError as exc:
                raise DSLParseError("data must be valid JSON") from exc
            if name in {"addAbstractSymbol", "editAbstractSymbol"}:
                symbol = FirstOrderSymbol.model_validate(data)
                if name == "addAbstractSymbol": result = self.memory.add_symbol(symbol)
                else:
                    if a.get("uid", symbol.uid) != symbol.uid: raise DSLParseError("identity is immutable")
                    result = self.memory.edit_symbol(symbol.uid, symbol.sensory_representations)
            elif name in {"addElement", "editElement"}:
                element = MemoryElement.model_validate(data)
                result = self.memory.add_element(a.get("section", "P"), element) if name == "addElement" else self.memory.edit_element(a.get("uid", element.uid), element)
            elif name in {"addProperty", "editProperty"}:
                prop = Property.model_validate(data)
                if a.get("meta", "false") not in {"true", "false"}: raise DSLParseError("meta must be true or false")
                meta = a.get("meta", "false") == "true"
                result = self.memory.add_property(a.get("uid", ""), prop, meta) if name == "addProperty" else self.memory.edit_property(a.get("uid", ""), a.get("name", prop.name), prop, meta)
            else:
                result = self.memory.add_link(AssociativeLink.model_validate(data))
            return result.model_dump(mode="json")
        if calls[0].name == "addElement":
            name = a.get("uid") or uid("m")
            return self.memory.add_element(a.get("section", "P"), MemoryElement(uid=name, payload=SecondOrderSymbol(uid=name))).model_dump(mode="json")
        if calls[0].name != "addLink": raise DSLParseError("allowed mutations: addElement, addLink")
        source, target = a.get("source"), a.get("target")
        if not source or not target: raise DSLParseError("addLink requires source and target")
        link = AssociativeLink(uid=uid("l"), type_id=a.get("type", "ASSOCIATES"), weight=float(a.get("weight", "1")), source_ref=(SReference if source in self.memory.symbols else ElementReference)(reference_uid=uid("ref"), target_uid=source), target_ref=(SReference if target in self.memory.symbols else ElementReference)(reference_uid=uid("ref"), target_uid=target))
        self.memory.add_link(link)
        return link.model_dump(mode="json")
