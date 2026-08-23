from __future__ import annotations

import re
from dataclasses import dataclass
from .core import AHMemory
from .models import AssociativeLink, SReference, uid


class DSLParseError(ValueError): pass


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, str]


def _split_top(text: str, sep: str) -> list[str]:
    out, start, depth, quote = [], 0, 0, None
    for i, ch in enumerate(text):
        if ch in "\"'": quote = None if quote == ch else (ch if quote is None else quote)
        elif quote is None:
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            elif ch == sep and depth == 0: out.append(text[start:i].strip()); start = i + 1
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

    def query(self, expression: str):
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
                nested = self.query(call.args["expr"])
                left = result or []
                right_ids = {getattr(x, "uid", None) for x in nested}
                result = [x for x in left if getattr(x, "uid", None) in right_ids]
            elif call.name == "follow":
                depth = min(int(call.args.get("depth", "1")), 20)
                current = {getattr(x, "uid", None) for x in (result or [])}; seen = set(current)
                for _ in range(depth):
                    nxt = {l.target_ref.target_uid for l in self.memory.links.values() if l.type_id == "FOLLOW" and l.source_ref.target_uid in current}
                    current = nxt - seen; seen |= nxt
                result = sorted(seen)
            else: raise DSLParseError(f"unknown read operation: {call.name}")
        values = result if isinstance(result, (list, tuple, set)) else ([] if result is None else [result])
        return [x.model_dump(mode="json") if hasattr(x, "model_dump") else x for x in values]

    def mutate(self, expression: str):
        calls = parse(expression)
        if len(calls) != 1 or calls[0].name != "addLink": raise DSLParseError("allowed mutation: addLink(...) only")
        a = calls[0].args
        source, target = a.get("source"), a.get("target")
        if not source or not target: raise DSLParseError("addLink requires source and target")
        link = AssociativeLink(uid=uid("l"), type_id=a.get("type", "ASSOCIATES"), weight=float(a.get("weight", "1")), source_ref=SReference(reference_uid=uid("sr"), target_uid=source), target_ref=SReference(reference_uid=uid("sr"), target_uid=target))
        self.memory.add_link(link)
        return link.model_dump(mode="json")
