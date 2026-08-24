from __future__ import annotations

import math
from dataclasses import dataclass
from .core import AHMemory, MemorySnapshot
from .models import *

ENGINE_STEPS = ("associative_impulses", "hypernode_impulses", "sum_impulses", "activation", "integration", "threshold", "hebbian", "commit")

def _clip(v: float) -> float: return max(0.0, min(1.0, v))

@dataclass
class EngineResult:
    run: IgnitionRun
    element_excitation: dict[str, float]

def build_candidate_snapshot(memory: AHMemory, seed_uids: list[str], max_nodes: int = 256, max_hops: int = 12) -> tuple[MemorySnapshot, list[str]]:
    """Build a bounded relevance projection; only question-grounded items remain ignition seeds."""
    snap = memory.snapshot(); selected = set(seed_uids); frontier = set(seed_uids)
    hypernodes = sorted(memory.find_hypernodes(), key=lambda item: item.uid)
    links = sorted(snap.links.values(), key=lambda item: item.uid)
    for _ in range(max_hops):
        discovered = set()
        for hypernode in hypernodes:
            members = {hypernode.uid, *(binding.target_ref.target_uid for binding in hypernode.role_bindings)}
            if members & frontier: discovered.update(members)
        for link in links:
            endpoints = {link.source_ref.target_uid, link.target_ref.target_uid}
            if link.weight > 0 and endpoints & frontier: discovered.update(endpoints)
        nxt = [uid_ for uid_ in sorted(discovered - selected) if uid_ in snap.symbols or uid_ in snap.elements]
        if not nxt: break
        room = max_nodes - len(selected)
        if room <= 0: break
        frontier = set(nxt[:room]); selected.update(frontier)
    elements = {k: v for k, v in snap.elements.items() if k in selected or isinstance(v.payload, ControlTemplate)}
    symbols = {k: v for k, v in snap.symbols.items() if k in selected}
    for e in elements.values():
        if isinstance(e.payload, Hypernode):
            for b in e.payload.role_bindings:
                if b.target_ref.kind == "S": symbols[b.target_ref.target_uid] = snap.symbols[b.target_ref.target_uid]
    links = {k: v for k, v in snap.links.items() if v.source_ref.target_uid in selected and v.target_ref.target_uid in selected}
    candidate = MemorySnapshot(snap.revision, snap.current_tick, symbols, elements, links, snap.templates)
    initial = sorted(set(seed_uids) & selected)
    return candidate, initial

class IgnitionEngine:
    steps = ENGINE_STEPS

    def run(self, snapshot: MemorySnapshot, seed_uids: list[str], config: IgnitionConfig | None = None, profile: str = "integrated_v1", run_uid: str | None = None) -> EngineResult:
        cfg = config or IgnitionConfig()
        if profile not in {"literal_2026", "integrated_v1"}: raise ValueError("unknown ignition profile")
        ids = sorted(set(snapshot.symbols) | set(snapshot.elements)); initial = set(seed_uids) & set(ids)
        current = {x: (1.0 if x in initial else 0.0) for x in ids}; link_weights = {k: v.weight for k, v in snapshot.links.items()}; hyper_weights = {k: e.payload.weight for k, e in snapshot.elements.items() if isinstance(e.payload, Hypernode)}
        traces: list[TickTrace] = []; last_trace: dict[str, int] = {}; working: set[str] = set(); evidence: set[str] = set(); stable_ticks = 0; period = max(1, round(1.0 / cfg.rhythm_hz)); dt = 1.0 / cfg.rhythm_hz
        for tick in range(cfg.max_ticks):
            impulses: dict[str, list[tuple[str | None, str | None, float, float | None, str]]] = {x: [] for x in ids}
            for lid, link in sorted(snapshot.links.items()):
                out = current.get(link.source_ref.target_uid, 0.0)
                if out > cfg.epsilon: impulses.setdefault(link.target_ref.target_uid, []).append((link.source_ref.target_uid, lid, out * link_weights[lid], link_weights[lid], "associative"))
            for eid, element in sorted(snapshot.elements.items()):
                if not isinstance(element.payload, Hypernode): continue
                actants = sorted({b.target_ref.target_uid for b in element.payload.role_bindings})
                if profile == "integrated_v1":
                    for source in actants:
                        out = current.get(source, 0.0)
                        if out > cfg.epsilon: impulses.setdefault(eid, []).append((source, eid, out * hyper_weights[eid] / max(1, len(actants)), hyper_weights[eid], "role_to_hypernode"))
                out = current.get(eid, 0.0)
                if out <= cfg.epsilon: continue
                evidence.update(ev.parser_run_uid for ev in element.payload.evidence)
                for target in actants: impulses.setdefault(target, []).append((eid, eid, out * hyper_weights[eid], hyper_weights[eid], "hypernode"))
            if tick % period == 0:
                for target in sorted(initial): impulses.setdefault(target, []).append((None, None, min(1.0, cfg.epsilon * 10), None, "rhythm_pulse"))
            next_exc: dict[str, float] = {}
            for target in ids:
                prev = current[target]; items = sorted(impulses.get(target, []), key=lambda x: (x[1] or "", x[0] or "")); z = sum(x[2] for x in items); activation = _clip(prev + z); nxt = prev * math.exp(-cfg.decay_lambda * dt) if profile == "literal_2026" else activation * math.exp(-cfg.decay_lambda * dt); next_exc[target] = _clip(nxt)
                entered = target not in working and nxt >= cfg.working_memory_threshold; exited = target in working and nxt < cfg.working_memory_threshold
                if entered: working.add(target)
                if exited: working.discard(target)
                for source, edge_uid, value, weight, impulse_type in items:
                    trace = TickTrace(tick=tick, source_uid=source, target_uid=target, link_or_hypernode_uid=edge_uid, impulse_type=impulse_type, impulse_value=value, previous_excitation=prev, next_excitation=nxt, activation=activation, threshold_entry=entered, threshold_exit=exited, previous_weight=weight, next_weight=weight, parent_trace=last_trace.get(source) if source else None); traces.append(trace); last_trace[target] = len(traces) - 1
                if not items and (prev > cfg.epsilon or nxt > cfg.epsilon): traces.append(TickTrace(tick=tick, target_uid=target, impulse_type="decay", impulse_value=0.0, previous_excitation=prev, next_excitation=nxt, activation=activation, threshold_entry=entered, threshold_exit=exited, parent_trace=last_trace.get(target)))
            if cfg.hebbian_eta:
                for lid, link in sorted(snapshot.links.items()):
                    old = link_weights[lid]; new = _clip(old + cfg.hebbian_eta * current.get(link.source_ref.target_uid, 0.0) * current.get(link.target_ref.target_uid, 0.0)); link_weights[lid] = new
                    if new != old: traces.append(TickTrace(tick=tick, source_uid=link.source_ref.target_uid, target_uid=link.target_ref.target_uid, link_or_hypernode_uid=lid, impulse_type="hebbian_link", impulse_value=new-old, previous_excitation=current.get(link.source_ref.target_uid, 0), next_excitation=next_exc.get(link.source_ref.target_uid, 0), activation=current.get(link.source_ref.target_uid, 0), previous_weight=old, next_weight=new, parent_trace=last_trace.get(link.source_ref.target_uid)))
                for eid in sorted(hyper_weights):
                    old = hyper_weights[eid]; new = _clip(old + cfg.hebbian_eta * current.get(eid, 0.0) ** 2); hyper_weights[eid] = new
                    if new != old: traces.append(TickTrace(tick=tick, target_uid=eid, link_or_hypernode_uid=eid, impulse_type="hebbian_hypernode", impulse_value=new-old, previous_excitation=current.get(eid, 0), next_excitation=next_exc.get(eid, 0), activation=current.get(eid, 0), previous_weight=old, next_weight=new, parent_trace=last_trace.get(eid)))
            change = max((abs(next_exc[k] - current[k]) for k in ids), default=0.0); stable_ticks = stable_ticks + 1 if change <= cfg.epsilon else 0; current = next_exc
            if stable_ticks >= 2 or (not any(v > cfg.epsilon for v in current.values()) and not working): break
        weight_deltas = {k: v - snapshot.links[k].weight for k, v in link_weights.items() if v != snapshot.links[k].weight}; weight_deltas.update({k: v - snapshot.elements[k].payload.weight for k, v in hyper_weights.items() if v != snapshot.elements[k].payload.weight})
        path = tuple(sorted(set(initial) | {x for t in traces if t.impulse_type in {"associative", "role_to_hypernode", "hypernode"} and t.impulse_value > cfg.epsilon for x in (t.source_uid, t.target_uid, t.link_or_hypernode_uid) if x}))
        run = IgnitionRun(run_uid=run_uid or uid("ign"), profile=profile, source_revision=snapshot.revision, ticks=tuple(traces), working_memory=tuple(sorted(working)), status="completed", evidence_uids=tuple(sorted(evidence)), effective_config=cfg, minimal_path=path, trace_complete=bool(traces and path), weight_deltas=weight_deltas)
        return EngineResult(run, current)

def gc_preview(memory: AHMemory, config: IgnitionConfig | None = None) -> dict:
    cfg = config or IgnitionConfig(); elements = memory.elements; connected = set(memory.symbols); changed = True
    while changed:
        changed = False
        for l in memory.links.values():
            if l.weight > 0 and l.source_ref.target_uid in connected and l.target_ref.target_uid not in connected: connected.add(l.target_ref.target_uid); changed = True
            if l.weight > 0 and l.target_ref.target_uid in connected and l.source_ref.target_uid not in connected: connected.add(l.source_ref.target_uid); changed = True
        for e in elements.values():
            if isinstance(e.payload, Hypernode) and (e.uid in connected or any(b.target_ref.target_uid in connected for b in e.payload.role_bindings)):
                connected.add(e.uid)
                for b in e.payload.role_bindings: connected.add(b.target_ref.target_uid)
    deletable = []
    for eid, e in elements.items():
        if eid in memory.templates: continue
        incident = [l for l in memory.links.values() if l.source_ref.target_uid == eid or l.target_ref.target_uid == eid]; hyper_incident = [h for h in elements.values() if isinstance(h.payload, Hypernode) and any(b.target_ref.target_uid == eid for b in h.payload.role_bindings)]; age = memory.current_tick - getattr(e.payload, "created_tick", 0)
        weights = [link.weight for link in incident] + [item.payload.weight for item in hyper_incident] + ([e.payload.weight] if isinstance(e.payload, Hypernode) else [])
        orphan = eid not in connected or bool(weights) and all(weight == 0 for weight in weights)
        if age >= cfg.initial_life_ticks and orphan: deletable.append(eid)
    token = uid("gc"); memory._gc_previews[token] = {"revision": memory.revision, "uids": tuple(sorted(deletable))}; return {"preview_token": token, "deletable_uids": sorted(deletable), "orphan_count_before": len(deletable), "grace_ticks": cfg.initial_life_ticks, "memory_revision": memory.revision}

def gc_commit(memory: AHMemory, token: str, expected_uids: list[str] | None = None) -> dict:
    preview = memory._gc_previews.get(token)
    if preview is None: raise ValueError("unknown or expired GC preview token")
    if preview["revision"] != memory.revision: raise ValueError("stale GC preview")
    ids = preview["uids"]
    if expected_uids is not None and tuple(sorted(expected_uids)) != ids: raise ValueError("GC token/set mismatch")
    delete = set(ids); sections = {s: {k: v for k, v in vals.items() if k not in delete} for s, vals in memory.sections.items()}; links = {k: v for k, v in memory.links.items() if v.source_ref.target_uid not in delete and v.target_ref.target_uid not in delete}; memory._commit(dict(memory.symbols), sections, links, dict(memory.templates)); del memory._gc_previews[token]; return {"deleted_uids": sorted(delete), "deleted_count": len(delete), "revision": memory.revision}
