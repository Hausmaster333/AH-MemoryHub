from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from .core import AHMemory, MemorySnapshot, memory_locked
from .models import *

ENGINE_STEPS = ("associative_impulses", "hypernode_impulses", "sum_impulses", "activation", "integration", "threshold", "hebbian", "commit")

def _clip(v: float) -> float: return max(0.0, min(1.0, v))

@dataclass
class EngineResult:
    run: IgnitionRun
    element_excitation: dict[str, float]

@memory_locked
def build_candidate_snapshot(memory: AHMemory, seed_uids: list[str], max_nodes: int = 256, max_hops: int = 12) -> tuple[MemorySnapshot, list[str]]:
    """Build a bounded relevance projection; only question-grounded items remain ignition seeds."""
    snap = memory.snapshot(); selected = set(seed_uids); frontier = set(seed_uids)
    hypernodes = sorted(memory.find_hypernodes(), key=lambda item: item.uid)
    links = sorted(snap.links.values(), key=lambda item: item.uid)
    @lru_cache(maxsize=None)
    def symbol_label(uid_: str) -> str:
        symbol = snap.symbols.get(uid_)
        if not symbol: return ""
        return "|".join(f"{rep.modality.casefold()}:{rep.value.casefold()}" for rep in sorted(symbol.sensory_representations, key=lambda rep: (rep.modality.casefold(), rep.value.casefold())))

    @lru_cache(maxsize=None)
    def label(uid_: str) -> str:
        direct = symbol_label(uid_)
        if direct: return direct
        element = snap.elements.get(uid_)
        if element and isinstance(element.payload, (SReference, MReference)):
            return symbol_label(element.payload.target_uid) or element.payload.kind
        if element and isinstance(element.payload, SecondOrderSymbol):
            for prop in element.payload.properties:
                if prop.name == "label": return str(prop.value).casefold()
        if element and isinstance(element.payload, FunctionalSymbol):
            return f"{element.payload.function_id}:" + ",".join(symbol_label(ref.target_uid) or ref.kind for ref in element.payload.ordered_operands)
        if element and isinstance(element.payload, MemoryList):
            return f"{element.payload.list_type}:" + ",".join(symbol_label(ref.target_uid) or ref.kind for ref in element.payload.ordered_members)
        return type(element.payload).__name__ if element else ""

    @lru_cache(maxsize=None)
    def content_key(uid_: str):
        element = snap.elements.get(uid_)
        if element and isinstance(element.payload, Hypernode):
            node = element.payload
            template = snap.templates.get(node.template_ref.target_uid)
            predicate = label(template.predicate_ref.target_uid) if template else ""
            roles = tuple(sorted((binding.role_id, label(binding.target_ref.target_uid)) for binding in node.role_bindings))
            return ("H", predicate, roles, uid_)
        return ("S" if uid_ in snap.symbols else "M", label(uid_), (), uid_)
    for _ in range(max_hops):
        discovered = set()
        for hypernode in hypernodes:
            members = {hypernode.uid, *(binding.target_ref.target_uid for binding in hypernode.role_bindings)}
            if members & frontier: discovered.update(members)
        for link in links:
            endpoints = {link.source_ref.target_uid, link.target_ref.target_uid}
            if link.weight > 0 and endpoints & frontier: discovered.update(endpoints)
        nxt = [uid_ for uid_ in sorted(discovered - selected, key=content_key) if uid_ in snap.symbols or uid_ in snap.elements]
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

def activation_output(name: str, z: float, x: float) -> float:
    if name == "clip_sum": return _clip(x + z)
    if name == "relu": return _clip(z)
    if name == "sigmoid": return 1 / (1 + math.exp(-max(-60, min(60, x + z))))
    raise ValueError(f"unknown activation function: {name}")


class IgnitionEngine:
    steps = ENGINE_STEPS

    def run(self, snapshot: MemorySnapshot, seed_uids: list[str], config: IgnitionConfig | None = None, profile: str = "integrated_v1", run_uid: str | None = None) -> EngineResult:
        cfg = config or IgnitionConfig()
        if profile not in {"literal_2026", "integrated_v1", "directed_v1"}: raise ValueError("unknown ignition profile")
        ids = sorted(set(snapshot.symbols) | set(snapshot.elements))
        initial = set(seed_uids) & set(ids)
        functions = {key: snapshot.elements[key].activation_function if key in snapshot.elements else "clip_sum" for key in ids}
        current = {key: max(float(key in initial), snapshot.elements[key].excitation if key in snapshot.elements else 0) for key in ids}
        outputs = {key: activation_output(functions[key], 0, current[key]) for key in ids}
        link_weights = {key: link.weight for key, link in snapshot.links.items()}
        explicit_pairs = {(link.source_ref.target_uid, link.target_ref.target_uid) for link in snapshot.links.values()}
        hypernodes = {key: element.payload for key, element in snapshot.elements.items() if isinstance(element.payload, Hypernode)}
        hyper_weights = {key: node.weight for key, node in hypernodes.items()}
        traces = []
        previous_updates = {}
        for key in ids:
            if outputs[key] > cfg.epsilon:
                previous_updates[key] = len(traces)
                traces.append(TickTrace(tick=-1, target_uid=key, impulse_type="seed" if key in initial else "initial_state", impulse_value=current[key], previous_excitation=0, next_excitation=current[key], activation=outputs[key]))
        working = {key for key in ids if current[key] > cfg.working_memory_threshold}
        elapsed = 0
        # A deterministic autonomous pacemaker; it does not create query evidence by itself.
        rhythm_targets = sorted(initial) or sorted(snapshot.symbols)[:1] or ids[:1]
        rhythm_phase = 1.0
        for tick in range(cfg.max_ticks):
            impulses = {key: [] for key in ids}
            for key, link in sorted(snapshot.links.items()):
                source, target = link.source_ref.target_uid, link.target_ref.target_uid
                value = outputs.get(source, 0) * link_weights[key]
                if value > cfg.epsilon and target in impulses:
                    impulses[target].append((source, key, value, link_weights[key], "associative"))
            for key, node in sorted(hypernodes.items()):
                actants = sorted({binding.target_ref.target_uid for binding in node.role_bindings})
                if profile == "integrated_v1":
                    # Explicit compatibility extension, not the monograph's N -> actant rule.
                    for source in actants:
                        if (source, key) in explicit_pairs: continue
                        value = outputs.get(source, 0) * hyper_weights[key] / max(1, len(actants))
                        if value > cfg.epsilon: impulses[key].append((source, key, value, hyper_weights[key], "role_to_hypernode"))
                value = outputs.get(key, 0) * hyper_weights[key]
                if value > cfg.epsilon:
                    for target in actants:
                        if target in impulses: impulses[target].append((key, key, value, hyper_weights[key], "hypernode"))
            if cfg.rhythm_hz and cfg.rhythm_amplitude:
                pulses = int(rhythm_phase + 1e-12)
                rhythm_phase -= pulses
                if pulses:
                    for target in rhythm_targets: impulses[target].append((None, None, pulses * cfg.rhythm_amplitude, None, "rhythm_pulse"))
                rhythm_phase += cfg.rhythm_hz * cfg.tick_seconds
            next_exc, next_output, next_updates = {}, {}, {}
            for target in ids:
                prev = current[target]
                items = impulses[target]
                z = sum(item[2] for item in items)
                output = activation_output(functions[target], z, prev)
                # Literal mode keeps g(x); integrated mode documents g(clip(x+z)).
                excitation = prev if profile == "literal_2026" else _clip(prev + z)
                nxt = _clip(excitation * math.exp(-cfg.decay_lambda * cfg.tick_seconds))
                next_exc[target], next_output[target] = nxt, output
                entered = target not in working and nxt > cfg.working_memory_threshold
                exited = target in working and nxt < cfg.working_memory_threshold
                if entered: working.add(target)
                if exited: working.discard(target)
                dependencies = []
                if prev > cfg.epsilon and target in previous_updates: dependencies.append(previous_updates[target])
                for source, edge, value, weight, kind in items:
                    parent = previous_updates.get(source)
                    dependencies.append(len(traces))
                    traces.append(TickTrace(tick=tick, source_uid=source, target_uid=target, link_or_hypernode_uid=edge, impulse_type=kind, impulse_value=value, previous_excitation=prev, next_excitation=nxt, activation=output, threshold_entry=entered, threshold_exit=exited, previous_weight=weight, next_weight=weight, parent_trace=parent, parent_traces=(parent,) if parent is not None else ()))
                if items or prev > cfg.epsilon or output > cfg.epsilon:
                    parents = tuple(dict.fromkeys(dependencies))
                    next_updates[target] = len(traces)
                    traces.append(TickTrace(tick=tick, target_uid=target, impulse_type="activation", impulse_value=z, previous_excitation=prev, next_excitation=nxt, activation=output, threshold_entry=entered, threshold_exit=exited, parent_trace=parents[0] if parents else None, parent_traces=parents))
            if cfg.hebbian_eta:
                for key, link in sorted(snapshot.links.items()):
                    old = link_weights[key]
                    source, target = link.source_ref.target_uid, link.target_ref.target_uid
                    new = _clip(old + cfg.hebbian_eta * current.get(source, 0) * current.get(target, 0))
                    link_weights[key] = new
                    if new != old: traces.append(TickTrace(tick=tick, source_uid=source, target_uid=target, link_or_hypernode_uid=key, impulse_type="hebbian_link", impulse_value=new-old, previous_excitation=current.get(target, 0), next_excitation=next_exc.get(target, 0), activation=next_output.get(target, 0), previous_weight=old, next_weight=new))
                for key, node in sorted(hypernodes.items()):
                    old = hyper_weights[key]
                    actants = {binding.target_ref.target_uid for binding in node.role_bindings}
                    average = sum(current.get(target, 0) for target in actants) / max(1, len(actants))
                    new = _clip(old + cfg.hebbian_eta * current[key] * average)
                    hyper_weights[key] = new
                    if new != old: traces.append(TickTrace(tick=tick, target_uid=key, link_or_hypernode_uid=key, impulse_type="hebbian_hypernode", impulse_value=new-old, previous_excitation=current[key], next_excitation=next_exc[key], activation=next_output[key], previous_weight=old, next_weight=new))
            current, outputs, previous_updates = next_exc, next_output, next_updates
            elapsed += 1
        deltas = {key: weight-snapshot.links[key].weight for key, weight in link_weights.items() if weight != snapshot.links[key].weight}
        deltas.update({key: weight-hypernodes[key].weight for key, weight in hyper_weights.items() if weight != hypernodes[key].weight})
        active = tuple(dict.fromkeys(value for event in traces if event.impulse_value > cfg.epsilon and event.impulse_type in {"seed", "associative", "hypernode", "role_to_hypernode"} for value in (event.source_uid, event.link_or_hypernode_uid, event.target_uid) if value))
        run = IgnitionRun(run_uid=run_uid or uid("ign"), profile=profile, source_revision=snapshot.revision, ticks=tuple(traces), working_memory=tuple(sorted(working)), status="completed", effective_config=cfg, minimal_path=active, activated_path=active, trace_complete=False, weight_deltas=deltas, elapsed_ticks=elapsed)
        return EngineResult(run, current)


def evidence_trace(run: IgnitionRun, goals: tuple[str, ...], seeds: list[str]) -> tuple[str, ...]:
    """Chronological dependency closure, not a shortest path or all activity in a run."""
    latest = {event.target_uid: index for index, event in enumerate(run.ticks) if event.impulse_type in {"seed", "initial_state", "activation"} and event.activation > run.effective_config.epsilon}
    if not goals or any(goal not in latest for goal in goals): return ()
    selected = set()
    for goal in goals:
        pending, branch, seed_found = [latest[goal]], set(), False
        while pending:
            index = pending.pop()
            if index in branch: continue
            branch.add(index)
            event = run.ticks[index]
            seed_found |= event.impulse_type == "seed" and event.target_uid in seeds
            for parent in event.parent_traces:
                if parent < 0 or parent >= index or run.ticks[parent].tick > event.tick: return ()
                pending.append(parent)
        if not seed_found: return ()
        selected.update(branch)
    return tuple(dict.fromkeys(value for index in sorted(selected) for value in (run.ticks[index].source_uid, run.ticks[index].link_or_hypernode_uid, run.ticks[index].target_uid) if value))


@memory_locked
def gc_preview(memory: AHMemory, config: IgnitionConfig | None = None) -> dict:
    cfg = config or IgnitionConfig()
    elements = memory.elements
    adjacency = {key: set() for key in set(memory.symbols) | set(elements)}
    weights = {key: [] for key in adjacency}
    def join(left, right):
        adjacency[left].add(right); adjacency[right].add(left)
    for link in memory.links.values():
        left, right = link.source_ref.target_uid, link.target_ref.target_uid
        weights[left].append(link.weight); weights[right].append(link.weight)
        if link.weight > 0: join(left, right)
    for key, element in elements.items():
        payload = element.payload
        if isinstance(payload, Hypernode):
            weights[key].append(payload.weight)
            for binding in payload.role_bindings:
                target = binding.target_ref.target_uid
                weights[target].append(payload.weight)
                if payload.weight > 0: join(key, target)
        elif isinstance(payload, (SReference, MReference)):
            join(key, payload.target_uid)
        elif isinstance(payload, MemoryList):
            for ref in payload.ordered_members: join(key, ref.target_uid)
        elif isinstance(payload, FunctionalSymbol):
            for ref in payload.ordered_operands: join(key, ref.target_uid)
        elif isinstance(payload, ControlTemplate):
            join(key, payload.predicate_ref.target_uid)
    connected = set(memory.symbols)
    pending = list(connected)
    while pending:
        for target in adjacency[pending.pop()] - connected:
            connected.add(target); pending.append(target)
    def dependencies(payload):
        if isinstance(payload, (SReference, MReference)): return (payload,)
        if isinstance(payload, FunctionalSymbol): return payload.ordered_operands
        if isinstance(payload, Hypernode): return tuple(binding.target_ref for binding in payload.role_bindings)
        return ()
    protected = {key for key in elements if memory.current_tick-memory.created_ticks.get(key, memory.current_tick) < cfg.initial_life_ticks}
    pending = list(protected)
    while pending:
        for ref in dependencies(elements[pending.pop()].payload):
            if ref.target_uid in elements and ref.target_uid not in protected:
                protected.add(ref.target_uid); pending.append(ref.target_uid)
    deletable = {key for key in elements if key not in memory.templates and key not in protected and (key not in connected or bool(weights[key]) and not any(weights[key]))}
    # Preserve referential integrity: lists lose dead members; other dependent values are indivisible.
    changed = True
    while changed:
        changed = False
        for key, element in elements.items():
            payload = element.payload
            refs = dependencies(payload)
            if key not in deletable and key not in protected and any(ref.target_uid in deletable for ref in refs):
                deletable.add(key)
                changed = True
    token = uid("gc")
    memory._gc_previews[token] = {"revision": memory.revision, "tick": memory.current_tick, "uids": tuple(sorted(deletable))}
    return {"preview_token": token, "deletable_uids": sorted(deletable), "orphan_count_before": len(deletable), "grace_ticks": cfg.initial_life_ticks, "memory_revision": memory.revision}


@memory_locked
def gc_commit(memory: AHMemory, token: str, expected_uids: list[str] | None = None) -> dict:
    preview = memory._gc_previews.get(token)
    if preview is None: raise ValueError("unknown or expired GC preview token")
    if preview["revision"] != memory.revision or preview["tick"] != memory.current_tick: raise ValueError("stale GC preview")
    ids = preview["uids"]
    if expected_uids is not None and tuple(sorted(expected_uids)) != ids: raise ValueError("GC token/set mismatch")
    delete = set(ids)
    sections = {section: {key: value for key, value in items.items() if key not in delete} for section, items in memory.sections.items()}
    for items in sections.values():
        for key, element in list(items.items()):
            if isinstance(element.payload, MemoryList):
                members = tuple(ref for ref in element.payload.ordered_members if ref.target_uid not in delete)
                if members != element.payload.ordered_members:
                    items[key] = element.model_copy(update={"payload": element.payload.model_copy(update={"ordered_members": members})})
    links = {key: link for key, link in memory.links.items() if link.source_ref.target_uid not in delete and link.target_ref.target_uid not in delete}
    if delete: memory._commit(dict(memory.symbols), sections, links, dict(memory.templates))
    del memory._gc_previews[token]
    return {"deleted_uids": sorted(delete), "deleted_count": len(delete), "revision": memory.revision}
