"""Small, deterministic activation demonstrations for monograph pp. 48, 51–52.

This fixture demonstrates traversal of supplied abstractions, not learning a new class.
"""
from .core import AHMemory
from .engine import IgnitionEngine, evidence_trace
from .models import (
    AssociativeLink, ElementReference, FirstOrderSymbol, IgnitionConfig,
    MemoryElement, Property, SecondOrderSymbol, SensoryRepresentation, SReference,
)


def senior_activation_memory() -> AHMemory:
    memory = AHMemory()
    memory.add_symbol(FirstOrderSymbol(uid="s_pump", sensory_representations=(SensoryRepresentation(modality="text", value="Насос Н-17"),)))
    for section, key, label in (("C", "c_pump", "Класс: насос"), ("P", "p_n17", "Модель насоса Н-17")):
        memory.add_element(section, MemoryElement(uid=key, payload=SecondOrderSymbol(uid=key, properties=(Property(name="label", value=label),))))
    for key, source, target, kind in (
        ("l_abstract", "s_pump", "c_pump", "ABSTRACTS"),
        ("l_specialize", "c_pump", "p_n17", "SPECIALIZES"),
        ("l_evoke", "p_n17", "s_pump", "EVOKES"),
    ):
        def ref(name):
            return (SReference if name in memory.symbols else ElementReference)(reference_uid=key + "_" + name, target_uid=name)
        memory.add_link(AssociativeLink(uid=key, type_id=kind, weight=.8, source_ref=ref(source), target_ref=ref(target)))
    return memory


def senior_activation_report() -> dict:
    memory = senior_activation_memory()
    engine = IgnitionEngine()
    config = IgnitionConfig(max_ticks=12, rhythm_hz=0, decay_lambda=.08)
    loop = engine.run(memory.snapshot(), ["s_pump"], config, "directed_v1")
    reverse = engine.run(memory.snapshot(), ["p_n17"], config, "directed_v1")
    isolated = memory.snapshot()
    isolated.links = {}
    decay = engine.run(isolated, ["s_pump"], IgnitionConfig(max_ticks=30, rhythm_hz=0, decay_lambda=.2), "directed_v1")
    rhythmic = engine.run(isolated, [], IgnitionConfig(max_ticks=12, rhythm_hz=1, rhythm_amplitude=.3), "directed_v1")
    checks = {
        "S_to_C_to_P": {"s_pump", "c_pump", "p_n17"} <= set(loop.run.working_memory),
        "P_to_S_to_C": {"s_pump", "c_pump", "p_n17"} <= set(reverse.run.working_memory),
        "loop_without_rhythm": min(loop.element_excitation.values()) > .8,
        "decay_without_links": not decay.run.working_memory,
        "autonomous_rhythm": "s_pump" in rhythmic.run.working_memory,
    }
    return {
        "profile": "directed_v1", "source_pages": [48, 51, 52],
        "scope": "activation over explicit pre-existing abstractions; not inductive concept learning",
        "checks": checks, "passed": all(checks.values()),
        "config": config.model_dump(mode="json"),
        "memory": memory.export(), "trace": [event.model_dump(mode="json") for event in loop.run.ticks],
        "dependency_uids": evidence_trace(loop.run, ("p_n17",), ["s_pump"]),
        "final_excitation": loop.element_excitation,
        "decayed_excitation": decay.element_excitation,
    }
