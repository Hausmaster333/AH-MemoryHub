from __future__ import annotations

import json

from .core import AHMemory
from .models import *


RABBIT_SOURCE = (
    "Заяц — маленький дикий зверёк, который обитает на лугу или в лесу. "
    "У него сильные задние лапы, поэтому бегает он очень быстро. "
    "Уши зайца длинные, а хвост — круглый и пушистый. "
    "Летом шерсть зайца коричневого цвета, а зимой — белого."
)
RABBIT_FACT_UIDS = tuple(f"rabbit_fact_{i}" for i in range(1, 9))


def build_junior_conformance_memory() -> AHMemory:
    """Manual figure-14 fixture plus the required hierarchy and H/FOLLOW episode."""
    memory = AHMemory()

    def add_reference(section: str, ref: SReference | MReference):
        memory.add_element(section, MemoryElement(uid=ref.reference_uid, payload=ref))
        return ref

    def concept(key: str, label: str) -> str:
        symbol_uid, concept_uid = f"s_{key}", f"m_{key}"
        memory.add_symbol(FirstOrderSymbol(uid=symbol_uid, sensory_representations=(SensoryRepresentation(modality="text", value=label),)))
        memory.add_element("C", MemoryElement(uid=concept_uid, payload=SecondOrderSymbol(uid=concept_uid, properties=(Property(name="label", value=label),))))
        sref = add_reference("C", SReference(reference_uid=f"sr_ground_{key}", target_uid=symbol_uid))
        mref = add_reference("C", MReference(reference_uid=f"mr_ground_{key}", target_uid=concept_uid))
        memory.add_link(AssociativeLink(uid=f"l_ground_{key}", type_id="LINK", weight=1, source_ref=sref, target_ref=mref))
        return concept_uid

    concepts = {
        key: concept(key, label)
        for key, label in {
            "hare": "заяц", "animal": "зверёк", "small": "маленький", "wild": "дикий",
            "meadow": "луг", "forest": "лес", "paw": "лапа", "hind": "задняя",
            "strong": "сильная", "fast": "быстро", "ear": "ухо", "long": "длинное",
            "tail": "хвост", "round": "круглый", "fluffy": "пушистый", "fur": "шерсть",
            "summer": "лето", "brown": "коричневый", "winter": "зима", "white": "белый",
            "observer": "наблюдатель",
        }.items()
    }

    def template(key: str, roles: tuple[Role, ...]) -> str:
        predicate_uid = f"pred_{key.lower()}"
        memory.add_symbol(FirstOrderSymbol(uid=predicate_uid, sensory_representations=(SensoryRepresentation(modality="text", value=key),)))
        predicate_ref = add_reference("C", SReference(reference_uid=f"sr_{predicate_uid}", target_uid=predicate_uid))
        template_uid = f"tpl_{key.lower()}"
        memory.add_template(ControlTemplate(uid=template_uid, predicate_ref=predicate_ref, ordered_roles=roles))
        return template_uid

    templates = {
        "IS": template("IS", (Role(role_id="SUBJECT", required=True), Role(role_id="OBJECT", required=True))),
        "LIVE": template("LIVE", (Role(role_id="SUBJECT", required=True), Role(role_id="LOCATION", required=True))),
        "HAS": template("HAS", (Role(role_id="SUBJECT", required=True), Role(role_id="OBJECT", required=True), Role(role_id="TIME"))),
        "RUN": template("RUN", (Role(role_id="SUBJECT", required=True), Role(role_id="HOW-TO", required=True))),
        "OBSERVED": template("OBSERVED", (Role(role_id="SUBJECT", required=True), Role(role_id="OBJECT", required=True), Role(role_id="LOCATION"))),
    }

    def mref(section: str, uid_: str, target_uid: str) -> MReference:
        return add_reference(section, MReference(reference_uid=uid_, target_uid=target_uid))

    or_operands = (mref("C", "mr_or_meadow", concepts["meadow"]), mref("C", "mr_or_forest", concepts["forest"]))
    memory.add_element("C", MemoryElement(uid="fn_or_place", payload=FunctionalSymbol(uid="fn_or_place", function_id="OR", ordered_operands=or_operands)))
    very_operand = mref("C", "mr_very_fast", concepts["fast"])
    memory.add_element("C", MemoryElement(uid="fn_very_fast", payload=FunctionalSymbol(uid="fn_very_fast", function_id="VERY", ordered_operands=(very_operand,))))
    paw_members = (mref("C", "mr_hind_paw_paw", concepts["paw"]), mref("C", "mr_hind_paw_hind", concepts["hind"]))
    memory.add_element("C", MemoryElement(uid="list_hind_paw", payload=MemoryList(uid="list_hind_paw", ordered_members=paw_members, list_type="ObjectModel")))
    hierarchy_members = (mref("C", "mr_hierarchy_hare", concepts["hare"]), mref("C", "mr_hierarchy_animal", concepts["animal"]))
    memory.add_element("C", MemoryElement(uid="hierarchy_hare_animal", payload=MemoryList(uid="hierarchy_hare_animal", ordered_members=hierarchy_members, list_type="Hierarchy")))

    def fact(uid_: str, template_key: str, bindings: tuple[tuple[str, str], ...], section: str = "C") -> str:
        role_bindings = []
        for ordinal, (role, target_uid) in enumerate(bindings):
            target = memory.elements[target_uid].payload if target_uid in memory.elements else None
            if isinstance(target, SecondOrderSymbol):
                ref = mref(section, f"mr_{uid_}_{ordinal}", target_uid)
            else:
                ref = ElementReference(reference_uid=f"er_{uid_}_{ordinal}", target_uid=target_uid)
            role_bindings.append(RoleBinding(role_id=role, target_ref=ref))
        hypernode = Hypernode(
            uid=uid_, weight=1,
            template_ref=ElementReference(reference_uid=f"tr_{uid_}", target_uid=templates[template_key]),
            role_bindings=tuple(role_bindings), origin="manual",
        )
        memory.add_element(section, MemoryElement(uid=uid_, payload=hypernode))
        return uid_

    fact("rabbit_fact_1", "IS", (("SUBJECT", concepts["hare"]), ("OBJECT", concepts["animal"])))
    fact("rabbit_fact_2", "LIVE", (("SUBJECT", concepts["hare"]), ("LOCATION", "fn_or_place")))
    fact("rabbit_fact_3", "HAS", (("SUBJECT", concepts["hare"]), ("OBJECT", "list_hind_paw")))
    fact("rabbit_fact_4", "RUN", (("SUBJECT", concepts["hare"]), ("HOW-TO", "fn_very_fast")))
    fact("rabbit_fact_5", "HAS", (("SUBJECT", concepts["hare"]), ("OBJECT", concepts["ear"])))
    fact("rabbit_fact_6", "HAS", (("SUBJECT", concepts["hare"]), ("OBJECT", concepts["tail"])))
    fact("rabbit_fact_7", "HAS", (("SUBJECT", concepts["hare"]), ("OBJECT", concepts["fur"]), ("TIME", concepts["summer"])))
    fact("rabbit_fact_8", "HAS", (("SUBJECT", concepts["hare"]), ("OBJECT", concepts["fur"]), ("TIME", concepts["winter"])))

    def linked_refs(key: str, source_uid: str, target_uid: str):
        source = mref("C", f"mr_{key}_source", source_uid) if isinstance(memory.elements[source_uid].payload, SecondOrderSymbol) else ElementReference(reference_uid=f"er_{key}_source", target_uid=source_uid)
        target = mref("C", f"mr_{key}_target", target_uid) if isinstance(memory.elements[target_uid].payload, SecondOrderSymbol) else ElementReference(reference_uid=f"er_{key}_target", target_uid=target_uid)
        return source, target

    for key, source, target, type_id in (
        ("hare_is_animal", concepts["hare"], concepts["animal"], "IS-A"),
        ("animal_small", concepts["animal"], concepts["small"], "ATTRIBUTE"),
        ("animal_wild", concepts["animal"], concepts["wild"], "ATTRIBUTE"),
        ("paw_strong", "list_hind_paw", concepts["strong"], "ATTRIBUTE"),
        ("ear_long", concepts["ear"], concepts["long"], "ATTRIBUTE"),
        ("tail_round", concepts["tail"], concepts["round"], "ATTRIBUTE"),
        ("tail_fluffy", concepts["tail"], concepts["fluffy"], "ATTRIBUTE"),
        ("summer_fur_brown", concepts["fur"], concepts["brown"], "ATTRIBUTE"),
        ("winter_fur_white", concepts["fur"], concepts["white"], "ATTRIBUTE"),
        ("strong_paw_causes_run", "rabbit_fact_3", "rabbit_fact_4", "CAUSE"),
    ):
        source_ref, target_ref = linked_refs(key, source, target)
        memory.add_link(AssociativeLink(uid=f"l_{key}", type_id=type_id, weight=1, source_ref=source_ref, target_ref=target_ref))

    episode_first = fact("episode_fact_1", "OBSERVED", (("SUBJECT", concepts["observer"]), ("OBJECT", concepts["hare"]), ("LOCATION", concepts["meadow"])), "H")
    episode_second = fact("episode_fact_2", "OBSERVED", (("SUBJECT", concepts["observer"]), ("OBJECT", concepts["hare"]), ("LOCATION", concepts["forest"])), "H")
    memory.add_element("H", MemoryElement(uid="episode_hare_observation", payload=MemoryList(
        uid="episode_hare_observation", list_type="Episode",
        ordered_members=(ElementReference(reference_uid="er_episode_1", target_uid=episode_first), ElementReference(reference_uid="er_episode_2", target_uid=episode_second)),
    )))
    memory.add_link(AssociativeLink(
        uid="l_episode_follow", type_id="FOLLOW", weight=1,
        source_ref=ElementReference(reference_uid="er_follow_source", target_uid=episode_first),
        target_ref=ElementReference(reference_uid="er_follow_target", target_uid=episode_second),
    ))
    memory.add_element("P", MemoryElement(uid="m_private_hare_interest", payload=SecondOrderSymbol(
        uid="m_private_hare_interest", properties=(Property(name="label", value="личный интерес к зайцам"),)
    )))
    memory.validate()
    return memory


def junior_conformance_report() -> dict:
    memory = build_junior_conformance_memory()
    payload_types = {type(element.payload).__name__ for element in memory.elements.values()}
    expected_payloads = {"SReference", "SecondOrderSymbol", "MReference", "FunctionalSymbol", "MemoryList", "ControlTemplate", "Hypernode"}
    expected_rabbit_facts = {
        "rabbit_fact_1": ("tpl_is", {"SUBJECT": "m_hare", "OBJECT": "m_animal"}),
        "rabbit_fact_2": ("tpl_live", {"SUBJECT": "m_hare", "LOCATION": "fn_or_place"}),
        "rabbit_fact_3": ("tpl_has", {"SUBJECT": "m_hare", "OBJECT": "list_hind_paw"}),
        "rabbit_fact_4": ("tpl_run", {"SUBJECT": "m_hare", "HOW-TO": "fn_very_fast"}),
        "rabbit_fact_5": ("tpl_has", {"SUBJECT": "m_hare", "OBJECT": "m_ear"}),
        "rabbit_fact_6": ("tpl_has", {"SUBJECT": "m_hare", "OBJECT": "m_tail"}),
        "rabbit_fact_7": ("tpl_has", {"SUBJECT": "m_hare", "OBJECT": "m_fur", "TIME": "m_summer"}),
        "rabbit_fact_8": ("tpl_has", {"SUBJECT": "m_hare", "OBJECT": "m_fur", "TIME": "m_winter"}),
    }
    rabbit_facts_match = all(
        (hypernode := memory.get_hypernode(uid_)) is not None
        and hypernode.template_ref.target_uid == template_uid
        and {binding.role_id: binding.target_ref.target_uid for binding in hypernode.role_bindings} == bindings
        for uid_, (template_uid, bindings) in expected_rabbit_facts.items()
    )
    restored = AHMemory.from_export(memory.export())
    hierarchy = memory.get_list("hierarchy_hare_animal")
    hierarchy_targets = {ref.target_uid for ref in hierarchy.ordered_members} if hierarchy else set()
    junior_operations = (
        "addAbstractSymbol", "editAbstractSymbol", "addElement", "getSymbol", "findLinks",
    )
    checks = {
        "tuple_sections": set(memory.sections) == {"C", "P", "H"},
        "all_seven_payload_variants": expected_payloads == payload_types,
        "s_references_target_S": all(ref.target_uid in memory.symbols for ref in memory.reference_index().values() if isinstance(ref, SReference)),
        "m_references_target_second_order": all(isinstance(memory.elements[ref.target_uid].payload, SecondOrderSymbol) for ref in memory.reference_index().values() if isinstance(ref, MReference)),
        "template_predicates_are_s_references": all(isinstance(template.predicate_ref, SReference) for template in memory.templates.values()),
        "hypernodes_reference_templates": all(isinstance(memory.elements[h.template_ref.target_uid].payload, ControlTemplate) for h in memory.find_hypernodes()),
        "rabbit_figure_14_has_eight_exact_facts": rabbit_facts_match,
        "is_a_hierarchy_present": hierarchy is not None and any(
            link.type_id == "IS-A" and {link.source_ref.target_uid, link.target_ref.target_uid} <= hierarchy_targets
            for link in memory.links.values()
        ),
        "history_episode_follow_dag_present": bool(memory.find_lists(list_type="Episode")) and any(link.type_id == "FOLLOW" for link in memory.links.values()),
        "junior_operations_present": all(callable(getattr(memory, name, None)) for name in junior_operations),
        "typed_get_symbol": isinstance(memory.getSymbol("m_hare"), SecondOrderSymbol) and memory.getSymbol("list_hind_paw") is None,
        "lossless_round_trip": restored.export()["dump"] == memory.export()["dump"],
    }
    exported = memory.export()
    return {
        "profile": "AH-2026-Junior",
        "status": "conformant" if all(checks.values()) else "non_conformant",
        "source": {"monograph_section": 7, "figure": 14, "text": RABBIT_SOURCE},
        "checks": checks,
        "payload_types": sorted(payload_types),
        "rabbit_fact_uids": list(RABBIT_FACT_UIDS),
        "stats": memory.stats(),
        "sha256": exported["sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(junior_conformance_report(), ensure_ascii=False, indent=2))
