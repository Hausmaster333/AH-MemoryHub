from app.core import AHMemory
from app.models import AssociativeLink, FirstOrderSymbol, MReference, MemoryElement, SReference, SensoryRepresentation
from app.engine import build_candidate_snapshot
import app.main as api


def test_equal_score_seed_cutoff_does_not_depend_on_random_ids(monkeypatch):
    selected=[]
    for reverse in (False,True):
        memory=AHMemory()
        for i in range(12):
            memory.add_symbol(FirstOrderSymbol(uid=f"s{11-i if reverse else i:02}",sensory_representations=(SensoryRepresentation(modality="text",value=f"Датчик Д-{i:02}"),)))
        monkeypatch.setattr(api,"memory",memory)
        selected.append([memory.label(uid) for uid in api._seed_ids("датчик")])
    assert selected[0]==selected[1] and len(selected[0])==8


def test_projection_cutoff_preserves_content_when_all_connected_ids_change():
    projections = []
    for reverse in (False, True):
        memory = AHMemory()
        seed = "seed-b" if reverse else "seed-a"
        memory.symbols[seed] = FirstOrderSymbol(uid=seed, sensory_representations=(SensoryRepresentation(modality="text", value="Исходный узел"),))
        for index in range(270):
            suffix = 269 - index if reverse else index
            target = f"symbol-{suffix:03}"
            link_uid = f"link-{suffix:03}"
            memory.symbols[target] = FirstOrderSymbol(uid=target, sensory_representations=(SensoryRepresentation(modality="text", value=f"Объект {index:03}"),))
            memory.links[link_uid] = AssociativeLink(
                uid=link_uid, type_id="ASSOCIATES", weight=0.5,
                source_ref=SReference(reference_uid=f"from-{suffix:03}", target_uid=seed),
                target_ref=SReference(reference_uid=f"to-{suffix:03}", target_uid=target),
            )
        snapshot, _ = build_candidate_snapshot(memory, [seed], max_nodes=256, max_hops=1)
        projections.append(sorted(symbol.sensory_representations[0].value for symbol in snapshot.symbols.values()))
    assert projections[0] == projections[1]
    assert len(projections[0]) == 256


def test_projection_key_handles_cyclic_memory_references():
    memory = AHMemory()
    memory.symbols["seed"] = FirstOrderSymbol(uid="seed", sensory_representations=(SensoryRepresentation(modality="text", value="Пуск"),))
    for current, target in (("a", "b"), ("b", "a")):
        memory.sections["C"][current] = MemoryElement(uid=current, payload=MReference(reference_uid=current, target_uid=target))
        memory.links[f"link-{current}"] = AssociativeLink(uid=f"link-{current}", type_id="ASSOCIATES", weight=0.5,
            source_ref=SReference(reference_uid=f"from-{current}", target_uid="seed"),
            target_ref=MReference(reference_uid=f"to-{current}", target_uid=current))
    snapshot, _ = build_candidate_snapshot(memory, ["seed"], max_nodes=2, max_hops=1)
    assert len(snapshot.elements) == 1
