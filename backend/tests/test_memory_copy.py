import pytest
from app.core import AHMemory, InvariantError
from app.models import FirstOrderSymbol, SensoryRepresentation, MemoryElement, SecondOrderSymbol, Property, AssociativeLink, SReference


def test_copy_on_write_preserves_snapshot_and_rolls_back_invalid_link():
    memory=AHMemory()
    memory.add_symbol(FirstOrderSymbol(uid="s",sensory_representations=(SensoryRepresentation(modality="text",value="test"),)))
    memory.add_element("C",MemoryElement(uid="m",payload=SecondOrderSymbol(uid="m",properties=(Property(name="nested",value={"list":[1]}),))))
    snapshot=memory.snapshot()
    before=memory.export()
    with pytest.raises(InvariantError):
        memory.add_link(AssociativeLink(uid="bad",type_id="CAUSE",weight=.5,source_ref=SReference(reference_uid="r1",target_uid="s"),target_ref=SReference(reference_uid="r2",target_uid="missing")))
    assert memory.export()==before
    memory.add_property("m",Property(name="new",value="value"))
    assert len(snapshot.elements["m"].payload.properties)==1
    memory.elements["m"].payload.properties[0].value["list"].append(2)
    assert snapshot.elements["m"].payload.properties[0].value=={"list":[1]}
    memory.validate()
