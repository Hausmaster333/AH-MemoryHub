from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.core import AHMemory
from app.dsl import Interpreter
from app.engine import IgnitionEngine, evidence_trace, gc_commit, gc_preview
from app.ingestion import ingest, RuleBasedProvider
from app.models import (
    AssociativeLink, ElementReference, IgnitionConfig, MemoryElement,
    SecondOrderSymbol,
)


def add_orphan(memory, name, **kwargs):
    memory.add_element("P", MemoryElement(uid=name, payload=SecondOrderSymbol(uid=name), **kwargs))


def test_parallel_writers_keep_every_commit():
    memory, barrier = AHMemory(), Barrier(8)
    def write(index):
        barrier.wait(timeout=5)
        add_orphan(memory, f"parallel_{index}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))
    assert len(memory.elements) == memory.revision == 8
    memory.validate()


def test_activation_output_is_previous_tick_signal_and_trace_is_causal():
    memory = AHMemory()
    add_orphan(memory, "a", activation_function="sigmoid")
    add_orphan(memory, "b")
    memory.add_link(AssociativeLink(uid="ab", type_id="ASSOCIATES", weight=.5,
        source_ref=ElementReference(reference_uid="sa", target_uid="a"),
        target_ref=ElementReference(reference_uid="tb", target_uid="b")))
    run = IgnitionEngine().run(memory.snapshot(), ["a"], IgnitionConfig(max_ticks=3, rhythm_hz=0)).run
    impulse = next(event for event in run.ticks if event.impulse_type == "associative")
    assert impulse.impulse_value == pytest.approx(.3655292893)
    assert run.ticks[impulse.parent_trace].tick == -1
    for index, event in enumerate(run.ticks):
        assert all(parent < index for parent in event.parent_traces)
        if event.impulse_type == "associative":
            assert run.ticks[event.parent_trace].tick < event.tick
    assert {"a", "ab", "b"} <= set(evidence_trace(run, ("b",), ["a"]))
    assert not evidence_trace(run, ("b",), ["unknown"])
    assert not IgnitionEngine().run(memory.snapshot(), [], IgnitionConfig(rhythm_hz=0)).run.trace_complete
    with pytest.raises(ValueError): IgnitionConfig(decay_lambda=float("inf"))


def test_api_ticks_collect_200_orphans_and_preserve_live_episode(monkeypatch):
    memory = AHMemory()
    ingest(memory, "Перегрев насоса вызвал остановку агрегата. Остановка агрегата вызвала аварийный сигнал.", provider=RuleBasedProvider())
    live = set(memory.elements)
    assert memory.find_lists(list_type="Episode")
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    for index in range(200):
        response = client.post("/api/v1/dsl/mutate", json={"expression": f"addElement(section=P,uid=orphan_{index})"})
        assert response.status_code == 200
    response = client.post("/api/v1/memory/ticks", json={"max_ticks": 50, "rhythm_hz": 0})
    assert response.status_code == 200, response.text
    assert response.json()["deleted_count"] == 200
    assert response.json()["current_tick"] == 50
    assert set(memory.elements) == live
    memory.validate()
    add_orphan(memory, "late")
    preview = gc_preview(memory)
    assert "late" not in preview["deletable_uids"]
    memory.advance_ticks(5)
    with pytest.raises(ValueError, match="stale"):
        gc_commit(memory, preview["preview_token"])
    assert gc_preview(memory)["deletable_uids"] == ["late"]
    assert AHMemory.from_export(memory.export()).created_ticks == memory.created_ticks


def test_query_requires_working_memory_and_dsl_intersects_objects(monkeypatch):
    memory = AHMemory()
    ingest(memory, "Перегрев насоса вызвал остановку агрегата.", provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    question = "Что вызвало остановку агрегата?"
    blocked = client.post("/api/v1/queries", json={"question": question, "ignition": {"working_memory_threshold": 1}}).json()
    assert blocked["status"] == "insufficient_evidence" and not blocked["trace_complete"]
    answer = client.post("/api/v1/queries", json={"question": question}).json()
    assert answer["status"] == "answered" and answer["trace_complete"]
    assert set(answer["answer_path"]) <= set(answer["minimal_path"])
    assert memory.current_tick == 16
    dsl = Interpreter(memory)
    facts = dsl.query("findHypernodes() | intersect(findHypernodes())")
    assert len(facts) == len(memory.find_hypernodes())
    name = facts[0]["uid"]
    assert dsl.query(f"getHypernode(uid={name}) | intersect(findLists(type=Episode))") == facts
