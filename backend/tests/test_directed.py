import math
import pytest
from fastapi.testclient import TestClient
import app.main as api
from app.core import AHMemory
from app.engine import IgnitionEngine
from app.ingestion import ingest, RuleBasedProvider
from app.models import IgnitionConfig
from app.senior_conformance import senior_activation_memory, senior_activation_report


def test_directed_numerical_ticks_and_demonstration():
    memory = senior_activation_memory()
    run = IgnitionEngine().run(memory.snapshot(), ["s_pump"], IgnitionConfig(max_ticks=2, rhythm_hz=0, decay_lambda=.08), "directed_v1")
    # tick 0: C receives .8 from S; tick 1: P receives .8 * f_C(0) = .64.
    assert run.element_excitation["p_n17"] == pytest.approx(.64 * math.exp(-.08))
    assert not any(event.target_uid == "p_n17" and event.impulse_value > 0 and event.tick == 0 for event in run.run.ticks)
    report = senior_activation_report()
    assert report["passed"], report["checks"]
    assert set(report["checks"]) == {"S_to_C_to_P", "P_to_S_to_C", "loop_without_rhythm", "decay_without_links", "autonomous_rhythm"}


def test_ingested_fact_requires_explicit_recall_link(monkeypatch):
    memory = AHMemory()
    ingest(memory, "Перегрев насоса вызвал остановку агрегата.", provider=RuleBasedProvider())
    node = memory.find_hypernodes()[0]
    seed = node.role_bindings[0].target_ref.target_uid
    config = IgnitionConfig(max_ticks=4, rhythm_hz=0)
    full = IgnitionEngine().run(memory.snapshot(), [seed], config, "directed_v1")
    assert node.uid in full.run.working_memory
    assert not any(event.impulse_type == "role_to_hypernode" for event in full.run.ticks)
    snapshot = memory.snapshot()
    snapshot.links = {key: link for key, link in snapshot.links.items() if link.type_id != "RECALLS"}
    broken = IgnitionEngine().run(snapshot, [seed], config, "directed_v1")
    assert node.uid not in broken.run.working_memory
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    response = client.post("/api/v1/queries", json={"question": "Что вызвало остановку агрегата?", "profile": "directed_v1"})
    assert response.status_code == 200
    assert response.json()["status"] == "answered" and response.json()["trace_complete"]
    assert client.get("/api/v1/conformance/senior/activation").json()["passed"]
