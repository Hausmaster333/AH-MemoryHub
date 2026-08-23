import time
from fastapi.testclient import TestClient
import pytest
from app.core import AHMemory, InvariantError
from app.dsl import Interpreter
from app.engine import IgnitionEngine, build_candidate_snapshot, gc_commit, gc_preview
from app.ingestion import ingest
from app.ingestion import RuleBasedProvider
from app.main import app
from app.models import *

def symbol(m, uid_, label):
    return m.add_symbol(FirstOrderSymbol(uid=uid_, sensory_representations=(SensoryRepresentation(modality="text", value=label), SensoryRepresentation(modality="text", value=label + "а"))))
def template(m, uid_="tpl_test"):
    return m.add_template(ControlTemplate(uid=uid_, predicate_ref="TEST", ordered_roles=(Role(role_id="SUBJECT"), Role(role_id="OBJECT", multiplicity=True))))
def hyper(m, uid_="h_test", origin="manual"):
    return m.add_element("C", MemoryElement(uid=uid_, payload=Hypernode(uid=uid_, weight=.8, template_ref="tpl_test", origin=origin, role_bindings=(RoleBinding(role_id="SUBJECT", target_ref=SReference(reference_uid=uid_+"_s", target_uid="s1")), RoleBinding(role_id="OBJECT", target_ref=SReference(reference_uid=uid_+"_o", target_uid="s2"))))))

def test_ingestion_templates_and_roundtrip():
    m = AHMemory(); r = ingest(m, "Перегрев насоса вызвал остановку агрегата в насосном зале.")
    assert r.accepted and not r.rejected and len(m.templates) >= 8
    exported = m.export(); restored = AHMemory.from_export(exported)
    assert restored.stats() == m.stats() and restored.export()["dump"] == exported["dump"]

def test_reference_identity_index_and_target_lookup():
    m = AHMemory(); symbol(m, "s1", "alpha"); symbol(m, "s2", "beta"); template(m); hyper(m)
    m.add_element("P", MemoryElement(uid="list", payload=MemoryList(uid="list", ordered_members=(MReference(reference_uid="list_ref", target_uid="h_test"),))))
    m.validate(); assert m.get_s_reference("h_test_s") is not None; assert m.get_s_reference("s1") is None
    assert m.find_s_references("s1")[0].reference_uid == "h_test_s"; assert m.get_m_reference("list_ref").target_uid == "h_test"
    assert AHMemory.from_export(m.export()).get_s_reference("h_test_s").target_uid == "s1"

def test_repeated_text_wordforms_allowed_but_exact_duplicate_rejected():
    s = FirstOrderSymbol(uid="s", sensory_representations=(SensoryRepresentation(modality="text", value="насос"), SensoryRepresentation(modality="text", value="насоса")))
    assert len(s.sensory_representations) == 2
    with pytest.raises(ValueError): FirstOrderSymbol(uid="x", sensory_representations=(SensoryRepresentation(modality="text", value="x"), SensoryRepresentation(modality="text", value="x")))

def test_all_payload_types_and_manual_evidence_exception():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); template(m); hyper(m)
    m.add_element("C", MemoryElement(uid="so", payload=SecondOrderSymbol(uid="so", properties=(Property(name="label", value="so"),))))
    m.add_element("C", MemoryElement(uid="fn", payload=FunctionalSymbol(uid="fn", function_id="f", ordered_operands=(SReference(reference_uid="fn_ref", target_uid="s1"),))))
    m.validate(); assert isinstance(m.get_symbol("fn").payload, FunctionalSymbol)

def test_atomic_cycle_and_reference_conflict():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); before = m.revision
    m.add_link(AssociativeLink(uid="l1", type_id="IS-A", weight=1, source_ref=SReference(reference_uid="r1", target_uid="s1"), target_ref=SReference(reference_uid="r2", target_uid="s2")))
    with pytest.raises(InvariantError): m.add_link(AssociativeLink(uid="l2", type_id="IS-A", weight=1, source_ref=SReference(reference_uid="r3", target_uid="s2"), target_ref=SReference(reference_uid="r4", target_uid="s1")))
    assert "l2" not in m.links and m.revision > before

def test_retrieval_precedes_one_way_ignition():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); template(m); hyper(m)
    raw = IgnitionEngine().run(m.snapshot(), ["s1"], IgnitionConfig(max_ticks=2)); assert not any(t.impulse_type == "hypernode" for t in raw.run.ticks)
    snap, seeds = build_candidate_snapshot(m, ["s1"]); assert "h_test" in seeds
    prepared = IgnitionEngine().run(snap, seeds, IgnitionConfig(max_ticks=2)); assert any(t.impulse_type == "hypernode" for t in prepared.run.ticks)

def test_profiles_rhythm_parent_trace_and_hebbian_deltas():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b")
    m.add_link(AssociativeLink(uid="l", type_id="CAUSE", weight=.5, source_ref=SReference(reference_uid="r1", target_uid="s1"), target_ref=SReference(reference_uid="r2", target_uid="s2")))
    cfg = IgnitionConfig(max_ticks=4, rhythm_hz=2, hebbian_eta=.5); out = IgnitionEngine().run(m.snapshot(), ["s1"], cfg, "integrated_v1")
    assert any(t.impulse_type == "rhythm_pulse" for t in out.run.ticks); assert out.run.weight_deltas and any(t.parent_trace is not None for t in out.run.ticks)
    assert IgnitionEngine().run(m.snapshot(), ["s1"], cfg, "literal_2026").run.profile == "literal_2026"; assert all(0 <= x.next_weight <= 1 for x in out.run.ticks if x.next_weight is not None)

def test_gc_grace_stale_preview_and_live_hypernode_preservation():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); template(m); hyper(m)
    m.add_element("P", MemoryElement(uid="orphan", payload=SecondOrderSymbol(uid="orphan", properties=(Property(name="label", value="o"),))))
    cfg = IgnitionConfig(initial_life_ticks=5); first = gc_preview(m, cfg); assert "orphan" not in first["deletable_uids"]
    m.advance_ticks(50); stale = first["preview_token"]; m.add_symbol(FirstOrderSymbol(uid="s3", sensory_representations=(SensoryRepresentation(modality="text", value="c"),)))
    with pytest.raises(ValueError): gc_commit(m, stale, first["deletable_uids"])
    second = gc_preview(m, cfg); assert "orphan" in second["deletable_uids"] and "h_test" not in second["deletable_uids"]
    gc_commit(m, second["preview_token"], second["deletable_uids"]); assert "orphan" not in m.elements and "h_test" in m.elements

def test_dsl_composition_and_mutation_validation():
    m = AHMemory(); symbol(m, "s1", "alpha"); symbol(m, "s2", "beta"); assert Interpreter(m).query("findSymbols(value=alpha)")
    result = Interpreter(m).mutate("addLink(type=CAUSE, source=s1, target=s2, weight=0.4)"); assert result["type_id"] == "CAUSE"
    with pytest.raises(ValueError): Interpreter(m).mutate("addNode(x=1)")

def test_api_end_to_end_all_critical_routes():
    c = TestClient(app); assert c.get("/health").status_code == 200
    ing = c.post("/api/v1/ingestions", json={"text": "Перегрев насоса вызвал остановку агрегата в насосном зале."}); assert ing.status_code == 200
    iid = ing.json()["ingestion_uid"]; assert c.get("/api/v1/ingestions/" + iid).status_code == 200
    q = c.post("/api/v1/queries", json={"question": "остановку агрегата", "ignition": {"max_ticks": 3, "rhythm_hz": 2}}); assert q.status_code == 200 and "effective_config" in q.json()
    if q.json().get("run_uid"): assert c.get("/api/v1/ignition-runs/" + q.json()["run_uid"] + "/ticks").status_code == 200
    assert c.post("/api/v1/dsl/query", json={"expression": "findSymbols(value=насос)"}).status_code == 200; assert c.get("/api/v1/memory/stats").status_code == 200 and c.post("/api/v1/memory/export").status_code == 200; assert c.post("/api/v1/evaluations", json={"fixture": "internal"}).status_code == 200

def test_demo_three_hop_answer_and_honest_evaluation():
    c = TestClient(app); d = c.post("/api/v1/demo/seed").json(); assert len(d["accepted"]) >= 5
    q = c.post("/api/v1/queries", json={"question": "почему остановился агрегат?", "max_ticks": 8}).json(); assert q["status"] == "answered" and q["trace_complete"] and q["evidence"]
    e = c.post("/api/v1/evaluations").json(); assert e["metrics"]["M4"]["status"] == "unavailable" and e["metrics"]["M5"]["status"] == "unavailable"

def test_m3_fixture_and_n1000_tick_benchmark():
    m = AHMemory()
    for i in range(1000): m.add_symbol(FirstOrderSymbol(uid=f"s{i}", sensory_representations=(SensoryRepresentation(modality="text", value=str(i)),)))
    for i in range(999): m.add_link(AssociativeLink(uid=f"l{i}", type_id="ASSOCIATES", weight=.5, source_ref=SReference(reference_uid=f"a{i}", target_uid=f"s{i}"), target_ref=SReference(reference_uid=f"b{i}", target_uid=f"s{i+1}")))
    start = time.perf_counter(); IgnitionEngine().run(m.snapshot(), ["s0"], IgnitionConfig(max_ticks=1)); assert (time.perf_counter() - start) < .5

def test_candidate_fact_atomic_rejection_does_not_leave_symbols():
    m = AHMemory(); result = ingest(m, "а", provider=type("Bad", (), {"extract": lambda self, text: [CandidateFact(predicate="UNKNOWN", bindings=(CandidateBinding(role_id="SUBJECT", value="leak"),), source_start=0, source_end=1, exact_text="а", confidence=.5)]})())
    assert result.rejected and not m.find_abstract_symbols("leak")

def test_properties_and_link_edit_identity():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); m.add_element("C", MemoryElement(uid="so", payload=SecondOrderSymbol(uid="so")))
    m.add_property("so", Property(name="label", value="x")); assert m.find_symbols("x");
    m.add_link(AssociativeLink(uid="l", type_id="CAUSE", weight=.2, source_ref=SReference(reference_uid="r1", target_uid="s1"), target_ref=SReference(reference_uid="r2", target_uid="s2")))
    m.edit_link("l", AssociativeLink(uid="l", type_id="CAUSE", weight=.9, source_ref=SReference(reference_uid="r1", target_uid="s1"), target_ref=SReference(reference_uid="r2", target_uid="s2"))); assert m.get_link("l").weight == .9

def test_structured_errors_and_reference_conflict_rejected():
    c = TestClient(app); response = c.post("/api/v1/dsl/query", json={"expression": ""}); assert response.status_code == 422 and response.json()["error"]["request_id"]
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b");
    with pytest.raises(InvariantError): m.add_link(AssociativeLink(uid="x", type_id="R", weight=.5, source_ref=SReference(reference_uid="same", target_uid="s1"), target_ref=SReference(reference_uid="same", target_uid="s2")))

def test_same_origin_frontend_assets_and_full_smoke_flow():
    c = TestClient(app)
    root = c.get("/"); assert root.status_code == 200 and "AH-MemoryHub" in root.text
    css = c.get("/assets/styles.css"); js = c.get("/assets/app.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert c.get("/health").status_code == 200
    seed = c.post("/api/v1/demo/seed"); assert seed.status_code == 200
    query_result = c.post("/api/v1/queries", json={"question": "почему остановился агрегат?", "ignition": {"max_ticks": 4}})
    assert query_result.status_code == 200 and "effective_config" in query_result.json()
    assert c.post("/api/v1/memory/export", json={}).status_code == 200
    assert c.post("/api/v1/evaluations", json={}).status_code == 200
    assert c.post("/api/v1/gc/preview", json={}).status_code == 200

def test_ingestion_and_demo_are_idempotent():
    c = TestClient(app)
    body = {"text": "Перегрев насоса вызвал остановку агрегата.", "document_uid": "idempotent_check"}
    first = c.post("/api/v1/ingestions", json=body).json(); before = c.get("/api/v1/memory/stats").json()
    second = c.post("/api/v1/ingestions", json=body).json(); after = c.get("/api/v1/memory/stats").json()
    assert second["reused"] is True and first["accepted"] == second["accepted"] and before == after
    demo_first = c.post("/api/v1/demo/seed").json(); demo_before = c.get("/api/v1/memory/stats").json(); demo_second = c.post("/api/v1/demo/seed").json(); demo_after = c.get("/api/v1/memory/stats").json()
    assert demo_second["reused"] is True and demo_before == demo_after

def test_query_answer_is_selected_causal_evidence_bound_phrase():
    c = TestClient(app)
    c.post("/api/v1/ingestions", json={"document_uid": "answer_check", "text": "Перегрев насоса вызвал остановку агрегата."})
    result = c.post("/api/v1/queries", json={"question": "Почему остановился агрегат?"}).json()
    assert result["status"] == "answered" and result["evidence"]
    assert "Подтверждено AH-памятью" not in result["answer"] and result["answer"].count("агрегат") == 1
    assert "останов" in result["answer"].lower() and "перегрев" in result["answer"].lower()
    assert result["evidence"][0]["exact_text"] in "Перегрев насоса вызвал остановку агрегата."

def test_rule_provider_does_not_make_location_from_word_suffix():
    facts = RuleBasedProvider().extract("Перегрев насоса вызвал остановку агрегата.")
    roles = {binding.role_id for binding in facts[0].bindings}
    assert roles == {"SUBJECT", "OBJECT"}

def test_rule_provider_demo_predicates_are_conservative_and_atomic():
    provider = RuleBasedProvider()
    facts = provider.extract("Оператор обнаружил перегрев насоса. Перегрев насоса вызвал остановку агрегата. После этого оператор применил ручной ключ.")
    assert [fact.predicate for fact in facts] == ["OBSERVED", "CAUSE", "USES_TOOL"]
    for fact in facts[:2]:
        values = {binding.role_id: binding.value for binding in fact.bindings}
        assert values["SUBJECT"] and values["OBJECT"] and values["SUBJECT"] != values["OBJECT"]
