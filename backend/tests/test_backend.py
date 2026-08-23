import json
import time
from fastapi.testclient import TestClient
import pytest
import app.ingestion as ingestion_module
from app.core import AHMemory, InvariantError
from app.conformance import RABBIT_FACT_UIDS, build_junior_conformance_memory, junior_conformance_report
from app.dsl import Interpreter
from app.engine import IgnitionEngine, build_candidate_snapshot, gc_commit, gc_preview
from app.evaluation import rabbit_ingestion_v2
from app.ingestion import ingest
from app.ingestion import Compiler, OpenAICompatibleProvider, RuleBasedProvider, canonicalize_candidates, extract_candidates, segment_text
from app.main import app
from app.models import *

def symbol(m, uid_, label):
    return m.add_symbol(FirstOrderSymbol(uid=uid_, sensory_representations=(SensoryRepresentation(modality="text", value=label), SensoryRepresentation(modality="text", value=label + "а"))))
def template(m, uid_="tpl_test"):
    predicate_uid = uid_ + "_predicate"; predicate_ref = SReference(reference_uid=uid_ + "_predicate_ref", target_uid=predicate_uid)
    m.add_symbol(FirstOrderSymbol(uid=predicate_uid, sensory_representations=(SensoryRepresentation(modality="text", value="TEST"),)))
    m.add_element("C", MemoryElement(uid=predicate_ref.reference_uid, payload=predicate_ref))
    return m.add_template(ControlTemplate(uid=uid_, predicate_ref=predicate_ref, ordered_roles=(Role(role_id="SUBJECT"), Role(role_id="OBJECT", multiplicity=True))))
def hyper(m, uid_="h_test", origin="manual"):
    return m.add_element("C", MemoryElement(uid=uid_, payload=Hypernode(uid=uid_, weight=.8, template_ref=ElementReference(reference_uid=uid_+"_tpl", target_uid="tpl_test"), origin=origin, role_bindings=(RoleBinding(role_id="SUBJECT", target_ref=SReference(reference_uid=uid_+"_s", target_uid="s1")), RoleBinding(role_id="OBJECT", target_ref=SReference(reference_uid=uid_+"_o", target_uid="s2"))))))

def test_ingestion_templates_and_roundtrip():
    m = AHMemory(); r = ingest(m, "Перегрев насоса вызвал остановку агрегата в насосном зале.")
    assert r.accepted and not r.rejected and len(m.templates) >= 8
    exported = m.export(); restored = AHMemory.from_export(exported)
    assert restored.stats() == m.stats() and restored.export()["dump"] == exported["dump"]

def test_openai_compatible_provider_anchors_exact_source_spans():
    text = "Датчик обнаружил перегрев насоса. Перегрев насоса вызвал остановку агрегата."
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    provider._completion = lambda _: {"choices": [{"message": {"content": '{"facts":[{"predicate":"CAUSE","bindings":[{"role_id":"SUBJECT","value":"перегрев насоса"},{"role_id":"OBJECT","value":"остановка агрегата"}],"quote":"Перегрев насоса вызвал остановку агрегата","confidence":0.91}]}'}}]}
    candidate = provider.extract(text)[0]
    assert text[candidate.source_start:candidate.source_end] == candidate.exact_text
    assert candidate.model_id == "test-model" and candidate.predicate == "CAUSE"

def test_openai_compatible_provider_requests_strict_json_schema():
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    requests = []
    provider._request_json = lambda payload: requests.append(payload) or {"choices": [{"message": {"content": '{"facts":[]}'}}]}
    provider._completion("Наблюдение")
    response_format = requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    json_provider = OpenAICompatibleProvider("http://local.test/v1", "test-model", response_format="json_object")
    json_requests = []
    json_provider._request_json = lambda payload: json_requests.append(payload) or {"choices": [{"message": {"content": '{"facts":[]}'}}]}
    json_provider._completion("Наблюдение")
    assert json_requests[0]["response_format"] == {"type": "json_object"}

def test_preview_and_candidate_decisions_are_server_side(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "rule")
    c = TestClient(app)
    response = c.post("/api/v1/ingestions/preview", json={"text": "Перегрев насоса вызвал остановку агрегата. Затем оператор применил ручной ключ."})
    assert response.status_code == 200
    preview = response.json(); assert preview["provider"]["active"] == "rule" and len(preview["candidates"]) == 2
    before = c.get("/api/v1/memory/stats").json()["revision"]
    rejected = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][1]["candidate_uid"], "decision": "reject"})
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"
    admitted = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][0]["candidate_uid"], "decision": "admit"})
    assert admitted.status_code == 200 and admitted.json()["accepted"] and c.get("/api/v1/memory/stats").json()["revision"] > before
    reused = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][0]["candidate_uid"], "decision": "admit"})
    assert reused.json()["reused"] is True

def test_reference_identity_index_and_target_lookup():
    m = AHMemory(); symbol(m, "s1", "alpha"); symbol(m, "s2", "beta"); template(m); hyper(m)
    m.add_element("C", MemoryElement(uid="so_ref_target", payload=SecondOrderSymbol(uid="so_ref_target")))
    m.add_element("P", MemoryElement(uid="list", payload=MemoryList(uid="list", ordered_members=(MReference(reference_uid="list_ref", target_uid="so_ref_target"),))))
    m.validate(); assert m.get_s_reference("h_test_s") is not None; assert m.get_s_reference("s1") is None
    assert m.find_s_references("s1")[0].reference_uid == "h_test_s"; assert m.get_m_reference("list_ref").target_uid == "so_ref_target"
    assert AHMemory.from_export(m.export()).get_s_reference("h_test_s").target_uid == "s1"

def test_junior_conformance_fixture_and_lossless_roundtrip():
    memory = build_junior_conformance_memory(); report = junior_conformance_report()
    assert report["status"] == "conformant" and all(report["checks"].values())
    assert set(report["payload_types"]) == {"SReference", "SecondOrderSymbol", "MReference", "FunctionalSymbol", "MemoryList", "ControlTemplate", "Hypernode"}
    assert len(RABBIT_FACT_UIDS) == 8 and all(memory.get_hypernode(uid_) for uid_ in RABBIT_FACT_UIDS)
    assert AHMemory.from_export(memory.export()).export()["dump"] == memory.export()["dump"]

def test_m_reference_cannot_target_an_arbitrary_memory_element():
    memory = AHMemory(); memory.add_element("C", MemoryElement(uid="concept", payload=SecondOrderSymbol(uid="concept")))
    memory.add_element("C", MemoryElement(uid="list", payload=MemoryList(uid="list")))
    with pytest.raises(InvariantError):
        memory.add_element("C", MemoryElement(uid="bad_m_ref", payload=MReference(reference_uid="bad_m_ref", target_uid="list")))

def test_repeated_text_wordforms_allowed_but_exact_duplicate_rejected():
    s = FirstOrderSymbol(uid="s", sensory_representations=(SensoryRepresentation(modality="text", value="насос"), SensoryRepresentation(modality="text", value="насоса")))
    assert len(s.sensory_representations) == 2
    with pytest.raises(ValueError): FirstOrderSymbol(uid="x", sensory_representations=(SensoryRepresentation(modality="text", value="x"), SensoryRepresentation(modality="text", value="x")))

def test_all_payload_types_and_manual_evidence_exception():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); template(m); hyper(m)
    m.add_element("C", MemoryElement(uid="so", payload=SecondOrderSymbol(uid="so", properties=(Property(name="label", value="so"),))))
    m.add_element("C", MemoryElement(uid="fn", payload=FunctionalSymbol(uid="fn", function_id="f", ordered_operands=(SReference(reference_uid="fn_ref", target_uid="s1"),))))
    m.validate(); assert isinstance(m.get_symbol("so"), SecondOrderSymbol) and m.get_symbol("fn") is None

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
    m = AHMemory(); symbol(m, "s1", "alpha"); symbol(m, "s2", "beta"); assert Interpreter(m).query("findAbstractSymbols(value=alpha)")
    result = Interpreter(m).mutate("addLink(type=CAUSE, source=s1, target=s2, weight=0.4)"); assert result["type_id"] == "CAUSE"
    with pytest.raises(ValueError): Interpreter(m).mutate("addNode(x=1)")

def test_api_end_to_end_all_critical_routes():
    c = TestClient(app); assert c.get("/health").status_code == 200
    ing = c.post("/api/v1/ingestions", json={"text": "Перегрев насоса вызвал остановку агрегата в насосном зале."}); assert ing.status_code == 200
    iid = ing.json()["ingestion_uid"]; assert c.get("/api/v1/ingestions/" + iid).status_code == 200
    q = c.post("/api/v1/queries", json={"question": "остановку агрегата", "ignition": {"max_ticks": 3, "rhythm_hz": 2}}); assert q.status_code == 200 and "effective_config" in q.json()
    if q.json().get("run_uid"): assert c.get("/api/v1/ignition-runs/" + q.json()["run_uid"] + "/ticks").status_code == 200
    assert c.post("/api/v1/dsl/query", json={"expression": "findSymbols(value=насос)"}).status_code == 200; assert c.get("/api/v1/memory/stats").status_code == 200 and c.post("/api/v1/memory/export").status_code == 200; assert c.post("/api/v1/evaluations", json={"fixture": "internal"}).status_code == 200
    junior = c.get("/api/v1/conformance/junior"); assert junior.status_code == 200 and junior.json()["status"] == "conformant"

def test_demo_three_hop_answer_and_honest_evaluation():
    c = TestClient(app); d = c.post("/api/v1/demo/seed").json(); assert len(d["accepted"]) + sum(item["reason"] == "duplicate_memory_fact" for item in d["rejected"]) >= 5
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
    assert root.headers["cache-control"] == css.headers["cache-control"] == js.headers["cache-control"] == "no-store"
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

def test_v2_segmentation_preserves_offsets_and_neighbor_context():
    text = "Первый факт.  Первый факт.\nТретий факт без точки"
    spans = segment_text(text)
    assert [text[span.start:span.end] for span in spans] == [span.text for span in spans]
    assert [span.start for span in spans] == [0, 14, 27]
    assert spans[1].context_before == spans[0].text and spans[1].context_after == spans[2].text

def test_v2_rejects_implicit_follow_resolves_safe_anaphora_and_deduplicates():
    text = "Насос перегрелся. Он остановился. После этого оператор применил ключ."
    first = CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="Насос"), CandidateBinding(role_id="STATE", value="перегрелся")), source_start=0, source_end=17, exact_text=text[0:17], confidence=.8)
    second_start = text.index("Он")
    second_end = second_start + len("Он остановился.")
    second = CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="Он"), CandidateBinding(role_id="STATE", value="остановился")), source_start=second_start, source_end=second_end, exact_text=text[second_start:second_end], confidence=.7, unresolved_entities=True)
    follow_start = text.index("После")
    follow = CandidateFact(predicate="FOLLOW", bindings=(CandidateBinding(role_id="SUBJECT", value="остановка насоса"), CandidateBinding(role_id="OBJECT", value="оператор применил ключ")), source_start=follow_start, source_end=len(text), exact_text=text[follow_start:], confidence=.9)
    accepted, rejected = canonicalize_candidates(text, [first, second, first.model_copy(update={"confidence": .4}), follow])
    assert len(accepted) == 2 and accepted[1].bindings[0].value == "Насос" and accepted[1].bindings[0].observed == "Он"
    assert {item["reason"] for item in rejected} == {"duplicate_fact", "implicit_follow"}

def test_v2_canonicalizes_provider_role_synonyms_before_template_validation():
    text = "Заяц бегает очень быстро."
    candidate = CandidateFact(predicate="RUN", bindings=(CandidateBinding(role_id="AGENT", value="Заяц"), CandidateBinding(role_id="HOW-TO", value="очень быстро")), source_start=0, source_end=len(text), exact_text=text, confidence=.9)
    accepted, rejected = canonicalize_candidates(text, [candidate])
    assert not rejected and [binding.role_id for binding in accepted[0].bindings] == ["SUBJECT", "HOW-TO"]
    isa = CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value="Заяц"), CandidateBinding(role_id="VALUE", value="зверёк")), source_start=0, source_end=len(text), exact_text=text, confidence=.8)
    accepted, rejected = canonicalize_candidates(text, [isa])
    assert not rejected and [binding.role_id for binding in accepted[0].bindings] == ["SUBJECT", "OBJECT"]

def test_v2_parser_confidence_is_not_activation_weight(monkeypatch):
    monkeypatch.setenv("AH_INGESTION_INITIAL_WEIGHT", "0.65")
    memory = AHMemory(); result = ingest(memory, "Перегрев вызвал остановку.", provider=RuleBasedProvider())
    hypernode = memory.get_hypernode(result.accepted[0])
    assert hypernode.weight == .65 and hypernode.evidence[0].parser_confidence == .78
    payloads = [element.payload for element in memory.elements.values()]
    assert any(isinstance(payload, SecondOrderSymbol) for payload in payloads)
    assert any(isinstance(payload, SReference) for payload in payloads) and any(isinstance(payload, MReference) for payload in payloads)

def test_v2_cache_key_uses_document_model_and_prompt(monkeypatch):
    class CountingProvider(RuleBasedProvider):
        model_id = "cache-model"
        calls = 0
        def extract(self, text):
            self.calls += 1
            return super().extract(text)
    provider = CountingProvider(); ingestion_module._CANDIDATE_CACHE.clear()
    monkeypatch.setattr(ingestion_module, "configured_provider", lambda: (provider, {"configured": "test", "active": "test", "model_id": provider.model_id, "fallback": False}))
    first = extract_candidates("Уникальный датчик обнаружил сигнал.")[1]
    second = extract_candidates("Уникальный датчик обнаружил сигнал.")[1]
    assert provider.calls == 1 and first["cached"] is False and second["cached"] is True

def test_v2_retries_only_transient_upstream_rate_limit(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError
    calls, delays = [], []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return b'{"choices":[]}'
    def fake_urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3: raise HTTPError("http://local", 429, "limited", {}, BytesIO(b'{"error":"temporarily rate-limited"}'))
        return Response()
    monkeypatch.setattr(ingestion_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(ingestion_module.time, "sleep", delays.append)
    provider = OpenAICompatibleProvider("http://local.test/v1", "retry-model")
    assert provider._request_json({}) == {"choices": []} and len(calls) == 3 and delays == [1, 2]

def test_v2_structured_rabbit_contract_admits_at_least_six_of_eight():
    from app.conformance import RABBIT_SOURCE
    sentences = [span.text for span in segment_text(RABBIT_SOURCE)]
    facts = [
        ("IS-A", sentences[0], [("SUBJECT", "заяц"), ("OBJECT", "зверёк")]),
        ("LIVE", sentences[0], [("SUBJECT", "заяц"), ("LOCATION", "луг или лес")]),
        ("HAS", sentences[1], [("SUBJECT", "заяц"), ("OBJECT", "сильные задние лапы")]),
        ("RUN", sentences[1], [("SUBJECT", "заяц"), ("HOW-TO", "очень быстро")]),
        ("HAS", sentences[2], [("SUBJECT", "заяц"), ("OBJECT", "длинные уши")]),
        ("HAS", sentences[2], [("SUBJECT", "заяц"), ("OBJECT", "круглый пушистый хвост")]),
        ("HAS", sentences[3], [("SUBJECT", "заяц"), ("OBJECT", "коричневая шерсть"), ("TIME", "лето")]),
        ("HAS", sentences[3], [("SUBJECT", "заяц"), ("OBJECT", "белая шерсть"), ("TIME", "зима")]),
    ]
    provider = OpenAICompatibleProvider("http://local.test/v1", "structured-rabbit")
    provider._completion = lambda _: {"choices": [{"message": {"content": json.dumps({"facts": [{"predicate": predicate, "bindings": [{"role_id": role, "value": value} for role, value in bindings], "quote": quote, "confidence": .9, "unresolved_entities": False} for predicate, quote, bindings in facts]}, ensure_ascii=False)}}]}
    candidates, meta = extract_candidates(RABBIT_SOURCE, provider); memory = AHMemory()
    accepted, rejected = Compiler(memory).compile(candidates, RABBIT_SOURCE, "rabbit_v2")
    score = rabbit_ingestion_v2(provider)
    assert not meta["rejection_log"] and not rejected and len(accepted) >= 6
    assert score["status"] == "pass" and score["score"] == 8
