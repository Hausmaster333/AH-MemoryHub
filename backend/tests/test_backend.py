import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import app.ingestion as ingestion_module
from app.core import AHMemory, InvariantError, lexical_score
from app.answering import generate_evidence_answer, select_local_answer, validate_evidence_answer
from app.conformance import RABBIT_FACT_UIDS, build_junior_conformance_memory, junior_conformance_report
from app.dsl import Interpreter
from app.engine import IgnitionEngine, build_candidate_snapshot, gc_commit, gc_preview
from app.evaluation import rabbit_ingestion_v2
from app.ingestion import ingest
from app.ingestion import Compiler, OpenAICompatibleProvider, RuleBasedProvider, _coordinated_property_candidates, _declarative_recovery_candidates, _effect_candidates, _explicit_follow_candidates, _stable_fact_candidates, _temporal_state_candidates, _tool_use_candidates, canonicalize_candidates, extract_candidates, route_candidates, segment_text
from app.main import app
from app.models import *

CORPUS_PATH = Path(__file__).resolve().parents[2] / "docs" / "testing" / "heterogeneous-corpus.json"

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

def test_heterogeneous_long_form_corpus_contract():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    documents = corpus["documents"]
    assert corpus["schema"] == "ah-memoryhub-corpus-v1" and len(documents) >= 10
    assert len({document["id"] for document in documents}) == len(documents)
    assert len({document["domain"] for document in documents}) >= 8
    for document in documents:
        assert 10 <= len(segment_text(document["text"])) <= 15, document["id"]
        assert len(document["required_claims"]) >= 5
        assert len(document["questions"]) >= 5
        assert {"CAUSE", "FOLLOW", "TOOL", "TIME"} <= set(document["expected_features"])


def test_versioned_corpus_metrics_are_role_aware_and_trace_depth_six():
    from app.evaluation import GOLD_PATH, corpus_m2, corpus_role_metrics, load_gold
    gold = load_gold()
    sources = {document["id"]: document["text"] for document in json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["documents"]}
    predictions = {}
    for document in gold["documents"]:
        candidates = []
        text = sources[document["id"]]
        for fact in document["facts"]:
            start = text.index(fact["quote"])
            candidates.append(CandidateFact(predicate=fact["predicate"], bindings=tuple(CandidateBinding(role_id=role, value=value) for role, values in fact["bindings"].items() for value in values), source_start=start, source_end=start + len(fact["quote"]), exact_text=fact["quote"], confidence=1))
        predictions[document["id"]] = candidates
    m1, m2 = corpus_role_metrics(predictions), corpus_m2()
    assert m1["micro"]["f1"] == 1
    assert m1["errors"] == []
    assert m1["benchmark"] == {"schema": "ah-memoryhub-gold-v3", "documents": 10, "source_sentences": 100, "labelled_facts": 106, "role_assignments": 237}
    assert m1["required_roles_weighted_f1"] == 5
    assert m1["normalized_required_roles_weighted_f1"] == 1
    assert len(m1["documents"]) == 10 and m1["document_macro_f1"] == m1["document_min_f1"] == m1["document_max_f1"] == 1
    assert m2["passed"] == m2["total"] == 100 and m2["max_depth"] == 6
    assert m2["trace_pass_rate"] == 1 and m2["explain_score"] == pytest.approx(8 / 15)
    assert m2["fold_count"] == 5 and m2["fold_size"] == 20 and m2["fold_scores"] == pytest.approx([8 / 15] * 5)
    assert m2["available_cases_score"] == pytest.approx(8 / 15) and m2["coverage"] == 1 and not m2["provisional"]


def test_openai_compatible_provider_anchors_exact_source_spans():
    text = "Датчик обнаружил перегрев насоса. Перегрев насоса вызвал остановку агрегата."
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    provider._completion = lambda _: {"choices": [{"message": {"content": '{"facts":[{"predicate":"CAUSE","bindings":[{"role_id":"SUBJECT","value":"перегрев насоса"},{"role_id":"OBJECT","value":"остановка агрегата"}],"quote":"Перегрев насоса вызвал остановку агрегата","confidence":0.91}]}'}}]}
    candidate = provider.extract(text)[0]
    assert text[candidate.source_start:candidate.source_end] == candidate.exact_text
    assert candidate.model_id == "test-model" and candidate.predicate == "CAUSE"


def test_provider_conservatively_realigns_punctuation_and_repairs_or_term():
    text = "Модуль находится в цехе А или в резервном зале."
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    provider._completion = lambda _: {"choices": [{"message": {"content": json.dumps({"mentions": [{"id": "x", "span_index": 0, "text": "Модуль", "type": "entity", "canonical_label": "модуль", "coref_to": None}, {"id": "a", "span_index": 0, "text": "цехе А", "type": "location", "canonical_label": "цех А", "coref_to": None}, {"id": "b", "span_index": 0, "text": "резервном зале", "type": "location", "canonical_label": "резервный зал", "coref_to": None}], "facts": [{"span_index": 0, "predicate": "LOCATED_AT", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["x"]}}, {"role_id": "LOCATION", "term": {"operator": "ATOM", "mention_ids": ["a", "b"]}}], "quote": "Модуль совсем в другом месте.", "confidence": .9, "unresolved_entities": False, "section_hint": "P", "section_confidence": .8, "section_reason": "test"}]}, ensure_ascii=False)}}]}
    candidate = provider.extract(text)[0]
    assert candidate.exact_text == text
    assert candidate.bindings[1].term.operator == "OR"


def test_provider_grounds_inflected_mention_to_exact_source_tokens():
    text = "Витрина ВТ-7 имеет бронзовую раму."
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    provider._completion = lambda _: {"choices": [{"message": {"content": json.dumps({"mentions": [{"id": "x", "span_index": 0, "text": "Витрина ВТ-7", "type": "entity", "canonical_label": "витрина ВТ-7", "coref_to": None}, {"id": "p", "span_index": 0, "text": "бронзовая рама", "type": "state", "canonical_label": "бронзовая рама", "coref_to": None}], "facts": [{"span_index": 0, "predicate": "HAS_STATE", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["x"]}}, {"role_id": "STATE", "term": {"operator": "ATOM", "mention_ids": ["p"]}}], "quote": text, "confidence": .9, "unresolved_entities": False, "section_hint": "P", "section_confidence": .8, "section_reason": "test"}]}, ensure_ascii=False)}}]}
    candidate = provider.extract(text)[0]
    mention = next(item for item in candidate.mentions if item.mention_id == "p")
    assert mention.observed_text == "бронзовую раму"
    accepted, rejected = canonicalize_candidates(text, [candidate])
    assert accepted and not rejected


def test_canonicalizer_splits_atomic_and_and_resolves_unique_short_entity():
    text = "Инженер Орлова остановила турбину. Инженер использовала виброметр и ключ."
    mentions = (
        CandidateMention(mention_id="named", observed_text="Инженер Орлова", source_start=0, source_end=14, mention_type="entity", canonical_label="инженер Орлова"),
        CandidateMention(mention_id="short", observed_text="Инженер", source_start=35, source_end=42, mention_type="entity", canonical_label="инженер"),
        CandidateMention(mention_id="first", observed_text="виброметр", source_start=56, source_end=65, mention_type="entity", canonical_label="виброметр"),
        CandidateMention(mention_id="second", observed_text="ключ", source_start=68, source_end=72, mention_type="entity", canonical_label="ключ"),
    )
    fact = CandidateFact(predicate="USES_TOOL", bindings=(CandidateBinding(role_id="SUBJECT", term=CandidateTerm(mention_ids=("short",))), CandidateBinding(role_id="TOOL", term=CandidateTerm(operator="AND", mention_ids=("first", "second")))), source_start=35, source_end=len(text), exact_text=text[35:], confidence=.9, mentions=mentions)
    accepted, rejected = canonicalize_candidates(text, [fact])
    assert not rejected and len(accepted) == 2
    assert {next(binding.value for binding in item.bindings if binding.role_id == "SUBJECT") for item in accepted} == {"инженер Орлова"}
    assert {next(binding.value for binding in item.bindings if binding.role_id == "TOOL") for item in accepted} == {"виброметр", "ключ"}


def test_openai_compatible_provider_reports_truncated_structured_output():
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    provider._completion = lambda _: {"choices": [{"finish_reason": "length", "message": {"content": '{"facts":['}}]}
    with pytest.raises(ValueError, match="truncated at AH_LLM_MAX_TOKENS"):
        provider.extract("Наблюдение")

def test_openai_compatible_provider_chunks_long_documents_with_absolute_offsets():
    text = " ".join(f"Событие {index} произошло." for index in range(9))
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    calls = []
    def completion(chunk):
        calls.append(chunk)
        quote = chunk.split(".", 1)[0] + "."
        return {"choices": [{"message": {"content": json.dumps({"facts": [{"predicate": "HAS_STATE", "bindings": [{"role_id": "SUBJECT", "value": quote[:-1]}, {"role_id": "STATE", "value": "произошло"}], "quote": quote, "confidence": .9}]}, ensure_ascii=False)}}]}
    provider._completion = completion
    candidates = provider.extract(text)
    assert len(calls) == 2 and len(candidates) == 2
    assert candidates[0].source_start == 0
    assert candidates[1].source_start == text.index("Событие 5")
    assert provider.last_warnings[-1] == "document parsed in 2 overlapping chunks"

def test_openai_compatible_provider_requests_strict_json_schema():
    provider = OpenAICompatibleProvider("http://local.test/v1", "test-model")
    requests = []
    provider._request_json = lambda payload: requests.append(payload) or {"choices": [{"message": {"content": '{"facts":[]}'}}]}
    provider._completion("Наблюдение")
    response_format = requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    required_fact_fields = response_format["json_schema"]["schema"]["properties"]["facts"]["items"]["required"]
    assert {"span_index", "section_hint", "section_confidence", "section_reason"} <= set(required_fact_fields)
    assert requests[0]["max_tokens"] == 8192 and "reasoning" not in requests[0]
    deepseek = OpenAICompatibleProvider("http://local.test/v1", "deepseek/deepseek-v4-flash-0731:nitro")
    deepseek_requests = []
    deepseek._request_json = lambda payload: deepseek_requests.append(payload) or {"choices": [{"message": {"content": '{"facts":[]}'}}]}
    deepseek._completion("Наблюдение")
    assert deepseek_requests[0]["reasoning"] == {"effort": "none", "exclude": True}
    json_provider = OpenAICompatibleProvider("http://local.test/v1", "test-model", response_format="json_object")
    json_requests = []
    json_provider._request_json = lambda payload: json_requests.append(payload) or {"choices": [{"message": {"content": '{"facts":[]}'}}]}
    json_provider._completion("Наблюдение")
    assert json_requests[0]["response_format"] == {"type": "json_object"}

def test_preview_and_candidate_decisions_are_server_side(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "rule")
    c = TestClient(app)
    text = "Перегрев насоса вызвал остановку агрегата. Затем оператор применил ручной ключ."
    response = c.post("/api/v1/ingestions/preview", json={"text": text})
    assert response.status_code == 200
    preview = response.json(); assert preview["provider"]["active"] == "rule" and len(preview["candidates"]) == 2
    assert [(group["start"], group["end"]) for group in preview["source_groups"]] == [(span.start, span.end) for span in segment_text(text)]
    before = c.get("/api/v1/memory/stats").json()["revision"]
    rejected = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][1]["candidate_uid"], "decision": "reject"})
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"
    admitted = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][0]["candidate_uid"], "decision": "admit"})
    assert admitted.status_code == 200 and admitted.json()["accepted"] and c.get("/api/v1/memory/stats").json()["revision"] > before
    reused = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": preview["candidates"][0]["candidate_uid"], "decision": "admit"})
    assert reused.json()["reused"] is True

def test_auto_admission_processes_only_pending_candidates(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "rule")
    c = TestClient(app)
    text = "Перегрев турбины вызвал остановку генератора. Затем техник применил ручной ключ."
    preview = c.post("/api/v1/ingestions/preview", json={"text": text}).json()
    first, second = (item["candidate_uid"] for item in preview["candidates"])
    manual = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": second, "decision": "reject"})
    assert manual.status_code == 200
    admitted = c.post("/api/v1/ingestions/auto-admit", json={"preview_uid": preview["preview_uid"]})
    assert admitted.status_code == 200
    result = admitted.json()
    assert result["processed"] == 1 and result["admitted"] == 1 and result["rejected"] == 1 and result["pending"] == 0
    assert next(item for item in result["candidates"] if item["candidate_uid"] == first)["status"] == "admitted"
    dump = c.post("/api/v1/memory/export", json={}).json()["dump"]
    assert any(item["uid"] in result["results"][0]["accepted"] for item in dump["H"])
    repeated = c.post("/api/v1/ingestions/auto-admit", json={"preview_uid": preview["preview_uid"]}).json()
    assert repeated["processed"] == 0 and repeated["reused"] is True

def test_memory_router_classifies_mixed_document_without_splitting_it():
    common = CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value="датчик температуры"), CandidateBinding(role_id="OBJECT", value="средство контроля")), source_start=0, source_end=52, exact_text="Датчик температуры относится к средствам контроля.", confidence=.9)
    private = CandidateFact(predicate="HAS", bindings=(CandidateBinding(role_id="SUBJECT", value="компрессор К-17"), CandidateBinding(role_id="OBJECT", value="датчик ДТ-4")), source_start=53, source_end=89, exact_text="Компрессор К-17 имеет датчик ДТ-4.", confidence=.9)
    history = CandidateFact(predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value="перегрев ДТ-4"), CandidateBinding(role_id="OBJECT", value="аварийный сигнал")), source_start=90, source_end=133, exact_text="Перегрев ДТ-4 вызвал аварийный сигнал.", confidence=.9)
    purpose = CandidateFact(predicate="PURPOSE", bindings=(CandidateBinding(role_id="SUBJECT", value="компрессор К-17"), CandidateBinding(role_id="PURPOSE", value="подача воздуха")), source_start=134, source_end=170, exact_text="используется для подачи воздуха.", confidence=.9, section_hint="C")
    transient = CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="датчик ДТ-4"), CandidateBinding(role_id="STATE", value="горячее состояние")), source_start=171, source_end=201, exact_text="находится в горячем состоянии", confidence=.9, section_hint="P")
    routed = route_candidates([common, private, history, purpose, transient])
    assert [candidate.section_hint for candidate in routed] == ["C", "P", "H", "P", "H"]

def test_compiler_populates_s_c_p_h_and_builds_episode_follow_dag():
    parts = [
        "Датчик температуры относится к средствам контроля.",
        "Компрессор К-17 имеет датчик ДТ-4.",
        "Перегрев ДТ-4 вызвал аварийный сигнал.",
        "Аварийный сигнал вызвал остановку К-17.",
    ]
    text = " ".join(parts); starts = []; cursor = 0
    for part in parts: starts.append(cursor); cursor += len(part) + 1
    candidates = [
        CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value="датчик температуры"), CandidateBinding(role_id="OBJECT", value="средство контроля")), source_start=starts[0], source_end=starts[0] + len(parts[0]), exact_text=parts[0], confidence=.9, section_hint="C"),
        CandidateFact(predicate="HAS", bindings=(CandidateBinding(role_id="SUBJECT", value="компрессор К-17"), CandidateBinding(role_id="OBJECT", value="датчик ДТ-4")), source_start=starts[1], source_end=starts[1] + len(parts[1]), exact_text=parts[1], confidence=.9, section_hint="P"),
        CandidateFact(predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value="перегрев ДТ-4"), CandidateBinding(role_id="OBJECT", value="аварийный сигнал")), source_start=starts[2], source_end=starts[2] + len(parts[2]), exact_text=parts[2], confidence=.9, section_hint="H"),
        CandidateFact(predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value="аварийный сигнал"), CandidateBinding(role_id="OBJECT", value="остановка К-17")), source_start=starts[3], source_end=starts[3] + len(parts[3]), exact_text=parts[3], confidence=.9, section_hint="H"),
    ]
    memory = AHMemory(); accepted, rejected = Compiler(memory).compile(candidates, text, "mixed_document")
    assert len(accepted) == 4 and not rejected
    assert memory.find_abstract_symbols("компрессор К-17")
    assert sum(isinstance(element.payload, Hypernode) for element in memory.sections["C"].values()) == 1
    assert sum(isinstance(element.payload, Hypernode) for element in memory.sections["P"].values()) == 1
    history_facts = [element.payload for element in memory.sections["H"].values() if isinstance(element.payload, Hypernode)]
    episodes = memory.find_lists(list_type="Episode")
    assert len(history_facts) == 2 and len(episodes) == 1 and len(episodes[0].ordered_members) == 2
    assert any(link.type_id == "FOLLOW" and link.source_ref.target_uid == history_facts[0].uid and link.target_ref.target_uid == history_facts[1].uid for link in memory.links.values())

def test_manual_section_correction_is_applied_before_admission(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "rule")
    c = TestClient(app)
    preview = c.post("/api/v1/ingestions/preview", json={"text": "Перегрев редуктора вызвал остановку стенда."}).json()
    candidate = preview["candidates"][0]
    result = c.post("/api/v1/ingestions/decision", json={"preview_uid": preview["preview_uid"], "candidate_uid": candidate["candidate_uid"], "decision": "admit", "section_override": "P"}).json()
    dump = c.post("/api/v1/memory/export", json={}).json()["dump"]
    assert result["section"] == "P" and any(item["uid"] in result["accepted"] for item in dump["P"])

def test_preview_reports_source_spans_silently_missed_by_provider():
    text = "Насос вызвал остановку. Двигатель принадлежит компании Промтех."
    first, second = segment_text(text)
    class SparseProvider(RuleBasedProvider):
        def extract(self, _):
            return [CandidateFact(predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value="насос"), CandidateBinding(role_id="OBJECT", value="остановка")), source_start=first.start, source_end=first.end, exact_text=first.text, confidence=.9, span_uid=first.uid, group_uid=first.uid)]
    candidates, meta = extract_candidates(text, SparseProvider())
    assert len(candidates) == 1
    assert meta["coverage_warnings"] == [{"group_uid": second.uid, "source_start": second.start, "source_end": second.end, "exact_text": second.text, "reason": "uncovered_source_span", "message": "parser returned no reviewable fact for this source span"}]

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

def test_retrieval_keeps_only_question_items_as_seeds():
    m = AHMemory(); symbol(m, "s1", "a"); symbol(m, "s2", "b"); template(m); hyper(m)
    snap, seeds = build_candidate_snapshot(m, ["s1"])
    assert seeds == ["s1"] and "h_test" in snap.elements
    prepared = IgnitionEngine().run(snap, seeds, IgnitionConfig(max_ticks=3))
    assert any(t.source_uid == "s1" and t.target_uid == "h_test" and t.impulse_type == "role_to_hypernode" for t in prepared.run.ticks)
    assert any(t.source_uid == "h_test" and t.target_uid == "s2" and t.impulse_type == "hypernode" for t in prepared.run.ticks)

def test_retrieval_and_ignition_follow_three_hypernodes_backwards_from_effect():
    m = AHMemory()
    for uid_, label in (("s1", "перегрев"), ("s2", "остановка"), ("s3", "снижение давления"), ("s4", "оповещение")): symbol(m, uid_, label)
    template(m)
    for index, (source, target) in enumerate((("s1", "s2"), ("s2", "s3"), ("s3", "s4")), 1):
        uid_ = f"h{index}"
        m.add_element("C", MemoryElement(uid=uid_, payload=Hypernode(uid=uid_, weight=.9, template_ref=ElementReference(reference_uid=uid_+"_tpl", target_uid="tpl_test"), role_bindings=(RoleBinding(role_id="SUBJECT", target_ref=SReference(reference_uid=uid_+"_s", target_uid=source)), RoleBinding(role_id="OBJECT", target_ref=SReference(reference_uid=uid_+"_o", target_uid=target))))))
    snapshot, seeds = build_candidate_snapshot(m, ["s4"])
    assert seeds == ["s4"] and {"h1", "h2", "h3"} <= set(snapshot.elements)
    run = IgnitionEngine().run(snapshot, seeds, IgnitionConfig(max_ticks=12, working_memory_threshold=.15)).run
    assert {"h1", "h2", "h3", "s1", "s2", "s3", "s4"} <= set(run.minimal_path)
    assert any(trace.target_uid == "h1" and trace.impulse_type == "role_to_hypernode" for trace in run.ticks)

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
    assert len([uid_ for uid_ in q["answer_path"] if uid_.startswith("h_")]) == 1
    e = c.post("/api/v1/evaluations").json(); assert e["metrics"]["M4"]["status"] == "unavailable" and e["metrics"]["M5"]["status"] == "unavailable"

def test_independent_three_hop_document_reaches_all_grounded_facts_from_final_effect():
    c = TestClient(app)
    text = "Короткое замыкание вызвало отключение сервера. Отключение сервера привело к остановке обработки. Остановка обработки вызвала задержку отчёта."
    ingested = c.post("/api/v1/ingestions", json={"document_uid": "three_hop_server_chain", "text": text}).json()
    assert len(ingested["accepted"]) + sum(item["reason"] == "duplicate_memory_fact" for item in ingested["rejected"]) >= 3
    result = c.post("/api/v1/queries", json={"question": "Почему задержался отчёт?", "max_ticks": 14}).json()
    assert result["status"] == "answered" and result["trace_complete"]
    assert result["grounded_fact_count"] >= 3
    assert len([uid_ for uid_ in result["minimal_path"] if uid_.startswith("h_")]) >= 3
    assert sum(item["impulse_type"] == "role_to_hypernode" for item in result["trace"]) >= 3

def test_m3_fixture_and_n1000_tick_benchmark():
    from app.evaluation import internal_m3
    m3 = internal_m3()
    assert m3["case_count"] == 10 and m3["total_examined_nodes"] == 4930
    assert m3["orphan_nodes_before_gc"] == m3["deleted"] == 4100
    assert m3["orphan_nodes_after_gc"] == m3["false_deletions"] == 0
    assert m3["gc_efficiency"] == m3["min_gc_efficiency"] == 1
    assert m3["live_nodes_before"] == m3["live_nodes_after"] == 830
    assert len(m3["official_200_cases"]) == 2 and m3["zero_weight_conformance"]["passed"]
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
    assert result["answer_mode"] == "deterministic_grounded" and result["grounded_fact_count"] >= 1
    assert "Подтверждено AH-памятью" not in result["answer"] and result["answer"].count("агрегат") == 1
    assert "останов" in result["answer"].lower() and "перегрев" in result["answer"].lower()
    assert result["evidence"][0]["exact_text"] in "Перегрев насоса вызвал остановку агрегата."

def test_evidence_answer_validator_requires_known_citation_on_every_sentence():
    facts = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]
    valid = validate_evidence_answer({"status": "answered", "answer": "Агрегат остановился из-за перегрева [E1].", "evidence_ids": ["E1"]}, facts)
    assert valid["evidence_ids"] == ["E1"]
    with pytest.raises(ValueError): validate_evidence_answer({"status": "answered", "answer": "Это внешний вывод [E3].", "evidence_ids": ["E3"]}, facts)
    with pytest.raises(ValueError): validate_evidence_answer({"status": "answered", "answer": "Первое. [E1] Второе без ссылки.", "evidence_ids": ["E1"]}, facts)

def test_external_evidence_answer_has_small_budget_and_validated_citations(monkeypatch):
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash-0731:nitro")
    requests = []
    provider._request_json = lambda payload: requests.append(payload) or {"choices": [{"message": {"content": '{"status":"answered","answer":"Остановка вызвана перегревом [E1].","evidence_ids":["E1"]}'}}]}
    monkeypatch.setattr("app.answering.configured_provider", lambda _: (provider, {"model_id": provider.model_id}))
    result = generate_evidence_answer("Почему остановка?", [{"evidence_id": "E1", "predicate": "CAUSE", "bindings": {"SUBJECT": "перегрев", "OBJECT": "остановка"}, "source_quotes": ["Перегрев вызвал остановку."]}], provider.model_id, "Перегрев вызвал остановку.")
    assert result["status"] == "answered" and result["evidence_ids"] == ["E1"]
    assert requests[0]["max_tokens"] == 2048 and requests[0]["reasoning"]["effort"] == "none"
    assert requests[0]["messages"][1]["content"].find("GROUNDED_DRAFT") > 0

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
    isa_text = "Заяц — зверёк."
    isa = CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value="Заяц"), CandidateBinding(role_id="VALUE", value="зверёк")), source_start=0, source_end=len(isa_text), exact_text=isa_text, confidence=.8)
    accepted, rejected = canonicalize_candidates(isa_text, [isa])
    assert not rejected and [binding.role_id for binding in accepted[0].bindings] == ["SUBJECT", "OBJECT"]
    state_text = "Заяц быстрый."
    state = CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="Заяц"), CandidateBinding(role_id="VALUE", value="быстрый")), source_start=0, source_end=len(state_text), exact_text=state_text, confidence=.8)
    accepted, rejected = canonicalize_candidates(state_text, [state])
    assert not rejected and [binding.role_id for binding in accepted[0].bindings] == ["SUBJECT", "STATE"]

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

def test_deepseek_ui_preset_reuses_server_key_and_enforces_json_schema(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "auto")
    monkeypatch.setenv("AH_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("AH_LLM_API_KEY", "server-secret")
    provider, meta = ingestion_module.configured_provider("~deepseek/deepseek-v4-flash-latest")
    assert provider.model_id == "~deepseek/deepseek-v4-flash-latest"
    assert provider.response_format == "json_schema" and provider.api_key == "server-secret"
    assert meta["configured"] == "ui_override"

def test_ox_alpha_ui_preset_uses_openrouter_json_mode(monkeypatch):
    monkeypatch.setenv("AH_PARSER_PROVIDER", "auto")
    monkeypatch.setenv("AH_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("AH_LLM_API_KEY", "server-secret")
    provider, meta = ingestion_module.configured_provider("stealth/ox-alpha")
    assert provider.model_id == "stealth/ox-alpha"
    assert provider.response_format == "json_object" and provider.api_key == "server-secret"
    assert meta["configured"] == "ui_override"

def test_semantic_canonicalization_repairs_function_follow_and_time():
    function_text = "Компрессор используется для подачи воздуха в производственную линию"
    function = CandidateFact(predicate="USES_TOOL", bindings=(CandidateBinding(role_id="SUBJECT", value="компрессор"), CandidateBinding(role_id="OBJECT", value="подача воздуха"), CandidateBinding(role_id="TOOL", value="производственная линия")), source_start=0, source_end=len(function_text), exact_text=function_text, confidence=.9)
    accepted, rejected = canonicalize_candidates(function_text, [function])
    assert not rejected and accepted[0].predicate == "PURPOSE"
    assert {binding.role_id: binding.value for binding in accepted[0].bindings}["PURPOSE"] == "подачи воздуха в производственную линию"

    follow_text = "После восстановления давления аварийный сигнал отключился"
    earlier, later = "восстановления давления", "аварийный сигнал"
    earlier_start, later_start = follow_text.index(earlier), follow_text.index(later)
    mentions = (CandidateMention(mention_id="before", observed_text=earlier, source_start=earlier_start, source_end=earlier_start + len(earlier), mention_type="event", canonical_label="восстановление давления"), CandidateMention(mention_id="after", observed_text=later, source_start=later_start, source_end=later_start + len(later), mention_type="event", canonical_label="аварийный сигнал"))
    follow = CandidateFact(predicate="FOLLOW", bindings=(CandidateBinding(role_id="SUBJECT", term=CandidateTerm(mention_ids=("after",))), CandidateBinding(role_id="OBJECT", term=CandidateTerm(mention_ids=("before",)))), source_start=0, source_end=len(follow_text), exact_text=follow_text, confidence=.9, mentions=mentions)
    accepted, rejected = canonicalize_candidates(follow_text, [follow])
    bindings = {binding.role_id: binding.value for binding in accepted[0].bindings}
    assert not rejected and bindings == {"SUBJECT": "восстановления давления", "OBJECT": "аварийный сигнал отключился"}

    state_text = "компрессор остался остановлен до завершения проверки."
    state = CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="компрессор"), CandidateBinding(role_id="STATE", value="остановленное состояние")), source_start=0, source_end=len(state_text), exact_text=state_text, confidence=.9)
    accepted, rejected = canonicalize_candidates(state_text, [state])
    assert not rejected and {binding.role_id: binding.value for binding in accepted[0].bindings}["TIME"] == "до завершения проверки"

def test_semantic_gate_uses_action_without_abusing_run_or_observed():
    action_text = "Марина добавила ягодное пюре."
    wrong_run = CandidateFact(predicate="RUN", bindings=(CandidateBinding(role_id="SUBJECT", value="Марина"), CandidateBinding(role_id="HOW-TO", value="ягодное пюре")), source_start=0, source_end=len(action_text), exact_text=action_text, confidence=.9)
    accepted, rejected = canonicalize_candidates(action_text, [wrong_run])
    assert not rejected and accepted[0].predicate == "ACTION"
    assert {binding.role_id: binding.value for binding in accepted[0].bindings} == {"SUBJECT": "Марина", "OBJECT": "добавила ягодное пюре"}

    observation_text = "За релиз отвечает команда Норд."
    observation = CandidateFact(predicate="OBSERVED", bindings=(CandidateBinding(role_id="SUBJECT", value="команда Норд"), CandidateBinding(role_id="OBJECT", value="релиз")), source_start=0, source_end=len(observation_text), exact_text=observation_text, confidence=.8)
    accepted, rejected = canonicalize_candidates(observation_text, [observation])
    assert not accepted and rejected[0]["reason"] == "invalid_observation"

    hallucinated = CandidateFact(predicate="CAUSE", bindings=(CandidateBinding(role_id="SUBJECT", value="откат"), CandidateBinding(role_id="OBJECT", value="падение интеграционных тестов")), source_start=0, source_end=len("Откат восстановил тесты."), exact_text="Откат восстановил тесты.", confidence=.9)
    accepted, rejected = canonicalize_candidates("Откат восстановил тесты.", [hallucinated])
    assert not accepted and rejected[0]["reason"] == "ungrounded_role"

def test_identifier_grounding_and_local_property_answer():
    assert lexical_score("Какие свойства у датчика ДТ4?", "датчик температуры ДТ-4") == 2
    memory = AHMemory()
    text = "Датчик ДТ-4 имеет красный корпус и находится в горячем состоянии."
    facts = [
        CandidateFact(predicate="HAS", bindings=(CandidateBinding(role_id="SUBJECT", value="датчик температуры ДТ-4"), CandidateBinding(role_id="OBJECT", value="красный корпус")), source_start=0, source_end=len(text), exact_text=text, confidence=.9),
        CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value="датчик температуры ДТ-4"), CandidateBinding(role_id="STATE", value="горячее состояние")), source_start=0, source_end=len(text), exact_text=text, confidence=.9),
    ]
    accepted, rejected = Compiler(memory).compile(facts, text, "sensor-properties")
    answer, selected = select_local_answer(memory, "Какие свойства у датчика ДТ4?", tuple(accepted))
    assert not rejected and len(selected) == 2
    assert "красный корпус" in answer and "горячее состояние" in answer

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

def test_ir_v3_general_composition_coreference_and_atomic_groups():
    text = "Двигатель установлен в цехе A или цехе B. У него есть датчик. Датчик красный и горячий. Перегрев вызвал остановку."
    raw = {
        "mentions": [
            {"id": "motor", "text": "Двигатель", "type": "entity", "canonical_label": "двигатель", "coref_to": None},
            {"id": "hall_a", "text": "цехе A", "type": "location", "canonical_label": "цех A", "coref_to": None},
            {"id": "hall_b", "text": "цехе B", "type": "location", "canonical_label": "цех B", "coref_to": None},
            {"id": "owner", "text": "него", "type": "entity", "canonical_label": None, "coref_to": "motor"},
            {"id": "sensor", "text": "датчик", "type": "device", "canonical_label": "датчик", "coref_to": None},
            {"id": "red", "text": "красный", "type": "state", "canonical_label": "красный", "coref_to": None},
            {"id": "hot", "text": "горячий", "type": "state", "canonical_label": "горячий", "coref_to": None},
            {"id": "overheat", "text": "Перегрев", "type": "event", "canonical_label": "перегрев", "coref_to": None},
            {"id": "stop", "text": "остановку", "type": "event", "canonical_label": "остановка", "coref_to": None},
        ],
        "facts": [
            {"predicate": "LOCATED_AT", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["motor"]}}, {"role_id": "LOCATION", "term": {"operator": "OR", "mention_ids": ["hall_a", "hall_b"]}}], "quote": text[0:44], "confidence": .9, "unresolved_entities": False},
            {"predicate": "HAS", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["owner"]}}, {"role_id": "OBJECT", "term": {"operator": "ATOM", "mention_ids": ["sensor"]}}], "quote": "У него есть датчик.", "confidence": .9, "unresolved_entities": False},
            {"predicate": "HAS_STATE", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["sensor"]}}, {"role_id": "STATE", "term": {"operator": "ATOM", "mention_ids": ["red"]}}], "quote": "Датчик красный и горячий.", "confidence": .85, "unresolved_entities": False},
            {"predicate": "HAS_STATE", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["sensor"]}}, {"role_id": "STATE", "term": {"operator": "ATOM", "mention_ids": ["hot"]}}], "quote": "Датчик красный и горячий.", "confidence": .85, "unresolved_entities": False},
            {"predicate": "CAUSE", "bindings": [{"role_id": "SUBJECT", "term": {"operator": "ATOM", "mention_ids": ["overheat"]}}, {"role_id": "OBJECT", "term": {"operator": "ATOM", "mention_ids": ["stop"]}}], "quote": "Перегрев вызвал остановку.", "confidence": .95, "unresolved_entities": False},
        ],
    }
    provider = OpenAICompatibleProvider("http://local.test/v1", "ir-v3-general")
    provider._completion = lambda _: {"choices": [{"message": {"content": json.dumps(raw, ensure_ascii=False)}}]}
    candidates, meta = extract_candidates(text, provider)
    memory = AHMemory(); accepted, rejected = Compiler(memory).compile(candidates, text, "general_ir_v3")
    functions = [element.payload for element in memory.elements.values() if isinstance(element.payload, FunctionalSymbol)]
    assert not meta["rejection_log"] and not rejected and len(accepted) == 5
    assert len({candidate.group_uid for candidate in candidates}) == 4
    assert any(function.function_id == "OR" and len(function.ordered_operands) == 2 for function in functions)
    has = next(candidate for candidate in candidates if candidate.predicate == "HAS")
    assert {binding.role_id: binding.value for binding in has.bindings}["SUBJECT"] == "двигатель"

def test_ir_v3_rejects_forward_and_cyclic_coreference():
    text = "Он увидел датчик."
    mentions = (
        CandidateMention(mention_id="a", observed_text="Он", source_start=0, source_end=2, coref_to="b"),
        CandidateMention(mention_id="b", observed_text="датчик", source_start=10, source_end=16, coref_to="a"),
    )
    candidate = CandidateFact(predicate="OBSERVED", bindings=(CandidateBinding(role_id="SUBJECT", term=CandidateTerm(mention_ids=("a",))), CandidateBinding(role_id="OBJECT", term=CandidateTerm(mention_ids=("b",)))), source_start=0, source_end=len(text), exact_text=text, confidence=.8, mentions=mentions)
    accepted, rejected = canonicalize_candidates(text, [candidate])
    assert not accepted and rejected[0]["reason"] in {"forward_coreference", "coreference_cycle"}

def test_general_recovery_for_coordinated_state_and_explicit_follow():
    properties = _coordinated_property_candidates("Этикетка E-9 имеет синий фон и находится в повреждённом состоянии.")
    assert [(item.predicate, {binding.role_id: binding.value for binding in item.bindings}) for item in properties] == [
        ("HAS", {"SUBJECT": "Этикетка E-9", "OBJECT": "синий фон"}),
        ("HAS_STATE", {"SUBJECT": "Этикетка E-9", "STATE": "повреждённом состоянии"}),
    ]
    simple_states = _coordinated_property_candidates("Датчик двигателя красный и горячий.")
    assert [{binding.role_id: binding.value for binding in item.bindings} for item in simple_states] == [
        {"SUBJECT": "Датчик двигателя", "STATE": "красный"},
        {"SUBJECT": "Датчик двигателя", "STATE": "горячий"},
    ]
    follow = _explicit_follow_candidates("После падения тестов инженер Ли откатил изменение схемы.")
    assert {binding.role_id: binding.value for binding in follow[0].bindings} == {"SUBJECT": "падения тестов", "OBJECT": "инженер Ли откатил изменение схемы"}
    assert not _explicit_follow_candidates("После этого инженер выполнил откат.")
    temporal = _temporal_state_candidates("Наблюдение оставалось приостановленным до следующего окна связи.")
    assert {binding.role_id: binding.value for binding in temporal[0].bindings} == {"SUBJECT": "Наблюдение", "STATE": "приостановленным", "TIME": "до следующего окна связи"}
    locations = _stable_fact_candidates("Кластер СК-31 размещён в зале Б или в резервной зоне.")
    assert {binding.role_id: binding.value for binding in locations[0].bindings} == {"SUBJECT": "Кластер СК-31", "LOCATION": "OR(зале Б, резервной зоне)"}

def test_general_declarative_recovery_is_not_rabbit_lexicon_specific():
    text = "Барсук — лесной зверь. У него мощные лапы, поэтому бегает он довольно быстро."
    facts = _declarative_recovery_candidates(text)
    assert [(fact.predicate, {binding.role_id: binding.value for binding in fact.bindings}) for fact in facts] == [
        ("IS-A", {"SUBJECT": "Барсук", "OBJECT": "лесной зверь"}),
        ("HAS", {"SUBJECT": "Барсук", "OBJECT": "мощные лапы"}),
        ("RUN", {"SUBJECT": "Барсук", "HOW-TO": "довольно быстро"}),
    ]
    tool = _tool_use_candidates("Исследователь Ким использовал анализатор ZX-8 для измерения спектра.")
    assert {binding.role_id: binding.value for binding in tool[0].bindings} == {"SUBJECT": "Исследователь Ким", "TOOL": "анализатор ZX-8"}

    stable = _stable_fact_candidates("Модуль R-2 принадлежит лаборатории «Вега» и используется для очистки сигнала. У модуля R-2 есть фильтр F-1.")
    assert [(fact.predicate, {binding.role_id: binding.value for binding in fact.bindings}) for fact in stable] == [
        ("HAS", {"SUBJECT": "лаборатории «Вега»", "OBJECT": "Модуль R-2"}),
        ("PURPOSE", {"SUBJECT": "Модуль R-2", "PURPOSE": "очистки сигнала"}),
        ("HAS", {"SUBJECT": "модуля R-2", "OBJECT": "фильтр F-1"}),
    ]
    effects = _effect_candidates("Калибровка уменьшила шум и повысила точность спектра.")
    assert [{binding.role_id: binding.value for binding in fact.bindings} for fact in effects] == [
        {"SUBJECT": "Калибровка", "OBJECT": "уменьшила шум"},
        {"SUBJECT": "Калибровка", "OBJECT": "повысила точность спектра"},
    ]

def test_functional_label_and_leading_citation_cleanup():
    memory = AHMemory()
    symbol(memory, "s_a", "цех А"); symbol(memory, "s_b", "резервный зал")
    memory.add_element("C", MemoryElement(uid="fn_or", payload=FunctionalSymbol(uid="fn_or", function_id="OR", ordered_operands=(SReference(reference_uid="r_a", target_uid="s_a"), SReference(reference_uid="r_b", target_uid="s_b")))))
    assert memory.label("fn_or") == "цех А или резервный зал"
    result = validate_evidence_answer({"status": "answered", "answer": "[E1] Причина подтверждена [E1].", "evidence_ids": ["E1"]}, [{"evidence_id": "E1"}])
    assert result["answer"] == "Причина подтверждена [E1]."
