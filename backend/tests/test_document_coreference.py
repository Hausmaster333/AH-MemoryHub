from fastapi.testclient import TestClient
from app.core import AHMemory
from app.ingestion import RuleBasedProvider, extract_candidates, Compiler
import app.main as api


def test_temporal_reference_has_identity_and_both_source_quotes(monkeypatch):
    text = "Насос НС-81 находится в зале. Насос оставался остановлен до завершения проверки."
    candidates, _ = extract_candidates(text, RuleBasedProvider())
    temporal = next(c for c in candidates if c.predicate == "HAS_STATE")
    subject = next(b for b in temporal.bindings if b.role_id == "SUBJECT")
    assert subject.value == "Насос НС-81" and subject.observed == "Насос" and subject.term
    memory = AHMemory()
    accepted, rejected = Compiler(memory).compile([temporal], text, "one")
    assert accepted and not rejected
    node = memory.get_hypernode(accepted[0])
    assert {e.exact_text for e in node.evidence} == {"Насос НС-81 находится в зале.", "Насос оставался остановлен до завершения проверки."}
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    good = client.post("/api/v1/queries", json={"question": "До какого момента насос НС-81 оставался остановлен?", "profile": "directed_v1"}).json()
    assert good["status"] == "answered" and good["trace_complete"] and len(good["evidence"]) == 2
    bad = client.post("/api/v1/queries", json={"question": "До какого момента насос НС-82 оставался остановлен?", "profile": "directed_v1"}).json()
    assert bad["status"] == "insufficient_evidence"
    anchor = next(m for m in temporal.mentions if m.mention_id.startswith("docref_") and not m.coref_to)
    corrupted = temporal.model_copy(update={"mentions": tuple(m.model_copy(update={"observed_text": "Насос НС-82"}) if m is anchor else m for m in temporal.mentions)})
    accepted, rejected = Compiler(AHMemory()).compile([corrupted], text, "bad")
    assert not accepted and rejected[0]["reason"] == "mention_span_mismatch"


def test_temporal_reference_does_not_cross_ambiguity_paragraph_or_generic_subject():
    first = "Насос НС-81 находится в зале. "
    last = "Насос оставался остановлен до завершения проверки."
    for text in (first + "Насос НС-82 находится в цехе. " + last,
                 first + "Другой насос установлен в цехе. " + last,
                 first + "\n\n" + last,
                 "Если насос НС-81 находится в зале, сигнал включён. " + last,
                 first + "Каждый насос оставался остановлен до завершения проверки.",
                 last + " " + first):
        candidates, _ = extract_candidates(text, RuleBasedProvider())
        temporal = next(c for c in candidates if c.predicate == "HAS_STATE")
        assert not any(m.mention_id.startswith("docref_") for m in temporal.mentions)
