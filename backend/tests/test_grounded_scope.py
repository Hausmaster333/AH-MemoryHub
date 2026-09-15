"""A scope qualifies an event only through a named, source-anchored assertion."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core import AHMemory
from app.ingestion import Compiler, _named_scope_declarations, segment_text
from app.models import CandidateBinding, CandidateFact
import app.main as api


def _memory(text, events):
    memory = AHMemory()
    candidates = [candidate for candidate, _, _ in _named_scope_declarations(text)]
    for quote, subject, clock in events:
        span = next(span for span in segment_text(text) if span.text == quote)
        candidates.append(CandidateFact(
            predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject),
                                              CandidateBinding(role_id="STATE", value="завершилась"),
                                              CandidateBinding(role_id="TIME", value=clock)),
            source_start=span.start, source_end=span.end, exact_text=quote, confidence=1,
            section_hint="H", model_id="test-explicit-event",
        ))
    accepted, rejected = Compiler(memory).compile(candidates, text, "scope-test")
    assert len(accepted) == len(candidates) and not rejected
    return memory


def _ask(memory, question):
    with patch.multiple(api, memory=memory, storage_mode="in-memory", runs={}, ingestions={}, document_index={}):
        response = TestClient(api.app).post("/api/v1/queries", json={"question": question, "profile": "directed_v1", "answer_model": "local"})
        response.raise_for_status()
        return response.json()


def test_named_scope_returns_event_and_assertion_with_complete_proof():
    text = "Акт поверки датчиков\n\nПоверка относится к смене М-11. В 01:20 завершилась сверка датчиков."
    memory = _memory(text, [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20")])
    response = _ask(memory, "Когда завершилась сверка датчиков в смене М-11?")
    assert response["status"] == "answered" and "01:20" in response["answer"]
    assert response["trace_complete"] and len(response["answer_path"]) > 0
    assert {ev["exact_text"] for ev in response["evidence"]} == {
        "Поверка относится к смене М-11.", "В 01:20 завершилась сверка датчиков."}
    assert response["answer_fact_count"] == 2


def test_unrelated_old_shift_mention_does_not_grant_scope():
    text = "Акт поверки датчиков\n\nСтарую смену М-10 упомянули в архиве. Поверка относится к смене М-11. В 01:20 завершилась сверка датчиков."
    event = [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20")]
    assert _ask(_memory(text, event), "Когда завершилась сверка датчиков в смене М-10?")["status"] == "insufficient_evidence"
    assert _ask(_memory(text, event), "Когда завершилась сверка датчиков в смене М-11?")["status"] == "answered"


def test_two_scope_assertions_in_one_section_are_ambiguous():
    text = "Акт поверки датчиков\n\nПоверка относится к смене М-11. Поверка относится к смене М-12. В 01:20 завершилась сверка датчиков."
    memory = _memory(text, [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20")])
    assert not [link for link in memory.links.values() if link.type_id == "IN_SCOPE"]
    for shift in ("М-11", "М-12"):
        assert _ask(memory, f"Когда завершилась сверка датчиков в смене {shift}?")["status"] == "insufficient_evidence"


def test_two_named_sections_keep_their_station_scopes_separate():
    text = ("Акт поверки оборудования\n\nПоверка относится к станции Р-11. В 01:20 завершилась сверка оптики.\n\n"
            "Акт поверки оборудования\n\nПоверка относится к станции Р-12. В 02:30 завершилась сверка насосов.")
    events = [("В 01:20 завершилась сверка оптики.", "сверка оптики", "01:20"),
              ("В 02:30 завершилась сверка насосов.", "сверка насосов", "02:30")]
    assert _ask(_memory(text, events), "Когда завершилась сверка оптики на станции Р-11?")["status"] == "answered"
    assert _ask(_memory(text, events), "Когда завершилась сверка оптики на станции Р-12?")["status"] == "insufficient_evidence"


def test_explicit_conflicting_shift_inside_event_blocks_inheritance():
    text = "Акт поверки датчиков\n\nПоверка относится к смене М-11. В 01:20 завершилась сверка датчиков в смене М-12."
    event = [("В 01:20 завершилась сверка датчиков в смене М-12.", "сверка датчиков", "01:20")]
    memory = _memory(text, event)
    assert not [link for link in memory.links.values() if link.type_id == "IN_SCOPE"]
    assert _ask(memory, "Когда завершилась сверка датчиков в смене М-11?")["status"] == "insufficient_evidence"


def test_unconfirmed_heading_and_historical_event_do_not_inherit_scope():
    unmatched = "Акт поверки датчиков\n\nПередача относится к смене М-11. В 01:20 завершилась сверка датчиков."
    assert not _named_scope_declarations(unmatched)
    assert _ask(_memory(unmatched, [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20")]),
                "Когда завершилась сверка датчиков в смене М-11?")["status"] == "insufficient_evidence"
    historical = "Акт поверки датчиков\n\nПоверка относится к смене М-11. В 01:20 завершилась сверка датчиков. Ранее в старой смене М-12 в 00:10 завершилась сверка архива."
    memory = _memory(historical, [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20"),
                                  ("Ранее в старой смене М-12 в 00:10 завершилась сверка архива.", "сверка архива", "00:10")])
    assert len([link for link in memory.links.values() if link.type_id == "IN_SCOPE"]) == 1
    assert _ask(memory, "Когда завершилась сверка архива в смене М-11?")["status"] == "insufficient_evidence"


def test_reingestion_removes_stale_derived_scope_link():
    text = "Акт поверки датчиков\n\nПоверка относится к смене М-11. В 01:20 завершилась сверка датчиков."
    memory = _memory(text, [("В 01:20 завершилась сверка датчиков.", "сверка датчиков", "01:20")])
    derived = next(link for link in memory.links.values() if link.type_id == "IN_SCOPE")
    manual = derived.model_copy(update={"uid": "manual_scope_reference"})
    memory.add_link(manual)
    changed = "Акт поверки датчиков\n\nПоверка относится к смене М-12. В 01:20 завершилась сверка датчиков."
    Compiler(memory).compile([], changed, "scope-test")
    assert not [link for link in memory.links.values() if link.uid.startswith("l_scope_")]
    assert memory.get_link("manual_scope_reference") == manual
