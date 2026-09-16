import pytest

from app import main as api
from app.core import AHMemory
from app.ingestion import RuleBasedProvider, ingest
from app.models import QueryRequest


@pytest.mark.parametrize("question", (
    "Какова первопричина события «отключение сервера»?",
    "Какова первопричина «отключение сервера»?",
))
def test_primary_cause_target_ignores_question_scaffolding(monkeypatch, question):
    memory = AHMemory()
    ingest(memory, "Короткое замыкание вызвало отключение сервера.", provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    response = api.query(QueryRequest(question=question, answer_model="local"))
    assert response["status"] == "answered"
    assert "короткое замыкание" in response["answer"].lower()
    assert response["trace_complete"]


def test_primary_cause_target_does_not_guess_missing_event(monkeypatch):
    memory = AHMemory()
    ingest(memory, "Короткое замыкание вызвало отключение сервера.", provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    response = api.query(QueryRequest(
        question="Какова первопричина события «несуществующий сбой»?",
        answer_model="local",
    ))
    assert response["status"] == "insufficient_evidence"
    assert response["answer"] == "insufficient_evidence"
    assert not response["trace_complete"]
