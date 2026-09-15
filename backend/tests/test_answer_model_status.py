import json

import pytest
from fastapi.testclient import TestClient

import app.answering as answering
import app.main as api
from app.core import AHMemory
from app.ingestion import Compiler, OpenAICompatibleProvider, RuleBasedProvider, extract_candidates


@pytest.mark.parametrize("model_result,expected_status", [
    ({"status": "insufficient_evidence", "claims": []}, "insufficient_evidence"),
    ({"status": "answered", "claims": [{"text": "Неподтверждённый вывод", "evidence_ids": ["E99"]}]}, "invalid_response"),
    (None, "invalid_response"),
    ([], "invalid_response"),
    ("malformed_envelope", "invalid_response"),
    ("transport_error", "provider_error"),
])
def test_api_preserves_refusal_and_diagnoses_other_model_failures(monkeypatch, model_result, expected_status):
    text = "Перегрев насоса вызвал остановку агрегата."
    memory = AHMemory()
    candidates, _ = extract_candidates(text, RuleBasedProvider())
    Compiler(memory).compile(candidates, text, "model-status-test")
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "runs", {})
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    provider = OpenAICompatibleProvider("http://unused.invalid", "fake-model")
    def fake_request(_payload):
        if model_result == "transport_error": raise ValueError("fake provider unavailable")
        if model_result == "malformed_envelope": return {"choices": "bad"}
        return {"choices": [{"message": {"content": json.dumps(model_result)}}]}
    monkeypatch.setattr(provider, "_request_json", fake_request)
    monkeypatch.setattr(answering, "configured_provider", lambda _: (provider, {"model_id": "fake-model"}))
    client = TestClient(api.app)
    question = "Почему остановился агрегат?"
    if expected_status != "insufficient_evidence":
        local = client.post("/api/v1/queries", json={"question": question, "answer_model": "local"}).json()
        assert local["status"] == "answered"
    result = client.post("/api/v1/queries", json={"question": question, "answer_model": "configured"}).json()
    assert result["answer_model_status"] == expected_status
    assert result["run_uid"] and result["grounded_fact_count"] >= 1
    if expected_status == "insufficient_evidence":
        assert result["status"] == "insufficient_evidence" and result["answer"] == "insufficient_evidence"
        assert result["answer_mode"] == "llm_refusal" and result["answer_provider"]["model_id"] == "fake-model"
        assert not result["evidence"] and not result["answer_path"] and not result["trace_complete"]
        assert result["answer_fact_count"] == 0 and result["answer_warning"] is None
    else:
        assert result["status"] == "answered" and result["answer"] == local["answer"]
        assert result["answer_mode"] == "deterministic_grounded" and result["evidence"]
        assert result["answer_warning"]
