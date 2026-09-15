import json
import pytest
from app import ingestion
from app.benchmark_recording import document_predictions


@pytest.mark.parametrize("perception_format", ["ir", "roles"])
def test_record_and_replay_normal_pipeline_without_network(tmp_path, monkeypatch, perception_format):
    text = "Перегрев насоса вызвал остановку агрегата."
    provider = ingestion.OpenAICompatibleProvider("https://unused.invalid/v1", "test-model", "secret-not-recorded", perception_format=perception_format)
    response = {"choices": [{"message": {"content": json.dumps({"facts": [{"predicate": "CAUSE", "bindings": [{"role_id": "SUBJECT", "value": "Перегрев насоса"}, {"role_id": "OBJECT", "value": "остановку агрегата"}], "quote": text, "confidence": .9}]}, ensure_ascii=False)}}]}
    monkeypatch.setattr(provider, "_request_json", lambda payload: response)
    monkeypatch.setattr(ingestion, "configured_provider", lambda *args: (provider, {"configured": "ui_override", "active": "openai_compatible", "model_id": "test-model", "fallback": False}))
    original, _ = document_predictions(text, tmp_path / "fresh.json", "test-model")
    record = json.loads((tmp_path / "fresh.json").read_text(encoding="utf-8"))
    assert original and record["status"] == "completed" and record["calls"]
    assert record["settings"]["perception_format"] == perception_format
    schema = record["calls"][0]["request"]["response_format"]["json_schema"]["schema"]
    assert ("mentions" in schema["properties"]) == (perception_format == "ir")
    assert "secret-not-recorded" not in json.dumps(record)
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted network")
    monkeypatch.setattr(ingestion, "urlopen", forbidden)
    replayed, _ = document_predictions(text, tmp_path / "replayed.json", "test-model", record)
    assert replayed == original
    with pytest.raises(ValueError, match="source/schema"):
        document_predictions(text + " Изменено.", tmp_path / "wrong.json", "test-model", record)
    record["calls"][0]["request"]["model"] = "another-model"
    with pytest.raises(ValueError, match="request mismatch"):
        document_predictions(text, tmp_path / "wrong-settings.json", "test-model", record)


def test_successful_response_survives_later_failure(tmp_path, monkeypatch):
    provider = ingestion.OpenAICompatibleProvider("https://unused.invalid/v1", "test-model")
    monkeypatch.setattr(ingestion, "configured_provider", lambda *args: (provider, {"configured": "ui_override", "active": "openai_compatible", "model_id": "test-model"}))
    monkeypatch.setattr(provider, "_request_json", lambda payload: {"choices": [{"message": {"content": "not JSON"}}]})
    path = tmp_path / "failed.json"
    with pytest.raises(ValueError): document_predictions("Короткий текст.", path, "test-model")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["calls"][0]["response"]["choices"]
