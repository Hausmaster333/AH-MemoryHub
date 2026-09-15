import pytest
import hashlib
import importlib.util
import json
from pathlib import Path
from app.ingestion import RuleBasedProvider, extract_candidates


@pytest.mark.parametrize("with_rejected", [False, True])
@pytest.mark.parametrize("invalid_response", [False, True])
def test_paired_pilot_replays_without_network_or_gold_leakage(tmp_path, monkeypatch, with_rejected, invalid_response):
    spec = importlib.util.spec_from_file_location("paired", Path(__file__).parents[1] / "scripts/evaluate_ah_rag.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    text = "Клапан К-2 оставался закрыт до завершения проверки."
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    questions = tmp_path / "qa.json"
    questions.write_text(json.dumps({"source_sha256": {"doc": source_hash}, "cases": [{"id": "case", "question": "До какого момента клапан К-2 оставался закрыт?", "answer_contains": ["GOLD_MUST_NOT_LEAK"]}, {"id": "unknown", "question": "Какие свойства имеет датчик ДМ-999?", "answer": None}]}), encoding="utf-8")
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps({"questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(), "cases": [{"id": key, "contexts": [{"document_id": "doc", "start": 0, "end": len(text), "text": text}]} for key in ("case", "unknown")]}), encoding="utf-8")
    facts, _ = extract_candidates(text, RuleBasedProvider())
    if with_rejected: facts.append(facts[0].model_copy(update={"exact_text": "REJECTED_MUST_NOT_LEAK"}))
    (tmp_path / "run.json").write_text(json.dumps({"documents": [{"id": "doc", "file": "doc.json"}]}))
    (tmp_path / "doc.json").write_text(json.dumps({"status": "completed", "settings": {"model": "test"}, "source_sha256": source_hash, "text": text, "predictions": [fact.model_dump(mode="json") for fact in facts]}), encoding="utf-8")
    calls = []
    def request(self, payload):
        assert "GOLD_MUST_NOT_LEAK" not in json.dumps(payload)
        assert "REJECTED_MUST_NOT_LEAK" not in json.dumps(payload)
        calls.append(payload)
        if invalid_response and len(calls) == 1:
            return {"choices": [{"message": {"content": "null"}}]}
        answer = {"status": "insufficient_evidence", "claims": []} if "ДМ-999" in payload["messages"][1]["content"] else {"status": "answered", "claims": [{"text": "До завершения проверки", "evidence_ids": ["E1"]}]}
        return {"choices": [{"message": {"content": json.dumps(answer)}}]}
    monkeypatch.setattr(cli.OpenAICompatibleProvider, "_request_json", request)
    argv = ["paired", "--ah-records", str(tmp_path), "--rag-contexts", str(rag), "--questions", str(questions), "--model", "test", "--out", str(tmp_path / "out.json")]
    monkeypatch.setattr(cli.sys, "argv", argv)
    cli.main()
    assert len(calls) == 4
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert len(report["ingestions"][0]["rejected"]) == int(with_rejected)
    assert report["ingestions"][0]["accepted"] == len(facts) - int(with_rejected)
    monkeypatch.setattr(cli.sys, "argv", argv + ["--resume", "--offline"])
    cli.main()
    assert len(calls) == 4
    rows = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["cases"]
    assert len(rows) == 4
    for row in rows:
        if invalid_response and row["id"] == "case" and row["arm"] == "ah":
            assert row["answer_status"] == "invalid_response" and row["validation_error"] and "validated" not in row
        else:
            assert row["answer_status"] == row["validated"]["status"] == ("answered" if row["id"] == "case" else "insufficient_evidence")
    monkeypatch.setattr(cli.sys, "argv", argv[:-1] + [str(tmp_path / "imported.json"), "--responses", str(tmp_path / "out.json"), "--offline"])
    cli.main()
    assert len(calls) == 4
    def provider_error(_self, _payload): raise ValueError("fake transport failure")
    monkeypatch.setattr(cli.OpenAICompatibleProvider, "_request_json", provider_error)
    failed = tmp_path / "failed.json"
    monkeypatch.setattr(cli.sys, "argv", argv[:-1] + [str(failed)])
    with pytest.raises(ValueError, match="fake transport failure"):
        cli.main()
    failed_row = json.loads(failed.read_text(encoding="utf-8"))["cases"][0]
    assert failed_row["answer_status"] == "provider_error" and failed_row["provider_error"] == "fake transport failure"
