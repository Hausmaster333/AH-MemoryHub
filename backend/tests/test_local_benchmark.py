import importlib.util
import json
import pytest
from pathlib import Path

from app import ingestion


@pytest.mark.parametrize("record_only", [False, True])
def test_local_cli_and_offline_replay(tmp_path, monkeypatch, record_only):
    spec = importlib.util.spec_from_file_location("evaluate_corpus", Path(__file__).parents[1] / "scripts/evaluate_corpus.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = "Перегрев насоса вызвал остановку агрегата."
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"documents": [{"id": "one", "text": source}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "CORPUS_PATH", corpus)
    monkeypatch.setattr(cli, "GOLD_PATH", corpus)
    monkeypatch.setattr(cli, "load_gold", lambda: {"documents": [{"id": "one"}]})
    monkeypatch.setattr(cli, "corpus_role_metrics", lambda *a, **k: {})
    monkeypatch.setattr(cli, "corpus_m2", lambda: {})
    monkeypatch.setattr(cli, "internal_m3", lambda: {})
    def request(provider, payload):
        assert provider.model_id == "local-test:7b"
        assert "127.0.0.1:11434" in provider.endpoint
        return {"choices": [{"message": {"content": json.dumps({"facts": [{"predicate": "CAUSE", "bindings": [{"role_id": "SUBJECT", "value": "Перегрев насоса"}, {"role_id": "OBJECT", "value": "остановку агрегата"}], "quote": source, "confidence": .9}]})}}]}
    monkeypatch.setattr(ingestion.OpenAICompatibleProvider, "_request_json", request)
    extra = ["--record-only", "--corpus", str(corpus)] if record_only else []
    monkeypatch.setattr(cli.sys, "argv", ["evaluate_corpus", "--provider", "local", "--model", "local-test:7b", "--out", str(tmp_path / "fresh.json"), *extra])
    # pytest's capture stream need not expose reconfigure().
    monkeypatch.setattr(cli.sys.stdout, "reconfigure", lambda **k: None, raising=False)
    cli.main()
    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted network")
    monkeypatch.setattr(ingestion.OpenAICompatibleProvider, "_request_json", forbidden)
    monkeypatch.setattr(cli.sys, "argv", ["evaluate_corpus", "--replay", str(tmp_path / "fresh.records"), "--out", str(tmp_path / "replay.json"), *extra])
    cli.main()
    report = json.loads((tmp_path / "replay.json").read_text())
    assert report["provider"] == "local"
    assert ("M1" in report) == (not record_only)
    assert report["providers"]["one"]["model_id"] == "local-test:7b"
    assert json.loads((tmp_path / "replay.records/document-000.json").read_text())["predictions"]

