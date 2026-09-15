import importlib.util
import json
import hashlib
from pathlib import Path
import pytest


def test_qa_regression_corpus():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("evaluate_qa", root / "backend/scripts/evaluate_qa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluate(json.loads((root / "docs/testing/qa-regression.json").read_text(encoding="utf-8")))
    assert report["passed"] == report["total"], [(row["id"], row["checks"]) for row in report["cases"] if not row["passed"]]
    report = module.evaluate(json.loads((root / "docs/testing/qa-independent-v1.json").read_text(encoding="utf-8")))
    assert report["passed"] == report["total"], [(row["id"], row["checks"]) for row in report["cases"] if not row["passed"]]


def test_recorded_qa_checks_source_and_proof(tmp_path):
    from app.ingestion import extract_candidates, RuleBasedProvider
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("evaluate_qa_recorded", root / "backend/scripts/evaluate_qa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = "Засор фильтра вызвал остановку насоса."
    candidates, _ = extract_candidates(source, RuleBasedProvider())
    digest = hashlib.sha256(source.encode()).hexdigest()
    record = {"status": "completed", "source_sha256": digest, "text": source, "predictions": [c.model_dump(mode="json") for c in candidates]}
    (tmp_path / "document.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"documents": [{"id": "one", "file": "document.json"}]}), encoding="utf-8")
    corpus = {"source_sha256": {"one": digest}, "cases": [{"id": "cause", "document_id": "one", "question": "Что вызвало остановку насоса?", "quotes": [source], "answer_contains": ["Засор фильтра"]}]}
    assert module.evaluate_recorded(corpus, tmp_path)["passed"] == 1
    record["text"] = "Другой текст."
    (tmp_path / "document.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="changed source"):
        module.evaluate_recorded(corpus, tmp_path)
