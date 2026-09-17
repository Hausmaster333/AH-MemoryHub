from app import ingestion
from app.ingestion import Provider, extract_candidates, segment_text
from app.models import CandidateBinding, CandidateFact


class SparseProvider(Provider):
    model_id = "sparse-test"

    def extract(self, text):
        first = segment_text(text)[0]
        return [CandidateFact(
            predicate="CAUSE",
            bindings=(CandidateBinding(role_id="SUBJECT", value="нагрев"), CandidateBinding(role_id="OBJECT", value="остановка")),
            source_start=first.start, source_end=first.end, exact_text=first.text,
            confidence=.9, span_uid=first.uid, group_uid=first.uid,
        )]


def semantic_facts(candidates):
    return [(item.predicate, tuple((binding.role_id, binding.value) for binding in item.bindings), item.exact_text) for item in candidates]


def test_explicit_and_configured_provider_share_postprocessing(monkeypatch):
    text = "Нагрев вызвал остановку. Оператор использовал ключ."
    provider = SparseProvider()
    explicit, explicit_meta = extract_candidates(text, provider)
    monkeypatch.setattr(ingestion, "configured_provider", lambda *_: (provider, {"configured": "test", "active": "sparse", "model_id": provider.model_id, "fallback": False}))
    ingestion._CANDIDATE_CACHE.clear()
    configured, configured_meta = extract_candidates(text)
    pure, _ = extract_candidates(text, provider, recovery=False)
    assert semantic_facts(explicit) == semantic_facts(configured)
    assert "USES_TOOL" in {item.predicate for item in explicit} - {item.predicate for item in pure}
    assert explicit_meta["configured"] == "explicit" and configured_meta["configured"] == "test"


def test_recovery_false_keeps_only_provider_facts():
    text = "Нагрев вызвал остановку. Оператор использовал ключ."
    candidates, meta = extract_candidates(text, SparseProvider(), recovery=False)
    assert semantic_facts(candidates) == [("CAUSE", (("SUBJECT", "нагрев"), ("OBJECT", "остановку")), "Нагрев вызвал остановку.")]
    assert meta["coverage_warnings"] and not meta["fallback"]


def test_recovery_false_does_not_hide_configured_provider_failure(monkeypatch):
    class FailingProvider(Provider):
        model_id = "failing-test"

        def extract(self, text):
            raise ValueError("model unavailable")

    provider = FailingProvider()
    monkeypatch.setattr(ingestion, "configured_provider", lambda *_: (provider, {"configured": "auto", "active": "model", "model_id": provider.model_id, "fallback": False}))
    ingestion._CANDIDATE_CACHE.clear()
    try:
        extract_candidates("Нагрев вызвал остановку.", recovery=False)
    except ValueError as exc:
        assert str(exc) == "model unavailable"
    else:
        raise AssertionError("recovery=False must expose provider failure")
