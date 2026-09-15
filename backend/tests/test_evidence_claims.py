import pytest
from app.answering import render_evidence_claims


def test_claim_renderer_preserves_attribution_and_rejects_missing_support():
    facts = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]
    claims = [{"text": "Насос остановлен", "evidence_ids": ["E1"]}, {"text": "Клапан закрыт.", "evidence_ids": ["E2"]}]
    result = render_evidence_claims({"status": "answered", "claims": claims}, facts)
    assert result["answer"] == "Насос остановлен [E1]. Клапан закрыт [E2]."
    assert result["evidence_ids"] == ["E1", "E2"]
    for claim in [{"text": "Насос остановлен", "evidence_ids": []}, {"text": "Насос остановлен", "evidence_ids": ["E99"]}, {"text": "Первое. Второе.", "evidence_ids": ["E1"]}]:
        with pytest.raises(ValueError): render_evidence_claims({"status": "answered", "claims": [claim]}, facts)
    with pytest.raises(ValueError): render_evidence_claims({"status": "insufficient_evidence", "claims": claims}, facts)
    with pytest.raises(ValueError): render_evidence_claims([], facts)
    assert render_evidence_claims({"status": "insufficient_evidence", "claims": []}, facts)["answer"] == ""
