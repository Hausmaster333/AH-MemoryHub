from app.ingestion import _declarative_recovery_candidates, canonicalize_candidates
from app.models import CandidateFact, CandidateBinding


def test_event_and_value_are_not_classes_in_either_perception_path():
    examples = [
        ("13:00", "Датчик ДС-62 был временно снят для анализа"),
        ("08:15", "Оператор проверил насос"),
        ("Срок закрытия наряда", "19 марта"),
        ("Правильное время инцидента", "14:00"),
        ("Датчик", "временно отключён"),
    ]
    for subject, category in examples:
        text = f"{subject} — {category}."
        assert not [f for f in _declarative_recovery_candidates(text) if f.predicate == "IS-A"]
        candidate = CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="OBJECT", value=category)), source_start=0, source_end=len(text), exact_text=text, confidence=1)
        accepted, rejected = canonicalize_candidates(text, [candidate])
        assert not accepted and rejected[0]["reason"] == "invalid_classification"
    for text in ("Заяц — маленький дикий зверёк.", "НБ-32 — резервный насос."):
        accepted, rejected = canonicalize_candidates(text, _declarative_recovery_candidates(text))
        assert len(accepted) == 1 and not rejected
