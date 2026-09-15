from app.evaluation import role_metrics
from app.ingestion import canonicalize_candidates
from app.models import CandidateBinding, CandidateFact


def test_predicate_mismatch_is_not_a_correct_role_match():
    bindings = [{"role": role, "value": value} for role, value in (("SUBJECT", "насос"), ("OBJECT", "остановку"), ("LOCATION", "цехе"))]
    gold = [{"predicate": "CAUSE", "bindings": bindings}]
    assert role_metrics(gold, gold)["weighted_f1"] == 5
    assert role_metrics(gold, [{"predicate": "HAS", "bindings": bindings}])["weighted_f1"] == 0


def test_cause_and_follow_keep_grounded_location_and_event_boundaries():
    for predicate, text, subject, obj in (
        ("CAUSE", "Перегрев насоса вызвал остановку агрегата в цехе.", "Перегрев насоса", "остановку агрегата"),
        ("FOLLOW", "После остановки агрегата оператор проверил насос в цехе.", "остановки агрегата", "оператор проверил насос"),
    ):
        fact = CandidateFact(predicate=predicate, bindings=tuple(CandidateBinding(role_id=role, value=value) for role, value in (("SUBJECT", subject), ("OBJECT", obj), ("LOCATION", "цехе"))), source_start=0, source_end=len(text), exact_text=text, confidence=1)
        accepted, rejected = canonicalize_candidates(text, [fact])
        assert not rejected and len(accepted) == 1
        roles = {binding.role_id: binding.value for binding in accepted[0].bindings}
        assert roles["LOCATION"] == "цехе" and roles["OBJECT"] == obj
