from app.ingestion import canonicalize_candidates, _stable_fact_candidates
from app.models import CandidateBinding, CandidateFact


def candidate(text, predicate, **roles):
    return CandidateFact(predicate=predicate, bindings=tuple(CandidateBinding(role_id=role, value=value) for role,value in roles.items()), source_start=0, source_end=len(text), exact_text=text, confidence=.95)


def test_event_location_keeps_event_identity_and_avoids_mode_complement():
    for tail in ("внутри помещения", "на выходе трубопровода"):
        text = f"Отказ вентилятора вызвал повышение температуры {tail}."
        facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="Отказ вентилятора", OBJECT=f"повышение температуры {tail}")])
        assert not errors
        roles = {binding.role_id: binding.value for binding in facts[0].bindings}
        assert roles == {"SUBJECT":"Отказ вентилятора", "OBJECT":"повышение температуры", "LOCATION":tail}
        assert facts[0].exact_text == text
    text = "Сбой вызвал переход в резервный режим."
    facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="Сбой", OBJECT="переход в резервный режим")])
    assert not errors and all(binding.role_id != "LOCATION" for binding in facts[0].bindings)


def test_ownership_and_property_boundaries_are_not_extended():
    text = "Установка принадлежит лаборатории «Зенит» и содержит резервный блок."
    owner = next(f for f in _stable_fact_candidates(text) if f.predicate == "HAS")
    assert {binding.role_id:binding.value for binding in owner.bindings}["SUBJECT"] == "лаборатории «Зенит»"
    text = "Датчик КД-91 имеет прочный корпус и сменный кабель."
    facts, errors = canonicalize_candidates(text, [candidate(text, "HAS", SUBJECT="Датчик КД-91", OBJECT="прочный корпус")])
    assert not errors and next(b.value for b in facts[0].bindings if b.role_id == "OBJECT") == "прочный корпус"
    facts, errors = canonicalize_candidates(text, [candidate(text, "HAS_STATE", SUBJECT="прочный корпус", STATE="прочный корпус")])
    assert not errors and next(b.value for b in facts[0].bindings if b.role_id == "SUBJECT") == "Датчик КД-91"


def test_do_complement_is_not_invented_time_and_explicit_time_survives():
    for boundary in ("до шести бар", "до 6 бар", "до резервуара"):
        text = f"Сбой вызвал повышение давления {boundary}."
        facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="Сбой", OBJECT=f"повышение давления {boundary}")])
        assert not errors and facts
        assert all(b.role_id != "TIME" for b in facts[0].bindings)
    text = "Насос оставался остановлен до завершения проверки."
    from app.ingestion import _temporal_state_candidates
    facts, errors = canonicalize_candidates(text, _temporal_state_candidates(text))
    assert not errors and facts
    assert next(b.value for b in facts[0].bindings if b.role_id == "TIME") == "до завершения проверки"
    text = "До полудня сбой вызвал остановку насоса."
    facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="сбой", OBJECT="остановку насоса", TIME="До полудня")])
    assert not errors and facts
    assert next(b.value for b in facts[0].bindings if b.role_id == "TIME") == "До полудня"


def test_negated_causation_is_not_positive_cause():
    for verb in ("вызвал", "привёл к"):
        text = f"Сбой не {verb} остановку насоса."
        facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="Сбой", OBJECT="остановку насоса")])
        assert not facts and errors[0]["reason"] == "invalid_cause"


def test_negated_model_candidates_are_rejected_before_compilation():
    text = "Насос остановился не из-за перегрева."
    facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT="перегрева", OBJECT="Насос остановился")])
    assert not facts and errors[0]["reason"] == "invalid_cause"
    text = "Инженер Ли не использовал ручной ключ."
    for predicate, roles in (("USES_TOOL", {"TOOL": "ручной ключ"}), ("ACTION", {"OBJECT": "использовал ручной ключ"})):
        facts, errors = canonicalize_candidates(text, [candidate(text, predicate, SUBJECT="Инженер Ли", **roles)])
        assert not facts and errors[0]["reason"] == "negated_action"


def test_effect_recovery_requires_positive_transitive_verb():
    from app.ingestion import _effect_candidates
    for text in ("Давление снизилось до двух бар.", "Режим изменился после проверки.", "Передача не изменила принадлежность прибора.", "Калибровка не снизила шум."):
        recovered, _ = canonicalize_candidates(text, _effect_candidates(text))
        assert not recovered
        facts, errors = canonicalize_candidates(text, [candidate(text, "CAUSE", SUBJECT=text.split()[0], OBJECT=text.split(maxsplit=1)[1].rstrip("."))])
        assert not facts and errors[0]["reason"] == "invalid_cause"
    for verb in ("снизил", "снизила", "снизило", "снизили"):
        text = f"Воздействие {verb} шум."
        facts, errors = canonicalize_candidates(text, _effect_candidates(text))
        assert len(facts) == 1 and not errors


def test_action_clock_is_recovered_only_from_an_unambiguous_action_phrase():
    for text, action, expected in [
        ("Команда завершилась подтверждением в 08:25.", "завершилась подтверждением в 08:25", "в 08:25"),
        ("Команда завершилась в 08:25, а проверка началась в 08:30.", "завершилась в 08:25", None),
        ("Команда завершилась, а проверка началась в 08:30.", "завершилась", None),
        ("Команда завершилась в 28:25.", "завершилась в 28:25", None),
    ]:
        facts, errors = canonicalize_candidates(text, [candidate(text, "ACTION", SUBJECT="Команда", OBJECT=action)])
        assert facts and not errors
        assert next((b.value for b in facts[0].bindings if b.role_id == "TIME"), None) == expected
