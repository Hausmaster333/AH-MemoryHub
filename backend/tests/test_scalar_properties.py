from app.ingestion import _declarative_recovery_candidates, canonicalize_candidates, Compiler
from app.core import AHMemory


def test_labelled_scalars_are_properties_and_keep_rejected_alternative_in_evidence():
    for text, value in [
        ('Уточнённое время осмотра — 09:35, а не 08:10 из черновика.', '09:35'),
        ('Температура образца — -12 °C.', '-12 °C'),
        ('Доля примеси — 3,5%.', '3,5%'),
        ('Время выдержки — 5 минут.', '5 минут'),
    ]:
        facts, errors = canonicalize_candidates(text, _declarative_recovery_candidates(text))
        assert not errors and len(facts) == 1
        assert facts[0].predicate == 'HAS_STATE'
        assert next(b.value for b in facts[0].bindings if b.role_id == 'STATE') == value
        memory = AHMemory()
        accepted, rejected = Compiler(memory).compile(facts, text, 'scalar')
        assert len(accepted) == 1 and not rejected
        assert memory.get_hypernode(accepted[0]).evidence[0].exact_text == text
    for text in ('09:35 — оператор осмотрел датчик.', 'Время осмотра — неизвестно.', 'Возможно время осмотра — 09:35.', 'Время осмотра — 29:35.'):
        assert not [f for f in _declarative_recovery_candidates(text) if f.predicate == 'HAS_STATE']
