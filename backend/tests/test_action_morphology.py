from app.ingestion import _finite_action, canonicalize_candidates
from app.models import CandidateFact, CandidateBinding


def test_finite_actions_do_not_start_at_noun_suffixes():
    for source, subject, expected in [
        ('Сигнал подтвердил остановку насоса.', 'Сигнал', 'подтвердил остановку насоса'),
        ('Отдел выполнил проверку.', 'Отдел', 'выполнил проверку'),
        ('Датчик был снят для осмотра.', 'Датчик', 'был снят для осмотра'),
    ]:
        fact = CandidateFact(predicate='ACTION', bindings=(CandidateBinding(role_id='SUBJECT', value=subject), CandidateBinding(role_id='OBJECT', value=source.rstrip('.'))), source_start=0, source_end=len(source), exact_text=source, confidence=1)
        accepted, rejected = canonicalize_candidates(source, [fact])
        assert accepted and not rejected
        assert next(b.value for b in accepted[0].bindings if b.role_id == 'OBJECT') == expected
    assert _finite_action('Сигнал, отдел, инструмент и кабель.') is None
