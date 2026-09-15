from app.answering import select_local_answer
from app.core import AHMemory
from app.ingestion import Compiler, _temporal_state_candidates, canonicalize_candidates
from app.models import CandidateFact, CandidateBinding


def test_temporal_state_cannot_substitute_time_of_another_event():
    memory = AHMemory()
    source = 'В 10:15 датчик ДК-90 был ослаблен. В 11:20 датчик ДК-90 был снят.'
    facts = []
    for quote, state, time in [(source.split('. ')[0]+'.', 'ослаблен', '10:15'), (source.split('. ')[1], 'снят', '11:20')]:
        start = source.index(quote)
        facts.append(CandidateFact(predicate='HAS_STATE', bindings=tuple(CandidateBinding(role_id=r,value=v) for r,v in [('SUBJECT','датчик ДК-90'),('STATE',state),('TIME',time)]), source_start=start,source_end=start+len(quote),exact_text=quote,confidence=1))
    accepted, rejected = Compiler(memory).compile(facts, source, 'event-match')
    assert len(accepted)==2 and not rejected
    answer, selected = select_local_answer(memory,'Когда сняли датчик ДК-90?',tuple(accepted))
    assert '11:20' in answer and len(selected)==1
    answer, selected = select_local_answer(memory,'Когда сняли датчик ДК-90?',tuple(accepted[:1]))
    assert answer=='insufficient_evidence' and not selected


def test_stamped_passive_status_keeps_its_own_subject_and_clock():
    source = ('12:00 - Датчик ДК-90 был отключён. '
              '13:00 - Датчик ДК-91 был временно снят для анализа.')
    facts, rejected = canonicalize_candidates(source, _temporal_state_candidates(source))
    assert len(facts) == 2 and not rejected
    memory = AHMemory()
    accepted, rejected = Compiler(memory).compile(facts, source, 'stamped-status')
    assert len(accepted) == 2 and not rejected
    answer, selected = select_local_answer(memory, 'Когда сняли датчик ДК-91?', tuple(accepted))
    assert answer == '13:00 - Датчик ДК-91 был временно снят для анализа.' and len(selected) == 1
    answer, selected = select_local_answer(memory, 'Когда сняли датчик ДК-90?', tuple(accepted))
    assert answer == 'insufficient_evidence' and not selected
    for subject, verb, status in [('Пластина ПЛ-7', 'была', 'снята'), ('Крышки КР-8', 'были', 'сняты')]:
        text = f'14:00 - {subject} {verb} {status}.'
        facts, rejected = canonicalize_candidates(text, _temporal_state_candidates(text))
        assert len(facts) == 1 and not rejected


def test_stamped_status_rejects_negation_plans_and_mixed_events():
    for source in (
        '13:00 - Датчик ДК-90 не был снят.',
        '13:00 - Датчик ДК-90 должен был быть снят.',
        '13:00 - Датчик ДК-90 был установлен инженером.',
        '13:00 - Был временно снят датчик ДК-90.',
        '13:00 - Датчик ДК-90 был снят предположительно.',
        '13:00 - Датчик ДК-90 был снят по плану.',
        '13:00 - Датчик ДК-90 был снят, но это не подтверждено.',
        '13:00 - Датчик ДК-90 был снят для проверки и был отключён.',
        '13:00 - Датчик ДК-90 был снят, но насос отключился.',
        '13:00 - Датчик ДК-90 был снят; 14:00 - насос остановлен.',
    ):
        assert not _temporal_state_candidates(source), source


def test_clocked_completed_event_uses_event_noun_and_its_own_time():
    source = 'В 08:40 завершилась калибровка реле РЛ-14. В 09:20 закончилась проверка клапана КЛ-11.'
    facts, rejected = canonicalize_candidates(source, _temporal_state_candidates(source))
    assert len(facts) == 2 and not rejected
    assert [{binding.role_id: binding.value for binding in fact.bindings} for fact in facts] == [
        {'SUBJECT': 'калибровка реле РЛ-14', 'STATE': 'завершилась', 'TIME': '08:40'},
        {'SUBJECT': 'проверка клапана КЛ-11', 'STATE': 'закончилась', 'TIME': '09:20'},
    ]
    memory = AHMemory()
    accepted, rejected = Compiler(memory).compile(facts, source, 'completed-events')
    assert len(accepted) == 2 and not rejected
    answer, selected = select_local_answer(memory, 'Когда завершилась калибровка реле РЛ-14?', tuple(accepted))
    assert answer == 'В 08:40 завершилась калибровка реле РЛ-14.' and len(selected) == 1
    answer, selected = select_local_answer(memory, 'Когда закончилась проверка клапана КЛ-11?', tuple(accepted))
    assert answer == 'В 09:20 закончилась проверка клапана КЛ-11.' and len(selected) == 1


def test_clocked_completed_event_rejects_modality_and_multiple_predicates():
    for source in (
        'В 08:40 не завершилась калибровка реле РЛ-14.',
        'В 08:40 должна завершиться калибровка реле РЛ-14.',
        'В 08:40 завершится калибровка реле РЛ-14.',
        'В 08:40 закончится проверка клапана КЛ-11.',
        'В 08:40 завершилась калибровка реле РЛ-14 предположительно.',
        'В 08:40 завершилась калибровка, но её не подтвердили.',
        'В 08:40 завершилась калибровка и началась проверка.',
        'В 08:40 завершилась калибровка; в 09:20 началась проверка.',
        'В 08:40 завершилась калибровка в 09:20.',
        'В 08:40 завершили калибровку реле РЛ-14.',
        'В 08:40 завершилась она.',
    ):
        assert not _temporal_state_candidates(source), source
