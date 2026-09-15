from app.answering import select_local_answer
from app.core import AHMemory, lexical_key, lexical_score
from app.evaluation import _value_matches
from app.ingestion import Compiler, RuleBasedProvider, canonicalize_candidates, ingest, _grounded_value
from app.models import CandidateBinding, CandidateFact, CandidateMention, CandidateTerm


def test_equipment_identity_and_grounding():
    memory = AHMemory()
    compiler = Compiler(memory)
    first = compiler._grounded_symbol('датчик ДТ-12345')
    second = compiler._grounded_symbol('датчик ДТ-12346')
    assert first != second
    assert compiler._grounded_symbol('датчик ДТ-12345') == first
    assert lexical_key('ДТ-12345') != lexical_key('ДТ-12346')
    assert lexical_score('ДТ12345', 'ДТ-12346') == 0
    assert lexical_score('ДТ12345', 'ДТ-12345') == 1
    assert not _value_matches('датчик ДТ-12345', 'датчик ДТ-12346')
    assert not _value_matches('датчик ДТ-12345', 'датчик')
    assert not _grounded_value('датчик ДТ-12346', 'датчик ДТ-12345')
    assert not _grounded_value('насос 2', 'насос 1')


def test_unverified_mention_label_cannot_replace_source_entity():
    text = 'Насос установлен в цехе.'
    for label in ('атомный реактор', 'насосная станция', 'Насос 2'):
        candidate = CandidateFact(
            predicate='LOCATED_AT',
            bindings=(CandidateBinding(role_id='SUBJECT', term=CandidateTerm(mention_ids=('subject',))),
                      CandidateBinding(role_id='LOCATION', value='цехе')),
            source_start=0, source_end=len(text), exact_text=text, confidence=1,
            mentions=(CandidateMention(mention_id='subject', observed_text='Насос',
                                       canonical_label=label, source_start=0, source_end=5),),
        )
        facts, rejected = canonicalize_candidates(text, [candidate])
        assert not rejected and facts[0].bindings[0].value == 'Насос'
        memory = AHMemory()
        accepted, rejected = Compiler(memory).compile(facts, text, 'source')
        assert accepted and not rejected
        fact = memory.get_hypernode(accepted[0])
        assert memory.label(fact.role_bindings[0].target_ref.target_uid) == 'Насос'


def test_root_cause_requires_same_event_uid():
    memory = AHMemory()
    ingest(memory, 'Перегрев насоса вызвал остановку агрегата. Ошибка оператора вызвала перегрев сервера.', 'events', RuleBasedProvider())
    answer, facts = select_local_answer(memory, 'Какова первопричина остановки агрегата?', tuple(h.uid for h in memory.find_hypernodes()))
    assert 'Перегрев насоса' in answer and 'Ошибка оператора' not in answer
    assert len(facts) == 1

    ingest(memory, 'Засор фильтра вызвал перегрев насоса.', 'continuation', RuleBasedProvider())
    answer, facts = select_local_answer(memory, 'Какова первопричина остановки агрегата?', tuple(h.uid for h in memory.find_hypernodes()))
    assert 'Засор фильтра' in answer and len(facts) == 2

    ingest(memory, 'Отказ вентилятора вызвал перегрев насоса.', 'branch', RuleBasedProvider())
    answer, facts = select_local_answer(memory, 'Какова первопричина остановки агрегата?', tuple(h.uid for h in memory.find_hypernodes()))
    assert answer == 'insufficient_evidence' and not facts
