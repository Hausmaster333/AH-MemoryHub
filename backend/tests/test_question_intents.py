from fastapi.testclient import TestClient
from app.core import AHMemory
from app.ingestion import ingest, RuleBasedProvider
from app.ingestion import Compiler
from app.models import CandidateFact, CandidateBinding
import app.main as api


def test_classification_answer_requires_entity_and_class_in_their_roles():
    from app.answering import select_local_answer
    rows = [
        ("Клапан КЛ-11 является резервным.", "клапан КЛ-11", "резервным"),
        ("Реле РЛ-11 является резервным.", "реле РЛ-11", "резервным"),
        ("Клапан КЛ-12 является основным.", "клапан КЛ-12", "основным"),
    ]
    source = " ".join(row[0] for row in rows)
    memory = AHMemory()
    candidates = []
    for quote, subject, category in rows:
        start = source.index(quote)
        candidates.append(CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="OBJECT", value=category)), source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(candidates, source, "classification-roles")
    assert len(accepted) == 3 and not rejected
    answer, chosen = select_local_answer(memory, "Какой клапан является резервным?", tuple(accepted))
    assert answer == rows[0][0] and len(chosen) == 1
    answer, chosen = select_local_answer(memory, "Какой клапан является аварийным?", tuple(accepted))
    assert answer == "insufficient_evidence" and not chosen


def test_identifier_match_distinguishes_nearby_cyrillic_device_ids():
    from app.answering import select_local_answer
    rows = [("Клапан КЛ-11 оставался открытым.", "клапан КЛ-11", "открытым"),
            ("Клапан КР-11 оставался закрытым.", "клапан КР-11", "закрытым")]
    source = " ".join(row[0] for row in rows)
    memory = AHMemory()
    candidates = []
    for quote, subject, state in rows:
        start = source.index(quote)
        candidates.append(CandidateFact(predicate="HAS_STATE", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="STATE", value=state)), source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(candidates, source, "near-ids")
    assert len(accepted) == 2 and not rejected
    answer, chosen = select_local_answer(memory, "Какое состояние имеет клапан КЛ-11?", tuple(accepted))
    assert "КЛ-11" in answer and "КР-11" not in answer and len(chosen) == 1


def test_classification_site_scope_must_be_in_the_selected_fact():
    from app.answering import select_local_answer
    rows = [
        ("Клапан КЛ-11 является резервным на станции Восток.", "клапан КЛ-11"),
        ("Клапан КР-11 является резервным на станции Запад.", "клапан КР-11"),
    ]
    source = " ".join(quote for quote, _ in rows)
    memory = AHMemory()
    candidates = []
    for quote, subject in rows:
        start = source.index(quote)
        candidates.append(CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="OBJECT", value="резервным")), source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(candidates, source, "two-sites")
    assert len(accepted) == 2 and not rejected
    for site, expected in (("Восток", rows[0][0]), ("Запад", rows[1][0])):
        answer, selected = select_local_answer(memory, f"Какой клапан является резервным на станции {site}?", tuple(accepted))
        assert answer == expected and len(selected) == 1
    answer, selected = select_local_answer(memory, "Какой клапан является резервным?", tuple(accepted))
    assert answer == "insufficient_evidence" and not selected

    # A title elsewhere in the document does not establish this fact's site relation.
    memory = AHMemory()
    quote = "Клапан КЛ-11 является резервным."
    accepted, _ = Compiler(memory).compile([CandidateFact(predicate="IS-A", bindings=(CandidateBinding(role_id="SUBJECT", value="клапан КЛ-11"), CandidateBinding(role_id="OBJECT", value="резервным")), source_start=16, source_end=16 + len(quote), exact_text=quote, confidence=1)], "Станция Восток. " + quote, "site-title")
    answer, selected = select_local_answer(memory, "Какой клапан является резервным на станции Восток?", tuple(accepted))
    assert answer == "insufficient_evidence" and not selected


def test_actor_question_does_not_match_ownership_inside_cell_word(monkeypatch):
    memory = AHMemory()
    quotes = ["Оператор Петров выполнял проверку ячейки Луч.", "Ячейка Луч принадлежит лаборатории."]
    source = " ".join(quotes)
    facts = []
    for quote, predicate, subject, obj in [(quotes[0], "ACTION", "Оператор Петров", "выполнял проверку ячейки Луч"), (quotes[1], "HAS", "лаборатории", "ячейка Луч")]:
        start = source.index(quote)
        facts.append(CandidateFact(predicate=predicate, bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="OBJECT", value=obj)), source_start=start, source_end=start+len(quote), exact_text=quote, confidence=1))
    assert len(Compiler(memory).compile(facts, source, "actor-boundary")[0]) == 2
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    for question, expected in [("Кто выполнял проверку ячейки Луч?", "Петров"), ("Чья ячейка Луч?", "лаборатории")]:
        result = client.post("/api/v1/queries", json={"question": question, "profile": "directed_v1"}).json()
        assert result["status"] == "answered" and expected in result["answer"] and result["trace_complete"]


def test_cause_question_uses_effect_not_next_cause(monkeypatch):
    memory = AHMemory()
    source = "Поломка вентилятора вызвала перегрев двигателя. Перегрев двигателя вызвал остановку насоса. Остановка насоса вызвала аварийный сигнал."
    ingest(memory, source, provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    result = TestClient(api.app).post("/api/v1/queries", json={"question":"Что вызвало остановку насоса?", "profile":"directed_v1"}).json()
    assert result["status"] == "answered" and result["trace_complete"]
    assert result["answer"] == "Перегрев двигателя вызвал остановку насоса."


def test_plural_tool_question_selects_tool_fact(monkeypatch):
    memory = AHMemory()
    text = "Инженер Ли использовал ручной ключ."
    ingest(memory, text, provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    response = TestClient(api.app).post("/api/v1/queries", json={"question":"Какие инструменты использовал инженер Ли?", "profile":"directed_v1"}).json()
    assert response["status"] == "answered" and response["answer"] == text


def test_tool_question_cannot_select_action_without_tool(monkeypatch):
    memory = AHMemory()
    text = "Инженер Ли использовал инструменты и открыл люк. Инженер Ли применил ручной ключ."
    first, second = text.split(". ")
    first += "."
    candidates = []
    for predicate, quote, subject, role, value in [("ACTION", first, "Инженер Ли", "OBJECT", "использовал инструменты и открыл люк"), ("USES_TOOL", second, "Инженер Ли", "TOOL", "ручной ключ")]:
        start = text.index(quote)
        candidates.append(CandidateFact(predicate=predicate, source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1, bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id=role, value=value))))
    accepted, rejected = Compiler(memory).compile(candidates, text, "tools")
    assert len(accepted) == 2 and not rejected
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    result = TestClient(api.app).post("/api/v1/queries", json={"question": "Какие инструменты использовал инженер Ли?", "profile": "directed_v1"}).json()
    assert result["status"] == "answered" and result["answer"] == second and result["trace_complete"]


def test_tool_action_must_match_requested_event():
    from app.answering import select_local_answer
    memory = AHMemory()
    rows = [
        ("Инженер Ли проверил клапан ключом К-2.", "проверил клапан ключом К-2", "ключом К-2"),
        ("Инженер Ли очистил фильтр щёткой Щ-3.", "очистил фильтр щёткой Щ-3", "щёткой Щ-3"),
    ]
    source = " ".join(row[0] for row in rows)
    facts = []
    for quote, action, tool in rows:
        start = source.index(quote)
        facts.append(CandidateFact(predicate="ACTION", bindings=(CandidateBinding(role_id="SUBJECT", value="Инженер Ли"), CandidateBinding(role_id="OBJECT", value=action), CandidateBinding(role_id="TOOL", value=tool)), source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(facts, source, "tool-events")
    assert len(accepted) == 2 and not rejected
    answer, selected = select_local_answer(memory, "Какие инструменты при очистке фильтра использовал инженер Ли?", tuple(accepted))
    assert answer == rows[1][0] and len(selected) == 1
    answer, selected = select_local_answer(memory, "Какие инструменты при наладке фильтра использовал инженер Ли?", tuple(accepted))
    assert answer == "insufficient_evidence" and not selected

    memory = AHMemory()
    rows = [
        ("Инженер Ли проверил клапан ключом К-2.", "проверил клапан ключом К-2", "ключом К-2"),
        ("Инженер Ли проверил фильтр щёткой Щ-3.", "проверил фильтр щёткой Щ-3", "щёткой Щ-3"),
    ]
    source = " ".join(row[0] for row in rows)
    facts = []
    for quote, action, tool in rows:
        start = source.index(quote)
        facts.append(CandidateFact(predicate="ACTION", bindings=(CandidateBinding(role_id="SUBJECT", value="Инженер Ли"), CandidateBinding(role_id="OBJECT", value=action), CandidateBinding(role_id="TOOL", value=tool)), source_start=start, source_end=start + len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(facts, source, "same-verb-tools")
    assert len(accepted) == 2 and not rejected
    answer, selected = select_local_answer(memory, "Какие инструменты при повторной проверке фильтра использовал инженер Ли?", tuple(accepted))
    assert answer == rows[1][0] and len(selected) == 1




def test_location_questions_search_the_requested_role_and_object_type(monkeypatch):
    memory = AHMemory()
    rows = [("Датчик ДК-71", "камере ХК-10"), ("Датчик ДК-72", "камере ХК-11"), ("Насос НР-7", "камере ХК-10")]
    texts = [f"{subject} установлен в {location}." for subject, location in rows]
    source = " ".join(texts)
    facts = []
    for (subject, location), quote in zip(rows, texts):
        start = source.index(quote)
        facts.append(CandidateFact(predicate="LOCATED_AT", bindings=(CandidateBinding(role_id="SUBJECT", value=subject), CandidateBinding(role_id="LOCATION", value=location)), source_start=start, source_end=start+len(quote), exact_text=quote, confidence=1))
    accepted, rejected = Compiler(memory).compile(facts, source, "locations")
    assert len(accepted) == 3 and not rejected
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    for question, expected in [("Какой датчик установлен в камере ХК-10?", "ДК-71"), ("Какой датчик находится в камере ХК-11?", "ДК-72"), ("Где установлен датчик ДК-71?", "ХК-10")]:
        data = client.post("/api/v1/queries", json={"question":question, "profile":"directed_v1", "answer_model":"local"}).json()
        assert data["status"] == "answered" and expected in data["answer"] and data["trace_complete"]
        assert "НР-7" not in data["answer"]
    for question in ("Какой датчик установлен в камере ХК-12?", "Какой клапан установлен в камере ХК-10?"):
        data = client.post("/api/v1/queries", json={"question":question, "profile":"directed_v1", "answer_model":"local"}).json()
        assert data["status"] == "insufficient_evidence" and not data["evidence"]


def test_time_question_variants_do_not_return_an_untimed_fact(monkeypatch):
    memory = AHMemory()
    text = "Датчик ДК-33 установлен в лаборатории."
    ingest(memory, text, provider=RuleBasedProvider())
    monkeypatch.setattr(api, "memory", memory)
    monkeypatch.setattr(api, "storage_mode", "in-memory")
    client = TestClient(api.app)
    for question in ("Когда установили датчик ДК-33?", "Во сколько установили датчик ДК-33?", "В какое время установили датчик ДК-33?"):
        data = client.post("/api/v1/queries", json={"question":question, "profile":"directed_v1", "answer_model":"local"}).json()
        assert data["status"] == "insufficient_evidence" and not data["evidence"]


def test_action_time_question_matches_event_and_participant(monkeypatch):
    from app.ingestion import canonicalize_candidates
    memory = AHMemory()
    quotes = ["Команда завершилась подтверждением от МК-44 в 08:25.", "Команда началась для МК-44 в 08:20."]
    text = " ".join(quotes)
    candidates=[]
    for quote in quotes:
        start=text.index(quote)
        candidates.append(CandidateFact(predicate="ACTION",bindings=(CandidateBinding(role_id="SUBJECT",value="Команда"),CandidateBinding(role_id="OBJECT",value=quote.removeprefix("Команда ").rstrip("."))),source_start=start,source_end=start+len(quote),exact_text=quote,confidence=1))
    facts, errors = canonicalize_candidates(text,candidates)
    assert not errors
    accepted,rejected=Compiler(memory).compile(facts,text,"timed-actions")
    assert len(accepted)==2 and not rejected
    monkeypatch.setattr(api,"memory",memory)
    monkeypatch.setattr(api,"storage_mode","in-memory")
    client=TestClient(api.app)
    data=client.post("/api/v1/queries",json={"question":"Когда команда завершилась подтверждением от МК-44?","profile":"directed_v1","answer_model":"local"}).json()
    assert data["answer"]==quotes[0] and data["trace_complete"]
    for question in ("Когда команда отменена для МК-44?", "Когда команда завершилась подтверждением от МК-45?"):
        data=client.post("/api/v1/queries",json={"question":question,"profile":"directed_v1","answer_model":"local"}).json()
        assert data["status"]=="insufficient_evidence" and not data["evidence"]


def test_actor_and_ownership_questions_do_not_substitute_related_actions(monkeypatch):
    memory=AHMemory()
    rows=[("Иван открыл клапан К-71.","Иван","открыл клапан К-71"),("Пётр закрыл клапан К-71.","Пётр","закрыл клапан К-71")]
    text=" ".join(x[0] for x in rows)
    facts=[CandidateFact(predicate="ACTION",bindings=(CandidateBinding(role_id="SUBJECT",value=subject),CandidateBinding(role_id="OBJECT",value=action)),source_start=text.index(quote),source_end=text.index(quote)+len(quote),exact_text=quote,confidence=1) for quote,subject,action in rows]
    accepted,rejected=Compiler(memory).compile(facts,text,"actor-test")
    assert len(accepted)==2 and not rejected
    monkeypatch.setattr(api,"memory",memory)
    monkeypatch.setattr(api,"storage_mode","in-memory")
    client=TestClient(api.app)
    for question,answer in [("Кто открыл клапан К-71?",rows[0][0]),("Кто закрыл клапан К-71?",rows[1][0])]:
        data=client.post("/api/v1/queries",json={"question":question,"profile":"directed_v1","answer_model":"local"}).json()
        assert data["answer"]==answer and data["trace_complete"]
    for question in ("Кто создал клапан К-71?","Кто является собственником клапана К-71?","Кто открыл клапан К-72?"):
        data=client.post("/api/v1/queries",json={"question":question,"profile":"directed_v1","answer_model":"local"}).json()
        assert data["status"]=="insufficient_evidence" and not data["evidence"]
