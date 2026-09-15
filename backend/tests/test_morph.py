from app.evaluation import value_matches_morph


def test_morphology_preserves_identity_negation_and_full_phrase():
    for left, right in [("северный гребень", "северном гребне"), ("машинный зал Б", "в машинном зале Б"), ("резервная зона миграции", "резервной зоне миграции")]:
        assert value_matches_morph(left, right), (left, right)
    for left, right in [("насос Н-17", "насос Н-18"), ("зал А", "зал Б"), ("насос работает", "насос не работает"), ("насос работает", "насос иногда работает"), ("высокое давление", "низкое давление"), ("насос", "насос резервного контура"), ("насос насоса", "насос")]:
        assert not value_matches_morph(left, right), (left, right)
