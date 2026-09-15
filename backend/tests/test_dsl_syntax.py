import pytest
from app.dsl import DSLParseError, parse


def test_dsl_rejects_ambiguous_syntax_before_execution():
    for expression in ['getSymbol(uid="s1)', 'getSymbol(uid=s1))', 'getSymbol(uid=s1', 'getSymbol(uid=s1, uid=s2)']:
        with pytest.raises(DSLParseError): parse(expression)
    assert parse('findSymbols(value="насос (основной), резерв | смена")')[0].args['value'] == 'насос (основной), резерв | смена'
