import json
import pytest
from app.core import AHMemory
from app.dsl import Interpreter
from app.models import MemoryElement, SecondOrderSymbol, ControlTemplate, SReference, Role, MemoryList, ElementReference, FunctionalSymbol


def test_typed_dsl_mutations_and_atomic_template_updates():
    memory = AHMemory()
    dsl = Interpreter(memory)
    def mutate(name, data, **args):
        extra = ''.join(f', {key}={value}' for key, value in args.items())
        return dsl.mutate(f'{name}(data={json.dumps(data, ensure_ascii=False)}{extra})')
    symbol = {'uid': 's_test', 'sensory_representations': [{'modality': 'text', 'value': 'насос'}]}
    mutate('addAbstractSymbol', symbol)
    symbol['sensory_representations'][0]['value'] = 'насос резервный'
    mutate('editAbstractSymbol', symbol)
    assert dsl.query('findAbstractSymbols(value="насос резервный")')[0]['uid'] == 's_test'
    element = MemoryElement(uid='m_test', payload=SecondOrderSymbol(uid='m_test'))
    mutate('addElement', element.model_dump(mode='json'), section='P')
    value = {'text': 'a, (b) | "c"', 'items': [1, 2]}
    mutate('addProperty', {'name': 'label', 'value': value}, uid='m_test')
    assert memory.get_symbol('m_test').properties[0].value == value
    mutate('editProperty', {'name': 'label', 'value': 'резерв'}, uid='m_test')
    assert memory.get_symbol('m_test').properties[0].value == 'резерв'
    template = ControlTemplate(uid='t_test', predicate_ref=SReference(reference_uid='r_pred', target_uid='s_test'), ordered_roles=(Role(role_id='SUBJECT'),))
    element = MemoryElement(uid='t_test', payload=template, excitation=.4)
    mutate('addElement', element.model_dump(mode='json'), section='C')
    assert memory.elements['t_test'].excitation == .4
    updated = template.model_copy(update={'ordered_roles': (Role(role_id='SUBJECT', required=True),)})
    mutate('editElement', MemoryElement(uid='t_test', payload=updated).model_dump(mode='json'))
    assert memory.get_template('t_test') == updated
    for payload in (MemoryList(uid='list_test', ordered_members=(ElementReference(reference_uid='r_list', target_uid='m_test'),)), FunctionalSymbol(uid='f_test', function_id='AND', ordered_operands=(SReference(reference_uid='r_fn', target_uid='s_test'),))):
        mutate('addElement', MemoryElement(uid=payload.uid, payload=payload).model_dump(mode='json'), section='C')
    before = memory.export()
    duplicate_roles = updated.model_copy(update={'ordered_roles': (Role(role_id='SUBJECT'), Role(role_id='SUBJECT'))})
    with pytest.raises(ValueError):
        mutate('editElement', MemoryElement(uid='t_test', payload=duplicate_roles).model_dump(mode='json'))
    assert memory.export() == before
    with pytest.raises(ValueError):
        dsl.mutate('addProperty(data={"name":"x","value":[1,2}, uid=m_test)')
    assert memory.export() == before
