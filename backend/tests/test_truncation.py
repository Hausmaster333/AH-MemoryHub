import pytest
from app.ingestion import OpenAICompatibleProvider


def test_truncated_chunks_split_without_losing_offsets_or_looping(monkeypatch):
    provider = OpenAICompatibleProvider("http://localhost/v1", "test")
    source = "Первое событие. Второе событие. Третье событие. Четвертое событие."
    calls=[]
    def extract(text, offset, spans):
        calls.append((text,offset))
        assert source[offset:offset+len(text)] == text
        if len([s for s in spans if offset <= s.start < offset+len(text)]) > 1:
            raise ValueError("LLM structured output was truncated at AH_LLM_MAX_TOKENS")
        return [(text,offset)]
    monkeypatch.setattr(provider,"_extract_single",extract)
    results=provider.extract(source)
    assert len(results)==4 and len(set(results))==4
    assert [r[1] for r in results]==sorted(r[1] for r in results)
    assert len(calls)==10
    calls.clear()
    def fail(*args):
        calls.append(args)
        raise ValueError("LLM structured output was truncated at AH_LLM_MAX_TOKENS")
    monkeypatch.setattr(provider,"_extract_single",fail)
    with pytest.raises(ValueError,match="truncated"): provider.extract("Одно событие.")
    assert len(calls)==2
