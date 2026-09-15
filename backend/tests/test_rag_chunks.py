import importlib.util
from pathlib import Path


def test_token_windows_keep_exact_source_and_tail():
    spec = importlib.util.spec_from_file_location("rag", Path(__file__).parents[1] / "scripts/build_rag_contexts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def tokenizer(text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
    text = "АБВГДЕЖЗИ"
    chunks = list(module.token_chunks(text, tokenizer, window=4, overlap=1))
    assert [c["text"] for c in chunks] == ["АБВГ", "ГДЕЖ", "ЖЗИ"]
    assert all(text[c["start"]:c["end"]] == c["text"] for c in chunks)
    assert list(module.token_chunks("", tokenizer)) == []
