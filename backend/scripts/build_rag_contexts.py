"""Freeze a plain vector-retrieval baseline. Requires sentence-transformers separately from the server."""
import argparse
import hashlib
import json
from pathlib import Path


def token_chunks(text, tokenizer, window=96, overlap=24):
    if not 0 <= overlap < window: raise ValueError("require 0 <= overlap < window")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
    for index in range(0, len(offsets), window - overlap):
        selected = offsets[index:index + window]
        start, end = selected[0][0], selected[-1][1]
        yield {"start": start, "end": end, "text": text[start:end]}
        if index + window >= len(offsets): break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="existing local SentenceTransformer snapshot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    if args.top_k < 1: parser.error("top-k must be positive")
    if args.out.exists(): parser.error("output already exists")
    from transformers import AutoModel, PreTrainedTokenizerFast
    import torch
    import numpy as np
    # Use the model card's masked mean pooling. Direct tokenizer JSON avoids an installed AutoTokenizer config bug.
    special = json.loads((args.model / "special_tokens_map.json").read_text())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.model / "tokenizer.json"), **{k: v["content"] if isinstance(v, dict) else v for k, v in special.items()})
    model = AutoModel.from_pretrained(str(args.model.resolve()), local_files_only=True).eval()
    pooling = json.loads((args.model / "1_Pooling/config.json").read_text())
    if not pooling["pooling_mode_mean_tokens"] or any(v for k, v in pooling.items() if k.startswith("pooling_mode_") and k != "pooling_mode_mean_tokens"):
        parser.error("this baseline expects masked mean pooling")
    max_length = json.loads((args.model / "sentence_bert_config.json").read_text())["max_seq_length"]
    def encode(texts):
        batches = []
        for start in range(0, len(texts), 16):
            inputs = tokenizer(texts[start:start + 16], padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False)
            if inputs["input_ids"].shape[1] > max_length: raise ValueError("embedding input exceeds context")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.concatenate(batches)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    chunks = [{"document_id": document["id"], **chunk} for document in corpus["documents"] for chunk in token_chunks(document["text"], tokenizer)]
    if not chunks: parser.error("empty corpus")
    if any(len(tokenizer.encode(chunk["text"])) > max_length for chunk in chunks):
        parser.error("chunk exceeds embedding context; refuse silent truncation")
    vectors = encode([chunk["text"] for chunk in chunks])
    query_vectors = encode([case["question"] for case in questions["cases"]])
    rows = []
    for case, query in zip(questions["cases"], query_vectors):
        scores = vectors @ query
        ranked = np.argsort(-scores, kind="stable")[:args.top_k]
        rows.append({"id": case["id"], "question": case["question"], "contexts": [{**chunks[int(i)], "score": float(scores[i])} for i in ranked]})
    report = {"scope": "plain cosine vector retrieval; no AH roles, graph or gold used for ranking", "model": str(args.model.resolve()), "model_weights_sha256": hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest(), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "window_tokens": 96, "overlap_tokens": 24, "top_k": args.top_k, "chunk_count": len(chunks), "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(chunks)} chunks, {len(rows)} questions")


if __name__ == "__main__": main()
