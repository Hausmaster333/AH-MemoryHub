"""Benchmark-only capture/replay. Never stores credentials or makes replay network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from . import ingestion


def save_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def document_predictions(text: str, path: Path, model: str, replay: dict | None = None):
    """Run the normal configured pipeline against a recorded request transport."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    if replay is not None:
        if replay.get("schema") != "ah-predictions-v1" or replay.get("source_sha256") != digest or replay.get("text") != text:
            raise ValueError("recording source/schema mismatch")
        settings = replay["settings"]
        provider = ingestion.OpenAICompatibleProvider("https://replay.invalid/v1", settings["model"], response_format=settings["response_format"], perception_format=settings.get("perception_format", "ir"))
        meta = {"configured": "replay", "active": "openai_compatible", "model_id": provider.model_id, "fallback": False}
    else:
        provider, meta = ingestion.configured_provider(model)
        settings = {"model": provider.model_id, "response_format": provider.response_format, "perception_format": provider.perception_format}
    record = {"schema": "ah-predictions-v1", "source_sha256": digest, "text": text, "settings": settings, "prompt_version": ingestion.PROMPT_VERSION, "calls": [], "status": "running"}
    cursor = 0
    transport = provider._request_json

    def request(payload):
        nonlocal cursor
        call = {"request": deepcopy(payload)}
        if replay is not None:
            calls = replay["calls"]
            if cursor >= len(calls) or calls[cursor]["request"] != payload:
                raise ValueError("replay request mismatch: prompt/settings/chunking changed")
            stored = calls[cursor]
            cursor += 1
            if "response" not in stored:
                call["error_type"] = stored.get("error_type", "ValueError")
                record["calls"].append(call)
                save_json(path, record)
                raise ValueError("recorded model request failed")
            response = deepcopy(stored["response"])
        else:
            try:
                response = transport(payload)
            except Exception as exc:
                call["error_type"] = type(exc).__name__
                record["calls"].append(call)
                save_json(path, record)
                raise
        call["response"] = deepcopy(response)
        record["calls"].append(call)
        save_json(path, record)
        return response

    save_json(path, record)
    try:
        # ponytail: process-global patch is confined to the sequential benchmark CLI, never the server.
        with patch.object(provider, "_request_json", request), patch.object(ingestion, "configured_provider", return_value=(provider, meta)), patch.dict(ingestion._CANDIDATE_CACHE, {}, clear=True):
            candidates, metadata = ingestion.extract_candidates(text, model_override=model)
        if replay is not None and cursor != len(replay["calls"]):
            raise ValueError("replay left unused model responses")
        record.update(status="completed", predictions=[candidate.model_dump(mode="json") for candidate in candidates], metadata=metadata)
        return candidates, metadata
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save_json(path, record)
