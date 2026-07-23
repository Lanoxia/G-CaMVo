"""Content-addressed, atomic response cache for provider calls."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from camvo.exceptions import ResponseCacheError
from camvo.types import AnnotationItem, ModelResponse

_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, _SCHEMA_VERSION}
_MAX_CACHED_RATIONALE_CHARS = 4_000


def _safe_raw_metadata(raw: Any) -> dict[str, Any]:
    """Persist only small, experiment-relevant metadata.

    Provider payloads can contain request headers, internal identifiers, or
    other fields that should never be copied into a research artifact.  TRACE
    needs the normalized confidence on cache replay, so the cache keeps a
    deliberately tiny allow-list instead of serializing ``ModelResponse.raw``.
    """

    if not isinstance(raw, dict):
        return {}
    safe: dict[str, Any] = {}
    confidence = raw.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        value = float(confidence)
        if 0.0 <= value <= 1.0:
            safe["confidence"] = value
    rationale = raw.get("rationale")
    if isinstance(rationale, str):
        safe["rationale"] = rationale[:_MAX_CACHED_RATIONALE_CHARS]
    for key in (
        "elapsed_seconds",
        "provider_latency_seconds",
        "elapsed_time_seconds",
        "wall_time_seconds",
    ):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if value >= 0:
                safe[key] = value
    for key in ("workflow_model", "provider_model", "finish_reason"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            safe[key] = value[:256]
    aggregate_tokens = raw.get("dify_total_tokens")
    if isinstance(aggregate_tokens, int) and not isinstance(aggregate_tokens, bool):
        if aggregate_tokens >= 0:
            safe["dify_total_tokens"] = aggregate_tokens
    token_split = raw.get("token_split")
    if isinstance(token_split, str) and token_split:
        safe["token_split"] = token_split[:128]
    return safe


@dataclass(frozen=True, slots=True)
class ResponseCacheKey:
    digest: str
    model_id: str
    item_id: str
    prompt_version: str
    text_sha256: str
    labels: tuple[str, ...]


class FileResponseCache:
    """Store one response per atomic JSON file, keyed by prompt content."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    @staticmethod
    def key_for(
        model_id: str,
        item: AnnotationItem,
        prompt_version: str,
    ) -> ResponseCacheKey:
        if not prompt_version.strip():
            raise ValueError("prompt_version must not be empty")
        text_sha256 = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
        payload = json.dumps(
            {
                "model_id": model_id,
                "item_id": item.item_id,
                "prompt_version": prompt_version,
                "text_sha256": text_sha256,
                "labels": item.labels,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ResponseCacheKey(
            digest=hashlib.sha256(payload).hexdigest(),
            model_id=model_id,
            item_id=item.item_id,
            prompt_version=prompt_version,
            text_sha256=text_sha256,
            labels=item.labels,
        )

    def _path(self, key: ResponseCacheKey) -> Path:
        return self.root / key.digest[:2] / f"{key.digest}.json"

    def get(self, key: ResponseCacheKey) -> ModelResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        with self._lock:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                schema_version = int(record["schema_version"])
                if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
                    raise ResponseCacheError("unsupported response-cache schema")
                persisted_key = record["key"]
                expected: dict[str, Any] = {
                    "digest": key.digest,
                    "model_id": key.model_id,
                    "item_id": key.item_id,
                    "prompt_version": key.prompt_version,
                    "text_sha256": key.text_sha256,
                    "labels": list(key.labels),
                }
                if persisted_key != expected:
                    raise ResponseCacheError("response-cache key mismatch")
                response = record["response"]
                label = str(response["label"])
                input_tokens = int(response["input_tokens"])
                output_tokens = int(response["output_tokens"])
                if label not in key.labels:
                    raise ResponseCacheError("cached response contains an unknown label")
                if input_tokens < 0 or output_tokens < 0:
                    raise ResponseCacheError("cached response contains negative token usage")
                raw = _safe_raw_metadata(response.get("raw", {}))
                raw.update({"cache_hit": True, "cache_key": key.digest})
                return ModelResponse(
                    label=label,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw=raw,
                )
            except ResponseCacheError:
                raise
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ResponseCacheError(f"invalid cached response {path}: {exc}") from exc

    def put(self, key: ResponseCacheKey, response: ModelResponse) -> Path:
        if response.label not in key.labels:
            raise ValueError("cannot cache a response with an unknown label")
        if response.input_tokens < 0 or response.output_tokens < 0:
            raise ValueError("cannot cache negative token usage")
        path = self._path(key)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "key": {
                "digest": key.digest,
                "model_id": key.model_id,
                "item_id": key.item_id,
                "prompt_version": key.prompt_version,
                "text_sha256": key.text_sha256,
                "labels": list(key.labels),
            },
            "response": {
                "label": response.label,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "raw": _safe_raw_metadata(response.raw),
            },
            "created_at_unix": time.time(),
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        return path
