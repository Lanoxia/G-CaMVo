"""Dependency-free adapter for Anthropic's native Messages API."""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from camvo.llms.base import LLMClient
from camvo.security.tasks import SecurityClassificationTask
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class AnthropicMessagesLLMClient(LLMClient):
    """Call one pinned Claude model without routing through an aggregator."""

    def __init__(
        self,
        model_id: str,
        pricing: ModelPricing,
        *,
        provider_model: str,
        api_key: str,
        task: SecurityClassificationTask,
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        max_output_tokens: int = 128,
        temperature: float | None = 0.0,
        extra_body: dict[str, Any] | None = None,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(model_id, pricing)
        if not provider_model.strip() or not api_key.strip() or not base_url.strip():
            raise ValueError("provider_model, api_key, and base_url must not be empty")
        if timeout_seconds <= 0 or max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("invalid timeout/retry configuration")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if temperature is not None and not 0.0 <= float(temperature) <= 1.0:
            raise ValueError("Anthropic temperature must be between 0 and 1, or null")
        self.provider_model = provider_model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.task = task
        self.anthropic_version = anthropic_version
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = None if temperature is None else float(temperature)
        self.extra_body = dict(extra_body or {})
        self._urlopen = urlopen or urllib.request.urlopen

    @property
    def prompt_version(self) -> str:
        contract = json.dumps(
            {
                "task_prompt_version": self.task.prompt_version,
                "provider_model": self.provider_model,
                "base_url": self.base_url,
                "anthropic_version": self.anthropic_version,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "extra_body": self.extra_body,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(contract.encode("utf-8")).hexdigest()[:16]
        return f"{self.task.prompt_version}:{digest}"

    def _body(self, item: AnnotationItem) -> bytes:
        prompt = self.task.render(item)
        body: dict[str, Any] = {
            **self.extra_body,
            "model": self.provider_model,
            "max_tokens": self.max_output_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def count_input_tokens(self, item: AnnotationItem) -> int:
        prompt = self.task.render(item)
        byte_count = len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))
        return max(1, (byte_count + 3) // 4)

    @staticmethod
    def _content(payload: dict[str, Any]) -> str:
        blocks = payload["content"]
        if not isinstance(blocks, list):
            raise ValueError("provider response content is not a list")
        parts = [
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not parts:
            raise ValueError("provider response does not contain textual content")
        return "".join(parts)

    def _request_once(self, item: AnnotationItem) -> ModelResponse:
        request = urllib.request.Request(
            f"{self.base_url}/messages",
            data=self._body(item),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json",
                "accept": "application/json",
            },
            method="POST",
        )
        with self._urlopen(request, timeout=self.timeout_seconds) as response:
            raw_bytes = response.read()
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
            prediction = self.task.parse(self._content(payload))
            usage = payload.get("usage", {})
            input_tokens = int(usage.get("input_tokens", self.count_input_tokens(item)))
            output_tokens = int(usage.get("output_tokens", self.max_output_tokens))
            if input_tokens < 0 or output_tokens < 0:
                raise ValueError("provider returned negative token usage")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid provider response: {exc}") from exc
        return ModelResponse(
            label=prediction.label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw={
                "confidence": prediction.confidence,
                "rationale": prediction.rationale,
                "provider_model": self.provider_model,
            },
        )

    def predict(self, item: AnnotationItem) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(item)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"provider HTTP error {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("provider request failed after retries") from exc
                delay = None
            if delay is None:
                delay = self.retry_backoff_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)
        raise RuntimeError("unreachable provider retry state")
