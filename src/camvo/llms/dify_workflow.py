"""Adapter for a published Dify Workflow used as a multi-model gateway."""

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


class DifyWorkflowLLMClient(LLMClient):
    """Call one fixed model branch in a published Dify Workflow.

    Every logical client points to the same Dify app API but supplies a different
    ``workflow_model`` input (for example ``cheap_a`` or ``strong``). The imported
    workflow maps that stable alias to a fixed LLM node, so the online router can
    select models programmatically without changing the Dify UI between calls.

    Dify's public Workflow response exposes aggregate ``total_tokens`` rather than
    a guaranteed prompt/completion split. The adapter therefore preserves the
    exact aggregate count and records an explicit estimated split in ``raw`` for
    proxy-cost accounting. It never presents the split as provider-billed usage.
    """

    def __init__(
        self,
        model_id: str,
        pricing: ModelPricing,
        *,
        workflow_model: str,
        base_url: str,
        api_key: str,
        task: SecurityClassificationTask,
        timeout_seconds: float = 100.0,
        output_key: str = "result",
        user: str = "g-camvo-experiment",
        platform_user: str | None = None,
        user_token: str | None = None,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(model_id, pricing)
        if not workflow_model.strip():
            raise ValueError("workflow_model must not be empty")
        if not base_url.strip() or not api_key.strip():
            raise ValueError("base_url and api_key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not output_key.strip() or not user.strip():
            raise ValueError("output_key and user must not be empty")
        platform_user = (platform_user or "").strip()
        user_token = (user_token or "").strip()
        if bool(platform_user) != bool(user_token):
            raise ValueError(
                "platform_user and user_token must either both be configured or both be empty"
            )
        self.workflow_model = workflow_model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._platform_user = platform_user
        self._user_token = user_token
        self.task = task
        self.timeout_seconds = float(timeout_seconds)
        self.output_key = output_key
        self.user = user
        self._urlopen = urlopen or urllib.request.urlopen

    @property
    def prompt_version(self) -> str:
        """Cache fingerprint that contains no API key."""

        contract = json.dumps(
            {
                "task_prompt_version": self.task.prompt_version,
                "workflow_model": self.workflow_model,
                "base_url": self.base_url,
                "output_key": self.output_key,
                "auth_mode": "adams" if self._platform_user else "dify",
                "adapter_contract": "dify-workflow-v2",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(contract.encode("utf-8")).hexdigest()[:16]
        return f"{self.task.prompt_version}:dify:{digest}"

    def _body(self, item: AnnotationItem) -> bytes:
        prompt = self.task.render(item)
        payload = {
            "inputs": {
                "model_tier": self.workflow_model,
                "system_prompt": prompt.system,
                "user_prompt": prompt.user,
            },
            "response_mode": "blocking",
            "user": self.user,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def count_input_tokens(self, item: AnnotationItem) -> int:
        prompt = self.task.render(item)
        byte_count = len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))
        return max(1, (byte_count + 3) // 4)

    @staticmethod
    def _token_split(total_tokens: int, result_text: str, input_estimate: int) -> tuple[int, int]:
        """Create a conservative, auditable split of Dify aggregate usage."""

        output_estimate = max(1, (len(result_text.encode("utf-8")) + 3) // 4)
        if total_tokens <= 0:
            return input_estimate, output_estimate
        output_tokens = min(total_tokens, output_estimate)
        return total_tokens - output_tokens, output_tokens

    def _request_once(self, item: AnnotationItem) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._platform_user:
            headers.update(
                {
                    "Adams-Platform-User": self._platform_user,
                    "Adams-User-Token": self._user_token,
                }
            )
        request = urllib.request.Request(
            f"{self.base_url}/workflows/run",
            data=self._body(item),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with self._urlopen(request, timeout=self.timeout_seconds) as response:
            raw_bytes = response.read()
        wall_time_seconds = time.perf_counter() - started
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
            data = payload["data"]
            if not isinstance(data, dict) or data.get("status") != "succeeded":
                status = data.get("status", "unknown") if isinstance(data, dict) else "invalid"
                raise ValueError(f"workflow did not succeed (status={status})")
            outputs = data["outputs"]
            if not isinstance(outputs, dict):
                raise ValueError("workflow outputs must be an object")
            result_text = outputs[self.output_key]
            if not isinstance(result_text, str) or not result_text.strip():
                raise ValueError(f"workflow output {self.output_key!r} must be non-empty text")
            prediction = self.task.parse(result_text)
            total_tokens = int(data.get("total_tokens", 0))
            if total_tokens < 0:
                raise ValueError("workflow returned negative total_tokens")
            input_estimate = self.count_input_tokens(item)
            input_tokens, output_tokens = self._token_split(
                total_tokens, result_text, input_estimate
            )
            elapsed_time = float(data.get("elapsed_time", wall_time_seconds))
            if elapsed_time < 0:
                raise ValueError("workflow returned negative elapsed_time")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Dify Workflow response: {exc}") from exc
        return ModelResponse(
            label=prediction.label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw={
                "confidence": prediction.confidence,
                "rationale": prediction.rationale,
                "workflow_model": self.workflow_model,
                "workflow_run_id": payload.get("workflow_run_id", data.get("id")),
                "task_id": payload.get("task_id"),
                "dify_total_tokens": total_tokens,
                "token_split": "estimated_from_dify_aggregate_total",
                "elapsed_time_seconds": elapsed_time,
                "wall_time_seconds": wall_time_seconds,
            },
        )

    def predict(self, item: AnnotationItem) -> ModelResponse:
        """Perform exactly one provider attempt; caller controls retries and cache."""

        try:
            return self._request_once(item)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Dify Workflow HTTP error {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError("Dify Workflow request failed") from exc
        except ValueError as exc:
            # A model branch can occasionally ignore the strict JSON contract.
            # This is an invocation failure, not an experiment-configuration
            # error: no response is cached and the budget ledger has already
            # recorded the failed attempt.  Surface it as recoverable so the
            # resumable runner can retry this one uncached cell instead of
            # aborting the entire frozen response-matrix collection.
            raise RuntimeError("Dify Workflow returned an invalid structured response") from exc
