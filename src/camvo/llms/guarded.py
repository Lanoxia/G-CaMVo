"""Composable cache and budget protection for any LLM client."""

from __future__ import annotations

import math

from camvo.budget import BudgetGuard
from camvo.llms.base import LLMClient
from camvo.llms.cache import FileResponseCache
from camvo.types import AnnotationItem, ModelResponse


class CachedBudgetedLLMClient(LLMClient):
    """Call a provider at most once per cache key and enforce a shared budget."""

    def __init__(
        self,
        delegate: LLMClient,
        cache: FileResponseCache,
        budget: BudgetGuard,
        *,
        prompt_version: str,
        max_output_tokens: int = 8,
        input_token_margin: float = 1.25,
    ) -> None:
        super().__init__(delegate.model_id, delegate.pricing)
        if not prompt_version.strip():
            raise ValueError("prompt_version must not be empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not math.isfinite(input_token_margin) or input_token_margin < 1:
            raise ValueError("input_token_margin must be finite and at least 1")
        self.delegate = delegate
        self.cache = cache
        self.budget = budget
        self.prompt_version = prompt_version
        self.max_output_tokens = max_output_tokens
        self.input_token_margin = float(input_token_margin)

    def count_input_tokens(self, item: AnnotationItem) -> int:
        return self.delegate.count_input_tokens(item)

    def _reserved_cost(self, item: AnnotationItem) -> float:
        estimated_input = math.ceil(
            self.delegate.count_input_tokens(item) * self.input_token_margin
        )
        return self.pricing.cost(estimated_input, self.max_output_tokens)

    @staticmethod
    def _validate_response(response: ModelResponse, item: AnnotationItem) -> None:
        if response.label not in item.labels:
            raise ValueError(f"provider returned unknown label {response.label!r}")
        if response.input_tokens < 0 or response.output_tokens < 0:
            raise ValueError("provider returned negative token usage")

    def predict(self, item: AnnotationItem) -> ModelResponse:
        key = self.cache.key_for(self.model_id, item, self.prompt_version)
        cached = self.cache.get(key)
        if cached is not None:
            self.budget.record_cache_hit()
            return cached

        reservation = self.budget.reserve(
            model_id=self.model_id,
            item_id=item.item_id,
            estimated_max_cost_usd=self._reserved_cost(item),
        )
        try:
            response = self.delegate.predict(item)
        except Exception as exc:
            self.budget.fail(reservation, exc)
            raise

        actual_cost = reservation.reserved_usd
        if response.input_tokens >= 0 and response.output_tokens >= 0:
            actual_cost = self.pricing.cost(response.input_tokens, response.output_tokens)
        try:
            self._validate_response(response, item)
            self.cache.put(key, response)
        except Exception:
            self.budget.commit(
                reservation,
                actual_cost_usd=actual_cost,
                input_tokens=max(0, response.input_tokens),
                output_tokens=max(0, response.output_tokens),
            )
            raise
        self.budget.commit(
            reservation,
            actual_cost_usd=actual_cost,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response
