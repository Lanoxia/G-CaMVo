"""Small adapter that turns an arbitrary provider function into an LLM client."""

from __future__ import annotations

from collections.abc import Callable

from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class CallableLLMClient(LLMClient):
    """Integrate an SDK without coupling the core package to that provider.

    The callable is responsible for retries, provider-specific prompts, and
    returning a normalized :class:`ModelResponse`.
    """

    def __init__(
        self,
        model_id: str,
        pricing: ModelPricing,
        predictor: Callable[[AnnotationItem], ModelResponse],
        token_counter: Callable[[AnnotationItem], int] | None = None,
    ) -> None:
        super().__init__(model_id, pricing)
        self._predictor = predictor
        self._token_counter = token_counter

    def predict(self, item: AnnotationItem) -> ModelResponse:
        return self._predictor(item)

    def count_input_tokens(self, item: AnnotationItem) -> int:
        if self._token_counter is None:
            return super().count_input_tokens(item)
        count = int(self._token_counter(item))
        if count <= 0:
            raise ValueError("token counter must return a positive value")
        return count

