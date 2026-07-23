"""Provider-independent LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class LLMClient(ABC):
    """Minimal contract required by the CaMVo router.

    Provider adapters should normalize their outputs to one of ``item.labels``
    and report actual token usage whenever the provider exposes it.
    """

    def __init__(self, model_id: str, pricing: ModelPricing) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        self._model_id = model_id
        self._pricing = pricing

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def pricing(self) -> ModelPricing:
        return self._pricing

    @abstractmethod
    def predict(self, item: AnnotationItem) -> ModelResponse:
        """Return one normalized classification response."""

    def count_input_tokens(self, item: AnnotationItem) -> int:
        """Conservative fallback when a provider tokenizer is unavailable."""

        return max(1, (len(item.text.encode("utf-8")) + 3) // 4)

    def estimate_cost(self, item: AnnotationItem) -> float:
        """Estimate cost before the model is called."""

        return self.pricing.cost(self.count_input_tokens(item), output_tokens=1)

