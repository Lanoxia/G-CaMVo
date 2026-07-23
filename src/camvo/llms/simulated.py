"""Deterministic, benchmark-calibrated model simulator."""

from __future__ import annotations

import hashlib
import math

from camvo.llms.base import LLMClient
from camvo.mathutils import logit, sigmoid
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class SimulatedLLMClient(LLMClient):
    """Offline LLM with context-dependent correctness.

    ``base_accuracy`` is converted to a latent ability. Item metadata may
    contain ``difficulty_score`` where positive values are harder. Predictions
    are deterministic for a fixed ``seed``, model id, and item id, which makes
    regression tests reproducible.
    """

    def __init__(
        self,
        model_id: str,
        pricing: ModelPricing,
        *,
        base_accuracy: float,
        seed: int = 0,
        difficulty_sensitivity: float = 1.0,
        prompt_overhead_tokens: int = 16,
        output_correlation: float = 0.0,
    ) -> None:
        super().__init__(model_id, pricing)
        if not 0 < base_accuracy < 1:
            raise ValueError("base_accuracy must be in (0, 1)")
        if difficulty_sensitivity < 0:
            raise ValueError("difficulty_sensitivity must be non-negative")
        if prompt_overhead_tokens < 0:
            raise ValueError("prompt_overhead_tokens must be non-negative")
        if not 0 <= output_correlation <= 1:
            raise ValueError("output_correlation must be in [0, 1]")
        self.base_accuracy = base_accuracy
        self.seed = seed
        self.difficulty_sensitivity = difficulty_sensitivity
        self.prompt_overhead_tokens = prompt_overhead_tokens
        self.output_correlation = output_correlation

    def _uniform(self, item_id: str, stream: str) -> float:
        payload = f"{self.seed}|{self.model_id}|{item_id}|{stream}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return (value + 0.5) / (2**64)

    def _shared_uniform(self, item_id: str, stream: str) -> float:
        payload = f"{self.seed}|shared|{item_id}|{stream}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return (value + 0.5) / (2**64)

    def probability_correct(self, item: AnnotationItem) -> float:
        difficulty = float(item.metadata.get("difficulty_score", 0.0))
        topic_bonus_by_model = item.metadata.get("topic_bonus_by_model", {})
        topic_bonus = float(topic_bonus_by_model.get(self.model_id, 0.0))
        ability = float(logit(self.base_accuracy))
        probability = sigmoid(ability - self.difficulty_sensitivity * difficulty + topic_bonus)
        return min(1 - 1e-6, max(1e-6, probability))

    def count_input_tokens(self, item: AnnotationItem) -> int:
        return super().count_input_tokens(item) + self.prompt_overhead_tokens

    def predict(self, item: AnnotationItem) -> ModelResponse:
        gold_label = item.metadata.get("gold_label")
        if gold_label not in item.labels:
            raise ValueError("simulation item must contain a valid metadata['gold_label']")

        shared_mode = (
            self._shared_uniform(item.item_id, "correlation-mode") < self.output_correlation
        )
        correctness_draw = (
            self._shared_uniform(item.item_id, "correct")
            if shared_mode
            else self._uniform(item.item_id, "correct")
        )
        if correctness_draw < self.probability_correct(item):
            label = str(gold_label)
        else:
            alternatives = [candidate for candidate in item.labels if candidate != gold_label]
            wrong_draw = (
                self._shared_uniform(item.item_id, "wrong-label")
                if shared_mode
                else self._uniform(item.item_id, "wrong-label")
            )
            index = min(
                len(alternatives) - 1,
                math.floor(wrong_draw * len(alternatives)),
            )
            label = alternatives[index]

        return ModelResponse(
            label=label,
            input_tokens=self.count_input_tokens(item),
            output_tokens=1,
        )
