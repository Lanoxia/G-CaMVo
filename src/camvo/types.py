"""Shared immutable data types used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AnnotationItem:
    """One unlabeled item presented to the router.

    ``metadata`` may contain a ``gold_label`` for simulation or evaluation. The
    router never reads that field; only a simulator/evaluator may use it.
    """

    item_id: str
    text: str
    labels: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if len(self.labels) < 2:
            raise ValueError("at least two candidate labels are required")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Provider pricing in US dollars per one million tokens."""

    input_per_million: float
    output_per_million: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("token prices must be non-negative")

    def cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized response returned by any model provider adapter."""

    label: str
    input_tokens: int
    output_tokens: int = 1
    raw: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModelScore:
    """All per-model quantities computed before subset selection."""

    model_id: str
    predicted_agreement: float
    uncertainty: float
    lcb_score: float
    calibrated_lower_bound: float
    smoothed_lower_bound: float
    historical_agreement: float
    vote_weight: float
    estimated_cost: float
    graph_regularized_lower_bound: float | None = None
    graph_neighbor_count: int = 0
    graph_neighbor_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Auditable output from one CaMVo round."""

    item_id: str
    label: str
    selected_models: tuple[str, ...]
    subset_confidence: float
    estimated_cost: float
    actual_cost: float
    responses: dict[str, str]
    scores: dict[str, ModelScore]
    errors: dict[str, str]
    warmup: bool
    posterior: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    decision_risk: float | None = None
    risk_tolerance: float | None = None
    graph_evidence_weight: float = 0.0
    routing_trace: tuple[dict[str, Any], ...] = ()
