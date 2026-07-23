"""Unified baseline/CaMVo/G-CaMVo evaluation for security classification tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Mapping

from camvo.aggregation import weighted_vote
from camvo.ccamvo_router import CorrelatedCaMVoConfig, CorrelatedCaMVoRouter
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.graph_router import GraphCaMVoConfig, GraphCaMVoRouter
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.trace_router import TraceGraphCaMVoConfig, TraceGraphCaMVoRouter
from camvo.security.metrics import ClassificationMetrics, classification_metrics
from camvo.types import AnnotationItem, RoutingResult


@dataclass(frozen=True, slots=True)
class ExperimentModelSpec:
    model_id: str
    prior_quality: float
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not 0 < self.prior_quality < 1:
            raise ValueError("prior_quality must be in (0, 1)")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class RoutingMethodSummary:
    method: str
    items: int
    metrics: ClassificationMetrics
    total_cost_usd: float
    average_cost_usd: float
    average_models: float
    escalation_rate: float
    full_pool_rate: float
    model_selection_rate: dict[str, float]
    average_parallel_latency_ms: float
    p95_parallel_latency_ms: float
    average_sequential_latency_ms: float
    average_subset_confidence: float | None
    metrics_by_difficulty: dict[str, ClassificationMetrics]
    abstention_rate: float = 0.0
    average_decision_risk: float | None = None
    graph_use_rate: float = 0.0
    average_graph_evidence_weight: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        payload["metrics_by_difficulty"] = {
            key: value.to_dict() for key, value in self.metrics_by_difficulty.items()
        }
        return payload


@dataclass(frozen=True, slots=True)
class SecurityExperimentReport:
    dataset: dict[str, object]
    model_specs: list[dict[str, object]]
    camvo_config: dict[str, object]
    graph_config: dict[str, object] | None
    ccamvo_config: dict[str, object] | None
    trace_config: dict[str, object] | None
    methods: dict[str, RoutingMethodSummary]
    cost_savings_vs_full: dict[str, float]
    macro_f1_delta_vs_full: dict[str, float]
    macro_f1_delta_gcamvo_vs_camvo: float | None
    disclaimer: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "model_specs": self.model_specs,
            "camvo_config": self.camvo_config,
            "graph_config": self.graph_config,
            "ccamvo_config": self.ccamvo_config,
            "trace_config": self.trace_config,
            "methods": {name: value.to_dict() for name, value in self.methods.items()},
            "cost_savings_vs_full": self.cost_savings_vs_full,
            "macro_f1_delta_vs_full": self.macro_f1_delta_vs_full,
            "macro_f1_delta_gcamvo_vs_camvo": self.macro_f1_delta_gcamvo_vs_camvo,
            "disclaimer": self.disclaimer,
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize(
    *,
    method: str,
    items: list[AnnotationItem],
    predictions: list[str],
    selections: list[tuple[str, ...]],
    costs: list[float],
    model_ids: tuple[str, ...],
    latency_ms: Mapping[str, float],
    min_models: int,
    subset_confidences: list[float] | None = None,
    abstentions: list[bool] | None = None,
    decision_risks: list[float | None] | None = None,
    graph_evidence_weights: list[float] | None = None,
) -> RoutingMethodSummary:
    if not (len(items) == len(predictions) == len(selections) == len(costs)):
        raise ValueError("evaluation vectors must have equal lengths")
    labels = items[0].labels
    gold = [str(item.metadata["gold_label"]) for item in items]
    selection_counts = Counter(model_id for subset in selections for model_id in subset)
    parallel_latency = [max(latency_ms[model_id] for model_id in subset) for subset in selections]
    sequential_latency = [sum(latency_ms[model_id] for model_id in subset) for subset in selections]
    by_difficulty: dict[str, ClassificationMetrics] = {}
    for difficulty in ("easy", "medium", "hard"):
        indices = [
            index
            for index, item in enumerate(items)
            if item.metadata.get("difficulty") == difficulty
        ]
        if indices:
            by_difficulty[difficulty] = classification_metrics(
                [gold[index] for index in indices],
                [predictions[index] for index in indices],
                labels,
            )
    return RoutingMethodSummary(
        method=method,
        items=len(items),
        metrics=classification_metrics(gold, predictions, labels),
        total_cost_usd=sum(costs),
        average_cost_usd=mean(costs),
        average_models=mean(len(subset) for subset in selections),
        escalation_rate=(
            sum(len(subset) > min_models for subset in selections) / len(selections)
        ),
        full_pool_rate=(
            sum(len(subset) == len(model_ids) for subset in selections) / len(selections)
        ),
        model_selection_rate={
            model_id: selection_counts[model_id] / len(items) for model_id in model_ids
        },
        average_parallel_latency_ms=mean(parallel_latency),
        p95_parallel_latency_ms=_percentile(parallel_latency, 0.95),
        average_sequential_latency_ms=mean(sequential_latency),
        average_subset_confidence=(
            mean(subset_confidences) if subset_confidences is not None else None
        ),
        metrics_by_difficulty=by_difficulty,
        abstention_rate=(
            mean(float(value) for value in abstentions) if abstentions is not None else 0.0
        ),
        average_decision_risk=(
            mean(value for value in decision_risks if value is not None)
            if decision_risks is not None and any(value is not None for value in decision_risks)
            else None
        ),
        graph_use_rate=(
            mean(float(value > 0) for value in graph_evidence_weights)
            if graph_evidence_weights is not None
            else 0.0
        ),
        average_graph_evidence_weight=(
            mean(graph_evidence_weights) if graph_evidence_weights is not None else 0.0
        ),
    )


def _evaluate_fixed(
    method: str,
    models: Mapping[str, LLMClient],
    items: list[AnnotationItem],
    selected_ids: tuple[str, ...],
    prior_quality: Mapping[str, float],
    latency_ms: Mapping[str, float],
    min_models: int,
) -> RoutingMethodSummary:
    predictions: list[str] = []
    costs: list[float] = []
    for item in items:
        responses = {model_id: models[model_id].predict(item) for model_id in selected_ids}
        for model_id, response in responses.items():
            if response.label not in item.labels:
                raise ValueError(f"model {model_id} returned an unknown label")
        prediction, _ties = weighted_vote(
            {model_id: response.label for model_id, response in responses.items()},
            {model_id: prior_quality[model_id] for model_id in selected_ids},
            item.labels,
        )
        predictions.append(prediction)
        costs.append(
            sum(
                models[model_id].pricing.cost(response.input_tokens, response.output_tokens)
                for model_id, response in responses.items()
            )
        )
    return _summarize(
        method=method,
        items=items,
        predictions=predictions,
        selections=[selected_ids] * len(items),
        costs=costs,
        model_ids=tuple(sorted(models)),
        latency_ms=latency_ms,
        min_models=min_models,
    )


def _evaluate_router(
    method: str,
    router: CaMVoRouter,
    items: list[AnnotationItem],
    latency_ms: Mapping[str, float],
) -> RoutingMethodSummary:
    results: list[RoutingResult] = router.route_many(items)
    return _summarize(
        method=method,
        items=items,
        predictions=[result.label for result in results],
        selections=[result.selected_models for result in results],
        costs=[result.actual_cost for result in results],
        model_ids=tuple(sorted(router.models)),
        latency_ms=latency_ms,
        min_models=router.config.min_models,
        subset_confidences=[result.subset_confidence for result in results],
        abstentions=[result.abstained for result in results],
        decision_risks=[result.decision_risk for result in results],
        graph_evidence_weights=[result.graph_evidence_weight for result in results],
    )


def evaluate_security_strategies(
    items: list[AnnotationItem],
    models: list[LLMClient],
    model_specs: list[ExperimentModelSpec],
    embedder: EmbeddingProvider,
    camvo_config: CaMVoConfig,
    *,
    dataset_metadata: dict[str, object],
    graph_neighborhood: Mapping[str, Mapping[str, float]] | None = None,
    graph_config: GraphCaMVoConfig | None = None,
    ccamvo_config: CorrelatedCaMVoConfig | None = None,
    trace_config: TraceGraphCaMVoConfig | None = None,
    fixed_cheap_models: int = 3,
    disclaimer: str,
) -> SecurityExperimentReport:
    """Evaluate all required policies on identical items and model outputs."""

    if not items:
        raise ValueError("items must not be empty")
    if len({item.labels for item in items}) != 1:
        raise ValueError("all items must use the same label space")
    if any(item.metadata.get("gold_label") not in item.labels for item in items):
        raise ValueError("every evaluation item must contain a valid gold_label")
    model_map = {model.model_id: model for model in models}
    specs = {spec.model_id: spec for spec in model_specs}
    if set(model_map) != set(specs):
        raise ValueError("model_specs must match the model pool exactly")
    if embedder.dimension != camvo_config.embedding_dim:
        raise ValueError("embedder dimension does not match CaMVo config")
    if not 1 <= fixed_cheap_models <= len(models):
        raise ValueError("fixed_cheap_models is outside the model-pool range")
    prior = {model_id: specs[model_id].prior_quality for model_id in model_map}
    latency = {model_id: specs[model_id].latency_ms for model_id in model_map}
    all_ids = tuple(sorted(model_map))
    cheapest_order = sorted(
        model_map,
        key=lambda model_id: (
            model_map[model_id].pricing.input_per_million
            + model_map[model_id].pricing.output_per_million,
            model_id,
        ),
    )
    strongest = max(model_map, key=lambda model_id: (prior[model_id], model_id))
    fixed = tuple(cheapest_order[:fixed_cheap_models])

    methods: dict[str, RoutingMethodSummary] = {}
    for name, subset in (
        ("cheapest_single", (cheapest_order[0],)),
        ("strongest_single", (strongest,)),
        ("fixed_cheap", fixed),
        ("full_ensemble", all_ids),
    ):
        methods[name] = _evaluate_fixed(
            name,
            model_map,
            items,
            subset,
            prior,
            latency,
            camvo_config.min_models,
        )
    methods["camvo"] = _evaluate_router(
        "camvo",
        CaMVoRouter(models, embedder, camvo_config),
        items,
        latency,
    )
    if ccamvo_config is not None:
        methods["ccamvo"] = _evaluate_router(
            "ccamvo",
            CorrelatedCaMVoRouter(models, embedder, camvo_config, ccamvo_config),
            items,
            latency,
        )
    if (graph_neighborhood is None) != (graph_config is None):
        raise ValueError("graph_neighborhood and graph_config must be supplied together")
    if graph_neighborhood is not None and graph_config is not None:
        from camvo.graph_router import StaticGraphNeighborhood

        static_neighborhood = StaticGraphNeighborhood(graph_neighborhood)
        methods["gcamvo"] = _evaluate_router(
            "gcamvo",
            GraphCaMVoRouter(
                models,
                embedder,
                camvo_config,
                graph_config,
                static_neighborhood,
            ),
            items,
            latency,
        )
        if trace_config is not None:
            methods["trace_gcamvo"] = _evaluate_router(
                "trace_gcamvo",
                TraceGraphCaMVoRouter(
                    models,
                    embedder,
                    camvo_config,
                    trace_config,
                    static_neighborhood,
                ),
                items,
                latency,
            )

    full = methods["full_ensemble"]
    savings = {
        name: (
            0.0
            if full.total_cost_usd == 0
            else 1.0 - summary.total_cost_usd / full.total_cost_usd
        )
        for name, summary in methods.items()
    }
    f1_delta = {
        name: summary.metrics.macro_f1 - full.metrics.macro_f1
        for name, summary in methods.items()
    }
    graph_delta = None
    if "gcamvo" in methods:
        graph_delta = methods["gcamvo"].metrics.macro_f1 - methods["camvo"].metrics.macro_f1
    model_payload = []
    for model_id in sorted(model_map):
        model = model_map[model_id]
        model_payload.append(
            {
                "model_id": model_id,
                "prior_quality": prior[model_id],
                "latency_ms": latency[model_id],
                "input_usd_per_million": model.pricing.input_per_million,
                "output_usd_per_million": model.pricing.output_per_million,
            }
        )
    return SecurityExperimentReport(
        dataset=dataset_metadata,
        model_specs=model_payload,
        camvo_config=camvo_config.to_dict(),
        graph_config=asdict(graph_config) if graph_config is not None else None,
        ccamvo_config=(asdict(ccamvo_config) if ccamvo_config is not None else None),
        trace_config=asdict(trace_config) if trace_config is not None else None,
        methods=methods,
        cost_savings_vs_full=savings,
        macro_f1_delta_vs_full=f1_delta,
        macro_f1_delta_gcamvo_vs_camvo=graph_delta,
        disclaimer=disclaimer,
    )
