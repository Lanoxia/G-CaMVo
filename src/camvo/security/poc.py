"""Reproducible CASIE security-routing proof of concept.

The corpus is real; model responses are simulated because no provider/API key
has been selected.  The simulation exposes class- and difficulty-dependent
abilities and correlated errors, then compares common routing baselines with
CaMVo on exactly the same items.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

from camvo.aggregation import weighted_vote
from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.simulated import SimulatedLLMClient
from camvo.router import CaMVoRouter
from camvo.security.casie import CASIE_LABELS, CasieLoadStats, load_casie_event_items
from camvo.security.metrics import ClassificationMetrics, classification_metrics
from camvo.types import AnnotationItem, ModelPricing

CASIE_MODEL_SPECS: tuple[tuple[str, float, float], ...] = (
    ("soc-tiny", 0.68, 0.05),
    ("soc-fast", 0.76, 0.25),
    ("soc-balanced", 0.82, 0.80),
    ("soc-strong", 0.88, 2.50),
    ("soc-expert", 0.91, 4.00),
)

_SPECIALTY_BONUSES: dict[str, dict[str, float]] = {
    "soc-tiny": {"Phishing": 0.15, "Ransom": 0.08, "DiscoverVulnerability": -0.10},
    "soc-fast": {"Phishing": 0.05, "Databreach": 0.08},
    "soc-balanced": {"PatchVulnerability": 0.06},
    "soc-strong": {"Databreach": 0.08, "Ransom": 0.07},
    "soc-expert": {"DiscoverVulnerability": 0.12, "PatchVulnerability": 0.10},
}


@dataclass(frozen=True, slots=True)
class SecurityEvaluationSummary:
    method: str
    items: int
    metrics: ClassificationMetrics
    average_models: float
    total_cost_usd: float
    normalized_cost_per_million_source_tokens: float
    model_selection_rate: dict[str, float]
    metrics_by_difficulty: dict[str, ClassificationMetrics]
    average_models_by_difficulty: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "items": self.items,
            "metrics": self.metrics.to_dict(),
            "average_models": self.average_models,
            "total_cost_usd": self.total_cost_usd,
            "normalized_cost_per_million_source_tokens": (
                self.normalized_cost_per_million_source_tokens
            ),
            "model_selection_rate": self.model_selection_rate,
            "metrics_by_difficulty": {
                name: metrics.to_dict() for name, metrics in self.metrics_by_difficulty.items()
            },
            "average_models_by_difficulty": self.average_models_by_difficulty,
        }


@dataclass(frozen=True, slots=True)
class CasiePocReport:
    seed: int
    dataset: dict[str, object]
    simulation_disclaimer: str
    model_specs: list[dict[str, object]]
    config: dict[str, object]
    methods: dict[str, SecurityEvaluationSummary]
    cost_savings_vs_full: dict[str, float]
    macro_f1_delta_vs_full: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "dataset": self.dataset,
            "simulation_disclaimer": self.simulation_disclaimer,
            "model_specs": self.model_specs,
            "config": self.config,
            "methods": {name: summary.to_dict() for name, summary in self.methods.items()},
            "cost_savings_vs_full": self.cost_savings_vs_full,
            "macro_f1_delta_vs_full": self.macro_f1_delta_vs_full,
        }


def build_casie_simulated_models(seed: int) -> list[SimulatedLLMClient]:
    return [
        SimulatedLLMClient(
            model_id,
            ModelPricing(input_per_million=price),
            base_accuracy=accuracy,
            seed=seed,
            difficulty_sensitivity=0.95,
            prompt_overhead_tokens=48,
            output_correlation=0.50,
        )
        for model_id, accuracy, price in CASIE_MODEL_SPECS
    ]


def _stratified_sample(
    items: tuple[AnnotationItem, ...] | list[AnnotationItem],
    max_items: int | None,
    seed: int,
) -> list[AnnotationItem]:
    if max_items is None or max_items >= len(items):
        selected = list(items)
        random.Random(seed).shuffle(selected)
        return selected
    if max_items < len(CASIE_LABELS):
        raise ValueError(f"max_items must be at least {len(CASIE_LABELS)}")

    groups: dict[str, list[AnnotationItem]] = defaultdict(list)
    for item in items:
        groups[str(item.metadata["gold_label"])].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    quota, remainder = divmod(max_items, len(CASIE_LABELS))
    selected: list[AnnotationItem] = []
    for index, label in enumerate(CASIE_LABELS):
        take = quota + int(index < remainder)
        selected.extend(groups[label][:take])

    # Future/filtered datasets may have a small class. Fill unused capacity
    # deterministically without duplicating already selected items.
    if len(selected) < max_items:
        selected_ids = {item.item_id for item in selected}
        remaining = [item for item in items if item.item_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_items - len(selected)])
    rng.shuffle(selected)
    return selected


def _attach_simulated_capabilities(items: list[AnnotationItem]) -> list[AnnotationItem]:
    prepared: list[AnnotationItem] = []
    for item in items:
        gold = str(item.metadata["gold_label"])
        bonuses = {
            model_id: _SPECIALTY_BONUSES.get(model_id, {}).get(gold, 0.0)
            for model_id, _accuracy, _price in CASIE_MODEL_SPECS
        }
        prepared.append(
            replace(
                item,
                metadata={**item.metadata, "topic_bonus_by_model": bonuses},
            )
        )
    return prepared


def _source_tokens(models: list[SimulatedLLMClient], items: list[AnnotationItem]) -> int:
    return sum(models[0].count_input_tokens(item) for item in items)


def _summary(
    *,
    method: str,
    items: list[AnnotationItem],
    predictions: list[str],
    selected: list[tuple[str, ...]],
    total_cost: float,
    source_tokens: int,
) -> SecurityEvaluationSummary:
    gold = [str(item.metadata["gold_label"]) for item in items]
    counts = Counter(model_id for subset in selected for model_id in subset)
    metrics_by_difficulty: dict[str, ClassificationMetrics] = {}
    average_models_by_difficulty: dict[str, float] = {}
    for difficulty in ("easy", "medium", "hard"):
        indices = [
            index
            for index, item in enumerate(items)
            if item.metadata.get("difficulty") == difficulty
        ]
        if not indices:
            continue
        metrics_by_difficulty[difficulty] = classification_metrics(
            [gold[index] for index in indices],
            [predictions[index] for index in indices],
            CASIE_LABELS,
        )
        average_models_by_difficulty[difficulty] = mean(
            len(selected[index]) for index in indices
        )
    return SecurityEvaluationSummary(
        method=method,
        items=len(items),
        metrics=classification_metrics(gold, predictions, CASIE_LABELS),
        average_models=mean(len(subset) for subset in selected),
        total_cost_usd=total_cost,
        normalized_cost_per_million_source_tokens=total_cost * 1_000_000 / source_tokens,
        model_selection_rate={
            model_id: counts[model_id] / len(items)
            for model_id, _accuracy, _price in CASIE_MODEL_SPECS
        },
        metrics_by_difficulty=metrics_by_difficulty,
        average_models_by_difficulty=average_models_by_difficulty,
    )


def _evaluate_fixed_subset(
    method: str,
    models: list[SimulatedLLMClient],
    items: list[AnnotationItem],
    selected_ids: tuple[str, ...],
    source_tokens: int,
) -> SecurityEvaluationSummary:
    by_id = {model.model_id: model for model in models}
    chosen = [by_id[model_id] for model_id in selected_ids]
    weights = {model.model_id: model.base_accuracy for model in chosen}
    predictions: list[str] = []
    total_cost = 0.0
    for item in items:
        responses = {model.model_id: model.predict(item) for model in chosen}
        prediction, _ = weighted_vote(
            {model_id: response.label for model_id, response in responses.items()},
            weights,
            item.labels,
        )
        predictions.append(prediction)
        total_cost += sum(
            model.pricing.cost(
                responses[model.model_id].input_tokens,
                responses[model.model_id].output_tokens,
            )
            for model in chosen
        )
    return _summary(
        method=method,
        items=items,
        predictions=predictions,
        selected=[selected_ids] * len(items),
        total_cost=total_cost,
        source_tokens=source_tokens,
    )


def evaluate_casie_items(
    items: list[AnnotationItem],
    *,
    seed: int = 17,
    confidence_threshold: float = 0.97,
    min_models: int = 2,
    warmup_rounds: int = 40,
    embedding_dim: int = 128,
    dataset_stats: CasieLoadStats | None = None,
) -> CasiePocReport:
    if not items:
        raise ValueError("at least one CASIE item is required")
    prepared = _attach_simulated_capabilities(items)
    models = build_casie_simulated_models(seed)
    source_tokens = _source_tokens(models, prepared)
    all_ids = tuple(model.model_id for model in models)
    cheapest_ids = tuple(
        model.model_id
        for model in sorted(
            models,
            key=lambda model: model.pricing.input_per_million,
        )[:3]
    )

    methods: dict[str, SecurityEvaluationSummary] = {}
    methods["cheapest_single"] = _evaluate_fixed_subset(
        "cheapest_single", models, prepared, ("soc-tiny",), source_tokens
    )
    methods["strongest_single"] = _evaluate_fixed_subset(
        "strongest_single", models, prepared, ("soc-expert",), source_tokens
    )
    methods["fixed_cheap_three"] = _evaluate_fixed_subset(
        "fixed_cheap_three", models, prepared, cheapest_ids, source_tokens
    )
    methods["full_ensemble"] = _evaluate_fixed_subset(
        "full_ensemble", models, prepared, all_ids, source_tokens
    )

    config = CaMVoConfig(
        embedding_dim=embedding_dim,
        confidence_threshold=confidence_threshold,
        min_models=min_models,
        exploration_alpha=0.20,
        linucb_regularization=1.0,
        laplace_regularization=1.0,
        warmup_rounds=warmup_rounds,
        confidence_method="exact",
    )
    router = CaMVoRouter(models, HashingTextEmbedder(embedding_dim), config)
    results = router.route_many(prepared)
    methods["camvo"] = _summary(
        method="camvo",
        items=prepared,
        predictions=[result.label for result in results],
        selected=[result.selected_models for result in results],
        total_cost=sum(result.actual_cost for result in results),
        source_tokens=source_tokens,
    )

    full = methods["full_ensemble"]
    cost_savings = {
        name: 1.0 - summary.total_cost_usd / full.total_cost_usd
        for name, summary in methods.items()
    }
    f1_delta = {
        name: summary.metrics.macro_f1 - full.metrics.macro_f1
        for name, summary in methods.items()
    }
    dataset_payload: dict[str, object] = {
        "name": "CASIE",
        "task": "five-way event-subtype classification from a marked event mention",
        "items_evaluated": len(prepared),
        "evaluated_label_counts": dict(
            Counter(str(item.metadata["gold_label"]) for item in prepared)
        ),
    }
    if dataset_stats is not None:
        dataset_payload["load_stats"] = asdict(dataset_stats)

    return CasiePocReport(
        seed=seed,
        dataset=dataset_payload,
        simulation_disclaimer=(
            "CASIE texts and gold labels are real. LLM predictions, prices, and latency proxies "
            "are simulated; this validates the routing/evaluation pipeline, not real API quality."
        ),
        model_specs=[
            {"model_id": model_id, "base_accuracy": accuracy, "input_usd_per_million": price}
            for model_id, accuracy, price in CASIE_MODEL_SPECS
        ],
        config=config.to_dict(),
        methods=methods,
        cost_savings_vs_full=cost_savings,
        macro_f1_delta_vs_full=f1_delta,
    )


def run_casie_poc(
    data_dir: str | Path,
    *,
    max_items: int | None = 1200,
    seed: int = 17,
    confidence_threshold: float = 0.97,
    min_models: int = 2,
    warmup_rounds: int = 40,
    context_chars: int = 320,
    strict_data: bool = False,
) -> CasiePocReport:
    dataset = load_casie_event_items(
        data_dir,
        context_chars=context_chars,
        strict=strict_data,
    )
    items = _stratified_sample(dataset.items, max_items, seed)
    return evaluate_casie_items(
        items,
        seed=seed,
        confidence_threshold=confidence_threshold,
        min_models=min_models,
        warmup_rounds=warmup_rounds,
        dataset_stats=dataset.stats,
    )
