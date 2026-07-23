"""Small offline validation harness calibrated from the paper's MMLU table."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean

from camvo.aggregation import weighted_vote
from camvo.config import CaMVoConfig, ConfidenceMethod
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.simulated import SimulatedLLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelPricing

MMLU_MODEL_SPECS: tuple[tuple[str, float, float], ...] = (
    ("o3-mini", 0.8592, 1.10),
    ("claude-3-7-sonnet", 0.8565, 3.00),
    ("o1-mini", 0.8482, 1.10),
    ("gpt-4o", 0.8358, 2.50),
    ("llama-3.3-70b", 0.8170, 0.59),
    ("llama-3.1-8b", 0.6801, 0.05),
    ("claude-3-5-haiku", 0.6409, 0.80),
)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    method: str
    items: int
    accuracy: float
    average_models: float
    normalized_cost_per_million_source_tokens: float
    total_actual_cost: float
    accuracy_by_difficulty: dict[str, float]
    average_models_by_difficulty: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationReport:
    seed: int
    config: dict[str, object]
    camvo: EvaluationSummary
    full_ensemble: EvaluationSummary
    cost_savings_fraction: float
    accuracy_difference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "config": self.config,
            "camvo": self.camvo.to_dict(),
            "full_ensemble": self.full_ensemble.to_dict(),
            "cost_savings_fraction": self.cost_savings_fraction,
            "accuracy_difference": self.accuracy_difference,
        }


def build_paper_calibrated_models(seed: int = 0) -> list[SimulatedLLMClient]:
    return [
        SimulatedLLMClient(
            model_id,
            ModelPricing(input_per_million=cost),
            base_accuracy=accuracy,
            seed=seed,
            difficulty_sensitivity=1.0,
            prompt_overhead_tokens=16,
            output_correlation=0.65,
        )
        for model_id, accuracy, cost in MMLU_MODEL_SPECS
    ]


def generate_synthetic_items(count: int, seed: int = 0) -> list[AnnotationItem]:
    """Create context-visible difficulty clusters with hidden gold labels."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    labels = ("A", "B", "C", "D")
    difficulties = (
        ("easy", -0.8, 0.35),
        ("medium", 0.0, 0.40),
        ("hard", 0.8, 0.25),
    )
    topics = ("mathematics", "history", "law", "computing", "science")
    cumulative = []
    running = 0.0
    for name, score, probability in difficulties:
        running += probability
        cumulative.append((running, name, score))

    model_ids = [spec[0] for spec in MMLU_MODEL_SPECS]
    items: list[AnnotationItem] = []
    for index in range(count):
        draw = rng.random()
        _, difficulty_name, difficulty_score = next(
            entry for entry in cumulative if draw <= entry[0]
        )
        topic = rng.choice(topics)
        gold_label = rng.choice(labels)
        topic_bonus = {
            model_id: 0.12 * (2.0 * rng.random() - 1.0) for model_id in model_ids
        }
        text = (
            f"benchmark question difficulty_{difficulty_name} topic_{topic} "
            f"reasoning profile {difficulty_name} {topic} item_{index % 17}"
        )
        items.append(
            AnnotationItem(
                item_id=f"synthetic-{seed}-{index}",
                text=text,
                labels=labels,
                metadata={
                    "gold_label": gold_label,
                    "difficulty": difficulty_name,
                    "difficulty_score": difficulty_score,
                    "topic": topic,
                    "topic_bonus_by_model": topic_bonus,
                },
            )
        )
    return items


def _base_source_tokens(model: SimulatedLLMClient, items: list[AnnotationItem]) -> int:
    return sum(model.count_input_tokens(item) for item in items)


def _group_metrics(
    items: list[AnnotationItem],
    predictions: list[str],
    model_counts: list[int],
) -> tuple[dict[str, float], dict[str, float]]:
    accuracy: dict[str, float] = {}
    average_models: dict[str, float] = {}
    for difficulty in ("easy", "medium", "hard"):
        indices = [
            index
            for index, item in enumerate(items)
            if item.metadata["difficulty"] == difficulty
        ]
        accuracy[difficulty] = mean(
            predictions[index] == items[index].metadata["gold_label"] for index in indices
        )
        average_models[difficulty] = mean(model_counts[index] for index in indices)
    return accuracy, average_models


def evaluate_full_ensemble(
    models: list[SimulatedLLMClient],
    items: list[AnnotationItem],
) -> EvaluationSummary:
    weights = {model.model_id: model.base_accuracy for model in models}
    predictions: list[str] = []
    total_cost = 0.0
    for item in items:
        responses = {model.model_id: model.predict(item) for model in models}
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
            for model in models
        )
    counts = [len(models)] * len(items)
    by_difficulty, counts_by_difficulty = _group_metrics(items, predictions, counts)
    source_tokens = _base_source_tokens(models[0], items)
    return EvaluationSummary(
        method="full_ensemble_oracle_weights",
        items=len(items),
        accuracy=mean(
            prediction == item.metadata["gold_label"]
            for prediction, item in zip(predictions, items, strict=True)
        ),
        average_models=float(len(models)),
        normalized_cost_per_million_source_tokens=total_cost * 1_000_000 / source_tokens,
        total_actual_cost=total_cost,
        accuracy_by_difficulty=by_difficulty,
        average_models_by_difficulty=counts_by_difficulty,
    )


def evaluate_camvo(
    models: list[SimulatedLLMClient],
    items: list[AnnotationItem],
    config: CaMVoConfig,
) -> tuple[EvaluationSummary, CaMVoRouter]:
    embedder = HashingTextEmbedder(config.embedding_dim)
    router = CaMVoRouter(models, embedder, config)
    results = router.route_many(items)
    predictions = [result.label for result in results]
    counts = [len(result.selected_models) for result in results]
    total_cost = sum(result.actual_cost for result in results)
    by_difficulty, counts_by_difficulty = _group_metrics(items, predictions, counts)
    source_tokens = _base_source_tokens(models[0], items)
    summary = EvaluationSummary(
        method="camvo",
        items=len(items),
        accuracy=mean(
            prediction == item.metadata["gold_label"]
            for prediction, item in zip(predictions, items, strict=True)
        ),
        average_models=mean(counts),
        normalized_cost_per_million_source_tokens=total_cost * 1_000_000 / source_tokens,
        total_actual_cost=total_cost,
        accuracy_by_difficulty=by_difficulty,
        average_models_by_difficulty=counts_by_difficulty,
    )
    return summary, router


def run_simulation(
    *,
    item_count: int = 600,
    seed: int = 7,
    confidence_threshold: float = 0.90,
    min_models: int = 3,
    warmup_rounds: int = 20,
    confidence_method: ConfidenceMethod = "exact",
) -> tuple[SimulationReport, CaMVoRouter]:
    items = generate_synthetic_items(item_count, seed)
    models = build_paper_calibrated_models(seed)
    config = CaMVoConfig(
        embedding_dim=128,
        confidence_threshold=confidence_threshold,
        min_models=min_models,
        exploration_alpha=0.25,
        linucb_regularization=1.0,
        laplace_regularization=1.0,
        warmup_rounds=warmup_rounds,
        confidence_method=confidence_method,
    )
    full = evaluate_full_ensemble(models, items)
    camvo, router = evaluate_camvo(models, items, config)
    savings = 1.0 - (
        camvo.normalized_cost_per_million_source_tokens
        / full.normalized_cost_per_million_source_tokens
    )
    report = SimulationReport(
        seed=seed,
        config=config.to_dict(),
        camvo=camvo,
        full_ensemble=full,
        cost_savings_fraction=savings,
        accuracy_difference=camvo.accuracy - full.accuracy,
    )
    return report, router
