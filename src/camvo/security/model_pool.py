"""Reproducible no-key security model pool and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from camvo.llms.simulated import SimulatedLLMClient
from camvo.security.experiment import ExperimentModelSpec
from camvo.types import AnnotationItem, ModelPricing


@dataclass(frozen=True, slots=True)
class SimulatedSecurityModelSpec:
    model_id: str
    base_accuracy: float
    input_usd_per_million: float
    output_usd_per_million: float
    latency_ms: float
    difficulty_sensitivity: float = 0.95


DEFAULT_SECURITY_MODEL_POOL: tuple[SimulatedSecurityModelSpec, ...] = (
    SimulatedSecurityModelSpec("soc-tiny", 0.68, 0.05, 0.20, 120.0, 1.05),
    SimulatedSecurityModelSpec("soc-fast", 0.76, 0.25, 1.00, 260.0, 1.00),
    SimulatedSecurityModelSpec("soc-balanced", 0.82, 0.80, 3.20, 520.0, 0.95),
    SimulatedSecurityModelSpec("soc-strong", 0.88, 2.50, 10.00, 1_050.0, 0.85),
    SimulatedSecurityModelSpec("soc-expert", 0.91, 4.00, 16.00, 1_800.0, 0.75),
)

CASIE_SPECIALTIES: dict[str, dict[str, float]] = {
    "soc-tiny": {"Phishing": 0.15, "Ransom": 0.08, "DiscoverVulnerability": -0.10},
    "soc-fast": {"Phishing": 0.05, "Databreach": 0.08},
    "soc-balanced": {"PatchVulnerability": 0.06},
    "soc-strong": {"Databreach": 0.08, "Ransom": 0.07},
    "soc-expert": {"DiscoverVulnerability": 0.12, "PatchVulnerability": 0.10},
}

OPTC_SPECIALTIES: dict[str, dict[str, float]] = {
    "soc-tiny": {"benign": 0.08, "malicious": -0.05},
    "soc-fast": {"benign": 0.04},
    "soc-balanced": {"malicious": 0.04},
    "soc-strong": {"malicious": 0.10},
    "soc-expert": {"malicious": 0.16},
}


def build_simulated_security_pool(
    seed: int,
    *,
    output_correlation: float = 0.50,
    specs: tuple[SimulatedSecurityModelSpec, ...] = DEFAULT_SECURITY_MODEL_POOL,
) -> tuple[list[SimulatedLLMClient], list[ExperimentModelSpec]]:
    models = [
        SimulatedLLMClient(
            spec.model_id,
            ModelPricing(
                input_per_million=spec.input_usd_per_million,
                output_per_million=spec.output_usd_per_million,
            ),
            base_accuracy=spec.base_accuracy,
            seed=seed,
            difficulty_sensitivity=spec.difficulty_sensitivity,
            prompt_overhead_tokens=48,
            output_correlation=output_correlation,
        )
        for spec in specs
    ]
    experiment_specs = [
        ExperimentModelSpec(spec.model_id, spec.base_accuracy, spec.latency_ms)
        for spec in specs
    ]
    return models, experiment_specs


def attach_simulated_specialties(
    items: list[AnnotationItem] | tuple[AnnotationItem, ...],
    specialties: Mapping[str, Mapping[str, float]],
) -> list[AnnotationItem]:
    prepared: list[AnnotationItem] = []
    for item in items:
        gold = str(item.metadata["gold_label"])
        bonuses = {
            model_id: float(per_label.get(gold, 0.0))
            for model_id, per_label in specialties.items()
        }
        prepared.append(
            replace(item, metadata={**item.metadata, "topic_bonus_by_model": bonuses})
        )
    return prepared
