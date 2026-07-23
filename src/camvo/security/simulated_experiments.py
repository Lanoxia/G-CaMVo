"""Turn-key no-API-key experiment suites for CASIE and OpTC."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import GraphCaMVoConfig
from camvo.security.casie import (
    CASIE_LABELS,
    build_casie_event_adjacency,
    load_casie_event_items,
)
from camvo.security.experiment import SecurityExperimentReport, evaluate_security_strategies
from camvo.security.model_pool import (
    CASIE_SPECIALTIES,
    OPTC_SPECIALTIES,
    attach_simulated_specialties,
    build_simulated_security_pool,
)
from camvo.security.optc import OptcCorrelationEdge
from camvo.security.optc_dataset import (
    OptcBinaryDataset,
    build_optc_label_graph_simulation_dataset,
    build_optc_real_binary_dataset,
)
from camvo.security.sampling import graph_preserving_group_sample
from camvo.types import AnnotationItem
from camvo.trace_router import TraceGraphCaMVoConfig


@dataclass(frozen=True, slots=True)
class SweepPoint:
    confidence_threshold: float
    graph_regularization: float
    camvo_macro_f1: float
    gcamvo_macro_f1: float
    graph_f1_gain: float
    camvo_cost_usd: float
    gcamvo_cost_usd: float
    camvo_average_models: float
    gcamvo_average_models: float
    camvo_escalation_rate: float
    gcamvo_escalation_rate: float


@dataclass(frozen=True, slots=True)
class StabilityMethodSummary:
    method: str
    runs: int
    macro_f1_mean: float
    macro_f1_std: float
    total_cost_usd_mean: float
    total_cost_usd_std: float
    average_models_mean: float
    average_models_std: float


@dataclass(frozen=True, slots=True)
class SeedStabilityReport:
    seeds: tuple[int, ...]
    methods: dict[str, StabilityMethodSummary]
    graph_f1_gain_mean: float
    graph_f1_gain_std: float
    individual_runs: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seeds": self.seeds,
            "methods": {name: asdict(summary) for name, summary in self.methods.items()},
            "graph_f1_gain_mean": self.graph_f1_gain_mean,
            "graph_f1_gain_std": self.graph_f1_gain_std,
            "individual_runs": self.individual_runs,
        }


def stratified_sample(
    items: tuple[AnnotationItem, ...],
    max_items: int,
    seed: int,
) -> list[AnnotationItem]:
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_items >= len(items):
        selected = list(items)
        random.Random(seed).shuffle(selected)
        return selected
    groups: dict[str, list[AnnotationItem]] = defaultdict(list)
    for item in items:
        groups[str(item.metadata["gold_label"])].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    labels = sorted(groups)
    quota, remainder = divmod(max_items, len(labels))
    selected: list[AnnotationItem] = []
    for index, label in enumerate(labels):
        selected.extend(groups[label][: quota + int(index < remainder)])
    if len(selected) < max_items:
        selected_ids = {item.item_id for item in selected}
        remaining = [item for item in items if item.item_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_items - len(selected)])
    rng.shuffle(selected)
    return selected


def optc_adjacency(
    correlations: tuple[OptcCorrelationEdge, ...],
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in correlations:
        left = edge.source_event_id
        right = edge.target_event_id
        adjacency[left][right] = adjacency[left].get(right, 0.0) + edge.weight
        adjacency[right][left] = adjacency[right].get(left, 0.0) + edge.weight
    return {node_id: dict(neighbors) for node_id, neighbors in adjacency.items()}


def _base_config(
    embedding_dim: int,
    confidence_threshold: float,
    min_models: int,
    warmup_rounds: int,
) -> CaMVoConfig:
    return CaMVoConfig(
        embedding_dim=embedding_dim,
        confidence_threshold=confidence_threshold,
        min_models=min_models,
        exploration_alpha=0.20,
        linucb_regularization=1.0,
        laplace_regularization=1.0,
        warmup_rounds=warmup_rounds,
        confidence_method="exact",
    )


def run_casie_six_strategy_experiment(
    data_dir: str | Path,
    *,
    max_items: int = 1_200,
    seed: int = 17,
    confidence_threshold: float = 0.97,
    graph_regularization: float = 1.0,
    min_models: int = 2,
    warmup_rounds: int = 40,
    embedding_dim: int = 64,
) -> SecurityExperimentReport:
    dataset = load_casie_event_items(data_dir, strict=False)
    items = graph_preserving_group_sample(
        dataset.items,
        min(max_items, len(dataset.items)),
        seed,
        group_key=lambda item: str(item.metadata["document_id"]),
        order_key=lambda item: int(item.metadata["start_offset"]),
    )
    items = attach_simulated_specialties(items, CASIE_SPECIALTIES)
    adjacency = build_casie_event_adjacency(items)
    models, model_specs = build_simulated_security_pool(seed)
    return evaluate_security_strategies(
        items,
        models,
        model_specs,
        HashingTextEmbedder(embedding_dim),
        _base_config(
            embedding_dim,
            confidence_threshold,
            min_models,
            warmup_rounds,
        ),
        dataset_metadata={
            "name": "CASIE",
            "task": "five-way marked event-subtype classification",
            "items": len(items),
            "labels": CASIE_LABELS,
            "source_documents": dataset.stats.files_scanned,
            "graph_semantics": "same hopper/document text relation; not provenance",
        },
        graph_neighborhood=adjacency,
        graph_config=GraphCaMVoConfig(regularization=graph_regularization),
        trace_config=TraceGraphCaMVoConfig(),
        fixed_cheap_models=3,
        disclaimer=(
            "CASIE text and gold labels are real; model predictions, prices, and latency are "
            "simulated. The CASIE graph is a document relation baseline, not an OpTC provenance "
            "graph."
        ),
    )


def _run_optc_dataset(
    dataset: OptcBinaryDataset,
    *,
    seed: int,
    confidence_threshold: float,
    graph_regularization: float,
    min_models: int,
    warmup_rounds: int,
    embedding_dim: int,
) -> SecurityExperimentReport:
    items = attach_simulated_specialties(dataset.items, OPTC_SPECIALTIES)
    models, model_specs = build_simulated_security_pool(seed)
    disclaimer = (
        "Model predictions, prices, and latency are simulated. "
        + (
            "Attack IDs/topology are real, but benign controls are synthetic; results validate "
            "only experiment mechanics."
            if dataset.simulation_only
            else "Raw telemetry and gold construction are real, but model quality is simulated."
        )
    )
    return evaluate_security_strategies(
        items,
        models,
        model_specs,
        HashingTextEmbedder(embedding_dim),
        _base_config(
            embedding_dim,
            confidence_threshold,
            min_models,
            warmup_rounds,
        ),
        dataset_metadata=dataset.stats,
        graph_neighborhood=optc_adjacency(dataset.correlations),
        graph_config=GraphCaMVoConfig(regularization=graph_regularization),
        trace_config=TraceGraphCaMVoConfig(),
        fixed_cheap_models=3,
        disclaimer=disclaimer,
    )


def run_optc_label_graph_experiment(
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    max_items: int = 1_200,
    seed: int = 17,
    confidence_threshold: float = 0.97,
    graph_regularization: float = 1.0,
    min_models: int = 2,
    warmup_rounds: int = 40,
    embedding_dim: int = 64,
) -> SecurityExperimentReport:
    dataset = build_optc_label_graph_simulation_dataset(
        labels_path,
        scenario_manifest,
        max_positive_items=max(1, max_items // 2),
        seed=seed,
    )
    return _run_optc_dataset(
        dataset,
        seed=seed,
        confidence_threshold=confidence_threshold,
        graph_regularization=graph_regularization,
        min_models=min_models,
        warmup_rounds=warmup_rounds,
        embedding_dim=embedding_dim,
    )


def run_optc_real_data_simulated_model_experiment(
    attack_path: str | Path,
    benign_path: str | Path,
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    max_items: int = 1_200,
    seed: int = 17,
    confidence_threshold: float = 0.97,
    graph_regularization: float = 1.0,
    min_models: int = 2,
    warmup_rounds: int = 40,
    embedding_dim: int = 64,
) -> SecurityExperimentReport:
    dataset = build_optc_real_binary_dataset(
        attack_path,
        benign_path,
        labels_path,
        scenario_manifest,
        max_positive_items=max(1, max_items // 2),
        negative_ratio=1.0,
        seed=seed,
    )
    return _run_optc_dataset(
        dataset,
        seed=seed,
        confidence_threshold=confidence_threshold,
        graph_regularization=graph_regularization,
        min_models=min_models,
        warmup_rounds=warmup_rounds,
        embedding_dim=embedding_dim,
    )


def run_parameter_sweep(
    dataset: OptcBinaryDataset,
    *,
    confidence_thresholds: tuple[float, ...],
    graph_regularizations: tuple[float, ...],
    seed: int = 17,
    min_models: int = 2,
    warmup_rounds: int = 20,
    embedding_dim: int = 32,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for confidence_threshold in confidence_thresholds:
        for graph_regularization in graph_regularizations:
            report = _run_optc_dataset(
                dataset,
                seed=seed,
                confidence_threshold=confidence_threshold,
                graph_regularization=graph_regularization,
                min_models=min_models,
                warmup_rounds=warmup_rounds,
                embedding_dim=embedding_dim,
            )
            camvo = report.methods["camvo"]
            gcamvo = report.methods["gcamvo"]
            points.append(
                SweepPoint(
                    confidence_threshold=confidence_threshold,
                    graph_regularization=graph_regularization,
                    camvo_macro_f1=camvo.metrics.macro_f1,
                    gcamvo_macro_f1=gcamvo.metrics.macro_f1,
                    graph_f1_gain=gcamvo.metrics.macro_f1 - camvo.metrics.macro_f1,
                    camvo_cost_usd=camvo.total_cost_usd,
                    gcamvo_cost_usd=gcamvo.total_cost_usd,
                    camvo_average_models=camvo.average_models,
                    gcamvo_average_models=gcamvo.average_models,
                    camvo_escalation_rate=camvo.escalation_rate,
                    gcamvo_escalation_rate=gcamvo.escalation_rate,
                )
            )
    return points


def run_seed_stability(
    dataset: OptcBinaryDataset,
    *,
    seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
    confidence_threshold: float = 0.92,
    graph_regularization: float = 1.0,
    min_models: int = 2,
    warmup_rounds: int = 20,
    embedding_dim: int = 32,
) -> SeedStabilityReport:
    """Repeat the identical dataset/configuration under independent model seeds."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique values")
    reports = [
        _run_optc_dataset(
            dataset,
            seed=seed,
            confidence_threshold=confidence_threshold,
            graph_regularization=graph_regularization,
            min_models=min_models,
            warmup_rounds=warmup_rounds,
            embedding_dim=embedding_dim,
        )
        for seed in seeds
    ]
    summaries: dict[str, StabilityMethodSummary] = {}
    for method in reports[0].methods:
        method_runs = [report.methods[method] for report in reports]
        f1_values = [summary.metrics.macro_f1 for summary in method_runs]
        costs = [summary.total_cost_usd for summary in method_runs]
        model_counts = [summary.average_models for summary in method_runs]
        summaries[method] = StabilityMethodSummary(
            method=method,
            runs=len(reports),
            macro_f1_mean=mean(f1_values),
            macro_f1_std=pstdev(f1_values),
            total_cost_usd_mean=mean(costs),
            total_cost_usd_std=pstdev(costs),
            average_models_mean=mean(model_counts),
            average_models_std=pstdev(model_counts),
        )
    graph_gains = [
        report.methods["gcamvo"].metrics.macro_f1
        - report.methods["camvo"].metrics.macro_f1
        for report in reports
    ]
    individual_runs = tuple(
        {
            "seed": seed,
            "camvo_macro_f1": report.methods["camvo"].metrics.macro_f1,
            "gcamvo_macro_f1": report.methods["gcamvo"].metrics.macro_f1,
            "graph_f1_gain": gain,
            "camvo_total_cost_usd": report.methods["camvo"].total_cost_usd,
            "gcamvo_total_cost_usd": report.methods["gcamvo"].total_cost_usd,
        }
        for seed, report, gain in zip(seeds, reports, graph_gains, strict=True)
    )
    return SeedStabilityReport(
        seeds=seeds,
        methods=summaries,
        graph_f1_gain_mean=mean(graph_gains),
        graph_f1_gain_std=pstdev(graph_gains),
        individual_runs=individual_runs,
    )
