"""Paper-oriented TRACE-GCaMVo risk curves, ablations, and stability checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

from camvo.aggregation import weighted_vote
from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import StaticGraphNeighborhood
from camvo.security.casie import build_casie_event_adjacency, load_casie_event_items
from camvo.security.experiment import RoutingMethodSummary, _summarize
from camvo.security.graph_diagnostics import graph_diagnostics
from camvo.security.metrics import classification_metrics
from camvo.security.model_pool import (
    CASIE_SPECIALTIES,
    attach_simulated_specialties,
    build_simulated_security_pool,
)
from camvo.security.sampling import graph_preserving_group_sample
from camvo.security.simulated_experiments import run_casie_six_strategy_experiment
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.trace_router import TraceGraphCaMVoConfig, TraceGraphCaMVoRouter
from camvo.types import AnnotationItem, RoutingResult


@dataclass(frozen=True, slots=True)
class TraceOperatingPoint:
    name: str
    risk_tolerance: float
    graph_strength: float
    diversity_penalty: float
    macro_f1: float
    accuracy: float
    total_cost_usd: float
    cost_savings_vs_full: float
    average_models: float
    full_pool_rate: float
    abstention_rate: float
    graph_use_rate: float


def _base_config(embedding_dim: int, warmup_rounds: int) -> CaMVoConfig:
    return CaMVoConfig(
        embedding_dim=embedding_dim,
        confidence_threshold=0.97,
        min_models=2,
        exploration_alpha=0.20,
        linucb_regularization=1.0,
        laplace_regularization=1.0,
        warmup_rounds=warmup_rounds,
        confidence_method="exact",
    )


def _items(
    data_dir: str | Path,
    max_items: int,
    seed: int,
) -> tuple[list[AnnotationItem], dict[str, dict[str, float]]]:
    dataset = load_casie_event_items(data_dir, strict=False)
    sampled = graph_preserving_group_sample(
        dataset.items,
        min(max_items, len(dataset.items)),
        seed,
        group_key=lambda item: str(item.metadata["document_id"]),
        order_key=lambda item: int(item.metadata["start_offset"]),
    )
    prepared = attach_simulated_specialties(sampled, CASIE_SPECIALTIES)
    return prepared, build_casie_event_adjacency(prepared)


def _full_ensemble_predictions(
    items: list[AnnotationItem],
    seed: int,
) -> tuple[list[str], float]:
    models, specs = build_simulated_security_pool(seed)
    weights = {spec.model_id: spec.prior_quality for spec in specs}
    predictions: list[str] = []
    total_cost = 0.0
    for item in items:
        responses = {model.model_id: model.predict(item) for model in models}
        prediction, _ties = weighted_vote(
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
    return predictions, total_cost


def _trace_run(
    items: list[AnnotationItem],
    adjacency: dict[str, dict[str, float]],
    *,
    seed: int,
    embedding_dim: int,
    warmup_rounds: int,
    trace_config: TraceGraphCaMVoConfig,
) -> tuple[RoutingMethodSummary, list[RoutingResult]]:
    models, specs = build_simulated_security_pool(seed)
    latency = {spec.model_id: spec.latency_ms for spec in specs}
    router = TraceGraphCaMVoRouter(
        models,
        HashingTextEmbedder(embedding_dim),
        _base_config(embedding_dim, warmup_rounds),
        trace_config,
        StaticGraphNeighborhood(adjacency),
    )
    results = router.route_many(items)
    summary = _summarize(
        method="trace_gcamvo",
        items=items,
        predictions=[result.label for result in results],
        selections=[result.selected_models for result in results],
        costs=[result.actual_cost for result in results],
        model_ids=tuple(sorted(router.models)),
        latency_ms=latency,
        min_models=router.config.min_models,
        subset_confidences=[result.subset_confidence for result in results],
        abstentions=[result.abstained for result in results],
        decision_risks=[result.decision_risk for result in results],
        graph_evidence_weights=[result.graph_evidence_weight for result in results],
    )
    return summary, results


def _operating_point(
    name: str,
    config: TraceGraphCaMVoConfig,
    summary: RoutingMethodSummary,
    full_cost: float,
) -> TraceOperatingPoint:
    return TraceOperatingPoint(
        name=name,
        risk_tolerance=config.risk_tolerance,
        graph_strength=config.graph_strength,
        diversity_penalty=config.diversity_penalty,
        macro_f1=summary.metrics.macro_f1,
        accuracy=summary.metrics.accuracy,
        total_cost_usd=summary.total_cost_usd,
        cost_savings_vs_full=(
            0.0 if full_cost == 0 else 1.0 - summary.total_cost_usd / full_cost
        ),
        average_models=summary.average_models,
        full_pool_rate=summary.full_pool_rate,
        abstention_rate=summary.abstention_rate,
        graph_use_rate=summary.graph_use_rate,
    )


def run_casie_trace_research_suite(
    data_dir: str | Path,
    *,
    max_items: int = 1_200,
    seed: int = 17,
    stability_seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
    risk_grid: tuple[float, ...] = (0.20, 0.10, 0.05, 0.03, 0.02, 0.01),
    embedding_dim: int = 64,
    warmup_rounds: int = 40,
    bootstrap_iterations: int = 2_000,
) -> dict[str, object]:
    """Run the complete no-key evidence package for the proposed method."""

    if not stability_seeds or len(set(stability_seeds)) != len(stability_seeds):
        raise ValueError("stability_seeds must be non-empty and unique")
    if not risk_grid or any(not 0 < risk < 1 for risk in risk_grid):
        raise ValueError("risk_grid must contain probabilities in (0, 1)")
    items, adjacency = _items(data_dir, max_items, seed)
    diagnostics = graph_diagnostics(items, adjacency)
    labels = items[0].labels
    gold = [str(item.metadata["gold_label"]) for item in items]
    clusters = [str(item.metadata["document_id"]) for item in items]
    full_predictions, full_cost = _full_ensemble_predictions(items, seed)
    full_metrics = classification_metrics(gold, full_predictions, labels)

    default_config = TraceGraphCaMVoConfig()
    main_summary, main_results = _trace_run(
        items,
        adjacency,
        seed=seed,
        embedding_dim=embedding_dim,
        warmup_rounds=warmup_rounds,
        trace_config=default_config,
    )
    main_predictions = [result.label for result in main_results]

    risk_curve: list[TraceOperatingPoint] = []
    for risk in risk_grid:
        config = replace(default_config, risk_tolerance=risk)
        summary, _results = _trace_run(
            items,
            adjacency,
            seed=seed,
            embedding_dim=embedding_dim,
            warmup_rounds=warmup_rounds,
            trace_config=config,
        )
        risk_curve.append(_operating_point(f"risk={risk:g}", config, summary, full_cost))

    ablation_configs = {
        "full_trace": default_config,
        "no_graph": replace(default_config, graph_strength=0.0),
        "no_redundancy_correction": replace(default_config, diversity_penalty=0.0),
        "no_graph_or_redundancy": replace(
            default_config,
            graph_strength=0.0,
            diversity_penalty=0.0,
        ),
    }
    ablations: list[TraceOperatingPoint] = []
    for name, config in ablation_configs.items():
        summary, _results = _trace_run(
            items,
            adjacency,
            seed=seed,
            embedding_dim=embedding_dim,
            warmup_rounds=warmup_rounds,
            trace_config=config,
        )
        ablations.append(_operating_point(name, config, summary, full_cost))

    stability_runs: list[dict[str, object]] = []
    for run_seed in stability_seeds:
        run_full_predictions, run_full_cost = _full_ensemble_predictions(items, run_seed)
        run_full_metrics = classification_metrics(gold, run_full_predictions, labels)
        summary, _results = _trace_run(
            items,
            adjacency,
            seed=run_seed,
            embedding_dim=embedding_dim,
            warmup_rounds=warmup_rounds,
            trace_config=default_config,
        )
        stability_runs.append(
            {
                "seed": run_seed,
                "trace_macro_f1": summary.metrics.macro_f1,
                "full_macro_f1": run_full_metrics.macro_f1,
                "macro_f1_delta": summary.metrics.macro_f1 - run_full_metrics.macro_f1,
                "trace_total_cost_usd": summary.total_cost_usd,
                "full_total_cost_usd": run_full_cost,
                "cost_savings_vs_full": 1.0 - summary.total_cost_usd / run_full_cost,
                "average_models": summary.average_models,
            }
        )
    f1_deltas = [float(run["macro_f1_delta"]) for run in stability_runs]
    savings = [float(run["cost_savings_vs_full"]) for run in stability_runs]

    baseline_report = run_casie_six_strategy_experiment(
        data_dir,
        max_items=max_items,
        seed=seed,
        confidence_threshold=0.97,
        graph_regularization=1.0,
        min_models=2,
        warmup_rounds=warmup_rounds,
        embedding_dim=embedding_dim,
    ).to_dict()
    return {
        "method": "TRACE-GCaMVo",
        "claim_scope": (
            "CASIE labels and document graph are real; LLM responses, prices, and latency are "
            "simulated. Results validate routing behavior, not real-model superiority."
        ),
        "dataset": {
            "name": "CASIE",
            "items": len(items),
            "sampling": "document-group preserving, causal within-document order",
            **diagnostics,
        },
        "main": {
            "trace": main_summary.to_dict(),
            "full_ensemble": {
                "metrics": full_metrics.to_dict(),
                "total_cost_usd": full_cost,
            },
            "cost_savings_vs_full": 1.0 - main_summary.total_cost_usd / full_cost,
            "macro_f1_delta_vs_full": main_summary.metrics.macro_f1 - full_metrics.macro_f1,
        },
        "paired_document_bootstrap": {
            "macro_f1": paired_cluster_bootstrap_delta(
                gold,
                main_predictions,
                full_predictions,
                labels,
                clusters,
                metric="macro_f1",
                iterations=bootstrap_iterations,
                seed=seed,
            ).to_dict(),
            "accuracy": paired_cluster_bootstrap_delta(
                gold,
                main_predictions,
                full_predictions,
                labels,
                clusters,
                metric="accuracy",
                iterations=bootstrap_iterations,
                seed=seed,
            ).to_dict(),
        },
        "risk_curve": [asdict(point) for point in risk_curve],
        "ablations": [asdict(point) for point in ablations],
        "seed_stability": {
            "seeds": stability_seeds,
            "runs": stability_runs,
            "macro_f1_delta_mean": mean(f1_deltas),
            "macro_f1_delta_std": pstdev(f1_deltas),
            "cost_savings_mean": mean(savings),
            "cost_savings_std": pstdev(savings),
        },
        "baseline_report": baseline_report,
    }
