"""Leakage-resistant, resumable real-model evaluation protocol.

The expensive stage collects one complete model-by-item response matrix.  All
policy comparisons then replay an immutable in-memory copy, so every method
sees identical outputs and no threshold search can trigger extra provider
calls.  Whole documents/incidents are assigned to calibration, validation,
and frozen test partitions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from camvo.aggregation import weighted_vote
from camvo.ccamvo_router import CorrelatedCaMVoConfig, CorrelatedCaMVoRouter
from camvo.config import CaMVoConfig
from camvo.credentials import load_api_keys
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import GraphCaMVoConfig, GraphCaMVoRouter, StaticGraphNeighborhood
from camvo.llms.base import LLMClient
from camvo.llms.guarded import CachedBudgetedLLMClient
from camvo.router import CaMVoRouter
from camvo.security.experiment import (
    ExperimentModelSpec,
    RoutingMethodSummary,
    _summarize,
)
from camvo.security.graph_diagnostics import graph_diagnostics
from camvo.security.metrics import classification_metrics
from camvo.security.operational_metrics import optc_operational_metrics
from camvo.security.real_experiment import (
    _dataset,
    _object,
    build_guarded_provider_pool,
    load_real_experiment_config,
)
from camvo.security.splits import stratified_grouped_split
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.trace_router import TraceGraphCaMVoConfig, TraceGraphCaMVoRouter
from camvo.types import AnnotationItem, ModelPricing, ModelResponse, RoutingResult


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    summary: RoutingMethodSummary
    predictions: tuple[str, ...]
    selections: tuple[tuple[str, ...], ...]
    abstentions: tuple[bool, ...]


class FrozenResponseMatrixClient(LLMClient):
    """Read-only client used after provider collection is frozen."""

    def __init__(
        self,
        model_id: str,
        pricing: ModelPricing,
        responses: Mapping[str, ModelResponse],
    ) -> None:
        super().__init__(model_id, pricing)
        self._responses = dict(responses)

    def predict(self, item: AnnotationItem) -> ModelResponse:
        try:
            return self._responses[item.item_id]
        except KeyError as exc:
            raise KeyError(
                f"frozen response matrix lacks {self.model_id}/{item.item_id}"
            ) from exc


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _group_id(item: AnnotationItem) -> str:
    document = str(item.metadata.get("document_id", "")).strip()
    if document:
        return f"document:{document}"
    # OpTC may contain only a few large source files.  Host/time windows keep
    # locally correlated events together without collapsing the split to two
    # file-level groups.
    hostname = str(item.metadata.get("hostname", "unknown"))
    timestamp = int(item.metadata.get("timestamp_ms", 0))
    source = Path(str(item.metadata.get("source_path", "unknown"))).name
    return f"incident-window:{source}:{hostname}:{timestamp // 1_800_000}"


def _partition_digest(partition: tuple[AnnotationItem, ...]) -> str:
    payload = "\n".join(sorted(item.item_id for item in partition)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _permute_group_blocks(
    items: tuple[AnnotationItem, ...],
    *,
    seed: int,
) -> tuple[AnnotationItem, ...]:
    grouped: dict[str, list[AnnotationItem]] = {}
    for item in items:
        grouped.setdefault(_group_id(item), []).append(item)
    group_ids = list(grouped)
    random.Random(seed).shuffle(group_ids)
    return tuple(item for group_id in group_ids for item in grouped[group_id])


def _load_manifest(
    path: Path,
    *,
    model_ids: tuple[str, ...],
    item_ids: set[str],
) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported response-matrix manifest schema")
        if tuple(payload.get("model_ids", ())) != model_ids:
            raise ValueError("response-matrix manifest model pool mismatch")
        expected_digest = hashlib.sha256(
            "\n".join(sorted(item_ids)).encode("utf-8")
        ).hexdigest()
        if payload.get("sample_sha256") != expected_digest:
            raise ValueError("response-matrix manifest dataset sample mismatch")
        return {str(item_id) for item_id in payload.get("completed_item_ids", [])}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid response-matrix manifest: {exc}") from exc


def collect_response_matrix(
    models: list[CachedBudgetedLLMClient],
    items: list[AnnotationItem],
    manifest_path: str | Path,
    *,
    workers: int = 4,
    model_circuit_breaker_failures: int = 3,
) -> tuple[dict[str, dict[str, ModelResponse]], dict[str, object]]:
    """Collect/cache each model-item cell once and persist item-level progress.

    Missing cells are scheduled across the whole matrix instead of waiting for
    all models on one item before starting the next item.  This matters for
    heterogeneous gateways: one slow or timing-out model must not leave the
    other workers idle.  The response cache remains the source of truth, so a
    failed run can be restarted without repeating successful provider calls.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if model_circuit_breaker_failures <= 0:
        raise ValueError("model_circuit_breaker_failures must be positive")
    manifest = Path(manifest_path)
    model_ids = tuple(sorted(model.model_id for model in models))
    model_map = {model.model_id: model for model in models}
    allowed_items = {item.item_id for item in items}
    sample_digest = hashlib.sha256(
        "\n".join(sorted(allowed_items)).encode("utf-8")
    ).hexdigest()
    completed = _load_manifest(
        manifest,
        model_ids=model_ids,
        item_ids=allowed_items,
    )
    if not completed <= allowed_items:
        raise ValueError("response-matrix manifest belongs to a different dataset sample")
    matrix: dict[str, dict[str, ModelResponse]] = {
        item.item_id: {} for item in items
    }

    def persist(*, complete: bool, last_error: str | None = None) -> None:
        _write_json(
            manifest,
            {
                "schema_version": 1,
                "model_ids": list(model_ids),
                "sample_sha256": sample_digest,
                "items": len(items),
                "total_cells": len(items) * len(models),
                "completed_items": len(completed),
                "completed_cells": len(completed) * len(models),
                "completed_item_ids": sorted(completed),
                "complete": complete,
                "last_error": last_error,
            },
        )

    pending: list[tuple[AnnotationItem, CachedBudgetedLLMClient]] = []
    for item in items:
        row = matrix[item.item_id]
        for model_id in model_ids:
            model = model_map[model_id]
            key = model.cache.key_for(model_id, item, model.prompt_version)
            cached = model.cache.get(key)
            if cached is None:
                pending.append((item, model))
            else:
                row[model_id] = cached
        if set(row) == set(model_ids):
            completed.add(item.item_id)
        else:
            completed.discard(item.item_id)

    persist(complete=not pending)

    def accept(
        future: Future[ModelResponse],
        item: AnnotationItem,
        model: CachedBudgetedLLMClient,
    ) -> None:
        matrix[item.item_id][model.model_id] = future.result()
        if set(matrix[item.item_id]) == set(model_ids):
            completed.add(item.item_id)
            persist(complete=False)

    failures: list[tuple[str, str, BaseException]] = []
    circuit_open_models: set[str] = set()
    if pending:
        # Process each model independently.  A persistently unhealthy gateway
        # branch must not cancel queued work for every healthy model.
        for model_id in model_ids:
            model_pending = [
                (item, model)
                for item, model in pending
                if model.model_id == model_id
            ]
            if not model_pending:
                continue
            executor = ThreadPoolExecutor(
                max_workers=min(workers, len(model_pending))
            )
            pending_iterator = iter(model_pending)
            futures: dict[
                Future[ModelResponse],
                tuple[AnnotationItem, CachedBudgetedLLMClient],
            ] = {}

            def submit_next() -> bool:
                try:
                    item, model = next(pending_iterator)
                except StopIteration:
                    return False
                futures[executor.submit(model.predict, item)] = (item, model)
                return True

            for _ in range(min(workers, len(model_pending))):
                submit_next()
            accepted: set[Future[ModelResponse]] = set()
            consecutive_failures = 0
            try:
                while futures:
                    future = next(as_completed(tuple(futures)))
                    item, model = futures.pop(future)
                    try:
                        accept(future, item, model)
                        accepted.add(future)
                        consecutive_failures = 0
                    except Exception as exc:
                        failures.append((model_id, item.item_id, exc))
                        consecutive_failures += 1
                        if consecutive_failures >= model_circuit_breaker_failures:
                            circuit_open_models.add(model_id)
                            for remaining in futures:
                                remaining.cancel()
                            break
                    submit_next()
            finally:
                # Requests already running are allowed to finish so cache and
                # ledger writes remain atomic.  Merely queued requests for an
                # open circuit are cancelled and retried on a later pass.
                executor.shutdown(wait=True, cancel_futures=True)

            # Fold successes that completed while shutdown waited into the
            # in-memory matrix.  Record extra failures once, without allowing
            # them to abort collection for the next model branch.
            for future, (item, model) in futures.items():
                if future in accepted or future.cancelled() or not future.done():
                    continue
                try:
                    accept(future, item, model)
                    accepted.add(future)
                except Exception as exc:
                    failures.append((model_id, item.item_id, exc))

        if failures:
            first_model, first_item, first_error = failures[0]
            circuit_text = ",".join(sorted(circuit_open_models)) or "none"
            detail = (
                f"response-matrix collection incomplete; failures={len(failures)}; "
                f"circuit_open_models={circuit_text}; first_cell={first_model}/{first_item}; "
                f"first_error={type(first_error).__name__}: {first_error}"
            )
            persist(complete=False, last_error=detail)
            raise RuntimeError(detail) from first_error

    incomplete = {
        item_id for item_id, row in matrix.items() if set(row) != set(model_ids)
    }
    if incomplete:
        raise RuntimeError(
            f"provider collection produced {len(incomplete)} incomplete response rows"
        )
    persist(complete=True)
    return matrix, {
        "manifest_path": str(manifest),
        "items": len(items),
        "models": len(models),
        "total_cells": len(items) * len(models),
        "complete": True,
    }


def _frozen_clients(
    guarded: list[CachedBudgetedLLMClient],
    matrix: Mapping[str, Mapping[str, ModelResponse]],
) -> list[FrozenResponseMatrixClient]:
    return [
        FrozenResponseMatrixClient(
            model.model_id,
            model.pricing,
            {item_id: row[model.model_id] for item_id, row in matrix.items()},
        )
        for model in guarded
    ]


def _calibrate_models(
    calibration: tuple[AnnotationItem, ...],
    models: list[LLMClient],
    fallback_latency_ms: Mapping[str, float],
) -> tuple[list[ExperimentModelSpec], dict[str, dict[str, float | int]], str]:
    labels = calibration[0].labels
    reports: dict[str, dict[str, float | int]] = {}
    specs: list[ExperimentModelSpec] = []
    for model in models:
        responses = [model.predict(item) for item in calibration]
        predictions = [response.label for response in responses]
        gold = [str(item.metadata["gold_label"]) for item in calibration]
        metrics = classification_metrics(gold, predictions, labels)
        correct = sum(left == right for left, right in zip(gold, predictions, strict=True))
        smoothed_accuracy = (correct + 1.0) / (len(calibration) + 2.0)
        observed_latency_ms = []
        for response in responses:
            if not isinstance(response.raw, Mapping):
                continue
            raw_latency = response.raw.get(
                "elapsed_time_seconds", response.raw.get("wall_time_seconds")
            )
            if isinstance(raw_latency, (int, float)) and raw_latency >= 0:
                observed_latency_ms.append(float(raw_latency) * 1_000.0)
        latency_ms = (
            mean(observed_latency_ms)
            if observed_latency_ms
            else float(fallback_latency_ms[model.model_id])
        )
        reports[model.model_id] = {
            "items": len(calibration),
            "correct": correct,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "smoothed_accuracy_for_voting": smoothed_accuracy,
            "average_input_tokens": mean(response.input_tokens for response in responses),
            "average_output_tokens": mean(response.output_tokens for response in responses),
            "average_observed_latency_ms": latency_ms,
        }
        specs.append(
            ExperimentModelSpec(
                model_id=model.model_id,
                prior_quality=smoothed_accuracy,
                latency_ms=latency_ms,
            )
        )
    best = max(
        models,
        key=lambda model: (
            float(reports[model.model_id]["macro_f1"]),
            float(reports[model.model_id]["accuracy"]),
            -(model.pricing.input_per_million + model.pricing.output_per_million),
            model.model_id,
        ),
    ).model_id
    return specs, reports, best


def _fixed_evaluation(
    method: str,
    items: tuple[AnnotationItem, ...],
    models: Mapping[str, LLMClient],
    selected: tuple[str, ...],
    weights: Mapping[str, float],
    latency: Mapping[str, float],
    min_models: int,
) -> MethodEvaluation:
    predictions: list[str] = []
    costs: list[float] = []
    for item in items:
        responses = {model_id: models[model_id].predict(item) for model_id in selected}
        prediction, _ties = weighted_vote(
            {model_id: response.label for model_id, response in responses.items()},
            {model_id: weights[model_id] for model_id in selected},
            item.labels,
        )
        predictions.append(prediction)
        costs.append(
            sum(
                models[model_id].pricing.cost(response.input_tokens, response.output_tokens)
                for model_id, response in responses.items()
            )
        )
    summary = _summarize(
        method=method,
        items=list(items),
        predictions=predictions,
        selections=[selected] * len(items),
        costs=costs,
        model_ids=tuple(sorted(models)),
        latency_ms=latency,
        min_models=min_models,
    )
    return MethodEvaluation(
        summary,
        tuple(predictions),
        tuple([selected] * len(items)),
        tuple(False for _ in items),
    )


def _online_weighted_majority_evaluation(
    items: tuple[AnnotationItem, ...],
    models: Mapping[str, LLMClient],
    latency: Mapping[str, float],
    min_models: int,
) -> MethodEvaluation:
    model_ids = tuple(sorted(models))
    agreements = {model_id: 1.0 for model_id in model_ids}
    observations = {model_id: 2.0 for model_id in model_ids}
    predictions: list[str] = []
    costs: list[float] = []
    for item in items:
        responses = {model_id: models[model_id].predict(item) for model_id in model_ids}
        weights = {
            model_id: agreements[model_id] / observations[model_id] for model_id in model_ids
        }
        prediction, _ties = weighted_vote(
            {model_id: response.label for model_id, response in responses.items()},
            weights,
            item.labels,
        )
        predictions.append(prediction)
        for model_id, response in responses.items():
            observations[model_id] += 1.0
            agreements[model_id] += float(response.label == prediction)
        costs.append(
            sum(
                models[model_id].pricing.cost(response.input_tokens, response.output_tokens)
                for model_id, response in responses.items()
            )
        )
    summary = _summarize(
        method="online_weighted_majority",
        items=list(items),
        predictions=predictions,
        selections=[model_ids] * len(items),
        costs=costs,
        model_ids=model_ids,
        latency_ms=latency,
        min_models=min_models,
    )
    return MethodEvaluation(
        summary,
        tuple(predictions),
        tuple([model_ids] * len(items)),
        tuple(False for _ in items),
    )


def _router_evaluation(
    method: str,
    router: CaMVoRouter,
    items: tuple[AnnotationItem, ...],
    latency: Mapping[str, float],
) -> MethodEvaluation:
    results: list[RoutingResult] = router.route_many(list(items))
    summary = _summarize(
        method=method,
        items=list(items),
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
    return MethodEvaluation(
        summary,
        tuple(result.label for result in results),
        tuple(result.selected_models for result in results),
        tuple(result.abstained for result in results),
    )


def _wilson_upper_bound(errors: int, observations: int, *, z_value: float = 1.96) -> float:
    if observations <= 0 or not 0 <= errors <= observations:
        raise ValueError("Wilson interval requires 0 <= errors <= positive observations")
    estimate = errors / observations
    denominator = 1.0 + z_value * z_value / observations
    center = estimate + z_value * z_value / (2.0 * observations)
    radius = z_value * math.sqrt(
        estimate * (1.0 - estimate) / observations
        + z_value * z_value / (4.0 * observations * observations)
    )
    return min(1.0, (center + radius) / denominator)


def _selective_risk(
    items: tuple[AnnotationItem, ...],
    evaluation: MethodEvaluation,
) -> dict[str, float | int]:
    accepted = [index for index, abstained in enumerate(evaluation.abstentions) if not abstained]
    if not accepted:
        return {
            "accepted": 0,
            "errors": 0,
            "coverage": 0.0,
            "empirical_error": 1.0,
            "wilson_95_upper": 1.0,
        }
    errors = sum(
        evaluation.predictions[index] != str(items[index].metadata["gold_label"])
        for index in accepted
    )
    return {
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(items),
        "empirical_error": errors / len(accepted),
        "wilson_95_upper": _wilson_upper_bound(errors, len(accepted)),
    }


def _warm_start(
    router: CaMVoRouter,
    calibration: tuple[AnnotationItem, ...],
    matrix: Mapping[str, Mapping[str, ModelResponse]],
) -> CaMVoRouter:
    for item in calibration:
        router.observe_complete_feedback(
            item,
            dict(matrix[item.item_id]),
            str(item.metadata["gold_label"]),
        )
    return router


def _base_router_config(config: dict[str, Any]) -> CaMVoConfig:
    raw = _object(config, "router")
    return CaMVoConfig(
        embedding_dim=int(raw.get("embedding_dim", 64)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.97)),
        min_models=int(raw.get("min_models", 2)),
        linucb_regularization=float(raw.get("linucb_regularization", 1.0)),
        exploration_alpha=float(raw.get("exploration_alpha", 0.2)),
        laplace_regularization=float(raw.get("laplace_regularization", 1.0)),
        warmup_rounds=int(raw.get("warmup_rounds", 20)),
        confidence_method=str(raw.get("confidence_method", "exact")),
    )


def _trace_config(
    config: dict[str, Any],
    *,
    risk_tolerance: float,
    fallback_model_id: str,
) -> TraceGraphCaMVoConfig:
    raw = dict(config.get("trace", {}))
    return TraceGraphCaMVoConfig(
        risk_tolerance=risk_tolerance,
        critical_risk_tolerance=min(
            risk_tolerance, float(raw.get("critical_risk_tolerance", 0.01))
        ),
        symmetric_reliability_prior=float(raw.get("symmetric_reliability_prior", 0.68)),
        reliability_prior_strength=float(raw.get("reliability_prior_strength", 6.0)),
        min_reliability_observations=int(raw.get("min_reliability_observations", 8)),
        context_reliability_weight=float(raw.get("context_reliability_weight", 0.25)),
        diversity_penalty=float(raw.get("diversity_penalty", 0.45)),
        graph_strength=float(raw.get("graph_strength", 0.8)),
        min_transition_observations=float(raw.get("min_transition_observations", 6.0)),
        abstain_when_risk_unmet=bool(raw.get("abstain_when_risk_unmet", True)),
        fallback_model_id=fallback_model_id,
        fallback_after_models=int(raw.get("fallback_after_models", 2)),
        fallback_trigger_risk=float(raw.get("fallback_trigger_risk", 0.20)),
    )


def _router_set(
    models: list[LLMClient],
    base_config: CaMVoConfig,
    graph_config: GraphCaMVoConfig,
    neighborhood: StaticGraphNeighborhood,
    correlated_config: CorrelatedCaMVoConfig,
    trace_config: TraceGraphCaMVoConfig,
) -> dict[str, CaMVoRouter]:
    embedder = HashingTextEmbedder(base_config.embedding_dim)
    return {
        "camvo": CaMVoRouter(models, embedder, base_config),
        "ccamvo": CorrelatedCaMVoRouter(
            models, embedder, base_config, correlated_config
        ),
        "gcamvo": GraphCaMVoRouter(
            models, embedder, base_config, graph_config, neighborhood
        ),
        "trace_gcamvo": TraceGraphCaMVoRouter(
            models, embedder, base_config, trace_config, neighborhood
        ),
    }


def run_formal_provider_experiment(config_path: str | Path) -> dict[str, object]:
    """Run matrix collection, validation-only selection, and one frozen test."""

    config = load_real_experiment_config(config_path)
    if config.get("pricing_verified") is not True:
        raise ValueError("pricing_verified must be true before a real-provider run")
    load_api_keys(Path(config.get("api_key_file", "config/api_keys.env")))
    items, adjacency, dataset_metadata, task = _dataset(config)
    formal = dict(config.get("formal", {}))
    split = stratified_grouped_split(
        items,
        group_key=_group_id,
        calibration_fraction=float(formal.get("calibration_fraction", 0.20)),
        validation_fraction=float(formal.get("validation_fraction", 0.10)),
        seed=int(config.get("seed", 17)),
    )
    minimum = int(formal.get("minimum_partition_items", 5))
    if min(len(split.calibration), len(split.validation), len(split.test)) < minimum:
        raise ValueError(
            "grouped split is too small for the formal protocol; increase dataset.max_items"
        )

    guarded, configured_specs, budget = build_guarded_provider_pool(config, task)
    budget_raw = _object(config, "budget")
    matrix, matrix_audit = collect_response_matrix(
        guarded,
        items,
        formal.get(
            "matrix_manifest_path",
            str(
                Path(str(budget_raw.get("cache_dir", "artifacts/responses"))).parent
                / "matrix.json"
            ),
        ),
        workers=int(formal.get("prefetch_workers", 4)),
    )
    models = _frozen_clients(guarded, matrix)
    model_map = {model.model_id: model for model in models}
    fallback_latency = {spec.model_id: spec.latency_ms for spec in configured_specs}
    calibrated_specs, calibration_report, best_model = _calibrate_models(
        split.calibration, models, fallback_latency
    )
    priors = {spec.model_id: spec.prior_quality for spec in calibrated_specs}
    configured_latency = {spec.model_id: spec.latency_ms for spec in calibrated_specs}
    base_config = _base_router_config(config)
    graph_raw = _object(config, "graph")
    graph_config = GraphCaMVoConfig(
        regularization=float(graph_raw.get("regularization", 1.0)),
        max_edge_weight=float(graph_raw.get("max_edge_weight", 2.0)),
        max_total_neighbor_weight=float(graph_raw.get("max_total_neighbor_weight", 8.0)),
    )
    neighborhood = StaticGraphNeighborhood(adjacency)
    cc_raw = dict(config.get("ccamvo", {}))
    correlated_config = CorrelatedCaMVoConfig(
        monte_carlo_samples=int(cc_raw.get("monte_carlo_samples", 4_096)),
        seed=int(cc_raw.get("seed", config.get("seed", 17))),
    )
    all_ids = tuple(sorted(model_map))
    cheapest = min(
        all_ids,
        key=lambda model_id: (
            model_map[model_id].pricing.input_per_million
            + model_map[model_id].pricing.output_per_million,
            model_id,
        ),
    )

    validation: dict[str, MethodEvaluation] = {
        "calibrated_best_single": _fixed_evaluation(
            "calibrated_best_single",
            split.validation,
            model_map,
            (best_model,),
            priors,
            configured_latency,
            base_config.min_models,
        ),
        "full_ensemble": _fixed_evaluation(
            "full_ensemble",
            split.validation,
            model_map,
            all_ids,
            priors,
            configured_latency,
            base_config.min_models,
        ),
    }
    base_trace = _trace_config(config, risk_tolerance=0.03, fallback_model_id=best_model)
    for name, router in _router_set(
        models,
        base_config,
        graph_config,
        neighborhood,
        correlated_config,
        base_trace,
    ).items():
        if name == "trace_gcamvo":
            continue
        validation[name] = _router_evaluation(
            name,
            router,
            split.validation,
            configured_latency,
        )
    calibrated_validation_routers = _router_set(
        models,
        base_config,
        graph_config,
        neighborhood,
        correlated_config,
        base_trace,
    )
    for base_name in ("camvo", "ccamvo", "gcamvo"):
        name = f"calibrated_{base_name}"
        validation[name] = _router_evaluation(
            name,
            _warm_start(
                calibrated_validation_routers[base_name], split.calibration, matrix
            ),
            split.validation,
            configured_latency,
        )
    graph_lambda_grid = tuple(
        sorted(
            {
                float(value)
                for value in formal.get(
                    "graph_regularization_grid", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
                )
            }
        )
    )
    if not graph_lambda_grid or any(value < 0 for value in graph_lambda_grid):
        raise ValueError("formal.graph_regularization_grid values must be non-negative")
    graph_candidates: dict[float, MethodEvaluation] = {}
    for graph_lambda in graph_lambda_grid:
        candidate_router = GraphCaMVoRouter(
            models,
            HashingTextEmbedder(base_config.embedding_dim),
            base_config,
            replace(graph_config, regularization=graph_lambda),
            neighborhood,
        )
        graph_candidates[graph_lambda] = _router_evaluation(
            "calibrated_tuned_gcamvo",
            _warm_start(candidate_router, split.calibration, matrix),
            split.validation,
            configured_latency,
        )
    selected_graph_lambda, selected_graph_validation = max(
        graph_candidates.items(),
        key=lambda value: (
            value[1].summary.metrics.macro_f1,
            value[1].summary.metrics.macro_recall,
            -value[1].summary.total_cost_usd,
            -value[0],
        ),
    )
    validation["calibrated_tuned_gcamvo"] = selected_graph_validation
    reference_name, reference = max(
        validation.items(),
        key=lambda value: (
            value[1].summary.metrics.macro_f1,
            value[1].summary.metrics.macro_recall,
            -value[1].summary.total_cost_usd,
            value[0],
        ),
    )
    risk_grid = tuple(
        sorted(
            {
                float(value)
                for value in formal.get(
                    "risk_grid", [0.20, 0.10, 0.05, 0.03, 0.02, 0.01]
                )
            },
            reverse=True,
        )
    )
    if not risk_grid or any(not 0 < value < 1 for value in risk_grid):
        raise ValueError("formal.risk_grid values must be in (0, 1)")
    trace_graph_grid = tuple(
        sorted(
            {
                float(value)
                for value in formal.get(
                    "trace_graph_strength_grid", [0.0, 0.25, 0.5, 0.8, 1.2]
                )
            }
        )
    )
    if not trace_graph_grid or any(value < 0 for value in trace_graph_grid):
        raise ValueError("formal.trace_graph_strength_grid values must be non-negative")
    f1_margin = float(formal.get("noninferiority_margin_macro_f1", 0.01))
    recall_margin = float(formal.get("noninferiority_margin_macro_recall", 0.01))
    risk_margin = float(formal.get("noninferiority_margin_selective_risk", 0.02))
    max_abstention = float(formal.get("max_abstention_rate", 0.05))
    reference_risk = _selective_risk(split.validation, reference)
    candidates: dict[tuple[float, float], MethodEvaluation] = {}
    candidate_risk: dict[tuple[float, float], dict[str, float | int]] = {}
    feasible: list[tuple[float, float, MethodEvaluation]] = []
    for graph_strength in trace_graph_grid:
        for risk in risk_grid:
            trace = TraceGraphCaMVoRouter(
                models,
                HashingTextEmbedder(base_config.embedding_dim),
                base_config,
                replace(
                    base_trace,
                    risk_tolerance=risk,
                    graph_strength=graph_strength,
                ),
                neighborhood,
            )
            evaluation = _router_evaluation(
                "trace_gcamvo",
                _warm_start(trace, split.calibration, matrix),
                split.validation,
                configured_latency,
            )
            key = (risk, graph_strength)
            candidates[key] = evaluation
            candidate_risk[key] = _selective_risk(split.validation, evaluation)
            if (
                evaluation.summary.metrics.macro_f1
                >= reference.summary.metrics.macro_f1 - f1_margin
                and evaluation.summary.metrics.macro_recall
                >= reference.summary.metrics.macro_recall - recall_margin
                and evaluation.summary.abstention_rate <= max_abstention
                and float(candidate_risk[key]["wilson_95_upper"])
                <= float(reference_risk["wilson_95_upper"]) + risk_margin
            ):
                feasible.append((risk, graph_strength, evaluation))
    if feasible:
        selected_risk, selected_trace_graph_strength, selected_validation = min(
            feasible,
            key=lambda value: (
                value[2].summary.total_cost_usd,
                -value[2].summary.metrics.macro_f1,
                value[2].summary.abstention_rate,
                value[0],
                value[1],
            ),
        )
        selection_reason = "lowest validation cost satisfying frozen non-inferiority guards"
    else:
        (selected_risk, selected_trace_graph_strength), selected_validation = max(
            candidates.items(),
            key=lambda value: (
                value[1].summary.metrics.macro_f1,
                value[1].summary.metrics.macro_recall,
                -value[1].summary.abstention_rate,
                -value[1].summary.total_cost_usd,
            ),
        )
        selection_reason = "no candidate met every guard; selected highest validation quality"

    fixed_count = int(config.get("fixed_cheap_models", 3))
    cheap_order = tuple(
        sorted(
            all_ids,
            key=lambda model_id: (
                model_map[model_id].pricing.input_per_million
                + model_map[model_id].pricing.output_per_million,
                model_id,
            ),
        )
    )
    test: dict[str, MethodEvaluation] = {}
    for name, selected in (
        ("cheapest_single", (cheapest,)),
        ("calibrated_best_single", (best_model,)),
        ("fixed_cheap", cheap_order[:fixed_count]),
        ("full_ensemble", all_ids),
    ):
        test[name] = _fixed_evaluation(
            name,
            split.test,
            model_map,
            selected,
            priors,
            configured_latency,
            base_config.min_models,
        )
    test["online_weighted_majority"] = _online_weighted_majority_evaluation(
        split.test, model_map, configured_latency, base_config.min_models
    )
    frozen_trace = replace(
        base_trace,
        risk_tolerance=selected_risk,
        graph_strength=selected_trace_graph_strength,
    )
    for name, router in _router_set(
        models,
        base_config,
        graph_config,
        neighborhood,
        correlated_config,
        frozen_trace,
    ).items():
        test[name] = _router_evaluation(
            name,
            (
                _warm_start(router, split.calibration, matrix)
                if name == "trace_gcamvo"
                else router
            ),
            split.test,
            configured_latency,
        )
    calibrated_test_routers = _router_set(
        models,
        base_config,
        graph_config,
        neighborhood,
        correlated_config,
        frozen_trace,
    )
    for base_name in ("camvo", "ccamvo", "gcamvo"):
        name = f"calibrated_{base_name}"
        test[name] = _router_evaluation(
            name,
            _warm_start(calibrated_test_routers[base_name], split.calibration, matrix),
            split.test,
            configured_latency,
        )
    no_graph_router = GraphCaMVoRouter(
        models,
        HashingTextEmbedder(base_config.embedding_dim),
        base_config,
        replace(graph_config, regularization=0.0),
        neighborhood,
    )
    test["gcamvo_no_graph"] = _router_evaluation(
        "gcamvo_no_graph",
        _warm_start(no_graph_router, split.calibration, matrix),
        split.test,
        configured_latency,
    )
    tuned_graph_router = GraphCaMVoRouter(
        models,
        HashingTextEmbedder(base_config.embedding_dim),
        base_config,
        replace(graph_config, regularization=selected_graph_lambda),
        neighborhood,
    )
    test["calibrated_tuned_gcamvo"] = _router_evaluation(
        "calibrated_tuned_gcamvo",
        _warm_start(tuned_graph_router, split.calibration, matrix),
        split.test,
        configured_latency,
    )

    trace_test = test["trace_gcamvo"]
    gold = [str(item.metadata["gold_label"]) for item in split.test]
    clusters = [_group_id(item) for item in split.test]
    bootstrap_iterations = int(formal.get("bootstrap_iterations", 2_000))
    comparisons: dict[str, object] = {}
    for reference_method in (
        "cheapest_single",
        "calibrated_best_single",
        "fixed_cheap",
        "online_weighted_majority",
        "full_ensemble",
        "camvo",
        "ccamvo",
        "gcamvo",
        "calibrated_camvo",
        "calibrated_ccamvo",
        "calibrated_gcamvo",
        "calibrated_tuned_gcamvo",
    ):
        baseline = test[reference_method]
        comparisons[reference_method] = {
            "macro_f1_delta": (
                trace_test.summary.metrics.macro_f1 - baseline.summary.metrics.macro_f1
            ),
            "cost_savings": (
                0.0
                if baseline.summary.total_cost_usd == 0
                else 1.0
                - trace_test.summary.total_cost_usd / baseline.summary.total_cost_usd
            ),
            "paired_cluster_bootstrap": paired_cluster_bootstrap_delta(
                gold,
                trace_test.predictions,
                baseline.predictions,
                split.test[0].labels,
                clusters,
                metric="macro_f1",
                iterations=bootstrap_iterations,
                seed=int(config.get("seed", 17)),
            ).to_dict(),
        }

    stability_seeds = tuple(
        int(value)
        for value in formal.get("order_stability_seeds", [101, 103, 107, 109, 113])
    )
    if not stability_seeds or len(set(stability_seeds)) != len(stability_seeds):
        raise ValueError("formal.order_stability_seeds must be non-empty and unique")
    stability_runs: list[dict[str, object]] = []
    for stability_seed in stability_seeds:
        permuted = _permute_group_blocks(split.test, seed=stability_seed)
        stability_routers: dict[str, CaMVoRouter] = {
            "camvo": CaMVoRouter(
                models,
                HashingTextEmbedder(base_config.embedding_dim),
                base_config,
            ),
            "calibrated_camvo": _warm_start(
                CaMVoRouter(
                    models,
                    HashingTextEmbedder(base_config.embedding_dim),
                    base_config,
                ),
                split.calibration,
                matrix,
            ),
            "calibrated_tuned_gcamvo": _warm_start(
                GraphCaMVoRouter(
                    models,
                    HashingTextEmbedder(base_config.embedding_dim),
                    base_config,
                    replace(graph_config, regularization=selected_graph_lambda),
                    neighborhood,
                ),
                split.calibration,
                matrix,
            ),
            "trace_gcamvo": _warm_start(
                TraceGraphCaMVoRouter(
                    models,
                    HashingTextEmbedder(base_config.embedding_dim),
                    base_config,
                    frozen_trace,
                    neighborhood,
                ),
                split.calibration,
                matrix,
            ),
        }
        run_methods = {
            name: _router_evaluation(name, router, permuted, configured_latency)
            for name, router in stability_routers.items()
        }
        stability_runs.append(
            {
                "seed": stability_seed,
                "methods": {
                    name: {
                        "macro_f1": evaluation.summary.metrics.macro_f1,
                        "total_cost_usd": evaluation.summary.total_cost_usd,
                        "average_models": evaluation.summary.average_models,
                    }
                    for name, evaluation in run_methods.items()
                },
            }
        )
    stability_summary: dict[str, object] = {}
    for method in (
        "camvo",
        "calibrated_camvo",
        "calibrated_tuned_gcamvo",
        "trace_gcamvo",
    ):
        f1_values = [float(run["methods"][method]["macro_f1"]) for run in stability_runs]
        cost_values = [
            float(run["methods"][method]["total_cost_usd"]) for run in stability_runs
        ]
        stability_summary[method] = {
            "macro_f1_mean": mean(f1_values),
            "macro_f1_std": pstdev(f1_values),
            "total_cost_usd_mean": mean(cost_values),
            "total_cost_usd_std": pstdev(cost_values),
        }

    dataset_audit = {
        **dataset_metadata,
        **graph_diagnostics(items, adjacency),
        "split_group_unit": "document" if "document_id" in items[0].metadata else "30m host window",
    }
    operational_metrics = (
        {
            name: optc_operational_metrics(
                split.test,
                evaluation.predictions,
                evaluation.abstentions,
                adjacency,
            )
            for name, evaluation in test.items()
        }
        if "malicious" in split.test[0].labels
        else {}
    )
    return {
        "schema_version": 1,
        "protocol": {
            "name": "G-CaMVo frozen grouped holdout v1",
            "selection_uses_test_labels": False,
            "response_matrix_reused_by_all_methods": True,
            "calibration_items": len(split.calibration),
            "validation_items": len(split.validation),
            "test_items": len(split.test),
            "partition_sha256": {
                "calibration": _partition_digest(split.calibration),
                "validation": _partition_digest(split.validation),
                "test": _partition_digest(split.test),
            },
            "validation_reference_method": reference_name,
            "noninferiority_margin_macro_f1": f1_margin,
            "noninferiority_margin_macro_recall": recall_margin,
            "noninferiority_margin_selective_risk": risk_margin,
            "max_abstention_rate": max_abstention,
            "selected_trace_risk_tolerance": selected_risk,
            "selected_trace_graph_strength": selected_trace_graph_strength,
            "selected_gcamvo_regularization": selected_graph_lambda,
            "selection_reason": selection_reason,
        },
        "dataset": dataset_audit,
        "response_matrix": matrix_audit,
        "provider_budget": budget.snapshot().to_dict(),
        "calibration": {
            "selected_best_single": best_model,
            "models": calibration_report,
        },
        "config": {
            "camvo": base_config.to_dict(),
            "ccamvo": asdict(correlated_config),
            "graph": asdict(graph_config),
            "trace": asdict(frozen_trace),
        },
        "validation": {
            "baselines": {
                name: evaluation.summary.to_dict() for name, evaluation in validation.items()
            },
            "gcamvo_regularization_sweep": {
                str(graph_lambda): evaluation.summary.to_dict()
                for graph_lambda, evaluation in graph_candidates.items()
            },
            "trace_risk_graph_sweep": {
                f"risk={risk:g},graph_strength={graph_strength:g}": {
                    "summary": evaluation.summary.to_dict(),
                    "selective_risk": candidate_risk[(risk, graph_strength)],
                }
                for (risk, graph_strength), evaluation in candidates.items()
            },
            "reference_selective_risk": reference_risk,
            "selected": selected_validation.summary.to_dict(),
        },
        "test": {
            "methods": {
                name: evaluation.summary.to_dict() for name, evaluation in test.items()
            },
            "operational_metrics": operational_metrics,
            "trace_comparisons": comparisons,
            "gcamvo_graph_ablation": {
                "selected_regularization": selected_graph_lambda,
                "macro_f1_delta_tuned_minus_no_graph": (
                    test["calibrated_tuned_gcamvo"].summary.metrics.macro_f1
                    - test["gcamvo_no_graph"].summary.metrics.macro_f1
                ),
                "cost_delta_tuned_minus_no_graph": (
                    test["calibrated_tuned_gcamvo"].summary.total_cost_usd
                    - test["gcamvo_no_graph"].summary.total_cost_usd
                ),
            },
            "document_block_order_stability": {
                "used_for_selection": False,
                "seeds": list(stability_seeds),
                "summary": stability_summary,
                "runs": stability_runs,
            },
        },
        "claim_guardrails": [
            (
                "Dataset text, labels, and provider responses are real; routing is evaluated "
                "offline on a frozen response matrix."
            ),
            (
                "Threshold/model selection uses calibration and validation only; test labels "
                "are touched once after freezing."
            ),
            (
                "Reported dollar cost is a configured token-price proxy when the internal "
                "platform is quota-based."
            ),
            (
                "CASIE is event-subtype classification; incident detection claims require "
                "the separate OpTC experiment."
            ),
        ],
    }
