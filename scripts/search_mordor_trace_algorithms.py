"""Offline search for stronger TRACE-G-CaMVo policies on cached Mordor outputs.

This script makes **zero provider calls**.  It treats the already collected
model-response matrix as a fixed panel of experts and asks a narrower question:
how much can a deployable, graph-aware router improve by learning only from the
calibration split and selecting hyperparameters only on validation?

The test labels are used exactly once for the frozen comparison and for clearly
marked diagnostic oracle ceilings.  Those ceilings are not deployable methods.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.router import CaMVoRouter
from camvo.security.formal_experiment import (
    FrozenResponseMatrixClient,
    _base_router_config,
    _calibrate_models,
    _group_id,
    _router_evaluation,
    _warm_start,
)
from camvo.security.metrics import ClassificationMetrics, classification_metrics
from camvo.security.mordor_dataset import build_mordor_cdb_binary_dataset
from camvo.security.optc import OptcCorrelationEdge
from camvo.security.simulated_experiments import optc_adjacency
from camvo.security.splits import DatasetSplit, stratified_grouped_split
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.types import AnnotationItem, ModelPricing, ModelResponse

from analyze_partial_mordor import _cache_rows, _read_json


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("benign", "malicious")


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        logits = np.clip(design @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class Candidate:
    family: str
    subset: tuple[int, ...]
    alpha: float
    l2: float
    threshold: float
    margin: float = 0.0
    conflict: float = 1.0
    model: LogisticModel | None = None
    graph_mode: str = "undirected"
    propagation_beta: float = 0.0
    transition: tuple[float, float] = (0.25, 0.75)
    transition_prior: float = 0.5
    message_direction: str = "none"


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    steps: int = 1_800,
    learning_rate: float = 0.08,
) -> LogisticModel:
    """Small dependency-free class-balanced logistic stacker."""

    if features.ndim != 2 or len(features) != len(targets) or not len(targets):
        raise ValueError("invalid feature/target matrix")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    weights = np.zeros(design.shape[1], dtype=float)
    positives = max(1, int(np.sum(targets == 1)))
    negatives = max(1, int(np.sum(targets == 0)))
    sample_weights = np.where(
        targets == 1,
        len(targets) / (2.0 * positives),
        len(targets) / (2.0 * negatives),
    )
    for step in range(steps):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - targets) * sample_weights) / len(targets)
        gradient[1:] += l2 * weights[1:]
        rate = learning_rate / math.sqrt(1.0 + step / 300.0)
        weights -= rate * gradient
    return LogisticModel(mean=mean, scale=scale, weights=weights)


def _response(raw: Mapping[str, Any]) -> ModelResponse:
    return ModelResponse(
        label=str(raw["label"]),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 1)),
        raw=raw.get("raw"),
    )


def _malicious_probability(response: ModelResponse) -> float:
    confidence = 0.75
    if isinstance(response.raw, Mapping):
        try:
            confidence = float(response.raw.get("confidence", confidence))
        except (TypeError, ValueError):
            confidence = 0.75
    confidence = min(0.99, max(0.51, confidence))
    return confidence if response.label == "malicious" else 1.0 - confidence


def _partition_arrays(
    items: Sequence[AnnotationItem],
    matrix: Mapping[str, Mapping[str, ModelResponse]],
    model_ids: Sequence[str],
    pricing: Mapping[str, ModelPricing],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    item_ids = [item.item_id for item in items]
    probabilities = np.asarray(
        [
            [_malicious_probability(matrix[item_id][model_id]) for model_id in model_ids]
            for item_id in item_ids
        ],
        dtype=float,
    )
    costs = np.asarray(
        [
            [
                pricing[model_id].cost(
                    matrix[item_id][model_id].input_tokens,
                    matrix[item_id][model_id].output_tokens,
                )
                for model_id in model_ids
            ]
            for item_id in item_ids
        ],
        dtype=float,
    )
    targets = np.asarray(
        [int(item.metadata["gold_label"] == "malicious") for item in items], dtype=int
    )
    return item_ids, probabilities, costs, targets


def graph_neighbor_mean(
    item_ids: Sequence[str],
    values: np.ndarray,
    adjacency: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted neighbor means using only nodes in the current split."""

    if values.ndim != 2 or len(item_ids) != len(values):
        raise ValueError("item_ids and values are inconsistent")
    index = {item_id: position for position, item_id in enumerate(item_ids)}
    means = values.copy()
    degree = np.zeros(len(item_ids), dtype=float)
    for row, item_id in enumerate(item_ids):
        weighted = np.zeros(values.shape[1], dtype=float)
        total = 0.0
        for neighbor, raw_weight in adjacency.get(item_id, {}).items():
            position = index.get(neighbor)
            weight = max(0.0, float(raw_weight))
            if position is None or weight == 0:
                continue
            weighted += weight * values[position]
            total += weight
        if total:
            means[row] = weighted / total
            degree[row] = total
    return means, degree


def directed_parent_adjacency(
    correlations: Sequence[Any],
    items: Mapping[str, AnnotationItem],
    *,
    required_reason: str | None = None,
) -> dict[str, dict[str, float]]:
    """Recover temporal direction lost by the symmetric graph representation."""

    parents: dict[str, dict[str, float]] = {}
    for edge in correlations:
        if required_reason is not None and required_reason not in edge.reasons:
            continue
        left = items.get(edge.source_event_id)
        right = items.get(edge.target_event_id)
        if left is None or right is None:
            continue
        left_time = int(left.metadata["timestamp_ms"])
        right_time = int(right.metadata["timestamp_ms"])
        if (left_time, left.item_id) <= (right_time, right.item_id):
            parent, child = left.item_id, right.item_id
        else:
            parent, child = right.item_id, left.item_id
        target = parents.setdefault(child, {})
        target[parent] = target.get(parent, 0.0) + float(edge.weight)
    return parents


def _logit_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def causal_parent_logit_score(
    partition: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    parent_adjacency: Mapping[str, Mapping[str, float]],
    *,
    base_index: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    item_ids, probabilities, _costs, _targets = partition
    parent_mean, degree = graph_neighbor_mean(item_ids, probabilities, parent_adjacency)
    score = _logit_array(probabilities[:, base_index]) + beta * _logit_array(
        parent_mean[:, base_index]
    )
    return score, degree


def _search_low_capacity_causal(
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    base_index: int,
) -> tuple[dict[str, Any], ClassificationMetrics]:
    best: tuple[tuple[float, float, float], dict[str, Any], ClassificationMetrics] | None = None
    for graph_mode in (
        "causal_parents",
        "causal_process",
        "causal_host",
        "causal_network",
    ):
        for beta in (0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50, 0.75, 1.0):
            scores, degree = causal_parent_logit_score(
                validation,
                adjacencies[graph_mode],
                base_index=base_index,
                beta=beta,
            )
            for threshold in tuple(
                sorted(
                    {
                        -4.0,
                        -3.5,
                        -3.0,
                        -2.5,
                        -2.0,
                        -1.5,
                        -1.0,
                        -0.5,
                        0.0,
                        *(float(value) for value in np.quantile(scores, np.linspace(0.02, 0.98, 49))),
                    }
                )
            ):
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores - threshold, -30, 30)))
                metrics = _score(validation[3], probabilities, 0.5)
                config = {
                    "family": "causal_parent_logit",
                    "graph_mode": graph_mode,
                    "beta": beta,
                    "score_threshold": threshold,
                    "validation_parent_coverage": float(np.mean(degree > 0)),
                }
                record = (_rank(metrics, 1.0), config, metrics)
                if best is None or record[0] > best[0]:
                    best = record
    assert best is not None
    return best[1], best[2]


def _apply_low_capacity_causal(
    config: Mapping[str, Any],
    partition: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    base_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, degree = causal_parent_logit_score(
        partition,
        adjacencies[str(config["graph_mode"])],
        base_index=base_index,
        beta=float(config["beta"]),
    )
    probability = 1.0 / (
        1.0
        + np.exp(
            -np.clip(scores - float(config["score_threshold"]), -30.0, 30.0)
        )
    )
    return probability, degree


def _search_base_threshold(
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    *,
    base_index: int,
) -> tuple[float, ClassificationMetrics]:
    probabilities = validation[1][:, base_index]
    best: tuple[tuple[float, float, float], float, ClassificationMetrics] | None = None
    candidates = {
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        *(float(value) for value in np.quantile(probabilities, np.linspace(0.02, 0.98, 49))),
    }
    for threshold in sorted(candidates):
        metrics = _score(validation[3], probabilities, threshold)
        record = (_rank(metrics, 1.0), threshold, metrics)
        if best is None or record[0] > best[0]:
            best = record
    assert best is not None
    return best[1], best[2]


def estimate_chain_transition(
    item_ids: Sequence[str],
    targets: np.ndarray,
    parents: Mapping[str, Mapping[str, float]],
) -> tuple[float, float]:
    """Estimate P(child=malicious | parent label) with a Beta(1,1) prior."""

    index = {item_id: row for row, item_id in enumerate(item_ids)}
    malicious_weight = [1.0, 1.0]
    total_weight = [2.0, 2.0]
    for child, child_parents in parents.items():
        child_row = index.get(child)
        if child_row is None:
            continue
        for parent, raw_weight in child_parents.items():
            parent_row = index.get(parent)
            if parent_row is None:
                continue
            weight = max(0.0, float(raw_weight))
            parent_label = int(targets[parent_row])
            total_weight[parent_label] += weight
            malicious_weight[parent_label] += weight * int(targets[child_row])
    return (
        malicious_weight[0] / total_weight[0],
        malicious_weight[1] / total_weight[1],
    )


def causal_chain_propagation(
    item_ids: Sequence[str],
    unary: np.ndarray,
    parents: Mapping[str, Mapping[str, float]],
    timestamps: Mapping[str, int],
    *,
    beta: float,
    transition: tuple[float, float],
    prior: float,
) -> np.ndarray:
    """One causal forward pass; it never reads children or future events."""

    if beta == 0:
        return unary.copy()
    index = {item_id: row for row, item_id in enumerate(item_ids)}
    posterior = unary.copy()
    eps = 1e-5

    def logit(value: float) -> float:
        clipped = min(1.0 - eps, max(eps, value))
        return math.log(clipped / (1.0 - clipped))

    for item_id in sorted(item_ids, key=lambda value: (timestamps[value], value)):
        row = index[item_id]
        weighted = 0.0
        total = 0.0
        for parent, raw_weight in parents.get(item_id, {}).items():
            parent_row = index.get(parent)
            if parent_row is None or timestamps[parent] > timestamps[item_id]:
                continue
            weight = max(0.0, float(raw_weight))
            parent_probability = float(posterior[parent_row])
            predicted_child = (
                (1.0 - parent_probability) * transition[0]
                + parent_probability * transition[1]
            )
            weighted += weight * predicted_child
            total += weight
        if total:
            message = weighted / total
            combined = logit(float(unary[row])) + beta * (logit(message) - logit(prior))
            posterior[row] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, combined))))
    return posterior


def estimate_same_label_potential(
    item_ids: Sequence[str],
    targets: np.ndarray,
    adjacency: Mapping[str, Mapping[str, float]],
) -> float:
    index = {item_id: row for row, item_id in enumerate(item_ids)}
    same = 1.0
    total = 2.0
    visited: set[tuple[str, str]] = set()
    for left, neighbors in adjacency.items():
        left_row = index.get(left)
        if left_row is None:
            continue
        for right, raw_weight in neighbors.items():
            right_row = index.get(right)
            pair = tuple(sorted((left, right)))
            if right_row is None or pair in visited:
                continue
            visited.add(pair)
            weight = max(0.0, float(raw_weight))
            total += weight
            same += weight * int(targets[left_row] == targets[right_row])
    return same / total


def bidirectional_belief_propagation(
    item_ids: Sequence[str],
    unary: np.ndarray,
    adjacency: Mapping[str, Mapping[str, float]],
    *,
    beta: float,
    same_label_probability: float,
    prior: float,
    iterations: int = 6,
) -> np.ndarray:
    """Label-free loopy BP approximation for offline incident reconstruction."""

    if beta == 0:
        return unary.copy()
    index = {item_id: row for row, item_id in enumerate(item_ids)}
    posterior = unary.copy()
    eps = 1e-5

    def logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))

    for _ in range(iterations):
        messages = np.full(len(item_ids), prior, dtype=float)
        for row, item_id in enumerate(item_ids):
            weighted = 0.0
            total = 0.0
            for neighbor, raw_weight in adjacency.get(item_id, {}).items():
                neighbor_row = index.get(neighbor)
                if neighbor_row is None:
                    continue
                weight = max(0.0, float(raw_weight))
                q = posterior[neighbor_row]
                predicted = q * same_label_probability + (1.0 - q) * (
                    1.0 - same_label_probability
                )
                weighted += weight * predicted
                total += weight
            if total:
                messages[row] = weighted / total
        posterior = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    logit(unary) + beta * (logit(messages) - logit(np.asarray([prior]))[0]),
                    -30.0,
                    30.0,
                )
            )
        )
    return posterior


def make_features(
    probabilities: np.ndarray,
    neighbor_probabilities: np.ndarray,
    degree: np.ndarray,
    subset: tuple[int, ...],
    alpha: float,
) -> np.ndarray:
    base = probabilities[:, subset]
    neighbor = neighbor_probabilities[:, subset]
    smooth = (1.0 - alpha) * base + alpha * neighbor
    vote_fraction = np.mean(base >= 0.5, axis=1, keepdims=True)
    columns = [
        base,
        smooth,
        smooth - base,
        np.mean(base, axis=1, keepdims=True),
        np.std(base, axis=1, keepdims=True),
        vote_fraction,
        np.abs(vote_fraction - 0.5),
        np.log1p(degree).reshape(-1, 1),
    ]
    return np.column_stack(columns)


def _predictions(probabilities: np.ndarray, threshold: float) -> list[str]:
    return ["malicious" if value >= threshold else "benign" for value in probabilities]


def _score(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> ClassificationMetrics:
    gold = ["malicious" if value else "benign" for value in targets]
    return classification_metrics(gold, _predictions(probabilities, threshold), LABELS)


def _threshold_grid(probabilities: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(probabilities, np.linspace(0.05, 0.95, 19))
    values = {0.5, *(float(value) for value in quantiles)}
    return tuple(sorted(min(0.95, max(0.05, value)) for value in values))


def _rank(metrics: ClassificationMetrics, average_models: float) -> tuple[float, float, float]:
    malicious = metrics.per_class["malicious"]
    return metrics.macro_f1, float(malicious["recall"]), -average_models


def _method_summary(
    name: str,
    targets: np.ndarray,
    predictions: list[str],
    costs: np.ndarray,
    selected: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    gold = ["malicious" if value else "benign" for value in targets]
    metrics = classification_metrics(gold, predictions, LABELS)
    total_cost = sum(float(np.sum(costs[row, list(indices)])) for row, indices in enumerate(selected))
    return {
        "name": name,
        "metrics": metrics.to_dict(),
        "total_cost_usd": total_cost,
        "average_models": float(np.mean([len(indices) for indices in selected])),
        "predictions": predictions,
    }


def _all_selected(rows: int, subset: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [subset for _ in range(rows)]


def _search_stacker(
    calibration: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    subset: tuple[int, ...],
    alpha: float,
    graph_mode: str,
) -> tuple[Candidate, ClassificationMetrics]:
    cal_ids, cal_p, _cal_cost, cal_y = calibration
    val_ids, val_p, _val_cost, val_y = validation
    adjacency = adjacencies[graph_mode]
    cal_neighbor, cal_degree = graph_neighbor_mean(cal_ids, cal_p, adjacency)
    val_neighbor, val_degree = graph_neighbor_mean(val_ids, val_p, adjacency)
    cal_x = make_features(cal_p, cal_neighbor, cal_degree, subset, alpha)
    val_x = make_features(val_p, val_neighbor, val_degree, subset, alpha)
    best: tuple[tuple[float, float, float], Candidate, ClassificationMetrics] | None = None
    for l2 in (0.0, 0.001, 0.01, 0.1, 1.0):
        model = fit_logistic(cal_x, cal_y, l2=l2)
        val_probability = model.predict(val_x)
        for threshold in _threshold_grid(val_probability):
            metrics = _score(val_y, val_probability, threshold)
            candidate = Candidate(
                "graph_stacker", subset, alpha, l2, threshold, model=model, graph_mode=graph_mode
            )
            record = (_rank(metrics, len(subset)), candidate, metrics)
            if best is None or record[0] > best[0]:
                best = record
    assert best is not None
    return best[1], best[2]


def _apply_stacker(
    candidate: Candidate,
    partition: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> np.ndarray:
    item_ids, probabilities, _costs, _targets = partition
    neighbor, degree = graph_neighbor_mean(
        item_ids, probabilities, adjacencies[candidate.graph_mode]
    )
    features = make_features(
        probabilities, neighbor, degree, candidate.subset, candidate.alpha
    )
    assert candidate.model is not None
    return candidate.model.predict(features)


def _candidate_probability(
    candidate: Candidate,
    partition: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    timestamps: Mapping[str, int],
) -> np.ndarray:
    unary = _apply_stacker(candidate, partition, adjacencies)
    if candidate.propagation_beta == 0:
        return unary
    if candidate.message_direction == "bidirectional":
        return bidirectional_belief_propagation(
            partition[0],
            unary,
            adjacencies["undirected"],
            beta=candidate.propagation_beta,
            same_label_probability=candidate.transition[1],
            prior=candidate.transition_prior,
        )
    return causal_chain_propagation(
        partition[0],
        unary,
        adjacencies["causal_parents"],
        timestamps,
        beta=candidate.propagation_beta,
        transition=candidate.transition,
        prior=candidate.transition_prior,
    )


def _search_causal_chain(
    stackers: Sequence[Candidate],
    calibration: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    timestamps: Mapping[str, int],
) -> tuple[Candidate, ClassificationMetrics]:
    transition = estimate_chain_transition(
        calibration[0], calibration[3], adjacencies["causal_parents"]
    )
    prior = float(np.mean(calibration[3]))
    best: tuple[tuple[float, float, float], Candidate, ClassificationMetrics] | None = None
    for stacker in stackers:
        unary = _apply_stacker(stacker, validation, adjacencies)
        for beta in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5):
            probability = causal_chain_propagation(
                validation[0],
                unary,
                adjacencies["causal_parents"],
                timestamps,
                beta=beta,
                transition=transition,
                prior=prior,
            )
            for threshold in _threshold_grid(probability):
                metrics = _score(validation[3], probability, threshold)
                candidate = Candidate(
                    "causal_trace_message_passing",
                    stacker.subset,
                    stacker.alpha,
                    stacker.l2,
                    threshold,
                    model=stacker.model,
                    graph_mode=stacker.graph_mode,
                    propagation_beta=beta,
                    transition=transition,
                    transition_prior=prior,
                    message_direction="causal",
                )
                record = (_rank(metrics, len(stacker.subset)), candidate, metrics)
                if best is None or record[0] > best[0]:
                    best = record
    assert best is not None
    return best[1], best[2]


def _search_bidirectional_crf(
    stackers: Sequence[Candidate],
    calibration: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[Candidate, ClassificationMetrics]:
    same_label = estimate_same_label_potential(
        calibration[0], calibration[3], adjacencies["undirected"]
    )
    prior = float(np.mean(calibration[3]))
    best: tuple[tuple[float, float, float], Candidate, ClassificationMetrics] | None = None
    for stacker in stackers:
        unary = _apply_stacker(stacker, validation, adjacencies)
        for beta in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
            probability = bidirectional_belief_propagation(
                validation[0],
                unary,
                adjacencies["undirected"],
                beta=beta,
                same_label_probability=same_label,
                prior=prior,
            )
            for threshold in _threshold_grid(probability):
                metrics = _score(validation[3], probability, threshold)
                candidate = Candidate(
                    "bidirectional_crf_message_passing",
                    stacker.subset,
                    stacker.alpha,
                    stacker.l2,
                    threshold,
                    model=stacker.model,
                    graph_mode=stacker.graph_mode,
                    propagation_beta=beta,
                    transition=(1.0 - same_label, same_label),
                    transition_prior=prior,
                    message_direction="bidirectional",
                )
                record = (_rank(metrics, len(stacker.subset)), candidate, metrics)
                if best is None or record[0] > best[0]:
                    best = record
    assert best is not None
    return best[1], best[2]


def _search_gated(
    stackers: Sequence[Candidate],
    validation: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    timestamps: Mapping[str, int],
    base_index: int,
) -> tuple[Candidate, ClassificationMetrics, np.ndarray, list[tuple[int, ...]]]:
    val_ids, val_p, _val_cost, val_y = validation
    neighbor, _degree = graph_neighbor_mean(
        val_ids, val_p, adjacencies["causal_parents"]
    )
    base_probability = val_p[:, base_index]
    graph_base = neighbor[:, base_index]
    best: tuple[tuple[float, float, float], Candidate, ClassificationMetrics, np.ndarray, list[tuple[int, ...]]] | None = None
    for stacker in stackers:
        stacked = _candidate_probability(stacker, validation, adjacencies, timestamps)
        for margin in (0.05, 0.10, 0.15, 0.20, 0.30):
            for conflict in (0.10, 0.20, 0.30, 0.45, 1.0):
                gate = (np.abs(base_probability - 0.5) <= margin) | (
                    np.abs(graph_base - base_probability) >= conflict
                )
                final = np.where(gate, stacked, base_probability)
                for threshold in _threshold_grid(final):
                    metrics = _score(val_y, final, threshold)
                    selected = [
                        stacker.subset if escalated else (base_index,)
                        for escalated in gate
                    ]
                    average_models = float(np.mean([len(value) for value in selected]))
                    candidate = Candidate(
                        "flash_first_graph_gate",
                        stacker.subset,
                        stacker.alpha,
                        stacker.l2,
                        threshold,
                        margin,
                        conflict,
                        stacker.model,
                        stacker.graph_mode,
                        stacker.propagation_beta,
                        stacker.transition,
                        stacker.transition_prior,
                        stacker.message_direction,
                    )
                    record = (_rank(metrics, average_models), candidate, metrics, final, selected)
                    if best is None or record[0] > best[0]:
                        best = record
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _apply_gated(
    candidate: Candidate,
    partition: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    adjacencies: Mapping[str, Mapping[str, Mapping[str, float]]],
    timestamps: Mapping[str, int],
    base_index: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    item_ids, probabilities, _costs, _targets = partition
    neighbor, _degree = graph_neighbor_mean(
        item_ids, probabilities, adjacencies["causal_parents"]
    )
    base = probabilities[:, base_index]
    stacked = _candidate_probability(candidate, partition, adjacencies, timestamps)
    gate = (np.abs(base - 0.5) <= candidate.margin) | (
        np.abs(neighbor[:, base_index] - base) >= candidate.conflict
    )
    final = np.where(gate, stacked, base)
    selected = [candidate.subset if value else (base_index,) for value in gate]
    return final, selected


def _candidate_dict(candidate: Candidate, model_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "family": candidate.family,
        "models": [model_ids[index] for index in candidate.subset],
        "graph_alpha": candidate.alpha,
        "l2": candidate.l2,
        "threshold": candidate.threshold,
        "uncertainty_margin": candidate.margin,
        "graph_conflict_threshold": candidate.conflict,
        "graph_mode": candidate.graph_mode,
        "causal_message_beta": candidate.propagation_beta,
        "learned_transition_p_child_malicious_given_parent_benign": candidate.transition[0],
        "learned_transition_p_child_malicious_given_parent_malicious": candidate.transition[1],
        "message_direction": candidate.message_direction,
    }


def _load_offline_bundle(bundle_dir: Path) -> tuple[
    dict[str, Any],
    Any,
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, dict[str, ModelResponse]],
    tuple[AnnotationItem, ...],
    DatasetSplit,
    int,
]:
    manifest = _read_json(bundle_dir / "manifest.json")
    safe_models = json.loads((bundle_dir / "models.json").read_text(encoding="utf-8"))
    model_config = {str(raw["model_id"]): raw for raw in safe_models}
    model_ids = tuple(model_config)
    raw_rows = [
        json.loads(line)
        for line in (bundle_dir / "items_with_responses.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    items = tuple(
        AnnotationItem(
            item_id=str(row["item_id"]),
            text=str(row["text"]),
            labels=tuple(row["labels"]),
            metadata=dict(row["metadata"]),
        )
        for row in raw_rows
    )
    raw_by_id = {str(row["item_id"]): row for row in raw_rows}
    complete_items = tuple(
        item
        for item in items
        if all(model_id in raw_by_id[item.item_id]["responses"] for model_id in model_ids)
    )
    complete_ids = {item.item_id for item in complete_items}
    matrix = {
        item.item_id: {
            model_id: _response(raw_by_id[item.item_id]["responses"][model_id])
            for model_id in model_ids
        }
        for item in complete_items
    }
    edge_rows = [
        json.loads(line)
        for line in (bundle_dir / "directed_edges.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    correlations = tuple(
        OptcCorrelationEdge(
            source_event_id=str(row["parent_id"]),
            target_event_id=str(row["child_id"]),
            weight=float(row["weight"]),
            reasons=tuple(row["reasons"]),
            time_gap_ms=int(row["time_gap_ms"]),
        )
        for row in edge_rows
    )
    split_ids = _read_json(bundle_dir / "splits.json")
    item_map = {item.item_id: item for item in items}

    def partition(name: str) -> tuple[AnnotationItem, ...]:
        return tuple(
            item_map[item_id]
            for item_id in split_ids[name]
            if item_id in complete_ids
        )

    split = DatasetSplit(
        calibration=partition("calibration"),
        validation=partition("validation"),
        test=partition("test"),
    )
    built = SimpleNamespace(items=items, correlations=correlations, stats=manifest.get("dataset_stats", {}))
    config = {"seed": 17, "router": {}}
    return (
        config,
        built,
        model_config,
        model_ids,
        matrix,
        complete_items,
        split,
        int(manifest["items"]),
    )


def analyze(
    run_dir: Path | None,
    *,
    bundle_dir: Path | None = None,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    if bundle_dir is not None:
        (
            config,
            built,
            model_config,
            model_ids,
            matrix,
            complete_items,
            split,
            requested,
        ) = _load_offline_bundle(bundle_dir)
    else:
        if run_dir is None:
            raise ValueError("run_dir is required when bundle_dir is omitted")
        config = _read_json(run_dir / "config.json")
        dataset = config["dataset"]
        requested = int(dataset["max_items"])
        built = build_mordor_cdb_binary_dataset(
            dataset["path"],
            dataset["flags_path"],
            max_positive_items=None if dataset.get("require_all_flags") else requested // 2,
            correlation_window_minutes=int(dataset.get("correlation_window_minutes", 30)),
            max_records_per_item=int(dataset.get("max_records_per_item", 6)),
            max_chars_per_record=int(dataset.get("max_chars_per_record", 1_200)),
        )
        item_map = {item.item_id: item for item in built.items}
        model_config = {str(raw["model_id"]): raw for raw in config["models"]}
        model_ids = tuple(model_config)
        cached = _cache_rows(Path(config["budget"]["cache_dir"]))
        complete_ids = {
            item_id
            for item_id, row in cached.items()
            if item_id in item_map and all(model_id in row for model_id in model_ids)
        }
        complete_items = tuple(item for item in built.items if item.item_id in complete_ids)
        matrix = {
            item.item_id: {
                model_id: _response(cached[item.item_id][model_id]) for model_id in model_ids
            }
            for item in complete_items
        }
        split = stratified_grouped_split(
            complete_items,
            group_key=_group_id,
            calibration_fraction=float(config["formal"].get("calibration_fraction", 0.20)),
            validation_fraction=float(config["formal"].get("validation_fraction", 0.10)),
            seed=int(config.get("seed", 17)),
        )
    if len(complete_items) < 100:
        raise RuntimeError("fewer than 100 complete Mordor rows")
    item_map = {item.item_id: item for item in built.items}
    pricing = {
        model_id: ModelPricing(
            input_per_million=float(raw.get("input_usd_per_million", 0.0)),
            output_per_million=float(raw.get("output_usd_per_million", 0.0)),
        )
        for model_id, raw in model_config.items()
    }
    partitions = {
        "calibration": _partition_arrays(split.calibration, matrix, model_ids, pricing),
        "validation": _partition_arrays(split.validation, matrix, model_ids, pricing),
        "test": _partition_arrays(split.test, matrix, model_ids, pricing),
    }
    timestamps = {
        item.item_id: int(item.metadata["timestamp_ms"]) for item in complete_items
    }
    adjacencies = {
        "undirected": optc_adjacency(built.correlations),
        "causal_parents": directed_parent_adjacency(built.correlations, item_map),
        "causal_process": directed_parent_adjacency(
            built.correlations, item_map, required_reason="process"
        ),
        "causal_host": directed_parent_adjacency(
            built.correlations, item_map, required_reason="host"
        ),
        "causal_network": directed_parent_adjacency(
            built.correlations, item_map, required_reason="network"
        ),
    }

    # The same calibration protocol chooses the one-model anchor; this is not
    # allowed to change after validation/test inspection.
    frozen_clients = [
        FrozenResponseMatrixClient(
            model_id,
            pricing[model_id],
            {item_id: row[model_id] for item_id, row in matrix.items()},
        )
        for model_id in model_ids
    ]
    fallback_latency = {
        model_id: float(raw.get("latency_ms", 0.0)) for model_id, raw in model_config.items()
    }
    _specs, calibration_report, best_model = _calibrate_models(
        split.calibration, frozen_clients, fallback_latency
    )
    base_index = model_ids.index(best_model)
    selected_base_threshold, base_threshold_validation = _search_base_threshold(
        partitions["validation"], base_index=base_index
    )
    low_capacity_causal, low_capacity_causal_validation = _search_low_capacity_causal(
        partitions["validation"], adjacencies, base_index=base_index
    )

    subsets = [
        tuple(indices)
        for size in range(1, len(model_ids) + 1)
        for indices in itertools.combinations(range(len(model_ids)), size)
        if base_index in indices
    ]
    stacker_records: list[tuple[Candidate, ClassificationMetrics]] = []
    for subset in subsets:
        for graph_mode in ("undirected", "causal_parents"):
            for alpha in (0.0, 0.25, 0.50, 0.75):
                stacker_records.append(
                    _search_stacker(
                        partitions["calibration"],
                        partitions["validation"],
                        adjacencies,
                        subset,
                        alpha,
                        graph_mode,
                    )
                )
    best_stacker, best_stacker_validation = max(
        stacker_records,
        key=lambda record: _rank(record[1], len(record[0].subset)),
    )
    best_causal, best_causal_validation = _search_causal_chain(
        [record[0] for record in stacker_records],
        partitions["calibration"],
        partitions["validation"],
        adjacencies,
        timestamps,
    )
    best_bidirectional, best_bidirectional_validation = _search_bidirectional_crf(
        [record[0] for record in stacker_records],
        partitions["calibration"],
        partitions["validation"],
        adjacencies,
    )
    best_gate, best_gate_validation, _val_gate_p, _val_selected = _search_gated(
        [best_causal, best_bidirectional, *(record[0] for record in stacker_records)],
        partitions["validation"],
        adjacencies,
        timestamps,
        base_index,
    )

    test_ids, test_p, test_costs, test_y = partitions["test"]
    gold = ["malicious" if value else "benign" for value in test_y]
    clusters = [_group_id(item) for item in split.test]
    methods: dict[str, dict[str, Any]] = {}
    for index, model_id in enumerate(model_ids):
        predictions = _predictions(test_p[:, index], 0.5)
        methods[f"single::{model_id}"] = _method_summary(
            f"single::{model_id}", test_y, predictions, test_costs, _all_selected(len(test_y), (index,))
        )
    calibrated_base_predictions = _predictions(
        test_p[:, base_index], selected_base_threshold
    )
    methods["validation_calibrated_single"] = _method_summary(
        "validation_calibrated_single",
        test_y,
        calibrated_base_predictions,
        test_costs,
        _all_selected(len(test_y), (base_index,)),
    )
    causal_logit_probability, causal_logit_degree = _apply_low_capacity_causal(
        low_capacity_causal,
        partitions["test"],
        adjacencies,
        base_index=base_index,
    )
    causal_logit_predictions = _predictions(causal_logit_probability, 0.5)
    methods["selected_causal_parent_logit"] = _method_summary(
        "selected_causal_parent_logit",
        test_y,
        causal_logit_predictions,
        test_costs,
        _all_selected(len(test_y), (base_index,)),
    )
    methods["selected_causal_parent_logit"]["parent_coverage"] = float(
        np.mean(causal_logit_degree > 0)
    )

    stack_probability = _candidate_probability(
        best_stacker, partitions["test"], adjacencies, timestamps
    )
    stack_predictions = _predictions(stack_probability, best_stacker.threshold)
    methods["selected_graph_stacker"] = _method_summary(
        "selected_graph_stacker",
        test_y,
        stack_predictions,
        test_costs,
        _all_selected(len(test_y), best_stacker.subset),
    )
    causal_probability = _candidate_probability(
        best_causal, partitions["test"], adjacencies, timestamps
    )
    causal_predictions = _predictions(causal_probability, best_causal.threshold)
    methods["selected_causal_trace_message_passing"] = _method_summary(
        "selected_causal_trace_message_passing",
        test_y,
        causal_predictions,
        test_costs,
        _all_selected(len(test_y), best_causal.subset),
    )
    bidirectional_probability = _candidate_probability(
        best_bidirectional, partitions["test"], adjacencies, timestamps
    )
    bidirectional_predictions = _predictions(
        bidirectional_probability, best_bidirectional.threshold
    )
    methods["selected_bidirectional_crf_message_passing"] = _method_summary(
        "selected_bidirectional_crf_message_passing",
        test_y,
        bidirectional_predictions,
        test_costs,
        _all_selected(len(test_y), best_bidirectional.subset),
    )
    gate_probability, gate_selected = _apply_gated(
        best_gate, partitions["test"], adjacencies, timestamps, base_index
    )
    gate_predictions = _predictions(gate_probability, best_gate.threshold)
    methods["selected_flash_first_trace_gate"] = _method_summary(
        "selected_flash_first_trace_gate", test_y, gate_predictions, test_costs, gate_selected
    )

    # Reproduce CaMVo on the identical split for a direct comparison.
    base_config = _base_router_config(config)
    latency = {model_id: fallback_latency[model_id] for model_id in model_ids}
    camvo = _router_evaluation(
        "calibrated_camvo",
        _warm_start(
            CaMVoRouter(frozen_clients, HashingTextEmbedder(base_config.embedding_dim), base_config),
            split.calibration,
            matrix,
        ),
        split.test,
        latency,
    )
    methods["calibrated_camvo"] = {
        "name": "calibrated_camvo",
        **camvo.summary.to_dict(),
        "predictions": list(camvo.predictions),
    }

    # Diagnostic ceiling only: for every test row, count it correct if any
    # cached expert is correct.  This uses test labels and is not a policy.
    raw_labels = np.asarray(
        [[matrix[item_id][model_id].label for model_id in model_ids] for item_id in test_ids]
    )
    oracle_predictions = []
    best_predictions = methods[f"single::{best_model}"]["predictions"]
    for row, label in enumerate(gold):
        oracle_predictions.append(label if label in raw_labels[row] else best_predictions[row])
    oracle_metrics = classification_metrics(gold, oracle_predictions, LABELS)
    unanimous_wrong = int(np.sum(np.all(raw_labels != np.asarray(gold)[:, None], axis=1)))
    disagreement = int(np.sum(np.any(raw_labels != raw_labels[:, :1], axis=1)))

    reference_flash = methods[f"single::{best_model}"]
    comparisons: dict[str, Any] = {}
    for name in (
        "validation_calibrated_single",
        "selected_causal_parent_logit",
        "selected_graph_stacker",
        "selected_causal_trace_message_passing",
        "selected_bidirectional_crf_message_passing",
        "selected_flash_first_trace_gate",
        "calibrated_camvo",
    ):
        candidate = methods[name]
        comparisons[f"{name}_vs_{best_model}"] = paired_cluster_bootstrap_delta(
            gold,
            candidate["predictions"],
            reference_flash["predictions"],
            LABELS,
            clusters,
            iterations=bootstrap_iterations,
            seed=int(config.get("seed", 17)),
        ).to_dict()
    comparisons["selected_flash_first_trace_gate_vs_calibrated_camvo"] = (
        paired_cluster_bootstrap_delta(
            gold,
            methods["selected_flash_first_trace_gate"]["predictions"],
            methods["calibrated_camvo"]["predictions"],
            LABELS,
            clusters,
            iterations=bootstrap_iterations,
            seed=int(config.get("seed", 17)) + 1,
        ).to_dict()
    )
    comparisons["selected_causal_parent_logit_vs_validation_calibrated_single"] = (
        paired_cluster_bootstrap_delta(
            gold,
            methods["selected_causal_parent_logit"]["predictions"],
            methods["validation_calibrated_single"]["predictions"],
            LABELS,
            clusters,
            iterations=bootstrap_iterations,
            seed=int(config.get("seed", 17)) + 2,
        ).to_dict()
    )
    comparisons["selected_causal_parent_logit_vs_calibrated_camvo"] = (
        paired_cluster_bootstrap_delta(
            gold,
            methods["selected_causal_parent_logit"]["predictions"],
            methods["calibrated_camvo"]["predictions"],
            LABELS,
            clusters,
            iterations=bootstrap_iterations,
            seed=int(config.get("seed", 17)) + 3,
        ).to_dict()
    )

    # Predictions are useful for audit but too noisy for the human-facing JSON.
    clean_methods = {
        name: {key: value for key, value in row.items() if key != "predictions"}
        for name, row in methods.items()
    }
    return {
        "status": "offline_cached_trace_algorithm_search",
        "provider_calls": 0,
        "claim_boundary": (
            "Exploratory complete-case result. Hyperparameters use validation only; test is frozen. "
            "Mordor 'benign' means unflagged EventID-matched control, not independently verified benign."
        ),
        "requested_items": requested,
        "complete_case_items": len(complete_items),
        "split": {
            "calibration": len(split.calibration),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "calibration_selected_anchor": best_model,
        "calibration_models": calibration_report,
        "validation_selection": {
            "calibrated_single_threshold": {
                "model": best_model,
                "malicious_probability_threshold": selected_base_threshold,
                "metrics": base_threshold_validation.to_dict(),
            },
            "causal_parent_logit": {
                "configuration": low_capacity_causal,
                "metrics": low_capacity_causal_validation.to_dict(),
            },
            "graph_stacker": {
                "configuration": _candidate_dict(best_stacker, model_ids),
                "metrics": best_stacker_validation.to_dict(),
            },
            "causal_trace_message_passing": {
                "configuration": _candidate_dict(best_causal, model_ids),
                "metrics": best_causal_validation.to_dict(),
            },
            "bidirectional_crf_message_passing": {
                "configuration": _candidate_dict(best_bidirectional, model_ids),
                "metrics": best_bidirectional_validation.to_dict(),
            },
            "flash_first_trace_gate": {
                "configuration": _candidate_dict(best_gate, model_ids),
                "metrics": best_gate_validation.to_dict(),
            },
        },
        "test_methods": clean_methods,
        "diagnostic_oracle_not_deployable": {
            "any_cached_raw_label_correct": oracle_metrics.to_dict(),
            "unanimous_wrong_rows": unanimous_wrong,
            "model_disagreement_rows": disagreement,
            "model_disagreement_rate": disagreement / len(test_y),
        },
        "paired_cluster_bootstrap": comparisons,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    methods = report["test_methods"]
    selected = report["validation_selection"]
    oracle = report["diagnostic_oracle_not_deployable"]
    lines = [
        "# Mordor 缓存输出上的 TRACE-G-CaMVo 离线算法搜索",
        "",
        "> **零新增模型调用。** Calibration 训练、validation 选型、test 只验收一次。",
        "",
        f"- 完整共同样本：{report['complete_case_items']}/{report['requested_items']}",
        (
            f"- 切分：{report['split']['calibration']} calibration / "
            f"{report['split']['validation']} validation / {report['split']['test']} test"
        ),
        f"- 校准集选出的首选模型：`{report['calibration_selected_anchor']}`",
        "",
        "## Validation 冻结的算法",
        "",
        f"- 单模型概率阈值：`{json.dumps(selected['calibrated_single_threshold'], ensure_ascii=False)}`",
        f"- 低容量 causal parent-logit：`{json.dumps(selected['causal_parent_logit'], ensure_ascii=False)}`",
        f"- Graph stacker：`{json.dumps(selected['graph_stacker']['configuration'], ensure_ascii=False)}`",
        f"- Causal message passing：`{json.dumps(selected['causal_trace_message_passing']['configuration'], ensure_ascii=False)}`",
        f"- Bidirectional CRF message passing：`{json.dumps(selected['bidirectional_crf_message_passing']['configuration'], ensure_ascii=False)}`",
        f"- Flash-first graph gate：`{json.dumps(selected['flash_first_trace_gate']['configuration'], ensure_ascii=False)}`",
        "",
        "## Frozen test",
        "",
        "| Method | Macro-F1 | Accuracy | Malicious precision | Malicious recall | Cost ($) | Avg models |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        *(name for name in methods if name.startswith("single::")),
        "validation_calibrated_single",
        "calibrated_camvo",
        "selected_causal_parent_logit",
        "selected_graph_stacker",
        "selected_causal_trace_message_passing",
        "selected_bidirectional_crf_message_passing",
        "selected_flash_first_trace_gate",
    ]
    for name in order:
        row = methods[name]
        metrics = row["metrics"]
        malicious = metrics["per_class"]["malicious"]
        lines.append(
            f"| {name} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} | "
            f"{malicious['precision']:.4f} | {malicious['recall']:.4f} | "
            f"{row['total_cost_usd']:.4f} | {row['average_models']:.3f} |"
        )
    ceiling = oracle["any_cached_raw_label_correct"]
    lines.extend(
        [
            "",
            "## 输出矩阵的可利用空间（诊断上限，不是算法成绩）",
            "",
            (
                f"- 任一模型原始离散标签答对的选择 oracle：Macro-F1 **{ceiling['macro_f1']:.4f}**；"
                f"全模型一致答错 {oracle['unanimous_wrong_rows']} 条。"
            ),
            (
                f"- 模型发生分歧 {oracle['model_disagreement_rows']} 条 "
                f"({oracle['model_disagreement_rate']:.1%})；只有这些区域路由器才有明显选择空间。"
            ),
            "",
            "## 统计比较",
            "",
        ]
    )
    for name, row in report["paired_cluster_bootstrap"].items():
        lines.append(
            f"- `{name}`：ΔMacro-F1 {row['candidate_minus_reference']:+.4f}，"
            f"95% CI [{row['lower']:+.4f}, {row['upper']:+.4f}]，"
            f"P(候选更好)={row['probability_candidate_better']:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"> {report['claim_boundary']}",
            "",
            "若门控法优于单模型，它证明的是：图邻域冲突可以帮助决定何时升级；"
            "若 graph stacker 仍不增益，则说明当前边定义或标签粒度不足，而不是图思想被否定。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Legacy Adams run directory; unnecessary when --bundle-dir is used.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "data" / "derived" / "mordor_offline_bundle",
        help="Credential-free offline bundle exported from Adams.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()
    bundle_dir = args.bundle_dir if args.bundle_dir.exists() else None
    report = analyze(
        args.run_dir,
        bundle_dir=bundle_dir,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    output_dir = bundle_dir or args.run_dir
    assert output_dir is not None
    json_path = output_dir / "TRACE_ALGORITHM_SEARCH.json"
    markdown_path = output_dir / "TRACE_ALGORITHM_SEARCH.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
