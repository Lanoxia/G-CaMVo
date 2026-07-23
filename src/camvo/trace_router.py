"""TRACE-GCaMVo: causal graph priors with cost-aware sequential escalation.

The original CaMVo chooses a model subset before observing any response.  This
module implements a complementary SOC-oriented policy: query one model at a
time, update a categorical incident posterior, and stop as soon as the
remaining decision risk is small enough.  The next model is selected by
expected information gain per estimated dollar, with an online redundancy
penalty.

Graph evidence is deliberately conservative.  Only already-routed neighbors
are visible, and a relation contributes only after its empirical transition
matrix is both sufficiently observed and more informative than a uniform
transition.  This avoids treating every provenance edge as a same-label edge.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.exceptions import CheckpointError, InsufficientResponsesError, ModelInvocationError
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelResponse, RoutingResult


@dataclass(frozen=True, slots=True)
class TraceGraphCaMVoConfig:
    """Controls risk, calibration, graph gating, and sequential escalation."""

    risk_tolerance: float = 0.03
    critical_risk_tolerance: float = 0.01
    critical_metadata_key: str = "severity"
    critical_metadata_values: tuple[str, ...] = ("critical", "high")
    symmetric_reliability_prior: float = 0.68
    reliability_prior_strength: float = 6.0
    reliability_lower_z: float = 1.28
    min_reliability_observations: int = 8
    context_reliability_weight: float = 0.25
    confidence_bins: int = 10
    confidence_prior_strength: float = 4.0
    min_confidence_bin_observations: int = 5
    diversity_penalty: float = 0.45
    min_pair_observations: int = 5
    graph_strength: float = 0.8
    transition_prior: float = 1.0
    min_transition_observations: float = 6.0
    max_edge_weight: float = 2.0
    max_total_neighbor_weight: float = 8.0
    min_graph_informativeness: float = 0.05
    abstain_when_risk_unmet: bool = True
    use_feedback_metadata: bool = False
    feedback_metadata_key: str = "feedback_label"
    fallback_model_id: str | None = None
    fallback_after_models: int = 2
    fallback_trigger_risk: float = 0.20

    def __post_init__(self) -> None:
        probability_fields = {
            "risk_tolerance": self.risk_tolerance,
            "critical_risk_tolerance": self.critical_risk_tolerance,
            "symmetric_reliability_prior": self.symmetric_reliability_prior,
            "context_reliability_weight": self.context_reliability_weight,
            "diversity_penalty": self.diversity_penalty,
            "min_graph_informativeness": self.min_graph_informativeness,
            "fallback_trigger_risk": self.fallback_trigger_risk,
        }
        for name, value in probability_fields.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.risk_tolerance == 0 or self.critical_risk_tolerance == 0:
            raise ValueError("risk tolerances must be positive")
        if not 0 < self.symmetric_reliability_prior < 1:
            raise ValueError("symmetric_reliability_prior must be in (0, 1)")
        positive_fields = {
            "reliability_prior_strength": self.reliability_prior_strength,
            "reliability_lower_z": self.reliability_lower_z,
            "confidence_bins": self.confidence_bins,
            "confidence_prior_strength": self.confidence_prior_strength,
            "min_confidence_bin_observations": self.min_confidence_bin_observations,
            "min_reliability_observations": self.min_reliability_observations,
            "min_pair_observations": self.min_pair_observations,
            "transition_prior": self.transition_prior,
            "min_transition_observations": self.min_transition_observations,
            "max_edge_weight": self.max_edge_weight,
            "max_total_neighbor_weight": self.max_total_neighbor_weight,
            "fallback_after_models": self.fallback_after_models,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.critical_metadata_key.strip() or not self.feedback_metadata_key.strip():
            raise ValueError("metadata keys must not be empty")
        if not self.critical_metadata_values:
            raise ValueError("critical_metadata_values must not be empty")
        if self.fallback_model_id is not None and not self.fallback_model_id.strip():
            raise ValueError("fallback_model_id must be non-empty when configured")


@dataclass(slots=True)
class SoftBetaReliability:
    """Beta posterior that accepts confidence-weighted soft correctness."""

    alpha: float
    beta: float
    observations: int = 0

    @classmethod
    def with_prior(cls, mean: float, strength: float) -> SoftBetaReliability:
        return cls(mean * strength, (1.0 - mean) * strength)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1.0))

    def conservative(self, z_value: float, min_observations: int) -> float:
        if self.observations < min_observations:
            return self.mean
        return float(np.clip(self.mean - z_value * math.sqrt(self.variance), 0.0, 1.0))

    def update(self, correctness: float, *, weight: float = 1.0) -> None:
        if not math.isfinite(correctness) or not 0 <= correctness <= 1:
            raise ValueError("correctness must be finite and in [0, 1]")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        self.alpha += weight * correctness
        self.beta += weight * (1.0 - correctness)
        self.observations += 1

    def state_dict(self) -> dict[str, float | int]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "observations": self.observations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> SoftBetaReliability:
        value = cls(
            alpha=float(state["alpha"]),
            beta=float(state["beta"]),
            observations=int(state["observations"]),
        )
        if value.alpha <= 0 or value.beta <= 0 or value.observations < 0:
            raise ValueError("invalid reliability checkpoint")
        return value


@dataclass(slots=True)
class OnlineConfidenceCalibrator:
    """Histogram-Beta calibrator for model-reported confidence values."""

    bins: int
    prior_strength: float
    alpha: np.ndarray = field(init=False, repr=False)
    beta: np.ndarray = field(init=False, repr=False)
    observations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.alpha = np.full(self.bins, self.prior_strength / 2.0, dtype=float)
        self.beta = np.full(self.bins, self.prior_strength / 2.0, dtype=float)
        self.observations = np.zeros(self.bins, dtype=np.int64)

    def _index(self, confidence: float) -> int:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        return min(self.bins - 1, int(confidence * self.bins))

    def estimate(
        self,
        confidence: float,
        *,
        fallback: float,
        min_observations: int,
        z_value: float,
    ) -> float:
        index = self._index(confidence)
        count = int(self.observations[index])
        if count < min_observations:
            return fallback
        alpha = float(self.alpha[index])
        beta = float(self.beta[index])
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1.0))
        return float(np.clip(mean - z_value * math.sqrt(variance), 0.0, 1.0))

    def update(self, confidence: float, correctness: float, *, weight: float = 1.0) -> None:
        index = self._index(confidence)
        if not math.isfinite(correctness) or not 0 <= correctness <= 1:
            raise ValueError("correctness must be finite and in [0, 1]")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        self.alpha[index] += weight * correctness
        self.beta[index] += weight * (1.0 - correctness)
        self.observations[index] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "prior_strength": self.prior_strength,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "observations": self.observations.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["bins"]) != self.bins:
            raise ValueError("confidence calibrator bin mismatch")
        alpha = np.asarray(state["alpha"], dtype=float)
        beta = np.asarray(state["beta"], dtype=float)
        observations = np.asarray(state["observations"], dtype=np.int64)
        if alpha.shape != (self.bins,) or beta.shape != alpha.shape:
            raise ValueError("invalid confidence calibrator checkpoint shape")
        if observations.shape != alpha.shape:
            raise ValueError("invalid confidence calibrator observation shape")
        if np.any(alpha <= 0) or np.any(beta <= 0) or np.any(observations < 0):
            raise ValueError("invalid confidence calibrator checkpoint values")
        self.alpha = alpha
        self.beta = beta
        self.observations = observations


class PairwiseRedundancyTracker:
    """Track how often two models emit the same label when co-queried."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], list[int]] = {}

    @staticmethod
    def _key(left: str, right: str) -> tuple[str, str]:
        return tuple(sorted((left, right)))

    def update(self, responses: Mapping[str, str]) -> None:
        model_ids = sorted(responses)
        for left_index, left in enumerate(model_ids):
            for right in model_ids[left_index + 1 :]:
                same, total = self._counts.setdefault(self._key(left, right), [0, 0])
                self._counts[self._key(left, right)] = [
                    same + int(responses[left] == responses[right]),
                    total + 1,
                ]

    def redundancy(self, left: str, right: str, min_observations: int) -> float:
        same, total = self._counts.get(self._key(left, right), [0, 0])
        if total < min_observations:
            return 0.0
        return (same + 1.0) / (total + 2.0)

    def state_dict(self) -> dict[str, list[int]]:
        return {f"{left}\u0000{right}": value for (left, right), value in self._counts.items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        counts: dict[tuple[str, str], list[int]] = {}
        for key, raw in state.items():
            left, right = str(key).split("\u0000", 1)
            same, total = int(raw[0]), int(raw[1])
            if same < 0 or total < same:
                raise ValueError("invalid redundancy checkpoint")
            counts[self._key(left, right)] = [same, total]
        self._counts = counts


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("posterior must be a finite non-empty vector")
    values = np.clip(values, 1e-12, None)
    return values / values.sum()


def _entropy(probabilities: np.ndarray) -> float:
    values = _normalize(probabilities)
    return float(-np.dot(values, np.log(values)))


def posterior_after_vote(
    prior: np.ndarray,
    response_index: int,
    accuracy: float,
) -> np.ndarray:
    """Bayesian categorical update under a symmetric-error confusion model."""

    probabilities = _normalize(prior)
    labels = len(probabilities)
    if not 0 <= response_index < labels:
        raise ValueError("response_index is outside the label space")
    random_accuracy = 1.0 / labels
    accuracy = float(np.clip(accuracy, random_accuracy + 1e-6, 1.0 - 1e-6))
    error_probability = (1.0 - accuracy) / (labels - 1)
    likelihood = np.full(labels, error_probability, dtype=float)
    likelihood[response_index] = accuracy
    return _normalize(probabilities * likelihood)


def expected_information_gain(prior: np.ndarray, accuracy: float) -> float:
    """Expected entropy reduction of one model response."""

    probabilities = _normalize(prior)
    labels = len(probabilities)
    random_accuracy = 1.0 / labels
    accuracy = float(np.clip(accuracy, random_accuracy + 1e-6, 1.0 - 1e-6))
    error_probability = (1.0 - accuracy) / (labels - 1)
    expected_entropy = 0.0
    for response_index in range(labels):
        response_probability = (
            probabilities[response_index] * accuracy
            + (1.0 - probabilities[response_index]) * error_probability
        )
        expected_entropy += response_probability * _entropy(
            posterior_after_vote(probabilities, response_index, accuracy)
        )
    return max(0.0, _entropy(probabilities) - expected_entropy)


@dataclass(slots=True)
class _GraphNodeBelief:
    posterior: np.ndarray
    confidence: float


class CausalTransitionGraphMemory:
    """Learn relation-specific label transitions from previously routed nodes."""

    def __init__(
        self,
        labels: tuple[str, ...],
        neighborhood: Callable[[AnnotationItem], Mapping[str, Any]],
        config: TraceGraphCaMVoConfig,
    ) -> None:
        self.labels = labels
        self.neighborhood = neighborhood
        self.config = config
        self._history: dict[str, _GraphNodeBelief] = {}
        self._transitions: dict[str, np.ndarray] = {}

    @staticmethod
    def node_id(item: AnnotationItem) -> str:
        return str(
            item.metadata.get("graph_node_id", item.metadata.get("event_id", item.item_id))
        )

    def _edge(self, raw: Any) -> tuple[float, str]:
        if isinstance(raw, Mapping):
            weight = float(raw.get("weight", 0.0))
            relation = str(raw.get("relation", "default")).strip() or "default"
        else:
            weight = float(raw)
            relation = "default"
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("neighborhood returned an invalid edge weight")
        return min(weight, self.config.max_edge_weight), relation

    def observed_neighbors(self, item: AnnotationItem) -> list[tuple[str, float, str]]:
        candidates: list[tuple[str, float, str]] = []
        for node_id, raw in self.neighborhood(item).items():
            weight, relation = self._edge(raw)
            node_id = str(node_id)
            if weight > 0 and node_id in self._history:
                candidates.append((node_id, weight, relation))
        candidates.sort(key=lambda value: (-value[1], value[0], value[2]))
        accepted: list[tuple[str, float, str]] = []
        total = 0.0
        for node_id, weight, relation in candidates:
            remaining = self.config.max_total_neighbor_weight - total
            if remaining <= 0:
                break
            bounded = min(weight, remaining)
            accepted.append((node_id, bounded, relation))
            total += bounded
        return accepted

    def _matrix(self, relation: str) -> np.ndarray:
        if relation not in self._transitions:
            self._transitions[relation] = np.full(
                (len(self.labels), len(self.labels)),
                self.config.transition_prior,
                dtype=float,
            )
        return self._transitions[relation]

    def prior(self, item: AnnotationItem) -> tuple[np.ndarray, dict[str, Any]]:
        uniform = np.full(len(self.labels), 1.0 / len(self.labels), dtype=float)
        neighbors = self.observed_neighbors(item)
        log_prior = np.log(uniform)
        used = 0
        evidence_weight = 0.0
        for node_id, edge_weight, relation in neighbors:
            matrix = self._matrix(relation)
            learned_mass = float(matrix.sum() - matrix.size * self.config.transition_prior)
            if learned_mass < self.config.min_transition_observations:
                continue
            transition = matrix / matrix.sum(axis=1, keepdims=True)
            neighbor = self._history[node_id]
            predicted = _normalize(neighbor.posterior @ transition)
            informativeness = 1.0 - _entropy(predicted) / math.log(len(self.labels))
            if informativeness < self.config.min_graph_informativeness:
                continue
            maturity = learned_mass / (
                learned_mass + self.config.min_transition_observations
            )
            gate = edge_weight * neighbor.confidence * informativeness * maturity
            log_prior += self.config.graph_strength * gate * np.log(predicted)
            evidence_weight += gate
            used += 1
        prior = _normalize(np.exp(log_prior - float(np.max(log_prior))))
        return prior, {
            "observed_neighbors": len(neighbors),
            "used_neighbors": used,
            "graph_evidence_weight": evidence_weight,
        }

    def update(self, item: AnnotationItem, posterior: np.ndarray) -> None:
        belief = _GraphNodeBelief(
            posterior=_normalize(posterior),
            confidence=float(np.max(posterior)),
        )
        for node_id, edge_weight, relation in self.observed_neighbors(item):
            neighbor = self._history[node_id]
            confidence_weight = min(neighbor.confidence, belief.confidence)
            self._matrix(relation)[:] += (
                edge_weight
                * confidence_weight
                * np.outer(neighbor.posterior, belief.posterior)
            )
        self._history[self.node_id(item)] = belief

    def state_dict(self) -> dict[str, Any]:
        return {
            "history": {
                node_id: {
                    "posterior": belief.posterior.tolist(),
                    "confidence": belief.confidence,
                }
                for node_id, belief in sorted(self._history.items())
            },
            "transitions": {
                relation: matrix.tolist()
                for relation, matrix in sorted(self._transitions.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        history: dict[str, _GraphNodeBelief] = {}
        for node_id, raw in state.get("history", {}).items():
            posterior = _normalize(np.asarray(raw["posterior"], dtype=float))
            if posterior.shape != (len(self.labels),):
                raise ValueError("graph checkpoint label dimension mismatch")
            confidence = float(raw["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("invalid graph checkpoint confidence")
            history[str(node_id)] = _GraphNodeBelief(posterior, confidence)
        transitions: dict[str, np.ndarray] = {}
        for relation, raw in state.get("transitions", {}).items():
            matrix = np.asarray(raw, dtype=float)
            if matrix.shape != (len(self.labels), len(self.labels)):
                raise ValueError("graph checkpoint transition shape mismatch")
            if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
                raise ValueError("invalid graph checkpoint transition values")
            transitions[str(relation)] = matrix
        self._history = history
        self._transitions = transitions


class TraceGraphCaMVoRouter(CaMVoRouter):
    """Sequential risk router with causal, quality-gated graph evidence."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        trace_config: TraceGraphCaMVoConfig,
        neighborhood: Callable[[AnnotationItem], Mapping[str, Any]],
    ) -> None:
        super().__init__(models, embedder, config)
        self.trace_config = trace_config
        self._reliability = {
            model_id: SoftBetaReliability.with_prior(
                trace_config.symmetric_reliability_prior,
                trace_config.reliability_prior_strength,
            )
            for model_id in self.models
        }
        self._confidence_calibrators = {
            model_id: OnlineConfidenceCalibrator(
                trace_config.confidence_bins,
                trace_config.confidence_prior_strength,
            )
            for model_id in self.models
        }
        self._redundancy = PairwiseRedundancyTracker()
        self._graph_by_labels: dict[tuple[str, ...], CausalTransitionGraphMemory] = {}
        self._neighborhood = neighborhood
        if trace_config.fallback_model_id not in {None, *self.models}:
            raise ValueError("fallback_model_id is not present in the model pool")

    def _graph(self, labels: tuple[str, ...]) -> CausalTransitionGraphMemory:
        if labels not in self._graph_by_labels:
            self._graph_by_labels[labels] = CausalTransitionGraphMemory(
                labels,
                self._neighborhood,
                self.trace_config,
            )
        return self._graph_by_labels[labels]

    def _risk_tolerance(self, item: AnnotationItem) -> float:
        severity = str(
            item.metadata.get(self.trace_config.critical_metadata_key, "")
        ).casefold()
        critical = {value.casefold() for value in self.trace_config.critical_metadata_values}
        return (
            self.trace_config.critical_risk_tolerance
            if severity in critical
            else self.trace_config.risk_tolerance
        )

    def _model_accuracy(self, model_id: str, context_score: float, labels: int) -> float:
        random_accuracy = 1.0 / labels
        global_estimate = self._reliability[model_id].conservative(
            self.trace_config.reliability_lower_z,
            self.trace_config.min_reliability_observations,
        )
        context_estimate = float(np.clip(context_score, random_accuracy, 1.0))
        weight = self.trace_config.context_reliability_weight
        estimate = (1.0 - weight) * global_estimate + weight * context_estimate
        return float(np.clip(estimate, random_accuracy + 1e-6, 1.0 - 1e-6))

    @staticmethod
    def _reported_confidence(response: ModelResponse) -> float | None:
        if not isinstance(response.raw, Mapping):
            return None
        try:
            value = float(response.raw["confidence"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value) or not 0 <= value <= 1:
            return None
        return value

    def _effective_accuracy(
        self,
        model_id: str,
        response: ModelResponse,
        fallback: float,
        labels: int,
    ) -> tuple[float, float | None]:
        reported = self._reported_confidence(response)
        if reported is None:
            return fallback, None
        calibrated = self._confidence_calibrators[model_id].estimate(
            reported,
            fallback=fallback,
            min_observations=self.trace_config.min_confidence_bin_observations,
            z_value=self.trace_config.reliability_lower_z,
        )
        random_accuracy = 1.0 / labels
        return float(np.clip(calibrated, random_accuracy + 1e-6, 1.0 - 1e-6)), reported

    def _next_model(
        self,
        posterior: np.ndarray,
        remaining: set[str],
        selected: list[str],
        accuracies: Mapping[str, float],
        costs: Mapping[str, float],
    ) -> tuple[str, float, float]:
        fallback = self.trace_config.fallback_model_id
        unresolved_risk = 1.0 - float(np.max(posterior))
        if (
            fallback is not None
            and fallback in remaining
            and len(selected) >= self.trace_config.fallback_after_models
            and unresolved_risk > self.trace_config.fallback_trigger_risk
        ):
            information_gain = expected_information_gain(posterior, accuracies[fallback])
            diversity = self._diversity_factor(fallback, selected)
            utility = information_gain * diversity / max(costs[fallback], 1e-12)
            return fallback, utility, information_gain
        ranked: list[tuple[float, float, float, str]] = []
        for model_id in remaining:
            information_gain = expected_information_gain(posterior, accuracies[model_id])
            diversity = self._diversity_factor(model_id, selected)
            utility = information_gain * diversity / max(costs[model_id], 1e-12)
            ranked.append((utility, information_gain, -costs[model_id], model_id))
        utility, information_gain, _negative_cost, model_id = max(ranked)
        return model_id, utility, information_gain

    def _diversity_factor(self, model_id: str, selected: list[str]) -> float:
        max_redundancy = max(
            (
                self._redundancy.redundancy(
                    model_id,
                    previous,
                    self.trace_config.min_pair_observations,
                )
                for previous in selected
            ),
            default=0.0,
        )
        return max(
            0.05,
            1.0 - self.trace_config.diversity_penalty * max_redundancy,
        )

    def _feedback_target(
        self,
        item: AnnotationItem,
        posterior: np.ndarray,
    ) -> tuple[str, bool]:
        if self.trace_config.use_feedback_metadata:
            feedback = item.metadata.get(self.trace_config.feedback_metadata_key)
            if feedback in item.labels:
                return str(feedback), True
        return item.labels[int(np.argmax(posterior))], False

    def route(self, item: AnnotationItem) -> RoutingResult:
        next_round = self.round_index + 1
        context = self._context(item)
        scores, bandit_scores = self._score_models(item, context, next_round)
        warmup = next_round <= self.config.warmup_rounds
        graph = self._graph(item.labels)
        posterior, graph_audit = graph.prior(item)
        risk_tolerance = self._risk_tolerance(item)
        model_accuracies = {
            model_id: self._model_accuracy(
                model_id,
                score.smoothed_lower_bound,
                len(item.labels),
            )
            for model_id, score in scores.items()
        }
        costs = {model_id: score.estimated_cost for model_id, score in scores.items()}

        remaining = set(self.models)
        selected: list[str] = []
        normalized_responses: dict[str, str] = {}
        raw_responses: dict[str, ModelResponse] = {}
        reported_confidences: dict[str, float] = {}
        errors: dict[str, str] = {}
        routing_trace: list[dict[str, Any]] = []
        while remaining:
            model_id, utility, information_gain = self._next_model(
                posterior,
                remaining,
                selected,
                model_accuracies,
                costs,
            )
            remaining.remove(model_id)
            try:
                response = self.models[model_id].predict(item)
                self._validate_response(model_id, response, item)
            except Exception as exc:  # provider boundaries must be isolated
                errors[model_id] = f"{type(exc).__name__}: {exc}"
                if not self.config.allow_partial_responses:
                    raise ModelInvocationError(errors[model_id]) from exc
                continue
            selected.append(model_id)
            raw_responses[model_id] = response
            normalized_responses[model_id] = response.label
            effective_accuracy, reported = self._effective_accuracy(
                model_id,
                response,
                model_accuracies[model_id],
                len(item.labels),
            )
            if reported is not None:
                reported_confidences[model_id] = reported
            diversity = self._diversity_factor(model_id, selected[:-1])
            random_accuracy = 1.0 / len(item.labels)
            effective_accuracy = random_accuracy + diversity * (
                effective_accuracy - random_accuracy
            )
            posterior = posterior_after_vote(
                posterior,
                item.labels.index(response.label),
                effective_accuracy,
            )
            decision_confidence = float(np.max(posterior))
            routing_trace.append(
                {
                    "step": len(selected),
                    "model_id": model_id,
                    "estimated_accuracy": model_accuracies[model_id],
                    "effective_accuracy": effective_accuracy,
                    "diversity_factor": diversity,
                    "expected_information_gain": information_gain,
                    "information_gain_per_cost": utility,
                    "reported_confidence": reported,
                    "decision_confidence": decision_confidence,
                    "decision_risk": 1.0 - decision_confidence,
                }
            )
            enough_models = len(selected) >= self.config.min_models
            if not warmup and enough_models and 1.0 - decision_confidence <= risk_tolerance:
                break

        if len(selected) < self.config.min_models:
            raise InsufficientResponsesError(
                f"received {len(selected)} valid responses; required {self.config.min_models}"
            )
        decision_index = int(np.argmax(posterior))
        label = item.labels[decision_index]
        decision_confidence = float(posterior[decision_index])
        decision_risk = 1.0 - decision_confidence
        abstained = self.trace_config.abstain_when_risk_unmet and decision_risk > risk_tolerance

        feedback_label, audited = self._feedback_target(item, posterior)
        feedback_weight = 1.0 if audited else decision_confidence
        for model_id in selected:
            matched = normalized_responses[model_id] == feedback_label
            state = self._states[model_id]
            state.arm.update(context, float(matched))
            state.calibrator.update(bandit_scores[model_id].prediction, matched)
            state.observations += 1
            state.agreements += int(matched)
            self._reliability[model_id].update(float(matched), weight=feedback_weight)
            if model_id in reported_confidences:
                self._confidence_calibrators[model_id].update(
                    reported_confidences[model_id],
                    float(matched),
                    weight=feedback_weight,
                )
        self._redundancy.update(normalized_responses)
        graph.update(item, posterior)

        actual_cost = sum(
            self.models[model_id].pricing.cost(response.input_tokens, response.output_tokens)
            for model_id, response in raw_responses.items()
        )
        self.round_index = next_round
        return RoutingResult(
            item_id=item.item_id,
            label=label,
            selected_models=tuple(selected),
            subset_confidence=decision_confidence,
            estimated_cost=sum(costs[model_id] for model_id in selected),
            actual_cost=actual_cost,
            responses=normalized_responses,
            scores=scores,
            errors=errors,
            warmup=warmup,
            posterior={
                label_name: float(posterior[index])
                for index, label_name in enumerate(item.labels)
            },
            abstained=abstained,
            decision_risk=decision_risk,
            risk_tolerance=risk_tolerance,
            graph_evidence_weight=float(graph_audit["graph_evidence_weight"]),
            routing_trace=tuple(routing_trace),
        )

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: Mapping[str, ModelResponse],
        gold_label: str,
    ) -> None:
        """Warm-start every online component from an audited calibration row.

        This method never invokes a provider.  A formal experiment first
        collects a complete cached response matrix, then supplies calibration
        labels here so reliability, confidence, correlation, contextual arms,
        and causal transition statistics all start from the same evidence.
        """

        if gold_label not in item.labels:
            raise ValueError("gold_label is outside the item's label space")
        if set(responses) != set(self.models):
            raise ValueError("complete feedback must contain the entire model pool")
        super().observe_complete_feedback(item, dict(responses), gold_label)
        labels: dict[str, str] = {}
        for model_id in sorted(self.models):
            response = responses[model_id]
            labels[model_id] = response.label
            matched = response.label == gold_label
            self._reliability[model_id].update(float(matched))
            reported = self._reported_confidence(response)
            if reported is not None:
                self._confidence_calibrators[model_id].update(reported, float(matched))
        self._redundancy.update(labels)
        posterior = np.zeros(len(item.labels), dtype=float)
        posterior[item.labels.index(gold_label)] = 1.0
        self._graph(item.labels).update(item, posterior)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["trace_config"] = asdict(self.trace_config)
        state["trace_reliability"] = {
            model_id: tracker.state_dict()
            for model_id, tracker in sorted(self._reliability.items())
        }
        state["trace_confidence_calibrators"] = {
            model_id: calibrator.state_dict()
            for model_id, calibrator in sorted(self._confidence_calibrators.items())
        }
        state["trace_redundancy"] = self._redundancy.state_dict()
        state["trace_graphs"] = {
            "\u0000".join(labels): graph.state_dict()
            for labels, graph in self._graph_by_labels.items()
        }
        return state

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            expected_config = asdict(self.trace_config)
            checkpoint_config = dict(state["trace_config"])
            checkpoint_config["critical_metadata_values"] = tuple(
                checkpoint_config["critical_metadata_values"]
            )
            if checkpoint_config != expected_config:
                raise CheckpointError("checkpoint TRACE configuration does not match")
            reliability = {
                model_id: SoftBetaReliability.from_state_dict(raw)
                for model_id, raw in state["trace_reliability"].items()
            }
            if set(reliability) != set(self.models):
                raise CheckpointError("checkpoint TRACE model pool does not match")
            calibrator_states = state["trace_confidence_calibrators"]
            if set(calibrator_states) != set(self.models):
                raise CheckpointError("checkpoint confidence calibrator pool does not match")
            redundancy = PairwiseRedundancyTracker()
            redundancy.load_state_dict(state["trace_redundancy"])
            graphs: dict[tuple[str, ...], CausalTransitionGraphMemory] = {}
            for encoded_labels, raw in state.get("trace_graphs", {}).items():
                labels = tuple(encoded_labels.split("\u0000"))
                graph = CausalTransitionGraphMemory(
                    labels,
                    self._neighborhood,
                    self.trace_config,
                )
                graph.load_state_dict(raw)
                graphs[labels] = graph
        except CheckpointError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid TRACE checkpoint: {exc}") from exc
        super().load_checkpoint(path)
        self._reliability = reliability
        for model_id, raw in calibrator_states.items():
            self._confidence_calibrators[model_id].load_state_dict(raw)
        self._redundancy = redundancy
        self._graph_by_labels = graphs
