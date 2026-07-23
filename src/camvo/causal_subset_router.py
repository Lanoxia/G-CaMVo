"""Causal TRACE-CaMVo while preserving the original CaMVo round structure.

The router inherits CaMVo's complete loop unchanged:

1. score every model with a contextual bandit;
2. select one cost-minimal *subset* before observing any current response;
3. query that subset and perform weighted majority voting;
4. update selected arms from agreement with the round consensus.

Its interventions are causal, relation-typed regularization of each model's
lower correctness bound and a CRF-style prior on the weighted vote.  For each
model and provenance relation it learns a two-state transition model
``P(child agreement | parent agreement, relation)``.  Only already processed
parents can contribute, so no child/future leakage is possible.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from camvo.algorithm.oracle import OracleSelection
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.exceptions import CheckpointError
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class CausalSubsetGraphConfig:
    """Controls conservative propagation of model reliability over the graph."""

    regularization: float = 0.5
    transition_prior: float = 1.0
    min_transition_observations: float = 6.0
    transition_lower_z: float = 1.28
    min_informativeness: float = 0.02
    max_edge_weight: float = 2.0
    max_total_parent_weight: float = 8.0
    online_transition_weight: float = 0.25
    label_vote_regularization: float = 0.0
    joint_graph_oracle: bool = False
    asymmetric_vote_calibration: bool = False
    confusion_prior: float = 1.0
    confusion_temperature: float = 1.0
    asymmetric_vote_blend: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "transition_prior",
            "transition_lower_z",
            "max_edge_weight",
            "max_total_parent_weight",
            "confusion_prior",
            "confusion_temperature",
        ):
            if float(values[name]) == 0:
                raise ValueError(f"{name} must be positive")
        if self.min_informativeness > 1:
            raise ValueError("min_informativeness must not exceed one")
        if not 0 <= self.asymmetric_vote_blend <= 1:
            raise ValueError("asymmetric_vote_blend must be in [0, 1]")


class StaticCausalTypedNeighborhood:
    """Read-only child -> past parents index with semantic edge types."""

    def __init__(self, parents: Mapping[str, Mapping[str, Any]]) -> None:
        self._parents = {
            str(child): {str(parent): raw for parent, raw in rows.items()}
            for child, rows in parents.items()
        }

    def __call__(self, item: AnnotationItem) -> Mapping[str, Any]:
        node_id = str(
            item.metadata.get("graph_node_id", item.metadata.get("event_id", item.item_id))
        )
        return self._parents.get(node_id, {})


class CausalSubsetGraphCaMVoRouter(CaMVoRouter):
    """CaMVo subset voting with past-only typed provenance regularization."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        graph_config: CausalSubsetGraphConfig,
        neighborhood: Callable[[AnnotationItem], Mapping[str, Any]],
    ) -> None:
        super().__init__(models, embedder, config)
        self.graph_config = graph_config
        self.neighborhood = neighborhood
        self._reward_history: dict[str, dict[str, float]] = {
            model_id: {} for model_id in self.models
        }
        self._transitions: dict[str, dict[str, np.ndarray]] = {
            model_id: {} for model_id in self.models
        }
        self._transition_observations: dict[str, dict[str, float]] = {
            model_id: {} for model_id in self.models
        }
        self._label_history: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self._label_transitions: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        self._label_transition_observations: dict[tuple[str, ...], dict[str, float]] = {}
        self._class_counts: dict[tuple[str, ...], np.ndarray] = {}
        self._structural_label_counts: dict[
            tuple[str, ...], dict[str, np.ndarray]
        ] = {}
        self._structural_observations: dict[tuple[str, ...], dict[str, float]] = {}
        self._confusion_counts: dict[
            tuple[str, ...], dict[str, np.ndarray]
        ] = {}
        self._last_graph_prior: np.ndarray | None = None
        self._last_graph_sources = 0
        self._last_labels: tuple[str, ...] | None = None

    @staticmethod
    def _node_id(item: AnnotationItem) -> str:
        return str(
            item.metadata.get("graph_node_id", item.metadata.get("event_id", item.item_id))
        )

    @staticmethod
    def _decode_edge(raw: Any) -> tuple[float, tuple[str, ...]]:
        if isinstance(raw, Mapping):
            weight = float(raw.get("weight", 0.0))
            raw_relations = raw.get("relations", raw.get("relation", "default"))
            if isinstance(raw_relations, str):
                relations = (raw_relations.strip() or "default",)
            else:
                relations = tuple(
                    sorted({str(value).strip() or "default" for value in raw_relations})
                )
        else:
            weight = float(raw)
            relations = ("default",)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("neighborhood returned an invalid edge weight")
        return weight, relations

    def _bounded_parents(self, item: AnnotationItem) -> list[tuple[str, float, tuple[str, ...]]]:
        candidates: list[tuple[str, float, tuple[str, ...]]] = []
        for parent_id, raw in self.neighborhood(item).items():
            weight, relations = self._decode_edge(raw)
            if weight > 0:
                candidates.append(
                    (
                        str(parent_id),
                        min(weight, self.graph_config.max_edge_weight),
                        relations,
                    )
                )
        candidates.sort(key=lambda row: (-row[1], row[0], row[2]))
        accepted: list[tuple[str, float, tuple[str, ...]]] = []
        total = 0.0
        for parent_id, weight, relations in candidates:
            remaining = self.graph_config.max_total_parent_weight - total
            if remaining <= 0:
                break
            bounded = min(weight, remaining)
            accepted.append((parent_id, bounded, relations))
            total += bounded
        return accepted

    def _matrix(self, model_id: str, relation: str) -> np.ndarray:
        matrices = self._transitions[model_id]
        if relation not in matrices:
            matrices[relation] = np.full(
                (2, 2), self.graph_config.transition_prior, dtype=float
            )
            self._transition_observations[model_id][relation] = 0.0
        return matrices[relation]

    def _label_matrix(self, labels: tuple[str, ...], relation: str) -> np.ndarray:
        matrices = self._label_transitions.setdefault(labels, {})
        observations = self._label_transition_observations.setdefault(labels, {})
        if relation not in matrices:
            matrices[relation] = np.full(
                (len(labels), len(labels)),
                self.graph_config.transition_prior,
                dtype=float,
            )
            observations[relation] = 0.0
        return matrices[relation]

    def _graph_label_prior(self, item: AnnotationItem) -> tuple[np.ndarray, int]:
        labels = item.labels
        counts = self._class_counts.get(labels)
        if counts is None:
            base = np.full(len(labels), 1.0 / len(labels), dtype=float)
        else:
            base = counts / counts.sum()
        log_base = np.log(np.clip(base, 1e-12, 1.0))
        evidence = np.zeros(len(labels), dtype=float)
        used = 0
        structural_counts = self._structural_label_counts.get(labels, {})
        structural_observations = self._structural_observations.get(labels, {})
        for feature, weight in self._structural_features(item).items():
            counts_for_feature = structural_counts.get(feature)
            observations = structural_observations.get(feature, 0.0)
            if (
                counts_for_feature is None
                or observations < self.graph_config.min_transition_observations
            ):
                continue
            predicted = counts_for_feature / counts_for_feature.sum()
            maturity = observations / (
                observations + self.graph_config.min_transition_observations
            )
            message = maturity * predicted + (1.0 - maturity) * base
            evidence += min(weight, self.graph_config.max_edge_weight) * (
                np.log(np.clip(message, 1e-12, 1.0)) - log_base
            )
            used += 1
        for parent_id, edge_weight, relations in self._bounded_parents(item):
            raw_parent = self._label_history.get(parent_id)
            if raw_parent is None or raw_parent[0] != labels:
                continue
            parent = raw_parent[1]
            relation_weight = edge_weight / len(relations)
            for relation in relations:
                matrix = self._label_transitions.get(labels, {}).get(relation)
                observations = self._label_transition_observations.get(labels, {}).get(
                    relation, 0.0
                )
                if matrix is None or observations < self.graph_config.min_transition_observations:
                    continue
                transition = matrix / matrix.sum(axis=1, keepdims=True)
                predicted = parent @ transition
                maturity = observations / (
                    observations + self.graph_config.min_transition_observations
                )
                message = maturity * predicted + (1.0 - maturity) * base
                evidence += relation_weight * (
                    np.log(np.clip(message, 1e-12, 1.0)) - log_base
                )
                used += 1
        if not used:
            return base, 0
        logits = log_base + evidence
        probabilities = np.exp(logits - float(np.max(logits)))
        return probabilities / probabilities.sum(), used

    def _structural_features(self, item: AnnotationItem) -> dict[str, float]:
        """Past-visible provenance features; no model name or future child is used."""

        parents = self._bounded_parents(item)
        features: defaultdict[str, float] = defaultdict(float)
        for _parent_id, weight, relations in parents:
            for relation in relations:
                features[f"relation:{relation}"] += weight / len(relations)
        degree_bucket = min(3, len(parents))
        features[f"indegree:{degree_bucket}"] = 1.0
        total_weight = sum(weight for _parent_id, weight, _relations in parents)
        weight_bucket = min(3, int(total_weight))
        features[f"parent_weight:{weight_bucket}"] = 1.0
        return dict(features)

    def _transition_lower_bound(
        self,
        model_id: str,
        relation: str,
        parent_reward: float,
    ) -> tuple[float, float] | None:
        observations = self._transition_observations[model_id].get(relation, 0.0)
        if observations < self.graph_config.min_transition_observations:
            return None
        matrix = self._matrix(model_id, relation)
        lower: list[float] = []
        for parent_state in (0, 1):
            alpha = float(matrix[parent_state, 1])
            beta = float(matrix[parent_state, 0])
            mean = alpha / (alpha + beta)
            variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
            lower.append(
                min(
                    1.0,
                    max(0.0, mean - self.graph_config.transition_lower_z * math.sqrt(variance)),
                )
            )
        prediction = (1.0 - parent_reward) * lower[0] + parent_reward * lower[1]
        informativeness = abs(prediction - 0.5) * 2.0
        if informativeness < self.graph_config.min_informativeness:
            return None
        maturity = observations / (
            observations + self.graph_config.min_transition_observations
        )
        return prediction, maturity * informativeness

    def _score_models(
        self,
        item: AnnotationItem,
        context: np.ndarray,
        round_index: int,
    ) -> tuple[dict[str, ModelScore], dict[str, Any]]:
        scores, bandit_scores = super()._score_models(item, context, round_index)
        self._last_labels = item.labels
        self._last_graph_prior, self._last_graph_sources = self._graph_label_prior(item)
        parents = self._bounded_parents(item)
        result: dict[str, ModelScore] = {}
        for model_id, score in scores.items():
            numerator = score.smoothed_lower_bound
            denominator = 1.0
            used_parents: set[str] = set()
            used_weight = 0.0
            history = self._reward_history[model_id]
            for parent_id, edge_weight, relations in parents:
                parent_reward = history.get(parent_id)
                if parent_reward is None:
                    continue
                relation_weight = edge_weight / len(relations)
                for relation in relations:
                    estimate = self._transition_lower_bound(
                        model_id, relation, parent_reward
                    )
                    if estimate is None:
                        continue
                    predicted_correctness, gate = estimate
                    weight = self.graph_config.regularization * relation_weight * gate
                    if weight <= 0:
                        continue
                    numerator += weight * predicted_correctness
                    denominator += weight
                    used_weight += weight
                    used_parents.add(parent_id)
            regularized = min(1.0, max(0.0, numerator / denominator))
            result[model_id] = replace(
                score,
                graph_regularized_lower_bound=regularized,
                graph_neighbor_count=len(used_parents),
                graph_neighbor_weight=used_weight,
            )
        return result, bandit_scores

    def _graph_vote_weight(self, prior: np.ndarray, total_model_weight: float) -> float:
        if self.graph_config.label_vote_regularization <= 0:
            return 0.0
        entropy = -float(np.sum(prior * np.log(np.clip(prior, 1e-12, 1.0))))
        confidence = 1.0 - entropy / math.log(len(prior))
        return (
            self.graph_config.label_vote_regularization
            * confidence
            * total_model_weight
        )

    def _confusion(self, labels: tuple[str, ...], model_id: str) -> np.ndarray:
        by_model = self._confusion_counts.setdefault(labels, {})
        if model_id not in by_model:
            by_model[model_id] = np.full(
                (len(labels), len(labels)),
                self.graph_config.confusion_prior,
                dtype=float,
            )
        matrix = by_model[model_id]
        return matrix / matrix.sum(axis=1, keepdims=True)

    def _regularized_vote_label(
        self,
        labels: tuple[str, ...],
        responses: Mapping[str, str],
        scores: Mapping[str, ModelScore],
        subset: Sequence[str],
        prior: np.ndarray,
        used_graph_sources: int,
    ) -> str:
        if self.graph_config.asymmetric_vote_calibration:
            graph_power = (
                self.graph_config.label_vote_regularization
                if used_graph_sources
                else 0.0
            )
            logits = graph_power * np.log(np.clip(prior, 1e-12, 1.0))
            ordinary = np.zeros(len(labels), dtype=float)
            total_ordinary_weight = 0.0
            mean_weight = max(
                1e-12,
                float(np.mean([scores[model_id].vote_weight for model_id in subset])),
            )
            for model_id in subset:
                vote_index = labels.index(responses[model_id])
                relative_weight = scores[model_id].vote_weight / mean_weight
                ordinary[vote_index] += scores[model_id].vote_weight
                total_ordinary_weight += scores[model_id].vote_weight
                likelihood = self._confusion(labels, model_id)[:, vote_index]
                logits += (
                    self.graph_config.confusion_temperature
                    * relative_weight
                    * np.log(np.clip(likelihood, 1e-12, 1.0))
                )
            calibrated = np.exp(logits - float(np.max(logits)))
            calibrated /= calibrated.sum()
            ordinary /= max(total_ordinary_weight, 1e-12)
            blend = self.graph_config.asymmetric_vote_blend
            combined = (1.0 - blend) * ordinary + blend * calibrated
            maximum = float(np.max(combined))
            return next(
                label
                for index, label in enumerate(labels)
                if math.isclose(float(combined[index]), maximum)
            )

        totals = {label: 0.0 for label in labels}
        total_model_weight = 0.0
        for model_id in subset:
            weight = scores[model_id].vote_weight
            totals[responses[model_id]] += weight
            total_model_weight += weight
        if used_graph_sources and self.graph_config.label_vote_regularization > 0:
            graph_weight = self._graph_vote_weight(prior, total_model_weight)
            for index, label in enumerate(labels):
                totals[label] += graph_weight * float(prior[index])
        maximum = max(totals.values())
        return next(label for label in labels if math.isclose(totals[label], maximum))

    def _joint_subset_confidence(
        self,
        subset: Sequence[str],
        scores: Mapping[str, ModelScore],
        prior: np.ndarray,
    ) -> float:
        """Exact binary confidence of graph-regularized weighted voting."""

        if self.graph_config.asymmetric_vote_calibration:
            label_space = self._last_labels
            if label_space is not None and len(label_space) ** len(subset) <= 65_536:
                confidence = 0.0
                for truth_index, truth_probability in enumerate(prior):
                    matrices = {
                        model_id: self._confusion(label_space, model_id)
                        for model_id in subset
                    }
                    for vote_indices in itertools.product(
                        range(len(label_space)), repeat=len(subset)
                    ):
                        probability = float(truth_probability)
                        responses = {}
                        for model_id, vote_index in zip(
                            subset, vote_indices, strict=True
                        ):
                            probability *= float(matrices[model_id][truth_index, vote_index])
                            responses[model_id] = label_space[vote_index]
                        winner = self._regularized_vote_label(
                            label_space,
                            responses,
                            scores,
                            subset,
                            prior,
                            self._last_graph_sources,
                        )
                        confidence += probability * float(
                            winner == label_space[truth_index]
                        )
                return min(1.0, max(0.0, confidence))

        if len(prior) != 2:
            return self._confidence_function(
                [self._selection_lower_bound(scores[model_id]) for model_id in subset],
                [scores[model_id].vote_weight for model_id in subset],
            )
        weights = {model_id: scores[model_id].vote_weight for model_id in subset}
        graph_weight = self._graph_vote_weight(prior, sum(weights.values()))
        confidence = 0.0
        for truth in (0, 1):
            for correctness in itertools.product((0, 1), repeat=len(subset)):
                probability = float(prior[truth])
                totals = [graph_weight * float(prior[0]), graph_weight * float(prior[1])]
                for model_id, correct in zip(subset, correctness, strict=True):
                    accuracy = self._selection_lower_bound(scores[model_id])
                    probability *= accuracy if correct else 1.0 - accuracy
                    vote = truth if correct else 1 - truth
                    totals[vote] += weights[model_id]
                winner = 0 if totals[0] >= totals[1] else 1
                if winner == truth:
                    confidence += probability
        return min(1.0, max(0.0, confidence))

    def _selection(
        self,
        scores: dict[str, ModelScore],
        warmup: bool,
    ) -> OracleSelection:
        prior = self._last_graph_prior
        if (
            warmup
            or not self.graph_config.joint_graph_oracle
            or prior is None
            or self._last_graph_sources == 0
            or self.graph_config.label_vote_regularization == 0
        ):
            return super()._selection(scores, warmup)
        model_ids = tuple(sorted(scores))
        feasible: list[OracleSelection] = []
        for size in range(self.config.min_models, len(model_ids) + 1):
            for subset in itertools.combinations(model_ids, size):
                confidence = self._joint_subset_confidence(subset, scores, prior)
                cost = sum(scores[model_id].estimated_cost for model_id in subset)
                if confidence >= self.config.confidence_threshold:
                    feasible.append(OracleSelection(subset, confidence, cost, True))
        if feasible:
            return min(
                feasible,
                key=lambda selection: (
                    selection.cost,
                    len(selection.model_ids),
                    selection.model_ids,
                ),
            )
        confidence = self._joint_subset_confidence(model_ids, scores, prior)
        return OracleSelection(
            model_ids,
            confidence,
            sum(scores[model_id].estimated_cost for model_id in model_ids),
            False,
        )

    def _selection_lower_bound(self, score: ModelScore) -> float:
        return (
            score.smoothed_lower_bound
            if score.graph_regularized_lower_bound is None
            else score.graph_regularized_lower_bound
        )

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        """CaMVo weighted vote plus a causal CRF-style graph prior term."""

        prior = self._last_graph_prior
        used = self._last_graph_sources
        if prior is None:
            prior, used = self._graph_label_prior(item)
        self._last_graph_prior = prior.copy()
        self._last_graph_sources = used
        return self._regularized_vote_label(
            item.labels,
            responses,
            scores,
            successful_ids,
            prior,
            used,
        )

    def _record_rewards(
        self,
        item: AnnotationItem,
        rewards: Mapping[str, float],
        *,
        transition_weight: float,
    ) -> None:
        node_id = self._node_id(item)
        parents = self._bounded_parents(item)
        for model_id, reward in rewards.items():
            value = min(1.0, max(0.0, float(reward)))
            history = self._reward_history[model_id]
            for parent_id, edge_weight, relations in parents:
                parent_reward = history.get(parent_id)
                if parent_reward is None:
                    continue
                parent_distribution = np.asarray(
                    [1.0 - parent_reward, parent_reward], dtype=float
                )
                child_distribution = np.asarray([1.0 - value, value], dtype=float)
                share = transition_weight * edge_weight / len(relations)
                for relation in relations:
                    self._matrix(model_id, relation)[:] += share * np.outer(
                        parent_distribution, child_distribution
                    )
                    self._transition_observations[model_id][relation] += share
            history[node_id] = value

    def _record_label(
        self,
        item: AnnotationItem,
        label: str,
        *,
        transition_weight: float,
        audited: bool,
    ) -> None:
        labels = item.labels
        label_index = labels.index(label)
        posterior = np.zeros(len(labels), dtype=float)
        posterior[label_index] = 1.0
        if audited:
            counts = self._class_counts.setdefault(labels, np.ones(len(labels), dtype=float))
            counts[label_index] += 1.0
            structural_counts = self._structural_label_counts.setdefault(labels, {})
            structural_observations = self._structural_observations.setdefault(labels, {})
            for feature, weight in self._structural_features(item).items():
                if feature not in structural_counts:
                    structural_counts[feature] = np.ones(len(labels), dtype=float)
                    structural_observations[feature] = 0.0
                structural_counts[feature][label_index] += weight
                structural_observations[feature] += weight
        for parent_id, edge_weight, relations in self._bounded_parents(item):
            raw_parent = self._label_history.get(parent_id)
            if raw_parent is None or raw_parent[0] != labels:
                continue
            parent = raw_parent[1]
            share = transition_weight * edge_weight / len(relations)
            for relation in relations:
                self._label_matrix(labels, relation)[:] += share * np.outer(
                    parent, posterior
                )
                observations = self._label_transition_observations[labels]
                observations[relation] += share
        self._label_history[self._node_id(item)] = (labels, posterior)

    def route(self, item: AnnotationItem) -> RoutingResult:
        """Run one unmodified CaMVo subset-vote round, then update graph memory."""

        result = super().route(item)
        rewards = {
            model_id: float(label == result.label)
            for model_id, label in result.responses.items()
        }
        self._record_rewards(
            item,
            rewards,
            transition_weight=self.graph_config.online_transition_weight,
        )
        self._record_label(
            item,
            result.label,
            transition_weight=self.graph_config.online_transition_weight,
            audited=False,
        )
        graph_prior = self._last_graph_prior
        applied_graph_sources = (
            self._last_graph_sources
            if self.graph_config.label_vote_regularization > 0
            else int(
                any(score.graph_neighbor_weight > 0 for score in result.scores.values())
            )
        )
        return replace(
            result,
            posterior=(
                {}
                if graph_prior is None
                else {
                    label: float(graph_prior[index])
                    for index, label in enumerate(item.labels)
                }
            ),
            graph_evidence_weight=float(applied_graph_sources),
        )

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        """Warm start CaMVo and typed transitions from an audited graph node."""

        super().observe_complete_feedback(item, responses, gold_label)
        gold_index = item.labels.index(gold_label)
        for model_id, response in responses.items():
            matrix = self._confusion_counts.setdefault(item.labels, {}).setdefault(
                model_id,
                np.full(
                    (len(item.labels), len(item.labels)),
                    self.graph_config.confusion_prior,
                    dtype=float,
                ),
            )
            matrix[gold_index, item.labels.index(response.label)] += 1.0
        self._record_rewards(
            item,
            {
                model_id: float(response.label == gold_label)
                for model_id, response in responses.items()
            },
            transition_weight=1.0,
        )
        self._record_label(
            item,
            gold_label,
            transition_weight=1.0,
            audited=True,
        )

    def reset_graph_history(self) -> None:
        """Remove node beliefs between splits while retaining learned transitions."""

        self._reward_history = {model_id: {} for model_id in self.models}
        self._label_history = {}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["causal_subset_graph_config"] = asdict(self.graph_config)
        state["causal_reward_history"] = {
            model_id: dict(sorted(history.items()))
            for model_id, history in sorted(self._reward_history.items())
        }
        state["causal_relation_transitions"] = {
            model_id: {
                relation: matrix.tolist()
                for relation, matrix in sorted(matrices.items())
            }
            for model_id, matrices in sorted(self._transitions.items())
        }
        state["causal_transition_observations"] = {
            model_id: dict(sorted(rows.items()))
            for model_id, rows in sorted(self._transition_observations.items())
        }
        state["causal_label_history"] = {
            node_id: {
                "labels": list(labels),
                "posterior": posterior.tolist(),
            }
            for node_id, (labels, posterior) in sorted(self._label_history.items())
        }
        state["causal_label_transitions"] = {
            "\u0000".join(labels): {
                relation: matrix.tolist()
                for relation, matrix in sorted(rows.items())
            }
            for labels, rows in sorted(self._label_transitions.items())
        }
        state["causal_label_transition_observations"] = {
            "\u0000".join(labels): dict(sorted(rows.items()))
            for labels, rows in sorted(self._label_transition_observations.items())
        }
        state["causal_class_counts"] = {
            "\u0000".join(labels): counts.tolist()
            for labels, counts in sorted(self._class_counts.items())
        }
        state["causal_structural_label_counts"] = {
            "\u0000".join(labels): {
                feature: values.tolist() for feature, values in sorted(rows.items())
            }
            for labels, rows in sorted(self._structural_label_counts.items())
        }
        state["causal_structural_observations"] = {
            "\u0000".join(labels): dict(sorted(rows.items()))
            for labels, rows in sorted(self._structural_observations.items())
        }
        state["causal_confusion_counts"] = {
            "\u0000".join(labels): {
                model_id: matrix.tolist()
                for model_id, matrix in sorted(rows.items())
            }
            for labels, rows in sorted(self._confusion_counts.items())
        }
        return state

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            if state.get("causal_subset_graph_config") != asdict(self.graph_config):
                raise CheckpointError("checkpoint causal graph configuration does not match")
            raw_history = state["causal_reward_history"]
            raw_transitions = state["causal_relation_transitions"]
            raw_observations = state["causal_transition_observations"]
            if not (
                set(raw_history)
                == set(raw_transitions)
                == set(raw_observations)
                == set(self.models)
            ):
                raise CheckpointError("checkpoint causal model pool does not match")
            history = {
                model_id: {str(node): float(value) for node, value in rows.items()}
                for model_id, rows in raw_history.items()
            }
            transitions: dict[str, dict[str, np.ndarray]] = {}
            for model_id, rows in raw_transitions.items():
                transitions[model_id] = {}
                for relation, raw_matrix in rows.items():
                    matrix = np.asarray(raw_matrix, dtype=float)
                    if matrix.shape != (2, 2) or np.any(matrix <= 0):
                        raise CheckpointError("invalid causal transition matrix")
                    transitions[model_id][str(relation)] = matrix
            observations = {
                model_id: {str(relation): float(value) for relation, value in rows.items()}
                for model_id, rows in raw_observations.items()
            }
            label_history = {
                str(node_id): (
                    tuple(raw["labels"]),
                    np.asarray(raw["posterior"], dtype=float),
                )
                for node_id, raw in state.get("causal_label_history", {}).items()
            }
            label_transitions = {
                tuple(encoded.split("\u0000")): {
                    str(relation): np.asarray(matrix, dtype=float)
                    for relation, matrix in rows.items()
                }
                for encoded, rows in state.get("causal_label_transitions", {}).items()
            }
            label_transition_observations = {
                tuple(encoded.split("\u0000")): {
                    str(relation): float(value) for relation, value in rows.items()
                }
                for encoded, rows in state.get(
                    "causal_label_transition_observations", {}
                ).items()
            }
            class_counts = {
                tuple(encoded.split("\u0000")): np.asarray(values, dtype=float)
                for encoded, values in state.get("causal_class_counts", {}).items()
            }
            structural_label_counts = {
                tuple(encoded.split("\u0000")): {
                    str(feature): np.asarray(values, dtype=float)
                    for feature, values in rows.items()
                }
                for encoded, rows in state.get(
                    "causal_structural_label_counts", {}
                ).items()
            }
            structural_observations = {
                tuple(encoded.split("\u0000")): {
                    str(feature): float(value) for feature, value in rows.items()
                }
                for encoded, rows in state.get(
                    "causal_structural_observations", {}
                ).items()
            }
            confusion_counts = {
                tuple(encoded.split("\u0000")): {
                    str(model_id): np.asarray(matrix, dtype=float)
                    for model_id, matrix in rows.items()
                }
                for encoded, rows in state.get("causal_confusion_counts", {}).items()
            }
        except CheckpointError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid causal subset checkpoint: {exc}") from exc
        super().load_checkpoint(path)
        self._reward_history = history
        self._transitions = transitions
        self._transition_observations = observations
        self._label_history = label_history
        self._label_transitions = label_transitions
        self._label_transition_observations = label_transition_observations
        self._class_counts = class_counts
        self._structural_label_counts = structural_label_counts
        self._structural_observations = structural_observations
        self._confusion_counts = confusion_counts
