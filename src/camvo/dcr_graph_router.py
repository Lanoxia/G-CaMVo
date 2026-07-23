"""Diversity-, causality-, and reliability-aware graph CaMVo.

The router keeps the defining CaMVo round contract:

1. score a pool of models before observing current responses;
2. choose a subset before any current vote is visible;
3. query that subset and aggregate exactly one decision;
4. update online state after the decision.

It corrects three failure modes that are acute in security streams.  Audited
feedback learns class-conditional confusion matrices rather than treating
agreement with a potentially biased majority as competence.  Pairwise error
contingencies penalize redundant model subsets.  Finally, a past-only typed
provenance graph supplies a causal prior both to subset selection and to the
decision head.  Missing, immature, or out-of-distribution graph evidence falls
back to the reliability-calibrated model posterior.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from camvo.adaptive_graph_router import (
    AdaptiveGraphCaMVoRouter,
    AdaptiveGraphConfig,
    _PendingGraphDecision,
)
from camvo.algorithm.oracle import OracleSelection
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class DCRGraphConfig:
    """Controls reliability calibration and diversity-aware subset selection."""

    confusion_prior: float = 1.0
    error_pair_prior: float = 1.0
    balanced_prior_strength: float = 0.75
    graph_prior_blend: float = 0.30
    redundancy_penalty: float = 0.75
    correlation_discount: float = 1.50
    cost_penalty: float = 0.015
    competence_bonus: float = 0.05
    max_subset_models: int = 0
    minimum_audited_rows: int = 8
    likelihood_temperature: float = 1.0
    pseudo_history_weight: float = 0.0
    dynamic_entity_edges: bool = False
    recent_nodes_per_entity: int = 8
    max_graph_blend: float = 0.20
    protect_base_margin: float = 0.20
    max_js_divergence: float = 0.35
    latest_parent_per_relation: bool = False

    def __post_init__(self) -> None:
        for name in ("confusion_prior", "error_pair_prior", "likelihood_temperature"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "balanced_prior_strength",
            "graph_prior_blend",
            "redundancy_penalty",
            "correlation_discount",
            "cost_penalty",
            "competence_bonus",
            "pseudo_history_weight",
            "max_graph_blend",
            "protect_base_margin",
            "max_js_divergence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(
            value > 1
            for value in (
                self.balanced_prior_strength,
                self.graph_prior_blend,
                self.max_graph_blend,
                self.protect_base_margin,
                self.max_js_divergence,
            )
        ):
            raise ValueError("prior blend strengths must be in [0, 1]")
        if self.pseudo_history_weight > 1:
            raise ValueError("pseudo_history_weight must be in [0, 1]")
        if self.max_subset_models < 0:
            raise ValueError("max_subset_models must be non-negative")
        if self.minimum_audited_rows < 0:
            raise ValueError("minimum_audited_rows must be non-negative")
        if self.recent_nodes_per_entity <= 0:
            raise ValueError("recent_nodes_per_entity must be positive")


class DCRGraphCaMVoRouter(AdaptiveGraphCaMVoRouter):
    """CaMVo with audited reliability, error diversity, and causal graph priors."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        graph_config: AdaptiveGraphConfig,
        dcr_config: DCRGraphConfig,
        neighborhood: Any = None,
    ) -> None:
        graph_config = replace(
            graph_config,
            pseudo_history_weight=dcr_config.pseudo_history_weight,
            dynamic_entity_edges=dcr_config.dynamic_entity_edges,
            recent_nodes_per_entity=dcr_config.recent_nodes_per_entity,
            max_parents=max(
                graph_config.max_parents, dcr_config.recent_nodes_per_entity
            ),
            max_graph_blend=dcr_config.max_graph_blend,
            protect_base_margin=dcr_config.protect_base_margin,
            max_js_divergence=dcr_config.max_js_divergence,
        )
        super().__init__(models, embedder, config, graph_config, neighborhood)
        self.dcr_config = dcr_config
        self._confusions: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        self._error_pairs: dict[
            tuple[str, ...], dict[tuple[str, str], np.ndarray]
        ] = {}
        self._dcr_class_counts: dict[tuple[str, ...], np.ndarray] = {}
        self._audited_rows = 0
        self._pending_votes: dict[str, tuple[AnnotationItem, dict[str, str]]] = {}
        self._selection_item: AnnotationItem | None = None
        self._selection_prior: np.ndarray | None = None
        self._last_selection_trace: dict[str, Any] = {}

    def _confusion(self, labels: tuple[str, ...], model_id: str) -> np.ndarray:
        rows = self._confusions.setdefault(labels, {})
        if model_id not in rows:
            rows[model_id] = np.full(
                (len(labels), len(labels)), self.dcr_config.confusion_prior, dtype=float
            )
        return rows[model_id]

    def _pair(self, labels: tuple[str, ...], left: str, right: str) -> np.ndarray:
        key = tuple(sorted((left, right)))
        rows = self._error_pairs.setdefault(labels, {})
        if key not in rows:
            rows[key] = np.full((2, 2), self.dcr_config.error_pair_prior, dtype=float)
        return rows[key]

    def _class_prior(self, labels: tuple[str, ...]) -> np.ndarray:
        counts = self._dcr_class_counts.get(labels)
        empirical = (
            np.full(len(labels), 1.0 / len(labels), dtype=float)
            if counts is None
            else counts / counts.sum()
        )
        uniform = np.full(len(labels), 1.0 / len(labels), dtype=float)
        strength = self.dcr_config.balanced_prior_strength
        prior = strength * uniform + (1.0 - strength) * empirical
        return prior / prior.sum()

    def _confusion_probabilities(
        self, labels: tuple[str, ...], model_id: str
    ) -> np.ndarray:
        matrix = self._confusion(labels, model_id)
        return matrix / matrix.sum(axis=1, keepdims=True)

    def _error_correlation(
        self, labels: tuple[str, ...], left: str, right: str
    ) -> float:
        table = self._pair(labels, left, right)
        n00, n01 = map(float, table[0])
        n10, n11 = map(float, table[1])
        numerator = n11 * n00 - n10 * n01
        denominator = math.sqrt(
            max((n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10), 1e-12)
        )
        return max(-1.0, min(1.0, numerator / denominator))

    @staticmethod
    def _mutual_information(prior: np.ndarray, confusion: np.ndarray) -> float:
        vote_prior = prior @ confusion
        total = 0.0
        for gold_index in range(len(prior)):
            for vote_index in range(len(vote_prior)):
                joint = prior[gold_index] * confusion[gold_index, vote_index]
                if joint <= 0 or vote_prior[vote_index] <= 0:
                    continue
                total += joint * math.log(
                    confusion[gold_index, vote_index] / vote_prior[vote_index]
                )
        return max(0.0, total)

    def _causal_prior(self, item: AnnotationItem) -> np.ndarray:
        prior = self._class_prior(item.labels)
        parents = self._candidate_parents(item)
        graph, evidence_weight, _relations = self._graph_message(item.labels, parents)
        if graph is None or evidence_weight <= 0:
            return prior
        strength = min(1.0, evidence_weight / self.graph_config.max_total_parent_weight)
        alpha = self.dcr_config.graph_prior_blend * strength
        fused = (1.0 - alpha) * prior + alpha * graph
        return fused / fused.sum()

    def _graph_message(
        self,
        labels: tuple[str, ...],
        parents: Sequence[tuple[str, float, tuple[str, ...]]],
    ) -> tuple[np.ndarray | None, float, tuple[str, ...]]:
        if not self.dcr_config.latest_parent_per_relation:
            return super()._graph_message(labels, parents)
        # Security state is often piecewise persistent.  Averaging a long run
        # of stale parents can obscure the latest phase transition, so retain
        # only the highest-confidence/recency candidate for each relation.
        best: dict[str, tuple[float, np.ndarray]] = {}
        for parent_id, edge_weight, relations in parents:
            raw = self._label_history.get(parent_id)
            if raw is None or raw[0] != labels:
                continue
            _history_labels, parent, audited = raw
            history_weight = 1.0 if audited else self.graph_config.pseudo_history_weight
            if history_weight <= 0:
                continue
            for relation in relations:
                utility = self._gate_utility(labels, relation)
                if utility <= 0:
                    continue
                weight = history_weight * edge_weight * utility
                candidate = (weight, self._relation_prediction(labels, relation, parent))
                if relation not in best or candidate[0] > best[relation][0]:
                    best[relation] = candidate
        if not best:
            return None, 0.0, ()
        total = sum(weight for weight, _posterior in best.values())
        pooled = sum(
            (weight * posterior for weight, posterior in best.values()),
            start=np.zeros(len(labels), dtype=float),
        )
        return pooled / total, total, tuple(sorted(best))

    def _score_models(
        self,
        item: AnnotationItem,
        context: np.ndarray,
        round_index: int,
    ) -> tuple[dict[str, ModelScore], dict[str, Any]]:
        scores, bandit = super()._score_models(item, context, round_index)
        self._selection_item = item
        self._selection_prior = self._causal_prior(item)
        return scores, bandit

    def _subset_utility(
        self,
        labels: tuple[str, ...],
        prior: np.ndarray,
        subset: tuple[str, ...],
        scores: Mapping[str, ModelScore],
        information: Mapping[str, float],
        cheapest_cost: float,
    ) -> tuple[float, float, float, float]:
        raw_information = sum(information[model_id] for model_id in subset)
        redundancy = 0.0
        for left, right in itertools.combinations(subset, 2):
            correlation = max(0.0, self._error_correlation(labels, left, right))
            redundancy += correlation * math.sqrt(
                information[left] * information[right]
            )
        cost_ratio = sum(scores[model_id].estimated_cost for model_id in subset) / max(
            cheapest_cost, 1e-12
        )
        competence = sum(scores[model_id].smoothed_lower_bound for model_id in subset) / len(
            subset
        )
        objective = (
            raw_information
            - self.dcr_config.redundancy_penalty * redundancy
            - self.dcr_config.cost_penalty * math.log1p(cost_ratio)
            + self.dcr_config.competence_bonus * competence
        )
        return objective, raw_information, redundancy, cost_ratio

    def _selection(
        self, scores: dict[str, ModelScore], warmup: bool
    ) -> OracleSelection:
        item = self._selection_item
        prior = self._selection_prior
        if (
            item is None
            or prior is None
            or self._audited_rows < self.dcr_config.minimum_audited_rows
        ):
            self._last_selection_trace = {"fallback": "insufficient_audited_rows"}
            return super()._selection(scores, warmup)

        model_ids = tuple(sorted(scores))
        max_models = self.dcr_config.max_subset_models or len(model_ids)
        max_models = min(max_models, len(model_ids))
        information = {
            model_id: self._mutual_information(
                prior, self._confusion_probabilities(item.labels, model_id)
            )
            for model_id in model_ids
        }
        cheapest = min(score.estimated_cost for score in scores.values())
        candidates: list[
            tuple[float, float, float, float, float, tuple[str, ...]]
        ] = []
        for size in range(self.config.min_models, max_models + 1):
            for subset in itertools.combinations(model_ids, size):
                objective, raw_info, redundancy, cost_ratio = self._subset_utility(
                    item.labels,
                    prior,
                    subset,
                    scores,
                    information,
                    cheapest,
                )
                actual_cost = sum(scores[model_id].estimated_cost for model_id in subset)
                candidates.append(
                    (objective, raw_info, -redundancy, -actual_cost, -size, subset)
                )
        if not candidates:
            raise ValueError("no feasible DCR model subset")
        selected = max(candidates)
        objective, raw_info, negative_redundancy, negative_cost, _negative_size, subset = selected
        confidence = min(1.0, float(np.max(prior)) + raw_info)
        self._last_selection_trace = {
            "fallback": None,
            "objective": objective,
            "information_gain": raw_info,
            "redundancy": -negative_redundancy,
            "prior": {
                label: float(prior[index]) for index, label in enumerate(item.labels)
            },
            "per_model_information": dict(sorted(information.items())),
        }
        return OracleSelection(
            model_ids=subset,
            confidence=confidence,
            cost=-negative_cost,
            feasible=confidence >= self.config.confidence_threshold,
        )

    def _reliability_posterior(
        self,
        item: AnnotationItem,
        responses: Mapping[str, str],
        successful_ids: Sequence[str],
    ) -> np.ndarray:
        prior = self._class_prior(item.labels)
        information = {
            model_id: self._mutual_information(
                prior, self._confusion_probabilities(item.labels, model_id)
            )
            for model_id in successful_ids
        }
        ordered = sorted(successful_ids, key=lambda model_id: (-information[model_id], model_id))
        log_posterior = np.log(np.clip(prior, 1e-12, 1.0))
        accepted: list[str] = []
        for model_id in ordered:
            correlation = sum(
                max(0.0, self._error_correlation(item.labels, model_id, previous))
                for previous in accepted
            )
            exponent = 1.0 / (1.0 + self.dcr_config.correlation_discount * correlation)
            exponent /= self.dcr_config.likelihood_temperature
            confusion = self._confusion_probabilities(item.labels, model_id)
            vote_index = item.labels.index(responses[model_id])
            log_posterior += exponent * np.log(
                np.clip(confusion[:, vote_index], 1e-12, 1.0)
            )
            accepted.append(model_id)
        log_posterior -= float(np.max(log_posterior))
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum()

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        model_posterior = self._reliability_posterior(item, responses, successful_ids)
        parents = self._candidate_parents(item)
        graph, evidence_weight, relations = self._graph_message(item.labels, parents)
        fused, fallback = self._safe_fuse(model_posterior, graph, evidence_weight)
        self._last_decision = _PendingGraphDecision(
            item=item,
            base_posterior=model_posterior,
            graph_posterior=None if graph is None else graph.copy(),
            fused_posterior=fused,
            parents=parents,
            applied_relations=relations if fallback is None else (),
            fallback_reason=fallback,
        )
        self._pending_votes[self._node_id(item)] = (item, dict(responses))
        return self._winner(item.labels, fused)

    def route(self, item: AnnotationItem) -> RoutingResult:
        result = super().route(item)
        trace = {
            "component": "dcr_subset",
            **self._last_selection_trace,
            "selected_models": list(result.selected_models),
        }
        return replace(result, routing_trace=result.routing_trace + (trace,))

    def _update_audited_stats(
        self,
        item: AnnotationItem,
        responses: Mapping[str, str],
        gold_label: str,
        *,
        weight: float = 1.0,
    ) -> None:
        labels = item.labels
        if gold_label not in labels:
            raise ValueError("gold_label is outside the item's label space")
        gold_index = labels.index(gold_label)
        counts = self._dcr_class_counts.setdefault(labels, np.ones(len(labels), dtype=float))
        counts[gold_index] += weight
        errors: dict[str, int] = {}
        for model_id, vote in responses.items():
            if model_id not in self.models or vote not in labels:
                continue
            vote_index = labels.index(vote)
            self._confusion(labels, model_id)[gold_index, vote_index] += weight
            errors[model_id] = int(vote != gold_label)
        for left, right in itertools.combinations(sorted(errors), 2):
            self._pair(labels, left, right)[errors[left], errors[right]] += weight
        self._audited_rows += 1

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        self._update_audited_stats(
            item,
            {model_id: response.label for model_id, response in responses.items()},
            gold_label,
        )
        super().observe_complete_feedback(item, responses, gold_label)

    def observe_graph_feedback(self, item: AnnotationItem, gold_label: str) -> None:
        node_id = self._node_id(item)
        pending = self._pending_votes.pop(node_id, None)
        if pending is None:
            raise ValueError(f"no pending DCR votes for node {node_id!r}")
        self._update_audited_stats(pending[0], pending[1], gold_label)
        super().observe_graph_feedback(item, gold_label)

    def reset_graph_history(self) -> None:
        super().reset_graph_history()
        self._pending_votes = {}
        self._selection_item = None
        self._selection_prior = None
        self._last_selection_trace = {}

    def dcr_diagnostics(self) -> dict[str, Any]:
        correlations: dict[str, Any] = {}
        for labels, rows in self._error_pairs.items():
            correlations["|".join(labels)] = {
                f"{left}|{right}": self._error_correlation(labels, left, right)
                for left, right in sorted(rows)
            }
        confusions = {
            "|".join(labels): {
                model_id: self._confusion_probabilities(labels, model_id).tolist()
                for model_id in sorted(rows)
            }
            for labels, rows in self._confusions.items()
        }
        return {
            "config": asdict(self.dcr_config),
            "audited_rows": self._audited_rows,
            "confusion_probabilities": confusions,
            "error_correlations": correlations,
            "graph": self.graph_diagnostics(),
        }
