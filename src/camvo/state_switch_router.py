"""Causal state-switch extension for graph-aware CaMVo.

The router preserves the CaMVo round contract: a subset is chosen before the
current responses are visible, exactly that subset is queried, one label is
emitted, and audited feedback is incorporated only afterwards.  Its decision
head treats the latest audited state of an entity as a persistence prior and
learns which subset-specific vote patterns reliably signal a phase change.

This is deliberately model agnostic.  It consumes categorical votes from any
provider pool and never special-cases a model family or model identifier.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from camvo.adaptive_graph_router import AdaptiveGraphConfig
from camvo.algorithm.oracle import OracleSelection
from camvo.config import CaMVoConfig
from camvo.dcr_graph_router import DCRGraphCaMVoRouter, DCRGraphConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class StateSwitchConfig:
    """Configuration of the audited state-transition decision head."""

    entity_metadata_key: str = "hostname"
    pattern_prior_strength: float = 16.0
    switch_threshold: float = 0.75
    information_cost_penalty: float = 0.01
    minimum_pattern_support: int = 2
    subset_size: int = 2
    transition_prior: float = 1.0

    def __post_init__(self) -> None:
        if not self.entity_metadata_key:
            raise ValueError("entity_metadata_key must be non-empty")
        for name in ("pattern_prior_strength", "transition_prior"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("switch_threshold", "information_cost_penalty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.switch_threshold > 1:
            raise ValueError("switch_threshold must be in [0, 1]")
        if self.minimum_pattern_support < 0:
            raise ValueError("minimum_pattern_support must be non-negative")
        if self.subset_size <= 0:
            raise ValueError("subset_size must be positive")


class CausalStateSwitchGraphCaMVoRouter(DCRGraphCaMVoRouter):
    """DCR-G-CaMVo with an audited, past-only entity state transition head."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        graph_config: AdaptiveGraphConfig,
        dcr_config: DCRGraphConfig,
        switch_config: StateSwitchConfig,
        neighborhood: Any = None,
    ) -> None:
        if switch_config.subset_size < config.min_models:
            raise ValueError("switch subset_size cannot be smaller than min_models")
        if switch_config.subset_size > len(models):
            raise ValueError("switch subset_size exceeds the model pool")
        super().__init__(
            models,
            embedder,
            config,
            graph_config,
            replace(dcr_config, max_subset_models=switch_config.subset_size),
            neighborhood,
        )
        self.switch_config = switch_config
        self._transition_counts: dict[
            tuple[tuple[str, ...], str], np.ndarray
        ] = {}
        self._pattern_counts: dict[
            tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]], np.ndarray
        ] = {}
        self._calibration_state: dict[tuple[tuple[str, ...], str], str] = {}
        self._stream_state: dict[tuple[tuple[str, ...], str], str] = {}
        self._pending_switch: dict[
            str, tuple[AnnotationItem, str | None, dict[str, str]]
        ] = {}
        self._switch_trace: dict[str, Any] = {"fallback": "no_entity_state"}

    def _entity(self, item: AnnotationItem) -> str:
        value = item.metadata.get(self.switch_config.entity_metadata_key, "unknown")
        return str(value)

    def _state_key(self, item: AnnotationItem) -> tuple[tuple[str, ...], str]:
        return item.labels, self._entity(item)

    def _state_transition(self, labels: tuple[str, ...], previous: str) -> np.ndarray:
        key = (labels, previous)
        if key not in self._transition_counts:
            self._transition_counts[key] = np.full(
                len(labels), self.switch_config.transition_prior, dtype=float
            )
        return self._transition_counts[key]

    def _pattern(
        self,
        labels: tuple[str, ...],
        previous: str,
        subset: tuple[str, ...],
        votes: tuple[str, ...],
    ) -> np.ndarray:
        key = (labels, previous, subset, votes)
        if key not in self._pattern_counts:
            self._pattern_counts[key] = np.zeros(len(labels), dtype=float)
        return self._pattern_counts[key]

    def _all_subsets(self, model_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            itertools.combinations(sorted(model_ids), self.switch_config.subset_size)
        )

    def _observe_switch(
        self,
        item: AnnotationItem,
        previous: str,
        responses: Mapping[str, str],
        gold_label: str,
        *,
        complete: bool,
    ) -> None:
        gold_index = item.labels.index(gold_label)
        self._state_transition(item.labels, previous)[gold_index] += 1.0
        subsets = (
            self._all_subsets(tuple(responses))
            if complete
            else (tuple(sorted(responses)),)
        )
        for subset in subsets:
            if len(subset) != self.switch_config.subset_size:
                continue
            votes = tuple(responses[model_id] for model_id in subset)
            self._pattern(item.labels, previous, subset, votes)[gold_index] += 1.0

    def _subset_information(
        self,
        labels: tuple[str, ...],
        previous: str,
        subset: tuple[str, ...],
    ) -> float:
        tables = [
            (votes, counts)
            for (space, state, candidate, votes), counts in self._pattern_counts.items()
            if space == labels and state == previous and candidate == subset
        ]
        total = sum(float(counts.sum()) for _votes, counts in tables)
        if total <= 0:
            return 0.0
        gold = sum((counts for _votes, counts in tables), start=np.zeros(len(labels)))
        information = 0.0
        for _votes, counts in tables:
            pattern_total = float(counts.sum())
            for index, count in enumerate(counts):
                if count <= 0 or gold[index] <= 0:
                    continue
                joint = float(count) / total
                information += joint * math.log(float(count) * total / (pattern_total * gold[index]))
        return information

    def _selection(
        self, scores: dict[str, ModelScore], warmup: bool
    ) -> OracleSelection:
        item = self._selection_item
        if item is None:
            return super()._selection(scores, warmup)
        previous = self._stream_state.get(self._state_key(item))
        if previous is None:
            return super()._selection(scores, warmup)
        subsets = self._all_subsets(tuple(scores))
        cheapest = min(
            sum(scores[model_id].estimated_cost for model_id in subset)
            for subset in subsets
        )
        ranked = []
        for subset in subsets:
            information = self._subset_information(item.labels, previous, subset)
            cost = sum(scores[model_id].estimated_cost for model_id in subset)
            objective = information - self.switch_config.information_cost_penalty * cost / max(
                cheapest, 1e-12
            )
            ranked.append((objective, information, -cost, subset))
        objective, information, negative_cost, subset = max(ranked)
        self._last_selection_trace = {
            "fallback": None,
            "component": "state_switch_information",
            "previous_state": previous,
            "objective": objective,
            "information_gain": information,
        }
        return OracleSelection(
            model_ids=subset,
            confidence=min(1.0, information + 0.5),
            cost=-negative_cost,
            feasible=True,
        )

    def _switch_posterior(
        self,
        item: AnnotationItem,
        previous: str,
        responses: Mapping[str, str],
    ) -> tuple[np.ndarray, int]:
        subset = tuple(sorted(responses))
        votes = tuple(responses[model_id] for model_id in subset)
        counts = self._pattern(item.labels, previous, subset, votes)
        support = int(round(float(counts.sum())))
        transition = self._state_transition(item.labels, previous)
        prior = transition / transition.sum()
        strength = self.switch_config.pattern_prior_strength
        posterior = (counts + strength * prior) / (float(counts.sum()) + strength)
        return posterior / posterior.sum(), support

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        base = super()._aggregate_responses(
            item, responses, raw_responses, scores, successful_ids
        )
        previous = self._stream_state.get(self._state_key(item))
        self._pending_switch[self._node_id(item)] = (item, previous, dict(responses))
        if previous is None:
            self._switch_trace = {"fallback": "no_entity_state", "base_label": base}
            return base
        posterior, support = self._switch_posterior(item, previous, responses)
        alternatives = tuple(label for label in item.labels if label != previous)
        candidate = max(
            alternatives,
            key=lambda label: (posterior[item.labels.index(label)], -item.labels.index(label)),
        )
        probability = float(posterior[item.labels.index(candidate)])
        switch = (
            support >= self.switch_config.minimum_pattern_support
            and probability >= self.switch_config.switch_threshold
        )
        self._switch_trace = {
            "fallback": None,
            "previous_state": previous,
            "candidate_state": candidate,
            "switch_probability": probability,
            "pattern_support": support,
            "switched": switch,
            "base_label": base,
        }
        return candidate if switch else previous

    def route(self, item: AnnotationItem) -> RoutingResult:
        result = super().route(item)
        trace = {
            "component": "causal_state_switch",
            **self._switch_trace,
        }
        return replace(result, routing_trace=result.routing_trace + (trace,))

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        key = self._state_key(item)
        previous = self._calibration_state.get(key)
        if previous is not None:
            self._observe_switch(
                item,
                previous,
                {model_id: response.label for model_id, response in responses.items()},
                gold_label,
                complete=True,
            )
        self._calibration_state[key] = gold_label
        super().observe_complete_feedback(item, responses, gold_label)

    def observe_graph_feedback(self, item: AnnotationItem, gold_label: str) -> None:
        node_id = self._node_id(item)
        pending = self._pending_switch.pop(node_id, None)
        if pending is None:
            raise ValueError(f"no pending state-switch decision for node {node_id!r}")
        pending_item, previous, responses = pending
        if previous is not None:
            self._observe_switch(
                pending_item,
                previous,
                responses,
                gold_label,
                complete=False,
            )
        self._stream_state[self._state_key(item)] = gold_label
        super().observe_graph_feedback(item, gold_label)

    def reset_graph_history(self) -> None:
        super().reset_graph_history()
        self._stream_state = {}
        self._pending_switch = {}
        self._switch_trace = {"fallback": "no_entity_state"}

    def state_switch_diagnostics(self) -> dict[str, Any]:
        return {
            "config": asdict(self.switch_config),
            "transition_contexts": len(self._transition_counts),
            "vote_patterns": len(self._pattern_counts),
            "stream_entities": len(self._stream_state),
            "dcr": self.dcr_diagnostics(),
        }
