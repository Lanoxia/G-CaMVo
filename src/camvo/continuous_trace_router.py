"""Continuous causal vote fusion on top of the original CaMVo online loop.

The model subset is still selected before current responses are visible.  Each
selected categorical vote is converted to a soft distribution using the
provider-independent confidence field in the task contract.  CaMVo's
``omega=mu*q`` weights pool those distributions, and already processed causal
parents contribute a bounded product-of-experts message.  The fused label is
the round consensus used by the unchanged CaMVo bandit/Beta updates.
"""

from __future__ import annotations

import math
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any
from pathlib import Path

import numpy as np

from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.exceptions import CheckpointError
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult
from camvo.aggregation import weighted_vote


@dataclass(frozen=True, slots=True)
class ContinuousTraceConfig:
    """Model-agnostic controls for conservative causal probability fusion."""

    graph_strength: float = 0.1
    trim_each_side: int = 0
    label_logit_bias: tuple[tuple[str, float], ...] = ()
    default_reported_confidence: float = 0.75
    confidence_floor: float = 0.51
    confidence_ceiling: float = 0.99
    max_edge_weight: float = 2.0
    max_total_parent_weight: float = 8.0

    def __post_init__(self) -> None:
        if not 0 <= self.graph_strength <= 1:
            raise ValueError("graph_strength must be in [0, 1]")
        if self.trim_each_side < 0:
            raise ValueError("trim_each_side must be non-negative")
        if not 0.5 < self.confidence_floor < self.confidence_ceiling <= 1:
            raise ValueError("invalid confidence bounds")
        if not self.confidence_floor <= self.default_reported_confidence <= self.confidence_ceiling:
            raise ValueError("default confidence is outside configured bounds")
        if self.max_edge_weight <= 0 or self.max_total_parent_weight <= 0:
            raise ValueError("graph weight bounds must be positive")
        labels = [label for label, _value in self.label_logit_bias]
        if len(labels) != len(set(labels)):
            raise ValueError("label_logit_bias contains duplicate labels")
        if any(not math.isfinite(float(value)) for _label, value in self.label_logit_bias):
            raise ValueError("label_logit_bias must be finite")


class ContinuousTraceCaMVoRouter(CaMVoRouter):
    """CaMVo with confidence-aware votes and past-only causal messages."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        trace_config: ContinuousTraceConfig,
        neighborhood: Callable[[AnnotationItem], Mapping[str, Any]],
    ) -> None:
        super().__init__(models, embedder, config)
        self.trace_config = trace_config
        self.neighborhood = neighborhood
        self._posterior_history: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self._pending_posterior: np.ndarray | None = None
        self._pending_local_posterior: np.ndarray | None = None
        self._pending_parent_count = 0
        self._pending_model_consensus: str | None = None

    @staticmethod
    def _node_id(item: AnnotationItem) -> str:
        return str(item.metadata.get("graph_node_id", item.metadata.get("event_id", item.item_id)))

    def _reported_confidence(self, response: ModelResponse) -> float:
        value = self.trace_config.default_reported_confidence
        if isinstance(response.raw, Mapping):
            try:
                value = float(response.raw.get("confidence", value))
            except (TypeError, ValueError):
                value = self.trace_config.default_reported_confidence
        if not math.isfinite(value):
            value = self.trace_config.default_reported_confidence
        return min(self.trace_config.confidence_ceiling, max(self.trace_config.confidence_floor, value))

    def _soft_vote(self, response: ModelResponse, labels: tuple[str, ...]) -> np.ndarray:
        confidence = self._reported_confidence(response)
        distribution = np.full(
            len(labels),
            (1.0 - confidence) / max(1, len(labels) - 1),
            dtype=float,
        )
        distribution[labels.index(response.label)] = confidence
        return distribution

    def _pool_votes(
        self,
        raw_responses: Mapping[str, ModelResponse],
        scores: Mapping[str, ModelScore],
        model_ids: tuple[str, ...],
        labels: tuple[str, ...],
    ) -> np.ndarray:
        rows = np.asarray([self._soft_vote(raw_responses[model_id], labels) for model_id in model_ids])
        weights = np.asarray([scores[model_id].vote_weight for model_id in model_ids], dtype=float)
        trim = self.trace_config.trim_each_side
        pooled = np.zeros(len(labels), dtype=float)
        for label_index in range(len(labels)):
            order = np.argsort(rows[:, label_index], kind="stable")
            if len(order) > 2 * trim:
                order = order[trim : len(order) - trim]
            selected_weights = weights[order]
            pooled[label_index] = float(
                np.average(rows[order, label_index], weights=selected_weights)
            )
        pooled = np.clip(pooled, 1e-9, None)
        return pooled / pooled.sum()

    def _parent_message(self, item: AnnotationItem) -> tuple[np.ndarray, int]:
        labels = item.labels
        weighted = np.zeros(len(labels), dtype=float)
        total = 0.0
        used = 0
        candidates: list[tuple[str, float]] = []
        for parent_id, raw in self.neighborhood(item).items():
            if isinstance(raw, Mapping):
                weight = float(raw.get("weight", 0.0))
            else:
                weight = float(raw)
            if math.isfinite(weight) and weight > 0:
                candidates.append((str(parent_id), min(weight, self.trace_config.max_edge_weight)))
        for parent_id, weight in sorted(candidates, key=lambda row: (-row[1], row[0])):
            remaining = self.trace_config.max_total_parent_weight - total
            if remaining <= 0:
                break
            history = self._posterior_history.get(parent_id)
            if history is None or history[0] != labels:
                continue
            accepted = min(weight, remaining)
            weighted += accepted * history[1]
            total += accepted
            used += 1
        if total == 0:
            return np.full(len(labels), 1.0 / len(labels), dtype=float), 0
        return weighted / total, used

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        local = self._pool_votes(raw_responses, scores, successful_ids, item.labels)
        self._pending_local_posterior = local.copy()
        self._pending_model_consensus, _ties = weighted_vote(
            responses,
            {model_id: scores[model_id].vote_weight for model_id in successful_ids},
            item.labels,
        )
        parent, used = self._parent_message(item)
        logits = np.log(np.clip(local, 1e-12, 1.0))
        if used:
            logits += self.trace_config.graph_strength * np.log(
                np.clip(parent, 1e-12, 1.0)
            )
        bias = dict(self.trace_config.label_logit_bias)
        logits += np.asarray([float(bias.get(label, 0.0)) for label in item.labels])
        posterior = np.exp(logits - float(np.max(logits)))
        posterior /= posterior.sum()
        self._pending_posterior = posterior
        self._pending_parent_count = used
        maximum = float(np.max(posterior))
        return next(
            label
            for index, label in enumerate(item.labels)
            if math.isclose(float(posterior[index]), maximum)
        )

    def _reward_reference_label(
        self,
        item: AnnotationItem,
        final_label: str,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        """Keep CaMVo competence rewards tied to model-only consensus.

        The task head may incorporate graph evidence, but the LinUCB/Beta head
        continues to estimate agreement within the queried model panel.  This
        prevents a graph prior from creating a self-reinforcing selection loop.
        """

        return self._pending_model_consensus or final_label

    def route(self, item: AnnotationItem) -> RoutingResult:
        self._pending_posterior = None
        self._pending_local_posterior = None
        self._pending_parent_count = 0
        self._pending_model_consensus = None
        result = super().route(item)
        posterior = self._pending_posterior
        if posterior is None:
            posterior = np.zeros(len(item.labels), dtype=float)
            posterior[item.labels.index(result.label)] = 1.0
        history_posterior = self._pending_local_posterior
        if history_posterior is None:
            history_posterior = posterior
        self._posterior_history[self._node_id(item)] = (
            item.labels,
            history_posterior.copy(),
        )
        return replace(
            result,
            posterior={label: float(posterior[index]) for index, label in enumerate(item.labels)},
            graph_evidence_weight=float(self._pending_parent_count),
        )

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        super().observe_complete_feedback(item, responses, gold_label)
        posterior = np.zeros(len(item.labels), dtype=float)
        posterior[item.labels.index(gold_label)] = 1.0
        self._posterior_history[self._node_id(item)] = (item.labels, posterior)

    def reset_graph_history(self) -> None:
        self._posterior_history = {}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["continuous_trace_config"] = json.loads(
            json.dumps(asdict(self.trace_config))
        )
        state["continuous_trace_history"] = {
            node_id: {"labels": list(labels), "posterior": posterior.tolist()}
            for node_id, (labels, posterior) in sorted(self._posterior_history.items())
        }
        return state

    def load_checkpoint(self, path: str | Path) -> None:
        super().load_checkpoint(path)
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            expected = json.loads(json.dumps(asdict(self.trace_config)))
            if state["continuous_trace_config"] != expected:
                raise CheckpointError(
                    "checkpoint continuous TRACE configuration does not match"
                )
            self._posterior_history = {
                str(node_id): (
                    tuple(raw["labels"]),
                    np.asarray(raw["posterior"], dtype=float),
                )
                for node_id, raw in state["continuous_trace_history"].items()
            }
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid continuous TRACE checkpoint: {exc}") from exc
