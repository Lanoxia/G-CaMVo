"""Causal online graph-regularized extension of the CaMVo router."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.exceptions import CheckpointError
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.security.optc import OptcCorrelationEdge
from camvo.types import AnnotationItem, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class GraphCaMVoConfig:
    """Controls causal neighbor propagation on top of independent CaMVo scores."""

    regularization: float = 1.0
    max_edge_weight: float = 2.0
    max_total_neighbor_weight: float = 8.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_edge_weight == 0 or self.max_total_neighbor_weight == 0:
            raise ValueError("graph weight caps must be positive")


class StaticGraphNeighborhood:
    """Read-only adjacency index constructed from OpTC correlation edges."""

    def __init__(self, adjacency: Mapping[str, Mapping[str, float]]) -> None:
        self._adjacency = {
            str(node_id): {str(neighbor): float(weight) for neighbor, weight in neighbors.items()}
            for node_id, neighbors in adjacency.items()
        }

    @classmethod
    def from_optc_edges(cls, edges: Iterable[OptcCorrelationEdge]) -> "StaticGraphNeighborhood":
        adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in edges:
            if edge.weight <= 0 or not math.isfinite(edge.weight):
                raise ValueError("correlation weights must be finite and positive")
            left = edge.source_event_id
            right = edge.target_event_id
            adjacency[left][right] = adjacency[left].get(right, 0.0) + edge.weight
            adjacency[right][left] = adjacency[right].get(left, 0.0) + edge.weight
        return cls(adjacency)

    def __call__(self, item: AnnotationItem) -> Mapping[str, float]:
        node_id = str(
            item.metadata.get(
                "graph_node_id",
                item.metadata.get("event_id", item.item_id),
            )
        )
        return self._adjacency.get(node_id, {})


class GraphCaMVoRouter(CaMVoRouter):
    r"""CaMVo with a causal online approximation to Laplacian smoothing.

    For current node ``v`` and already observed neighbors ``N^-(v)``, each
    model's independent bound ``y_v`` becomes

    ``f_v = (y_v + lambda sum_u w_uv f_u) / (1 + lambda sum_u w_uv)``.

    Only previously routed nodes are used, so an offline experiment cannot
    leak future alerts into earlier decisions. The full batch closed form is
    available separately through ``laplacian_smooth_scores``.
    """

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        graph_config: GraphCaMVoConfig,
        neighborhood: Callable[[AnnotationItem], Mapping[str, float]],
    ) -> None:
        super().__init__(models, embedder, config)
        self.graph_config = graph_config
        self.neighborhood = neighborhood
        self._graph_history: dict[str, dict[str, float]] = {
            model_id: {} for model_id in self.models
        }

    @staticmethod
    def _node_id(item: AnnotationItem) -> str:
        return str(
            item.metadata.get(
                "graph_node_id",
                item.metadata.get("event_id", item.item_id),
            )
        )

    def _bounded_neighbors(self, item: AnnotationItem) -> list[tuple[str, float]]:
        raw = self.neighborhood(item)
        candidates: list[tuple[str, float]] = []
        for node_id, raw_weight in raw.items():
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("neighborhood returned an invalid edge weight")
            if weight > 0:
                candidates.append((str(node_id), min(weight, self.graph_config.max_edge_weight)))
        candidates.sort(key=lambda value: (-value[1], value[0]))
        bounded: list[tuple[str, float]] = []
        total = 0.0
        for node_id, weight in candidates:
            remaining = self.graph_config.max_total_neighbor_weight - total
            if remaining <= 0:
                break
            accepted = min(weight, remaining)
            bounded.append((node_id, accepted))
            total += accepted
        return bounded

    def _score_models(
        self,
        item: AnnotationItem,
        context: np.ndarray,
        round_index: int,
    ) -> tuple[dict[str, ModelScore], dict[str, Any]]:
        scores, bandit_scores = super()._score_models(item, context, round_index)
        neighbors = self._bounded_neighbors(item)
        graph_scores: dict[str, ModelScore] = {}
        for model_id, score in scores.items():
            history = self._graph_history[model_id]
            observed = [
                (history[node_id], weight)
                for node_id, weight in neighbors
                if node_id in history
            ]
            total_weight = sum(weight for _value, weight in observed)
            numerator = score.smoothed_lower_bound + self.graph_config.regularization * sum(
                value * weight for value, weight in observed
            )
            denominator = 1.0 + self.graph_config.regularization * total_weight
            regularized = min(1.0, max(0.0, numerator / denominator))
            graph_scores[model_id] = replace(
                score,
                graph_regularized_lower_bound=regularized,
                graph_neighbor_count=len(observed),
                graph_neighbor_weight=total_weight,
            )
        return graph_scores, bandit_scores

    def _selection_lower_bound(self, score: ModelScore) -> float:
        if score.graph_regularized_lower_bound is None:
            return score.smoothed_lower_bound
        return score.graph_regularized_lower_bound

    def route(self, item: AnnotationItem) -> RoutingResult:
        result = super().route(item)
        node_id = self._node_id(item)
        for model_id, score in result.scores.items():
            self._graph_history[model_id][node_id] = self._selection_lower_bound(score)
        return result

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, Any],
        gold_label: str,
    ) -> None:
        super().observe_complete_feedback(item, responses, gold_label)
        context = self._context(item)
        scores, _bandit_scores = self._score_models(item, context, self.round_index)
        node_id = self._node_id(item)
        for model_id, score in scores.items():
            self._graph_history[model_id][node_id] = self._selection_lower_bound(score)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["graph_config"] = asdict(self.graph_config)
        state["graph_history"] = {
            model_id: dict(sorted(history.items()))
            for model_id, history in sorted(self._graph_history.items())
        }
        return state

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            if state.get("graph_config") != asdict(self.graph_config):
                raise CheckpointError("checkpoint graph configuration does not match this router")
            graph_history = state["graph_history"]
            if set(graph_history) != set(self.models):
                raise CheckpointError("checkpoint graph model pool does not match this router")
            validated: dict[str, dict[str, float]] = {}
            for model_id, history in graph_history.items():
                if not isinstance(history, dict):
                    raise CheckpointError("checkpoint graph history must be an object")
                converted = {str(node_id): float(value) for node_id, value in history.items()}
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in converted.values()
                ):
                    raise CheckpointError("checkpoint graph history contains invalid probabilities")
                validated[model_id] = converted
        except CheckpointError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid graph checkpoint: {exc}") from exc
        super().load_checkpoint(path)
        self._graph_history = validated
