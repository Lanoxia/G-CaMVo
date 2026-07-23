"""Safe, actively learned causal graph augmentation for CaMVo.

The router preserves the paper-level CaMVo contract: a subset is chosen before
the current responses are visible, that subset performs one weighted vote, and
the ordinary CaMVo online states are updated from the model-only consensus.

The graph is a second, conservative decision head.  Candidate past edges may
come from a supplied provenance graph or be proposed online from repeated
entities.  Relation-specific label transitions and edge utility gates are
learned from delayed audited feedback.  A graph message is allowed to change a
decision only when the relation gate is mature and the message passes a set of
observable do-no-harm checks; otherwise the result is exactly ordinary CaMVo.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from camvo.aggregation import weighted_vote
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.exceptions import CheckpointError
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class AdaptiveGraphConfig:
    """Dataset- and model-agnostic controls for online graph learning."""

    transition_prior: float = 1.0
    gate_prior_success: float = 2.0
    gate_prior_failure: float = 2.0
    min_transition_observations: float = 8.0
    min_gate_observations: float = 12.0
    gate_lower_z: float = 1.0
    gate_activation_threshold: float = 0.5
    gate_decay: float = 0.995
    max_graph_blend: float = 0.35
    max_edge_weight: float = 2.0
    max_total_parent_weight: float = 6.0
    max_parents: int = 8
    protect_base_margin: float = 0.30
    max_js_divergence: float = 0.30
    require_entropy_reduction: bool = True
    minimum_graph_margin: float = 0.05
    pseudo_history_weight: float = 0.0
    dynamic_entity_edges: bool = True
    recent_nodes_per_entity: int = 2
    entity_memory_size: int = 4_096
    dynamic_edge_weight: float = 0.35

    def __post_init__(self) -> None:
        numeric = asdict(self)
        for key in (
            "transition_prior",
            "gate_prior_success",
            "gate_prior_failure",
            "min_transition_observations",
            "min_gate_observations",
            "gate_lower_z",
            "max_edge_weight",
            "max_total_parent_weight",
            "dynamic_edge_weight",
        ):
            value = float(numeric[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "gate_activation_threshold",
            "gate_decay",
            "max_graph_blend",
            "protect_base_margin",
            "max_js_divergence",
            "minimum_graph_margin",
            "pseudo_history_weight",
        ):
            value = float(numeric[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.max_parents <= 0:
            raise ValueError("max_parents must be positive")
        if self.recent_nodes_per_entity <= 0 or self.entity_memory_size <= 0:
            raise ValueError("entity memory sizes must be positive")


@dataclass(slots=True)
class _PendingGraphDecision:
    item: AnnotationItem
    base_posterior: np.ndarray
    graph_posterior: np.ndarray | None
    fused_posterior: np.ndarray
    parents: tuple[tuple[str, float, tuple[str, ...]], ...]
    applied_relations: tuple[str, ...]
    fallback_reason: str | None


class AdaptiveGraphCaMVoRouter(CaMVoRouter):
    """CaMVo with an online, relation-typed and safely gated causal graph."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        graph_config: AdaptiveGraphConfig,
        neighborhood: Callable[[AnnotationItem], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(models, embedder, config)
        self.graph_config = graph_config
        self.neighborhood = neighborhood or (lambda _item: {})
        self._label_history: dict[str, tuple[tuple[str, ...], np.ndarray, bool]] = {}
        self._transition_counts: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        self._transition_observations: dict[tuple[str, ...], dict[str, float]] = {}
        self._gate_counts: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        self._gate_observations: dict[tuple[str, ...], dict[str, float]] = {}
        self._class_counts: dict[tuple[str, ...], np.ndarray] = {}
        self._entity_index: dict[str, deque[str]] = defaultdict(deque)
        self._node_signatures: dict[str, tuple[str, ...]] = {}
        self._node_order: deque[str] = deque()
        self._pending: dict[str, _PendingGraphDecision] = {}
        self._last_decision: _PendingGraphDecision | None = None

    @staticmethod
    def _node_id(item: AnnotationItem) -> str:
        return str(item.metadata.get("graph_node_id", item.metadata.get("event_id", item.item_id)))

    @staticmethod
    def _decode_edge(raw: Any) -> tuple[float, tuple[str, ...]]:
        if isinstance(raw, Mapping):
            weight = float(raw.get("weight", 0.0))
            relation_values = raw.get("relations", raw.get("relation", "default"))
            if isinstance(relation_values, str):
                relations = (relation_values.strip() or "default",)
            else:
                relations = tuple(
                    sorted({str(value).strip() or "default" for value in relation_values})
                )
        else:
            weight = float(raw)
            relations = ("default",)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("neighborhood returned an invalid edge weight")
        return weight, relations or ("default",)

    @staticmethod
    def _values(metadata: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
        values: set[str] = set()
        for key in keys:
            raw = metadata.get(key)
            if raw is None:
                continue
            if isinstance(raw, (str, int, float)):
                candidates: Sequence[Any] = (raw,)
            elif isinstance(raw, Sequence):
                candidates = raw
            else:
                continue
            values.update(str(value).strip().casefold() for value in candidates if str(value).strip())
        return tuple(sorted(values))

    def _entity_signatures(self, item: AnnotationItem) -> tuple[str, ...]:
        metadata = item.metadata
        signatures: set[str] = set()
        document = str(metadata.get("document_id", "")).strip().casefold()
        if document:
            signatures.add(f"document:{document}")
            hopper = metadata.get("hopper_index")
            if hopper is not None:
                signatures.add(f"hopper:{document}:{hopper}")
        key_groups = {
            "host": ("hosts", "hostname", "host", "computer_name"),
            "process": ("process_ids", "process_id", "process_guid", "process_name"),
            "network": (
                "source_ip",
                "destination_ip",
                "src_ip",
                "dst_ip",
                "ip_addresses",
            ),
            "user": ("user", "username", "users"),
        }
        for relation, keys in key_groups.items():
            signatures.update(f"{relation}:{value}" for value in self._values(metadata, *keys))
        return tuple(sorted(signatures))

    @staticmethod
    def _signature_relation(signature: str) -> str:
        prefix = signature.split(":", 1)[0]
        return f"learned_{prefix}"

    def _candidate_parents(
        self, item: AnnotationItem
    ) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
        merged: dict[str, tuple[float, set[str]]] = {}
        for parent_id, raw in self.neighborhood(item).items():
            weight, relations = self._decode_edge(raw)
            if weight <= 0 or str(parent_id) == self._node_id(item):
                continue
            old_weight, old_relations = merged.get(str(parent_id), (0.0, set()))
            merged[str(parent_id)] = (old_weight + weight, old_relations | set(relations))
        if self.graph_config.dynamic_entity_edges:
            for signature in self._entity_signatures(item):
                history = self._entity_index.get(signature, ())
                for rank, parent_id in enumerate(
                    reversed(tuple(history)[-self.graph_config.recent_nodes_per_entity :]),
                    start=1,
                ):
                    if parent_id == self._node_id(item):
                        continue
                    weight = self.graph_config.dynamic_edge_weight / rank
                    old_weight, old_relations = merged.get(parent_id, (0.0, set()))
                    merged[parent_id] = (
                        old_weight + weight,
                        old_relations | {self._signature_relation(signature)},
                    )
        rows = [
            (parent_id, min(weight, self.graph_config.max_edge_weight), tuple(sorted(relations)))
            for parent_id, (weight, relations) in merged.items()
            if parent_id in self._label_history and weight > 0
        ]
        rows.sort(key=lambda row: (-row[1], row[0], row[2]))
        accepted: list[tuple[str, float, tuple[str, ...]]] = []
        total = 0.0
        for parent_id, weight, relations in rows:
            if len(accepted) >= self.graph_config.max_parents:
                break
            remaining = self.graph_config.max_total_parent_weight - total
            if remaining <= 0:
                break
            bounded = min(weight, remaining)
            accepted.append((parent_id, bounded, relations or ("default",)))
            total += bounded
        return tuple(accepted)

    def _register_node(self, item: AnnotationItem) -> None:
        node_id = self._node_id(item)
        signatures = self._entity_signatures(item)
        self._node_signatures[node_id] = signatures
        self._node_order.append(node_id)
        for signature in signatures:
            rows = self._entity_index[signature]
            if not rows or rows[-1] != node_id:
                rows.append(node_id)
        while len(self._node_order) > self.graph_config.entity_memory_size:
            expired = self._node_order.popleft()
            for signature in self._node_signatures.pop(expired, ()):
                rows = self._entity_index.get(signature)
                if rows is None:
                    continue
                try:
                    rows.remove(expired)
                except ValueError:
                    pass
                if not rows:
                    self._entity_index.pop(signature, None)

    def _transition(self, labels: tuple[str, ...], relation: str) -> np.ndarray:
        rows = self._transition_counts.setdefault(labels, {})
        observations = self._transition_observations.setdefault(labels, {})
        if relation not in rows:
            rows[relation] = np.full(
                (len(labels), len(labels)), self.graph_config.transition_prior, dtype=float
            )
            observations[relation] = 0.0
        return rows[relation]

    def _gate(self, labels: tuple[str, ...], relation: str) -> np.ndarray:
        rows = self._gate_counts.setdefault(labels, {})
        observations = self._gate_observations.setdefault(labels, {})
        if relation not in rows:
            rows[relation] = np.asarray(
                [self.graph_config.gate_prior_failure, self.graph_config.gate_prior_success],
                dtype=float,
            )
            observations[relation] = 0.0
        return rows[relation]

    def _base_prior(self, labels: tuple[str, ...]) -> np.ndarray:
        counts = self._class_counts.get(labels)
        if counts is None:
            return np.full(len(labels), 1.0 / len(labels), dtype=float)
        return counts / counts.sum()

    def _relation_prediction(
        self, labels: tuple[str, ...], relation: str, parent: np.ndarray
    ) -> np.ndarray:
        matrix = self._transition(labels, relation)
        transition = matrix / matrix.sum(axis=1, keepdims=True)
        estimate = parent @ transition
        observations = self._transition_observations[labels][relation]
        maturity = observations / (observations + self.graph_config.min_transition_observations)
        prediction = maturity * estimate + (1.0 - maturity) * self._base_prior(labels)
        return prediction / prediction.sum()

    def _gate_utility(self, labels: tuple[str, ...], relation: str) -> float:
        observations = self._gate_observations.get(labels, {}).get(relation, 0.0)
        if observations < self.graph_config.min_gate_observations:
            return 0.0
        failure, success = self._gate(labels, relation)
        total = failure + success
        mean = success / total
        variance = success * failure / (total * total * (total + 1.0))
        lower = mean - self.graph_config.gate_lower_z * math.sqrt(max(variance, 0.0))
        threshold = self.graph_config.gate_activation_threshold
        if lower <= threshold:
            return 0.0
        return min(1.0, (lower - threshold) / max(1e-12, 1.0 - threshold))

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        return -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))

    @staticmethod
    def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mean = 0.5 * (left + right)
        return 0.5 * float(
            np.sum(left * np.log(np.clip(left / mean, 1e-12, None)))
            + np.sum(right * np.log(np.clip(right / mean, 1e-12, None)))
        )

    @staticmethod
    def _margin(probabilities: np.ndarray) -> float:
        ordered = np.sort(probabilities)
        return float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0

    def _base_vote_posterior(
        self,
        labels: tuple[str, ...],
        responses: Mapping[str, str],
        scores: Mapping[str, ModelScore],
        subset: Sequence[str],
    ) -> np.ndarray:
        posterior = np.zeros(len(labels), dtype=float)
        for model_id in subset:
            posterior[labels.index(responses[model_id])] += scores[model_id].vote_weight
        if posterior.sum() <= 0:
            return np.full(len(labels), 1.0 / len(labels), dtype=float)
        return posterior / posterior.sum()

    def _graph_message(
        self,
        labels: tuple[str, ...],
        parents: Sequence[tuple[str, float, tuple[str, ...]]],
    ) -> tuple[np.ndarray | None, float, tuple[str, ...]]:
        total = 0.0
        pooled = np.zeros(len(labels), dtype=float)
        used: set[str] = set()
        for parent_id, edge_weight, relations in parents:
            history_labels, parent, audited = self._label_history[parent_id]
            if history_labels != labels:
                continue
            history_weight = 1.0 if audited else self.graph_config.pseudo_history_weight
            if history_weight <= 0:
                continue
            for relation in relations:
                utility = self._gate_utility(labels, relation)
                if utility <= 0:
                    continue
                weight = history_weight * edge_weight * utility / len(relations)
                pooled += weight * self._relation_prediction(labels, relation, parent)
                total += weight
                used.add(relation)
        if total <= 0:
            return None, 0.0, ()
        return pooled / total, total, tuple(sorted(used))

    def _safe_fuse(
        self, base: np.ndarray, graph: np.ndarray | None, evidence_weight: float
    ) -> tuple[np.ndarray, str | None]:
        if graph is None or evidence_weight <= 0:
            return base.copy(), "no_mature_helpful_edge"
        base_winner = int(np.argmax(base))
        graph_winner = int(np.argmax(graph))
        if self._margin(graph) < self.graph_config.minimum_graph_margin:
            return base.copy(), "graph_message_uncertain"
        if (
            graph_winner != base_winner
            and self._margin(base) >= self.graph_config.protect_base_margin
        ):
            return base.copy(), "protected_confident_base_vote"
        if self._js_divergence(base, graph) > self.graph_config.max_js_divergence:
            return base.copy(), "graph_message_out_of_distribution"
        if (
            self.graph_config.require_entropy_reduction
            and self._entropy(graph) >= self._entropy(base) - 1e-12
        ):
            return base.copy(), "graph_did_not_reduce_uncertainty"
        strength = min(1.0, evidence_weight / self.graph_config.max_total_parent_weight)
        alpha = self.graph_config.max_graph_blend * strength
        fused = (1.0 - alpha) * base + alpha * graph
        fused /= fused.sum()
        return fused, None

    @staticmethod
    def _winner(labels: tuple[str, ...], posterior: np.ndarray) -> str:
        maximum = float(np.max(posterior))
        return next(
            label
            for index, label in enumerate(labels)
            if math.isclose(float(posterior[index]), maximum)
        )

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        base = self._base_vote_posterior(item.labels, responses, scores, successful_ids)
        parents = self._candidate_parents(item)
        graph, evidence_weight, relations = self._graph_message(item.labels, parents)
        fused, fallback = self._safe_fuse(base, graph, evidence_weight)
        self._last_decision = _PendingGraphDecision(
            item=item,
            base_posterior=base,
            graph_posterior=None if graph is None else graph.copy(),
            fused_posterior=fused,
            parents=parents,
            applied_relations=relations if fallback is None else (),
            fallback_reason=fallback,
        )
        return self._winner(item.labels, fused)

    def _reward_reference_label(
        self,
        item: AnnotationItem,
        final_label: str,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        label, _ties = weighted_vote(
            responses,
            {model_id: scores[model_id].vote_weight for model_id in successful_ids},
            item.labels,
        )
        return label

    def route(self, item: AnnotationItem) -> RoutingResult:
        self._last_decision = None
        result = super().route(item)
        decision = self._last_decision
        if decision is None:
            raise RuntimeError("adaptive graph decision was not recorded")
        node_id = self._node_id(item)
        self._pending[node_id] = decision
        self._label_history[node_id] = (
            item.labels,
            decision.base_posterior.copy(),
            False,
        )
        self._register_node(item)
        trace = {
            "component": "adaptive_graph",
            "candidate_parents": len(decision.parents),
            "active_relations": list(decision.applied_relations),
            "fallback_reason": decision.fallback_reason,
            "base_label": self._winner(item.labels, decision.base_posterior),
            "graph_label": (
                None
                if decision.graph_posterior is None
                else self._winner(item.labels, decision.graph_posterior)
            ),
            "final_label": result.label,
        }
        return replace(
            result,
            posterior={
                label: float(decision.fused_posterior[index])
                for index, label in enumerate(item.labels)
            },
            graph_evidence_weight=float(len(decision.applied_relations)),
            routing_trace=result.routing_trace + (trace,),
        )

    def _update_graph_from_feedback(
        self,
        decision: _PendingGraphDecision,
        gold_label: str,
        *,
        weight: float,
    ) -> None:
        labels = decision.item.labels
        if gold_label not in labels:
            raise ValueError("gold_label is outside the item's label space")
        gold = np.zeros(len(labels), dtype=float)
        gold[labels.index(gold_label)] = 1.0
        class_prior_before = self._base_prior(labels).copy()
        class_counts = self._class_counts.setdefault(labels, np.ones(len(labels), dtype=float))
        class_counts += weight * gold
        for parent_id, edge_weight, relations in decision.parents:
            raw_parent = self._label_history.get(parent_id)
            if raw_parent is None or raw_parent[0] != labels:
                continue
            parent = raw_parent[1]
            parent_weight = 1.0 if raw_parent[2] else self.graph_config.pseudo_history_weight
            if parent_weight <= 0:
                continue
            share = weight * parent_weight * edge_weight / len(relations)
            for relation in relations:
                prediction = self._relation_prediction(labels, relation, parent)
                observations = self._transition_observations[labels][relation]
                if observations >= self.graph_config.min_transition_observations:
                    # Learn whether this relation carries information beyond the
                    # stream's class prior.  Comparing every edge directly with
                    # the often very strong model panel would permanently prune
                    # edges that are useful precisely on the panel's uncertain
                    # rounds.  The separate safe-fusion guard still decides
                    # whether that information may alter the current vote.
                    success = float(
                        prediction[labels.index(gold_label)]
                        > class_prior_before[labels.index(gold_label)] + 1e-12
                    )
                    gate = self._gate(labels, relation)
                    prior = np.asarray(
                        [
                            self.graph_config.gate_prior_failure,
                            self.graph_config.gate_prior_success,
                        ],
                        dtype=float,
                    )
                    gate[:] = prior + self.graph_config.gate_decay * (gate - prior)
                    gate[int(success)] += share
                    self._gate_observations[labels][relation] = (
                        self.graph_config.gate_decay
                        * self._gate_observations[labels][relation]
                        + share
                    )
                self._transition(labels, relation)[:] += share * np.outer(parent, gold)
                self._transition_observations[labels][relation] += share
        node_id = self._node_id(decision.item)
        self._label_history[node_id] = (labels, gold, True)

    def observe_graph_feedback(self, item: AnnotationItem, gold_label: str) -> None:
        """Update the active graph after delayed analyst feedback, with no model call."""

        node_id = self._node_id(item)
        decision = self._pending.pop(node_id, None)
        if decision is None:
            raise ValueError(f"no pending graph decision for node {node_id!r}")
        self._update_graph_from_feedback(decision, gold_label, weight=1.0)

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        """Warm-start CaMVo and the graph from one fully cached audited row."""

        parents = self._candidate_parents(item)
        super().observe_complete_feedback(item, responses, gold_label)
        context = self._context(item)
        scores, _bandit = self._score_models(item, context, self.round_index)
        response_labels = {model_id: response.label for model_id, response in responses.items()}
        model_ids = tuple(sorted(responses))
        base = self._base_vote_posterior(item.labels, response_labels, scores, model_ids)
        decision = _PendingGraphDecision(
            item=item,
            base_posterior=base,
            graph_posterior=None,
            fused_posterior=base.copy(),
            parents=parents,
            applied_relations=(),
            fallback_reason="calibration_feedback",
        )
        self._update_graph_from_feedback(decision, gold_label, weight=1.0)
        self._register_node(item)

    def reset_graph_history(self) -> None:
        """Clear split-local nodes while retaining learned relation parameters."""

        self._label_history = {}
        self._entity_index = defaultdict(deque)
        self._node_signatures = {}
        self._node_order = deque()
        self._pending = {}
        self._last_decision = None

    def graph_diagnostics(self) -> dict[str, Any]:
        relations: dict[str, Any] = {}
        for labels, rows in self._gate_counts.items():
            key = "|".join(labels)
            relations[key] = {}
            for relation, counts in sorted(rows.items()):
                failure, success = map(float, counts)
                relations[key][relation] = {
                    "gate_mean": success / (failure + success),
                    "gate_observations": self._gate_observations[labels][relation],
                    "transition_observations": self._transition_observations[labels][relation],
                    "active_utility": self._gate_utility(labels, relation),
                }
        return {"relations": relations, "pending_feedback": len(self._pending)}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["adaptive_graph_config"] = asdict(self.graph_config)
        state["adaptive_graph_transition_counts"] = {
            "\u0000".join(labels): {
                relation: matrix.tolist() for relation, matrix in sorted(rows.items())
            }
            for labels, rows in sorted(self._transition_counts.items())
        }
        state["adaptive_graph_transition_observations"] = {
            "\u0000".join(labels): dict(sorted(rows.items()))
            for labels, rows in sorted(self._transition_observations.items())
        }
        state["adaptive_graph_gate_counts"] = {
            "\u0000".join(labels): {
                relation: counts.tolist() for relation, counts in sorted(rows.items())
            }
            for labels, rows in sorted(self._gate_counts.items())
        }
        state["adaptive_graph_gate_observations"] = {
            "\u0000".join(labels): dict(sorted(rows.items()))
            for labels, rows in sorted(self._gate_observations.items())
        }
        state["adaptive_graph_class_counts"] = {
            "\u0000".join(labels): counts.tolist()
            for labels, counts in sorted(self._class_counts.items())
        }
        state["adaptive_graph_label_history"] = {
            node_id: {
                "labels": list(labels),
                "posterior": posterior.tolist(),
                "audited": audited,
            }
            for node_id, (labels, posterior, audited) in sorted(self._label_history.items())
        }
        return state

    def load_checkpoint(self, path: str | Path) -> None:
        super().load_checkpoint(path)
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if raw["adaptive_graph_config"] != asdict(self.graph_config):
                raise CheckpointError("checkpoint adaptive graph configuration does not match")

            def label_key(value: str) -> tuple[str, ...]:
                return tuple(value.split("\u0000"))

            self._transition_counts = {
                label_key(key): {
                    relation: np.asarray(matrix, dtype=float)
                    for relation, matrix in rows.items()
                }
                for key, rows in raw["adaptive_graph_transition_counts"].items()
            }
            self._transition_observations = {
                label_key(key): {relation: float(value) for relation, value in rows.items()}
                for key, rows in raw["adaptive_graph_transition_observations"].items()
            }
            self._gate_counts = {
                label_key(key): {
                    relation: np.asarray(counts, dtype=float)
                    for relation, counts in rows.items()
                }
                for key, rows in raw["adaptive_graph_gate_counts"].items()
            }
            self._gate_observations = {
                label_key(key): {relation: float(value) for relation, value in rows.items()}
                for key, rows in raw["adaptive_graph_gate_observations"].items()
            }
            self._class_counts = {
                label_key(key): np.asarray(counts, dtype=float)
                for key, counts in raw["adaptive_graph_class_counts"].items()
            }
            self._label_history = {
                str(node_id): (
                    tuple(value["labels"]),
                    np.asarray(value["posterior"], dtype=float),
                    bool(value["audited"]),
                )
                for node_id, value in raw["adaptive_graph_label_history"].items()
            }
            self._entity_index = defaultdict(deque)
            self._node_signatures = {}
            self._node_order = deque()
            self._pending = {}
            self._last_decision = None
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid adaptive graph checkpoint: {exc}") from exc
