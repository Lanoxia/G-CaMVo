"""Cluster-aware uncertainty estimates for correlated security events."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from camvo.security.metrics import classification_metrics


@dataclass(frozen=True, slots=True)
class PairedBootstrapDelta:
    metric: str
    candidate_minus_reference: float
    bootstrap_mean: float
    confidence_level: float
    lower: float
    upper: float
    probability_candidate_better: float
    iterations: int
    clusters: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_cluster_bootstrap_delta(
    gold: Sequence[str],
    candidate: Sequence[str],
    reference: Sequence[str],
    labels: tuple[str, ...],
    clusters: Sequence[str],
    *,
    metric: str = "macro_f1",
    iterations: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 17,
) -> PairedBootstrapDelta:
    """Paired bootstrap that resamples whole documents/incidents, not rows."""

    if not (len(gold) == len(candidate) == len(reference) == len(clusters)):
        raise ValueError("paired bootstrap vectors must have equal lengths")
    if not gold:
        raise ValueError("paired bootstrap vectors must not be empty")
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be 'accuracy' or 'macro_f1'")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        key = str(cluster).strip()
        if not key:
            raise ValueError("cluster IDs must not be empty")
        grouped[key].append(index)
    cluster_ids = sorted(grouped)
    rng = random.Random(seed)

    def score(indices: Sequence[int], predictions: Sequence[str]) -> float:
        metrics = classification_metrics(
            [gold[index] for index in indices],
            [predictions[index] for index in indices],
            labels,
        )
        return float(getattr(metrics, metric))

    full_indices = list(range(len(gold)))
    observed = score(full_indices, candidate) - score(full_indices, reference)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        indices = [index for cluster in sampled_clusters for index in grouped[cluster]]
        deltas.append(score(indices, candidate) - score(indices, reference))
    ordered = sorted(deltas)
    tail = (1.0 - confidence_level) / 2.0
    return PairedBootstrapDelta(
        metric=metric,
        candidate_minus_reference=observed,
        bootstrap_mean=float(np.mean(deltas)),
        confidence_level=confidence_level,
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        probability_candidate_better=sum(delta > 0 for delta in deltas) / iterations,
        iterations=iterations,
        clusters=len(cluster_ids),
    )
