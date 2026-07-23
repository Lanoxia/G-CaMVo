"""Defensible OpTC binary-task builders for real and no-key experiments."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from camvo.security.optc import (
    OptcAttackLabel,
    OptcCorrelationEdge,
    OptcEvent,
    build_optc_event_correlations,
    load_optc_attack_labels,
    load_optc_scenarios,
    sample_optc_events_by_hash,
)
from camvo.types import AnnotationItem

OPTC_BINARY_LABELS: tuple[str, ...] = ("benign", "malicious")


@dataclass(frozen=True, slots=True)
class OptcBinaryDataset:
    items: tuple[AnnotationItem, ...]
    correlations: tuple[OptcCorrelationEdge, ...]
    stats: dict[str, object]
    simulation_only: bool

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("OpTC binary dataset must not be empty")
        if any(item.labels != OPTC_BINARY_LABELS for item in self.items):
            raise ValueError("OpTC binary item has unexpected labels")


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _difficulty(event: OptcEvent) -> float:
    pair = (event.object_type, event.action)
    if pair in {("THREAD", "REMOTE_CREATE"), ("REGISTRY", "ADD"), ("SHELL", "COMMAND")}:
        return -0.25
    if pair in {("PROCESS", "CREATE"), ("FLOW", "START"), ("FILE", "CREATE")}:
        return 0.05
    if event.object_type in {"FLOW", "FILE", "MODULE"}:
        return 0.35
    return 0.20


def _difficulty_bucket(value: float) -> str:
    if value <= -0.1:
        return "easy"
    if value >= 0.3:
        return "hard"
    return "medium"


def _to_item(event: OptcEvent, gold_label: str, *, simulation_source: bool) -> AnnotationItem:
    item = event.to_annotation_item(labels=OPTC_BINARY_LABELS, gold_label=gold_label)
    difficulty = _difficulty(event)
    return replace(
        item,
        metadata={
            **item.metadata,
            "difficulty_score": difficulty,
            "difficulty": _difficulty_bucket(difficulty),
            "simulation_source": simulation_source,
        },
    )


def _balanced_interleave(
    positives: Iterable[AnnotationItem],
    negatives: Iterable[AnnotationItem],
) -> tuple[AnnotationItem, ...]:
    positive = list(positives)
    negative = list(negatives)
    output: list[AnnotationItem] = []
    for index in range(max(len(positive), len(negative))):
        if index < len(negative):
            output.append(negative[index])
        if index < len(positive):
            output.append(positive[index])
    return tuple(output)


def _select_labels(
    labels: tuple[OptcAttackLabel, ...],
    count: int,
    seed: int,
) -> list[OptcAttackLabel]:
    if count <= 0:
        raise ValueError("count must be positive")
    return sorted(labels, key=lambda label: _stable_rank(seed, label.event_id))[:count]


def build_optc_label_graph_simulation_dataset(
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    scenario_id: str = "optc-day3-malicious-upgrade",
    utc_offset_minutes: int = -240,
    padding_minutes: int = 10,
    max_positive_items: int = 2_000,
    seed: int = 17,
    correlation_window_minutes: int = 30,
) -> OptcBinaryDataset:
    """Use real attack IDs/topology plus matched synthetic benign controls.

    This fallback is explicitly simulation-only. It exists so every routing,
    graph, metric, and sweep path can be validated while public raw downloads
    are quota-limited; it must never be presented as a real OpTC classifier
    result.
    """

    scenario = load_optc_scenarios(scenario_manifest)[scenario_id]
    start_ms, end_ms = scenario.epoch_window(
        utc_offset_minutes=utc_offset_minutes,
        padding_minutes=padding_minutes,
    )
    index = load_optc_attack_labels(
        labels_path,
        hostnames=scenario.hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        strict=True,
    )
    selected_labels = _select_labels(
        index.labels,
        min(max_positive_items, len(index.labels)),
        seed,
    )
    positive_events = [
        label.to_minimal_event(source_path=str(labels_path)) for label in selected_labels
    ]
    base_time = min(event.timestamp_ms for event in positive_events) - 8 * 24 * 60 * 60 * 1000
    benign_events: list[OptcEvent] = []
    for position, positive in enumerate(positive_events):
        group = _stable_rank(seed + 1, positive.event_id) % 80
        benign_events.append(
            OptcEvent(
                timestamp_ms=base_time + position * 1_000,
                event_id=f"synthetic-benign-{position:06d}",
                hostname=f"SYNTHETIC-BENIGN-{group % 8:02d}",
                object_id=f"synthetic-object-{position:06d}",
                object_type=positive.object_type,
                action=positive.action,
                actor_id=f"synthetic-actor-{group:03d}",
                pid=1_000 + group,
                principal="SYNTHETIC\\benign-user",
                properties={"simulation_control": True},
                source_path="synthetic matched benign control",
            )
        )
    window_ms = correlation_window_minutes * 60 * 1000
    correlations = (
        *build_optc_event_correlations(positive_events, max_time_gap_ms=window_ms),
        *build_optc_event_correlations(benign_events, max_time_gap_ms=window_ms),
    )
    positives = [_to_item(event, "malicious", simulation_source=True) for event in positive_events]
    negatives = [_to_item(event, "benign", simulation_source=True) for event in benign_events]
    return OptcBinaryDataset(
        items=_balanced_interleave(positives, negatives),
        correlations=tuple(correlations),
        stats={
            "name": "OpTC label-graph simulation fallback",
            "scenario_id": scenario_id,
            "positive_items": len(positives),
            "negative_items": len(negatives),
            "positive_source": "real best-effort OpTC event IDs and entity topology",
            "negative_source": "synthetic object/action-matched controls",
            "label_load_stats": asdict(index.stats),
            "correlation_edges": len(correlations),
        },
        simulation_only=True,
    )


def _matched_negative_sample(
    candidates: Iterable[OptcEvent],
    positive_events: list[OptcEvent],
    target_count: int,
    seed: int,
) -> list[OptcEvent]:
    positive_counts = Counter((event.object_type, event.action) for event in positive_events)
    total_positive = sum(positive_counts.values())
    quotas = {
        pair: max(1, round(target_count * count / total_positive))
        for pair, count in positive_counts.items()
    }
    by_pair: dict[tuple[str, str], list[OptcEvent]] = defaultdict(list)
    for event in candidates:
        by_pair[(event.object_type, event.action)].append(event)
    selected: list[OptcEvent] = []
    used: set[str] = set()
    for pair in sorted(quotas):
        ordered = sorted(
            by_pair.get(pair, []),
            key=lambda event: _stable_rank(seed, event.event_id),
        )
        for event in ordered[: quotas[pair]]:
            selected.append(event)
            used.add(event.event_id)
    if len(selected) < target_count:
        remaining = sorted(
            (event for event in candidates if event.event_id not in used),
            key=lambda event: _stable_rank(seed + 1, event.event_id),
        )
        selected.extend(remaining[: target_count - len(selected)])
    return sorted(selected[:target_count], key=lambda event: (event.timestamp_ms, event.event_id))


def build_optc_real_binary_dataset(
    attack_path: str | Path,
    benign_path: str | Path,
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    scenario_id: str = "optc-day3-malicious-upgrade",
    utc_offset_minutes: int = -240,
    padding_minutes: int = 10,
    max_positive_items: int = 2_000,
    negative_ratio: float = 1.0,
    seed: int = 17,
    correlation_window_minutes: int = 30,
) -> OptcBinaryDataset:
    """Build the main binary task from labeled attack rows and benign-period rows."""

    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    scenario = load_optc_scenarios(scenario_manifest)[scenario_id]
    start_ms, end_ms = scenario.epoch_window(
        utc_offset_minutes=utc_offset_minutes,
        padding_minutes=padding_minutes,
    )
    label_index = load_optc_attack_labels(
        labels_path,
        hostnames=scenario.hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        strict=True,
    )
    positive_sample = sample_optc_events_by_hash(
        attack_path,
        sample_size=max_positive_items,
        seed=seed,
        hostnames=scenario.hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        predicate=lambda event: label_index.contains(event.event_id),
        strict=False,
    )
    positive_events = list(positive_sample.events)
    negative_count = max(1, round(len(positive_events) * negative_ratio))
    positive_pairs = {(event.object_type, event.action) for event in positive_events}
    benign_oversample = sample_optc_events_by_hash(
        benign_path,
        sample_size=max(negative_count, negative_count * 4),
        seed=seed + 1,
        predicate=lambda event: (event.object_type, event.action) in positive_pairs,
        strict=False,
    )
    benign_events = _matched_negative_sample(
        benign_oversample.events,
        positive_events,
        negative_count,
        seed + 2,
    )
    if len(benign_events) < negative_count:
        raise ValueError(
            f"only {len(benign_events)} matched benign events were available; "
            f"needed {negative_count}"
        )
    window_ms = correlation_window_minutes * 60 * 1000
    correlations = (
        *build_optc_event_correlations(positive_events, max_time_gap_ms=window_ms),
        *build_optc_event_correlations(benign_events, max_time_gap_ms=window_ms),
    )
    positives = [_to_item(event, "malicious", simulation_source=False) for event in positive_events]
    negatives = [_to_item(event, "benign", simulation_source=False) for event in benign_events]
    return OptcBinaryDataset(
        items=_balanced_interleave(positives, negatives),
        correlations=tuple(correlations),
        stats={
            "name": "DARPA OpTC binary incident-event detection",
            "scenario_id": scenario_id,
            "utc_offset_minutes": utc_offset_minutes,
            "padding_minutes": padding_minutes,
            "correlation_window_minutes": correlation_window_minutes,
            "positive_items": len(positives),
            "negative_items": len(negatives),
            "positive_source": "community-labeled attack-period eCAR IDs",
            "negative_source": "official benign-period eCAR, object/action matched",
            "unlabeled_attack_rows_treated_as_benign": False,
            "temporal_domain_shift_warning": (
                "Benign controls come from the earlier benign collection period."
            ),
            "positive_sample_stats": asdict(positive_sample.stats),
            "benign_sample_stats": asdict(benign_oversample.stats),
            "correlation_edges": len(correlations),
        },
        simulation_only=False,
    )
