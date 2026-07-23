"""Structure-preserving sampling for graph security experiments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from camvo.types import AnnotationItem


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def graph_preserving_group_sample(
    items: Iterable[AnnotationItem],
    max_items: int,
    seed: int,
    *,
    group_key: Callable[[AnnotationItem], str],
    order_key: Callable[[AnnotationItem], Any],
    label_key: str = "gold_label",
    connectivity_bonus: float = 0.35,
) -> list[AnnotationItem]:
    """Select whole correlated groups while approximately preserving label mix.

    Random row-level sampling destroys sparse provenance/document graphs.  This
    sampler greedily packs complete groups, balancing the source label
    distribution and rewarding multi-node groups.  If the final free capacity
    cannot fit any complete group, it takes an ordered prefix from one group;
    the prefix remains contiguous under the caller-provided causal order.

    The output order is the group-selection order followed by ``order_key``
    within each group, so an online router sees earlier events before later
    events from the same incident/document.
    """

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if not math.isfinite(connectivity_bonus) or connectivity_bonus < 0:
        raise ValueError("connectivity_bonus must be finite and non-negative")
    grouped: dict[str, list[AnnotationItem]] = defaultdict(list)
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"duplicate item ID: {item.item_id}")
        seen.add(item.item_id)
        group = str(group_key(item)).strip()
        if not group:
            raise ValueError("group_key returned an empty value")
        if item.metadata.get(label_key) not in item.labels:
            raise ValueError(f"item {item.item_id!r} lacks a valid {label_key!r}")
        grouped[group].append(item)
    if not grouped:
        raise ValueError("items must not be empty")

    ordered_groups = {
        group: sorted(values, key=lambda item: (order_key(item), item.item_id))
        for group, values in grouped.items()
    }
    total_items = sum(len(values) for values in ordered_groups.values())
    if max_items >= total_items:
        return [
            item
            for group in sorted(ordered_groups, key=lambda key: _stable_rank(seed, key))
            for item in ordered_groups[group]
        ]

    source_counts = Counter(
        str(item.metadata[label_key])
        for values in ordered_groups.values()
        for item in values
    )
    target = {
        label: max_items * count / total_items for label, count in source_counts.items()
    }
    selected_counts: Counter[str] = Counter()
    selected: list[AnnotationItem] = []
    available = set(ordered_groups)

    def balance_error(counts: Counter[str]) -> float:
        return sum(abs(counts[label] - target[label]) for label in target)

    while available and len(selected) < max_items:
        remaining = max_items - len(selected)
        fitting = [group for group in available if len(ordered_groups[group]) <= remaining]
        if not fitting:
            break
        before = balance_error(selected_counts)
        candidates: list[tuple[float, int, int, str]] = []
        for group in fitting:
            values = ordered_groups[group]
            proposed = selected_counts.copy()
            proposed.update(str(item.metadata[label_key]) for item in values)
            balance_gain = before - balance_error(proposed)
            graph_gain = connectivity_bonus * max(0, len(values) - 1)
            candidates.append(
                (
                    balance_gain + graph_gain,
                    len(values),
                    -_stable_rank(seed, group),
                    group,
                )
            )
        _score, _size, _rank, chosen = max(candidates)
        block = ordered_groups[chosen]
        selected.extend(block)
        selected_counts.update(str(item.metadata[label_key]) for item in block)
        available.remove(chosen)

    remaining = max_items - len(selected)
    if remaining and available:
        before = balance_error(selected_counts)
        prefixes: list[tuple[float, int, str, list[AnnotationItem]]] = []
        for group in available:
            prefix = ordered_groups[group][:remaining]
            proposed = selected_counts.copy()
            proposed.update(str(item.metadata[label_key]) for item in prefix)
            balance_gain = before - balance_error(proposed)
            graph_gain = connectivity_bonus * max(0, len(prefix) - 1)
            prefixes.append(
                (
                    balance_gain + graph_gain,
                    -_stable_rank(seed, group),
                    group,
                    prefix,
                )
            )
        _score, _rank, _group, prefix = max(prefixes)
        selected.extend(prefix)

    if len(selected) != max_items:
        raise RuntimeError("graph-preserving sampler failed to fill requested capacity")
    return selected
