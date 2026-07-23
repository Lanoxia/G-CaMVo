"""Deterministic group-aware splits that prevent document/scenario leakage."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from camvo.types import AnnotationItem


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    calibration: tuple[AnnotationItem, ...]
    validation: tuple[AnnotationItem, ...]
    test: tuple[AnnotationItem, ...]

    def __post_init__(self) -> None:
        partitions = (self.calibration, self.validation, self.test)
        ids = [{item.item_id for item in partition} for partition in partitions]
        if any(ids[left] & ids[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("dataset split contains duplicate item IDs across partitions")


def _stable_uniform(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def grouped_split(
    items: Iterable[AnnotationItem],
    *,
    group_key: Callable[[AnnotationItem], str],
    calibration_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> DatasetSplit:
    """Assign whole groups by stable hash, never individual correlated rows."""

    if not 0 <= calibration_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if calibration_fraction + validation_fraction >= 1:
        raise ValueError("calibration and validation fractions must sum to less than 1")
    grouped: dict[str, list[AnnotationItem]] = defaultdict(list)
    seen_ids: set[str] = set()
    for item in items:
        if item.item_id in seen_ids:
            raise ValueError(f"duplicate item ID: {item.item_id}")
        seen_ids.add(item.item_id)
        key = str(group_key(item)).strip()
        if not key:
            raise ValueError("group_key returned an empty value")
        grouped[key].append(item)
    if not grouped:
        raise ValueError("items must not be empty")

    calibration: list[AnnotationItem] = []
    validation: list[AnnotationItem] = []
    test: list[AnnotationItem] = []
    boundary = calibration_fraction + validation_fraction
    for key in sorted(grouped):
        value = _stable_uniform(seed, key)
        if value < calibration_fraction:
            calibration.extend(grouped[key])
        elif value < boundary:
            validation.extend(grouped[key])
        else:
            test.extend(grouped[key])
    return DatasetSplit(
        calibration=tuple(calibration),
        validation=tuple(validation),
        test=tuple(test),
    )


def stratified_grouped_split(
    items: Iterable[AnnotationItem],
    *,
    group_key: Callable[[AnnotationItem], str],
    label_key: str = "gold_label",
    calibration_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> DatasetSplit:
    """Approximate label stratification while keeping every group indivisible.

    Security documents/incidents often contain label-homogeneous blocks.  A
    pure group hash can therefore omit entire classes from calibration or
    validation.  This deterministic iterative allocator processes the rarest
    remaining label first and assigns its most informative group to the split
    with the greatest proportional label deficit, then overall label and size
    deficit.  Exact proportions are impossible for indivisible large groups,
    so the method is intentionally approximate and auditable.
    """

    if not 0 <= calibration_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if calibration_fraction + validation_fraction >= 1:
        raise ValueError("calibration and validation fractions must sum to less than 1")
    ordered_items = list(items)
    if not ordered_items:
        raise ValueError("items must not be empty")
    groups: dict[str, list[AnnotationItem]] = defaultdict(list)
    seen: set[str] = set()
    total_labels: Counter[str] = Counter()
    for item in ordered_items:
        if item.item_id in seen:
            raise ValueError(f"duplicate item ID: {item.item_id}")
        seen.add(item.item_id)
        group = str(group_key(item)).strip()
        if not group:
            raise ValueError("group_key returned an empty value")
        label = str(item.metadata.get(label_key, "")).strip()
        if label not in item.labels:
            raise ValueError(f"item {item.item_id!r} lacks a valid {label_key!r}")
        groups[group].append(item)
        total_labels[label] += 1

    fractions = {
        "calibration": calibration_fraction,
        "validation": validation_fraction,
        "test": 1.0 - calibration_fraction - validation_fraction,
    }
    target_size = {name: len(ordered_items) * fraction for name, fraction in fractions.items()}
    target_labels = {
        name: {label: total * fraction for label, total in total_labels.items()}
        for name, fraction in fractions.items()
    }
    current_size = {name: 0 for name in fractions}
    current_labels = {name: Counter() for name in fractions}
    group_labels = {
        group: Counter(str(item.metadata[label_key]) for item in values)
        for group, values in groups.items()
    }
    unassigned = set(groups)
    assignment: dict[str, str] = {}
    partition_order = ("calibration", "validation", "test")

    while unassigned:
        label_group_frequency = {
            label: sum(group_labels[group][label] > 0 for group in unassigned)
            for label in total_labels
            if any(group_labels[group][label] > 0 for group in unassigned)
        }
        rare_label = min(
            label_group_frequency,
            key=lambda label: (
                label_group_frequency[label],
                total_labels[label],
                label,
            ),
        )
        group = max(
            (group for group in unassigned if group_labels[group][rare_label] > 0),
            key=lambda value: (
                group_labels[value][rare_label],
                len(groups[value]),
                -int(_stable_uniform(seed, value) * (2**53)),
                value,
            ),
        )

        def partition_score(name: str) -> tuple[float, float, float, float, str]:
            rare_target = max(target_labels[name][rare_label], 1e-9)
            rare_need = (
                target_labels[name][rare_label] - current_labels[name][rare_label]
            ) / rare_target
            aggregate_need = sum(
                count
                * (target_labels[name][label] - current_labels[name][label])
                / max(target_labels[name][label], 1e-9)
                for label, count in group_labels[group].items()
            ) / len(groups[group])
            size_need = (target_size[name] - current_size[name]) / max(target_size[name], 1e-9)
            # Larger target is the final deterministic tie-break at the empty
            # state, after which proportional deficits drive the allocation.
            return rare_need, aggregate_need, size_need, target_size[name], name

        selected_partition = max(partition_order, key=partition_score)
        assignment[group] = selected_partition
        current_size[selected_partition] += len(groups[group])
        current_labels[selected_partition].update(group_labels[group])
        unassigned.remove(group)

    label_group_totals = {
        label: sum(counts[label] > 0 for counts in group_labels.values())
        for label in total_labels
    }
    group_counts = Counter(assignment.values())

    def allocation_objective(
        size_delta: dict[str, int] | None = None,
        label_delta: dict[tuple[str, str], int] | None = None,
        group_delta: dict[str, int] | None = None,
    ) -> float:
        """Score a small state delta without rebuilding all group assignments."""

        size_delta = size_delta or {}
        label_delta = label_delta or {}
        group_delta = group_delta or {}
        score = 0.0
        for name in fractions:
            size = current_size[name] + size_delta.get(name, 0)
            score += 4.0 * (size - target_size[name]) ** 2 / (
                target_size[name] + 1.0
            )
            for label in total_labels:
                label_count = current_labels[name][label] + label_delta.get(
                    (name, label), 0
                )
                score += (
                    label_count - target_labels[name][label]
                ) ** 2 / (target_labels[name][label] + 1.0)
                if (
                    label_group_totals[label] >= len(fractions)
                    and not label_count
                ):
                    score += 1_000_000.0
            if group_counts[name] + group_delta.get(name, 0) == 0:
                score += 1_000_000.0
        return score

    def move_group(group: str, target: str) -> None:
        source = assignment[group]
        if source == target:
            return
        size = len(groups[group])
        current_size[source] -= size
        current_size[target] += size
        for label, count in group_labels[group].items():
            current_labels[source][label] -= count
            current_labels[target][label] += count
        group_counts[source] -= 1
        group_counts[target] += 1
        assignment[group] = target

    # The rare-label pass guarantees a strong starting point.  Deterministic
    # single moves and pair swaps then repair size/label proportion distortion
    # caused by large indivisible documents.
    ordered_groups = sorted(groups, key=lambda value: (_stable_uniform(seed, value), value))
    for _iteration in range(100):
        current_objective = allocation_objective()
        best: tuple[float, str, str] | None = None
        for group in ordered_groups:
            source = assignment[group]
            size = len(groups[group])
            for target in partition_order:
                if target == source:
                    continue
                label_delta: dict[tuple[str, str], int] = {}
                for label, count in group_labels[group].items():
                    label_delta[(source, label)] = -count
                    label_delta[(target, label)] = count
                objective = allocation_objective(
                    {source: -size, target: size},
                    label_delta,
                    {source: -1, target: 1},
                )
                candidate = (objective, group, target)
                if objective < current_objective - 1e-9 and (best is None or candidate < best):
                    best = candidate
        if best is not None:
            _objective, group, target = best
            move_group(group, target)
            continue

        # Pair swaps are useful for small pilots, but a quadratic scan is not a
        # defensible preprocessing cost for a 1,000-document corpus.  The rare-
        # label allocation plus exact single-move refinement above is already
        # deterministic, group-safe and label-aware for larger datasets.
        if len(ordered_groups) > 200:
            break

        best_swap: tuple[float, str, str] | None = None
        for left_index, left in enumerate(ordered_groups):
            for right in ordered_groups[left_index + 1 :]:
                left_partition = assignment[left]
                right_partition = assignment[right]
                if left_partition == right_partition:
                    continue
                left_size = len(groups[left])
                right_size = len(groups[right])
                size_delta = {
                    left_partition: right_size - left_size,
                    right_partition: left_size - right_size,
                }
                label_delta: dict[tuple[str, str], int] = defaultdict(int)
                for label, count in group_labels[left].items():
                    label_delta[(left_partition, label)] -= count
                    label_delta[(right_partition, label)] += count
                for label, count in group_labels[right].items():
                    label_delta[(right_partition, label)] -= count
                    label_delta[(left_partition, label)] += count
                objective = allocation_objective(size_delta, label_delta)
                candidate = (objective, left, right)
                if objective < current_objective - 1e-9 and (
                    best_swap is None or candidate < best_swap
                ):
                    best_swap = candidate
        if best_swap is None:
            break
        _objective, left, right = best_swap
        left_partition = assignment[left]
        right_partition = assignment[right]
        left_size = len(groups[left])
        right_size = len(groups[right])
        current_size[left_partition] += right_size - left_size
        current_size[right_partition] += left_size - right_size
        for label, count in group_labels[left].items():
            current_labels[left_partition][label] -= count
            current_labels[right_partition][label] += count
        for label, count in group_labels[right].items():
            current_labels[right_partition][label] -= count
            current_labels[left_partition][label] += count
        assignment[left], assignment[right] = right_partition, left_partition

    partitions: dict[str, list[AnnotationItem]] = {name: [] for name in fractions}
    for item in ordered_items:
        partitions[assignment[str(group_key(item)).strip()]].append(item)
    return DatasetSplit(
        calibration=tuple(partitions["calibration"]),
        validation=tuple(partitions["validation"]),
        test=tuple(partitions["test"]),
    )
