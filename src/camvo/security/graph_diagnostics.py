"""Diagnostics that make graph suitability explicit before model calls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from camvo.types import AnnotationItem


def graph_diagnostics(
    items: Sequence[AnnotationItem],
    adjacency: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Report connectivity and optional evaluation-only label assortativity."""

    node_by_id = {str(item.item_id): item for item in items}
    node_ids = set(node_by_id)
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    edges: set[tuple[str, str]] = set()
    for left, raw_neighbors in adjacency.items():
        if left not in node_ids:
            continue
        for right, raw in raw_neighbors.items():
            weight = float(raw.get("weight", 0.0)) if isinstance(raw, Mapping) else float(raw)
            if right not in node_ids or right == left or weight <= 0:
                continue
            edge = tuple(sorted((str(left), str(right))))
            edges.add(edge)
            neighbors[str(left)].add(str(right))
            neighbors[str(right)].add(str(left))

    unseen = set(node_ids)
    component_sizes: list[int] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            discovered = neighbors[node] & unseen
            unseen.difference_update(discovered)
            stack.extend(discovered)
        component_sizes.append(size)

    possible_edges = len(node_ids) * (len(node_ids) - 1) // 2
    degrees = [len(neighbors[node_id]) for node_id in node_ids]
    gold_available = all(item.metadata.get("gold_label") in item.labels for item in items)
    same_label_edges = None
    label_agreement_rate = None
    if gold_available and edges:
        same_label_edges = sum(
            node_by_id[left].metadata["gold_label"]
            == node_by_id[right].metadata["gold_label"]
            for left, right in edges
        )
        label_agreement_rate = same_label_edges / len(edges)
    labels = Counter(
        str(item.metadata["gold_label"])
        for item in items
        if item.metadata.get("gold_label") in item.labels
    )
    return {
        "graph_nodes": len(node_ids),
        "graph_edges": len(edges),
        "graph_density": 0.0 if possible_edges == 0 else len(edges) / possible_edges,
        "graph_nonisolated_items": sum(degree > 0 for degree in degrees),
        "graph_isolated_items": sum(degree == 0 for degree in degrees),
        "graph_components": len(component_sizes),
        "graph_largest_component": max(component_sizes, default=0),
        "graph_mean_degree": 0.0 if not degrees else sum(degrees) / len(degrees),
        "same_label_edges": same_label_edges,
        "edge_label_agreement_rate": label_agreement_rate,
        "label_counts": dict(sorted(labels.items())),
    }
