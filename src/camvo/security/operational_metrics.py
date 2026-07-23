"""Operational SOC metrics derived from frozen OpTC event predictions."""

from __future__ import annotations

from statistics import mean, median
from typing import Mapping, Sequence

from camvo.types import AnnotationItem


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def optc_operational_metrics(
    items: Sequence[AnnotationItem],
    predictions: Sequence[str],
    abstentions: Sequence[bool],
    adjacency: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Measure host detection, delay, false-alert rate, and malicious-subgraph recovery.

    The edge score is an induced-subgraph proxy: gold edges join two gold-malicious
    events and predicted edges join two accepted malicious predictions.  It is not
    an ordered attack-timeline metric and is labelled accordingly in the report.
    """

    if not (len(items) == len(predictions) == len(abstentions)):
        raise ValueError("items, predictions, and abstentions must have equal length")
    if not items or "malicious" not in items[0].labels:
        raise ValueError("OpTC operational metrics require a non-empty malicious-label task")

    gold_malicious: set[str] = set()
    predicted_malicious: set[str] = set()
    malicious_by_host: dict[str, list[tuple[int, str]]] = {}
    accepted_true_positives_by_host: dict[str, list[int]] = {}
    false_alerts = 0
    benign_windows: set[tuple[str, int]] = set()
    node_by_item: dict[str, str] = {}

    for item, prediction, abstained in zip(items, predictions, abstentions, strict=True):
        node = str(item.metadata.get("graph_node_id", item.item_id))
        node_by_item[item.item_id] = node
        gold = str(item.metadata["gold_label"])
        hostname = str(item.metadata.get("hostname", "unknown"))
        timestamp = int(item.metadata.get("timestamp_ms", 0))
        if gold == "malicious":
            gold_malicious.add(node)
            malicious_by_host.setdefault(hostname, []).append((timestamp, node))
        else:
            benign_windows.add((hostname, timestamp // 1_800_000))
        if not abstained and prediction == "malicious":
            predicted_malicious.add(node)
            if gold == "malicious":
                accepted_true_positives_by_host.setdefault(hostname, []).append(timestamp)
            else:
                false_alerts += 1

    detected_hosts = 0
    delays: list[float] = []
    for hostname, events in malicious_by_host.items():
        true_positive_times = accepted_true_positives_by_host.get(hostname, [])
        if not true_positive_times:
            continue
        detected_hosts += 1
        first_ground = min(timestamp for timestamp, _node in events)
        delays.append(max(0.0, (min(true_positive_times) - first_ground) / 1000.0))

    test_nodes = set(node_by_item.values())

    def induced_edges(nodes: set[str]) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for left in nodes:
            for right in adjacency.get(left, {}):
                if right not in nodes or right not in test_nodes or left == right:
                    continue
                edges.add(tuple(sorted((left, right))))
        return edges

    gold_edges = induced_edges(gold_malicious)
    predicted_edges = induced_edges(predicted_malicious)
    correct_edges = len(gold_edges & predicted_edges)
    edge_precision = _safe_ratio(correct_edges, len(predicted_edges))
    edge_recall = _safe_ratio(correct_edges, len(gold_edges))
    benign_host_hours = len(benign_windows) * 0.5

    return {
        "incident_unit": "hostname within frozen test partition",
        "malicious_hosts": len(malicious_by_host),
        "detected_malicious_hosts": detected_hosts,
        "host_detection_rate": _safe_ratio(detected_hosts, len(malicious_by_host)),
        "time_to_first_true_positive_seconds": {
            "observations": len(delays),
            "mean": None if not delays else mean(delays),
            "median": None if not delays else median(delays),
            "maximum": None if not delays else max(delays),
        },
        "false_alerts": false_alerts,
        "observed_benign_host_hours": benign_host_hours,
        "false_alerts_per_benign_host_hour": (
            None if benign_host_hours == 0 else false_alerts / benign_host_hours
        ),
        "malicious_correlation_subgraph_proxy": {
            "definition": "induced undirected correlation edges; not ordered timeline ground truth",
            "gold_edges": len(gold_edges),
            "predicted_edges": len(predicted_edges),
            "correct_edges": correct_edges,
            "precision": edge_precision,
            "recall": edge_recall,
            "f1": _f1(edge_precision, edge_recall),
        },
    }

