"""Positive-only G-CaMVo scalability smoke test on the real OpTC attack graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from camvo.algorithm.graph_regularization import WeightedGraphEdge, laplacian_smooth_scores
from camvo.security.optc import (
    build_optc_event_correlations,
    load_optc_attack_labels,
    load_optc_scenarios,
)


@dataclass(frozen=True, slots=True)
class GraphSmokePoint:
    regularization: float
    positive_recall_at_threshold: float
    mean_score: float
    rescued_false_negatives: int
    lost_true_positives: int
    objective_before: float
    objective_after: float
    solver: str
    iterations: int


def _simulated_independent_scores(event_ids: list[str], miss_rate: float) -> dict[str, float]:
    """Create reproducible heterogeneous confidence with controlled false negatives."""

    scores: dict[str, float] = {}
    for event_id in event_ids:
        digest = hashlib.sha256(f"optc-graph-smoke-v1:{event_id}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        detail = int.from_bytes(digest[8:16], "big") / (2**64 - 1)
        if unit < miss_rate:
            scores[event_id] = 0.15 + 0.25 * detail
        else:
            scores[event_id] = 0.68 + 0.27 * detail
    return scores


def run_optc_graph_smoke(
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    scenario_id: str = "optc-day3-malicious-upgrade",
    utc_offset_minutes: int = -240,
    padding_minutes: int = 10,
    correlation_window_minutes: int = 30,
    miss_rate: float = 0.28,
    threshold: float = 0.5,
    regularizations: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0, 3.0),
) -> dict[str, object]:
    if not 0 < miss_rate < 1 or not 0 < threshold < 1:
        raise ValueError("miss_rate and threshold must be in (0, 1)")
    scenario = load_optc_scenarios(scenario_manifest)[scenario_id]
    start_ms, end_ms = scenario.epoch_window(
        utc_offset_minutes=utc_offset_minutes,
        padding_minutes=padding_minutes,
    )
    labels = load_optc_attack_labels(
        labels_path,
        hostnames=scenario.hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        strict=True,
    )
    events = [label.to_minimal_event(source_path=str(labels_path)) for label in labels.labels]
    correlations = build_optc_event_correlations(
        events,
        max_time_gap_ms=correlation_window_minutes * 60 * 1000,
    )
    edges = [
        WeightedGraphEdge(edge.source_event_id, edge.target_event_id, edge.weight)
        for edge in correlations
    ]
    event_ids = [event.event_id for event in events]
    raw = _simulated_independent_scores(event_ids, miss_rate)
    raw_positive = {event_id for event_id, score in raw.items() if score >= threshold}
    points: list[GraphSmokePoint] = []
    for regularization in regularizations:
        result = laplacian_smooth_scores(
            raw,
            edges,
            regularization=regularization,
            solver="cg",
            tolerance=1e-7,
            max_iterations=1_000,
        )
        predicted = {
            event_id
            for event_id, score in result.smoothed_scores.items()
            if score >= threshold
        }
        points.append(
            GraphSmokePoint(
                regularization=regularization,
                positive_recall_at_threshold=len(predicted) / len(event_ids),
                mean_score=mean(result.smoothed_scores.values()),
                rescued_false_negatives=len(predicted - raw_positive),
                lost_true_positives=len(raw_positive - predicted),
                objective_before=result.objective_before,
                objective_after=result.objective_after,
                solver=result.solver,
                iterations=result.iterations,
            )
        )
    return {
        "experiment": "OpTC positive-only graph-regularization scalability smoke test",
        "scenario_id": scenario.scenario_id,
        "real_data": {
            "positive_event_ids": len(event_ids),
            "correlation_edges": len(edges),
            "hosts": scenario.hosts,
        },
        "simulation": {
            "independent_score_generator": "deterministic SHA-256 split mixture",
            "injected_miss_rate": miss_rate,
            "decision_threshold": threshold,
        },
        "critical_limitation": (
            "The event IDs and graph topology are real best-effort OpTC attack positives, but "
            "model "
            "scores are simulated and no benign negatives are present. Recall changes validate "
            "mechanics/scalability only; they do not establish precision, F1, or paper improvement."
        ),
        "points": [asdict(point) for point in points],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/raw/optc-labels/labels.csv"),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("config/optc_scenarios.json"),
    )
    parser.add_argument("--miss-rate", type=float, default=0.28)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_optc_graph_smoke(
        args.labels,
        args.scenario_manifest,
        miss_rate=args.miss_rate,
        threshold=args.threshold,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
