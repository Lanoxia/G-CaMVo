"""Audit event-level OpTC attack labels before the raw eCAR shard is available."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from camvo.security.optc import (
    build_optc_event_correlations,
    build_optc_provenance_graph,
    load_optc_attack_labels,
    load_optc_scenarios,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit best-effort malicious OpTC event labels for one scenario"
    )
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
    parser.add_argument(
        "--scenario-id",
        default="optc-day3-malicious-upgrade",
    )
    parser.add_argument(
        "--utc-offset-minutes",
        type=int,
        default=-240,
        help="The community labels encode -04:00; override only for another label source",
    )
    parser.add_argument("--padding-minutes", type=int, default=10)
    parser.add_argument("--correlation-window-minutes", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = load_optc_scenarios(args.scenario_manifest)
    try:
        scenario = scenarios[args.scenario_id]
    except KeyError as exc:
        raise SystemExit(f"unknown scenario ID: {args.scenario_id}") from exc
    start_ms, end_ms = scenario.epoch_window(
        utc_offset_minutes=args.utc_offset_minutes,
        padding_minutes=args.padding_minutes,
    )
    index = load_optc_attack_labels(
        args.labels,
        hostnames=scenario.hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        strict=True,
    )
    events = [label.to_minimal_event(source_path=str(args.labels)) for label in index.labels]
    graph = build_optc_provenance_graph(events)
    correlations = build_optc_event_correlations(
        events,
        max_time_gap_ms=args.correlation_window_minutes * 60 * 1000,
    )
    reason_counts: Counter[str] = Counter()
    for edge in correlations:
        reason_counts.update(edge.reasons)
    report = {
        "dataset": "DARPA OpTC community best-effort positive host-event labels",
        "important_semantics": (
            "Rows are high-confidence attack-related positives. An absent event ID is unknown, "
            "not automatically benign."
        ),
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "activities": len(scenario.activities),
            "utc_offset_minutes": args.utc_offset_minutes,
            "padding_minutes": args.padding_minutes,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
        "load_stats": {
            field: getattr(index.stats, field)
            for field in index.stats.__dataclass_fields__
        },
        "labels_by_host": dict(sorted(Counter(label.hostname for label in index.labels).items())),
        "labels_by_object_action": dict(
            sorted(
                Counter(
                    f"{label.object_type}:{label.action}" for label in index.labels
                ).items()
            )
        ),
        "label_time_range_ms": [index.labels[0].timestamp_ms, index.labels[-1].timestamp_ms],
        "provenance_graph": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "event_correlation_graph": {
            "nodes": len(events),
            "edges": len(correlations),
            "reason_counts": dict(sorted(reason_counts.items())),
            "window_minutes": args.correlation_window_minutes,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
