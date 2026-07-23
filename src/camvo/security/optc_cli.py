"""CLI for auditing a bounded DARPA OpTC eCAR subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from camvo.security.optc import (
    build_optc_event_correlations,
    build_optc_provenance_graph,
    load_optc_events,
    load_optc_scenarios,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate OpTC eCAR records and build graph statistics for a bounded subset"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--host", action="append", dest="hosts")
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--end-ms", type=int)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--scenario-id")
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("config/optc_scenarios.json"),
    )
    parser.add_argument(
        "--utc-offset-minutes",
        type=int,
        help="Explicit offset needed to align scenario-local ground truth with epoch telemetry",
    )
    parser.add_argument("--padding-minutes", type=int, default=10)
    parser.add_argument("--correlation-window-minutes", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hosts = args.hosts
    start_ms = args.start_ms
    end_ms = args.end_ms
    scenario_payload: dict[str, object] | None = None
    if args.scenario_id:
        scenarios = load_optc_scenarios(args.scenario_manifest)
        try:
            scenario = scenarios[args.scenario_id]
        except KeyError as exc:
            raise SystemExit(f"unknown scenario ID: {args.scenario_id}") from exc
        if args.utc_offset_minutes is None:
            raise SystemExit(
                "--utc-offset-minutes is required with --scenario-id because the release does "
                "not document the ground-truth clock's UTC offset"
            )
        scenario_start, scenario_end = scenario.epoch_window(
            utc_offset_minutes=args.utc_offset_minutes,
            padding_minutes=args.padding_minutes,
        )
        hosts = list(dict.fromkeys([*(hosts or []), *scenario.hosts]))
        start_ms = scenario_start if start_ms is None else max(start_ms, scenario_start)
        end_ms = scenario_end if end_ms is None else min(end_ms, scenario_end)
        scenario_payload = {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "hosts": scenario.hosts,
            "activities": len(scenario.activities),
            "resolved_start_ms": start_ms,
            "resolved_end_ms": end_ms,
            "utc_offset_minutes": args.utc_offset_minutes,
            "padding_minutes": args.padding_minutes,
        }

    dataset = load_optc_events(
        args.input,
        hostnames=hosts,
        start_ms=start_ms,
        end_ms=end_ms,
        max_events=args.max_events,
        strict=args.strict,
    )
    graph = build_optc_provenance_graph(dataset.events)
    correlations = build_optc_event_correlations(
        dataset.events,
        max_time_gap_ms=args.correlation_window_minutes * 60 * 1000,
    )
    report: dict[str, object] = {
        "load_stats": {
            field: getattr(dataset.stats, field)
            for field in dataset.stats.__dataclass_fields__
        },
        "hosts": dict(sorted(Counter(event.hostname for event in dataset.events).items())),
        "object_actions": dict(
            sorted(
                Counter(
                    f"{event.object_type}:{event.action}" for event in dataset.events
                ).items()
            )
        ),
        "time_range_ms": [dataset.events[0].timestamp_ms, dataset.events[-1].timestamp_ms],
        "provenance_graph": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "event_correlation_graph": {
            "nodes": len(dataset.events),
            "edges": len(correlations),
            "window_minutes": args.correlation_window_minutes,
        },
        "scenario": scenario_payload,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
