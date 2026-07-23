"""CLI for the no-provider DARPA OpTC readiness gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.security.optc_readiness import build_optc_readiness_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-path", type=Path, required=True)
    parser.add_argument("--benign-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("config/optc_scenarios.json"),
    )
    parser.add_argument("--scenario-id", default="optc-day3-malicious-upgrade")
    parser.add_argument("--utc-offset-minutes", type=int, default=-240)
    parser.add_argument("--padding-minutes", type=int, default=10)
    parser.add_argument("--sample-events", type=int, default=32)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_optc_readiness_report(
        args.attack_path,
        args.benign_path,
        args.labels_path,
        args.scenario_manifest,
        scenario_id=args.scenario_id,
        utc_offset_minutes=args.utc_offset_minutes,
        padding_minutes=args.padding_minutes,
        sample_events=args.sample_events,
        minimum_free_bytes=max(0, round(args.minimum_free_gib * 1024**3)),
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.json_out)
    return 0 if report["ready"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

