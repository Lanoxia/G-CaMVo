"""Command-line entry point for offline validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.simulation import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline CaMVo simulation")
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delta", type=float, default=0.90)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument(
        "--confidence-method",
        choices=("exact", "beta_cdf"),
        default="exact",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, router = run_simulation(
        item_count=args.items,
        seed=args.seed,
        confidence_threshold=args.delta,
        min_models=args.min_models,
        warmup_rounds=args.warmup_rounds,
        confidence_method=args.confidence_method,
    )
    serialized = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    if args.checkpoint_out:
        router.save_checkpoint(args.checkpoint_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
