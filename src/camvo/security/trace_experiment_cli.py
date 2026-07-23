"""CLI for the complete TRACE-GCaMVo no-key research suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.security.trace_experiments import run_casie_trace_research_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TRACE-GCaMVo risk curves, ablations, bootstrap, and seed stability"
    )
    parser.add_argument("--casie-dir", type=Path, default=Path("data/raw/casie/data"))
    parser.add_argument("--max-items", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stability-seeds", default="11,17,23,29,31")
    parser.add_argument("--risk-grid", default="0.20,0.10,0.05,0.03,0.02,0.01")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=40)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/trace_20260720/casie_trace_research_suite.json"),
    )
    return parser


def _tuple(raw: str, converter: object) -> tuple[object, ...]:
    try:
        values = tuple(converter(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise SystemExit("comma-separated numeric argument is invalid") from exc
    if not values:
        raise SystemExit("comma-separated numeric argument must not be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_casie_trace_research_suite(
        args.casie_dir,
        max_items=args.max_items,
        seed=args.seed,
        stability_seeds=_tuple(args.stability_seeds, int),
        risk_grid=_tuple(args.risk_grid, float),
        embedding_dim=args.embedding_dim,
        warmup_rounds=args.warmup_rounds,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
