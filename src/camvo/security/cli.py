"""CLI for the CASIE security-routing proof of concept."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.security.poc import run_casie_poc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CaMVo on real CASIE text with simulated LLM abilities"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/casie/data"))
    parser.add_argument("--max-items", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--delta", type=float, default=0.97)
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=40)
    parser.add_argument("--context-chars", type=int, default=320)
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_casie_poc(
        args.data_dir,
        max_items=args.max_items,
        seed=args.seed,
        confidence_threshold=args.delta,
        min_models=args.min_models,
        warmup_rounds=args.warmup_rounds,
        context_chars=args.context_chars,
        strict_data=args.strict_data,
    )
    serialized = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
