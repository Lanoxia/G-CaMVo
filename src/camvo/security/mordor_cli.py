"""Command-line audit for OTRF Security Datasets / Mordor NDJSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.security.mordor import audit_mordor_ndjson


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Mordor NDJSON schema, graph fields and label availability"
    )
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--lenient", action="store_true", help="count malformed rows and continue")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_mordor_ndjson(args.input, strict=not args.lenient)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

