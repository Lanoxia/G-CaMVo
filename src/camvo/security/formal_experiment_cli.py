"""CLI for the frozen grouped-holdout real-provider protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camvo.security.formal_experiment import run_formal_provider_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_formal_provider_experiment(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report["test"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
