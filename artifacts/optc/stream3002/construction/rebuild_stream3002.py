#!/usr/bin/env python3
"""Rebuild the zero-provider-call OpTC Stream3002 construction stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
CONFIG = HERE / "config"


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "camvo" / "security" / "optc.py").is_file():
            return parent
    raise RuntimeError("cannot locate G-CaMVo repository root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-pdf", type=Path, required=True)
    parser.add_argument("--exact-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CONFIG / "optc_core_acquisition_v1.tsv",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CONFIG / "optc_core_cases_v1.json",
    )
    parser.add_argument(
        "--skip-stage",
        action="store_true",
        help="reuse an existing, already validated stream_stage.sqlite",
    )
    return parser.parse_args()


def run(*arguments: object) -> None:
    command = [sys.executable, "-u", *(str(item) for item in arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=repository_root(), check=True)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_e2(report: dict[str, Any]) -> None:
    require(report.get("state") == "complete", "E2 is not complete")
    require(report.get("label_blind") is True, "E2 is not label blind")
    require(int(report.get("events_indexed", 0)) == 16_902_846, "unexpected E2 event count")
    require(len(report.get("files", [])) == 23, "E2 did not audit 23 source files")
    require(int(report.get("canonical_order_reversals", -1)) == 0, "E2 order gate failed")
    require(int(report.get("future_edges_materialized", -1)) == 0, "E2 future gate failed")


def validate_e3(report: dict[str, Any]) -> None:
    require(report.get("state") == "complete", "E3 is not complete")
    require(report.get("label_blind") is True, "E3 is not label blind")
    require(int(report.get("candidate_checkpoints", 0)) == 3_002, "unexpected E3 checkpoint count")
    require(int(report.get("events_scanned", 0)) == 16_902_846, "unexpected E3 scan count")


def validate_e4(report: dict[str, Any]) -> None:
    require(report.get("state") == "complete", "E4 is not complete")
    require(int(report.get("official_steps", 0)) == 101, "unexpected official-step count")
    require(int(report.get("attack_path_edges", 0)) == 73, "unexpected transition count")


def validate_e5(report: dict[str, Any]) -> None:
    require(report.get("state") == "complete", "E5 is not complete")
    require(report.get("past_only") is True, "E5 is not past only")
    require(int(report.get("checkpoints", 0)) == 3_002, "unexpected E5 checkpoint count")
    leakage = report.get("leakage", {})
    require(int(leakage.get("future_edges", -1)) == 0, "future evidence leaked")
    require(not leakage.get("forbidden_metadata_hits"), "prompt metadata leaked")
    require(int(leakage.get("raw_uuid_hits", -1)) == 0, "raw UUID leaked")
    require(not leakage.get("router_forbidden_metadata_hits"), "router metadata leaked")
    require(int(leakage.get("router_raw_uuid_hits", -1)) == 0, "router UUID leaked")
    require(int(leakage.get("provider_prompt_absolute_date_hits", -1)) == 0, "date leaked")
    require(int(leakage.get("source_group_split_overlap", -1)) == 0, "split groups overlap")


def main() -> int:
    args = parse_args()
    root = repository_root()
    for path in (args.manifest, args.cases, args.ground_truth_pdf, args.exact_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    evaluator = output / "evaluator"
    prompt_visible = output / "prompt_visible"
    router = output / "router"
    for path in (output, evaluator, prompt_visible, router):
        path.mkdir(parents=True, exist_ok=True)

    stage_db = output / "stream_stage.sqlite"
    e2_report = output / "e2_parser_report.json"
    candidates_db = output / "candidates.sqlite"
    e3_report = output / "e3_candidate_report.json"
    gold_db = evaluator / "gold.sqlite"
    e4_report = evaluator / "e4_gold_report.json"
    e5_report = output / "e5_bundle_audit.json"

    if not args.skip_stage:
        run(
            SCRIPTS / "build_optc_stream_stage.py",
            "--manifest", args.manifest,
            "--cases", args.cases,
            "--stage-db", stage_db,
            "--report", e2_report,
        )
    if not e2_report.is_file() or not stage_db.is_file():
        raise FileNotFoundError("E2 stage database/report missing")
    validate_e2(load(e2_report))

    run(
        SCRIPTS / "generate_optc_candidates.py",
        "--stage-db", stage_db,
        "--output-db", candidates_db,
        "--report", e3_report,
    )
    validate_e3(load(e3_report))

    run(
        SCRIPTS / "build_optc_evaluator_gold.py",
        "--stage-db", stage_db,
        "--cases", args.cases,
        "--ground-truth-pdf", args.ground_truth_pdf,
        "--exact-labels", args.exact_labels,
        "--output-db", gold_db,
        "--report", e4_report,
    )
    validate_e4(load(e4_report))

    run(
        SCRIPTS / "build_optc_evidence_bundles.py",
        "--stage-db", stage_db,
        "--candidates-db", candidates_db,
        "--evaluator-gold", gold_db,
        "--prompt-jsonl", prompt_visible / "checkpoints.jsonl",
        "--evaluator-map", evaluator / "checkpoint_map.jsonl",
        "--router-feature-map", router / "checkpoint_features.jsonl",
        "--split-manifest", evaluator / "splits.jsonl",
        "--report", e5_report,
    )
    validate_e5(load(e5_report))

    summary = {
        "state": "complete",
        "provider_calls": 0,
        "source_events": 16_902_846,
        "checkpoints": 3_002,
        "output_dir": str(output.relative_to(root) if output.is_relative_to(root) else output),
        "next_stage": "freeze prompts and collect the 3,002 x 5 model matrix",
    }
    (output / "construction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
