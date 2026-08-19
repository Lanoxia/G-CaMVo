#!/usr/bin/env python3
"""Statistical audit for the PPT-aligned final OpTC endpoints.

This script performs no provider calls.  It combines the frozen 30-minute
window results with step- and transition-level reconstruction audits.  Wilson
intervals describe finite-step uncertainty; a campaign-stratified paired
bootstrap is reported as sensitivity because OpTC has only three top-level
attack campaigns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RELEASE_ROOT / "results/audit_inputs"
DEFAULT_OUTPUT = RELEASE_ROOT / "results/recomputed"
ITERATIONS = 10_000
SEED = 20260819
CAMVO = "camvo_mordor_transfer"
FULL = "camvo_weighted_online_full_vote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def stratified_samples(
    by_campaign: dict[str, list[str]], iterations: int, seed: int
) -> Iterable[list[str]]:
    generator = random.Random(seed)
    campaigns = sorted(by_campaign)
    for _ in range(iterations):
        sampled: list[str] = []
        for campaign in (generator.choice(campaigns) for _ in campaigns):
            items = by_campaign[campaign]
            sampled.extend(generator.choice(items) for _ in items)
        yield sampled


def transition_f1(rows: list[dict[str, str]]) -> float:
    tp = sum(row["gold_transition"] == "True" and row["predicted_transition"] == "True" for row in rows)
    fp = sum(row["gold_transition"] != "True" and row["predicted_transition"] == "True" for row in rows)
    fn = sum(row["gold_transition"] == "True" and row["predicted_transition"] != "True" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def adjusted_timeline(rows: list[dict[str, str]]) -> float:
    concordant = sum(int(row["timeline_concordant_pairs"]) for row in rows)
    total = sum(int(row["timeline_total_official_pairs"]) for row in rows)
    return concordant / total if total else 0.0


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    steps = read_csv(input_dir / "e14_step_detection_audit.csv")
    transitions = read_csv(input_dir / "e14_transition_audit.csv")
    reconstruction = read_csv(input_dir / "e14_all_campaign_main_table.csv")
    campaigns = read_csv(input_dir / "e14_all_campaign_by_campaign.csv")
    window_main = read_csv(input_dir / "nested30_main_table.csv")
    window_pairs = read_csv(input_dir / "nested30_paired_bootstrap.csv")

    step_by_method: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    campaign_step_ids: dict[str, list[str]] = defaultdict(list)
    for row in steps:
        step_by_method[row["method"]][row["step_id"]] = row
        if row["method"] == CAMVO:
            campaign_step_ids[row["campaign"]].append(row["step_id"])

    methods = sorted(step_by_method)
    step_rows: list[dict[str, Any]] = []
    bootstrap_recall: dict[str, list[float]] = defaultdict(list)
    sampled_step_sets = list(stratified_samples(campaign_step_ids, ITERATIONS, SEED))
    for method in methods:
        method_rows = step_by_method[method]
        detected = sum(row["detected"] == "True" for row in method_rows.values())
        reachable = sum(row["reachable_under_alignment"] == "True" for row in method_rows.values())
        total = len(method_rows)
        for sample in sampled_step_sets:
            bootstrap_recall[method].append(
                sum(method_rows[step]["detected"] == "True" for step in sample) / len(sample)
            )
        low, high = wilson(detected, total)
        step_rows.append({
            "method": method,
            "detected_steps": detected,
            "observable_steps": total,
            "reachable_steps": reachable,
            "end_to_end_recall": detected / total,
            "wilson_ci_low": low,
            "wilson_ci_high": high,
            "campaign_stratified_bootstrap_low": percentile(bootstrap_recall[method], 0.025),
            "campaign_stratified_bootstrap_high": percentile(bootstrap_recall[method], 0.975),
            "top_level_campaigns": len(campaign_step_ids),
        })

    strongest_step_single = max(
        (method for method in methods if method.startswith("single::")),
        key=lambda method: sum(row["detected"] == "True" for row in step_by_method[method].values()),
    )
    pair_rows: list[dict[str, Any]] = []
    for candidate, reference in ((CAMVO, FULL), (CAMVO, strongest_step_single), (FULL, strongest_step_single)):
        differences = [
            left - right
            for left, right in zip(bootstrap_recall[candidate], bootstrap_recall[reference], strict=True)
        ]
        point = (
            sum(row["detected"] == "True" for row in step_by_method[candidate].values())
            - sum(row["detected"] == "True" for row in step_by_method[reference].values())
        ) / len(step_by_method[candidate])
        pair_rows.append({
            "endpoint": "official_step_recall",
            "candidate": candidate,
            "reference": reference,
            "difference": point,
            "ci_low": percentile(differences, 0.025),
            "ci_high": percentile(differences, 0.975),
            "probability_candidate_better": sum(value > 0 for value in differences) / len(differences),
            "statistically_distinguishable_at_95pct": percentile(differences, 0.025) > 0 or percentile(differences, 0.975) < 0,
        })

    campaign_names = sorted(campaign_step_ids)
    campaign_rows_by_method: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in campaigns:
        campaign_rows_by_method[row["method"]][row["campaign"]] = row
    timeline_samples: dict[str, list[float]] = defaultdict(list)
    generator = random.Random(SEED + 2)
    for _ in range(ITERATIONS):
        sampled_campaigns = [generator.choice(campaign_names) for _ in campaign_names]
        for method in methods:
            timeline_samples[method].append(
                adjusted_timeline([campaign_rows_by_method[method][campaign] for campaign in sampled_campaigns])
            )
    timeline_rows: list[dict[str, Any]] = []
    for method in methods:
        method_campaigns = list(campaign_rows_by_method[method].values())
        timeline_rows.append({
            "method": method,
            "coverage_adjusted_timeline": adjusted_timeline(method_campaigns),
            "campaign_cluster_bootstrap_low": percentile(timeline_samples[method], 0.025),
            "campaign_cluster_bootstrap_high": percentile(timeline_samples[method], 0.975),
            "top_level_campaigns": len(campaign_names),
        })
    for candidate, reference in ((CAMVO, FULL), (CAMVO, strongest_step_single), (FULL, strongest_step_single)):
        differences = [
            left - right
            for left, right in zip(timeline_samples[candidate], timeline_samples[reference], strict=True)
        ]
        point = adjusted_timeline(list(campaign_rows_by_method[candidate].values())) - adjusted_timeline(
            list(campaign_rows_by_method[reference].values())
        )
        pair_rows.append({
            "endpoint": "coverage_adjusted_timeline",
            "candidate": candidate,
            "reference": reference,
            "difference": point,
            "ci_low": percentile(differences, 0.025),
            "ci_high": percentile(differences, 0.975),
            "probability_candidate_better": sum(value > 0 for value in differences) / len(differences),
            "statistically_distinguishable_at_95pct": percentile(differences, 0.025) > 0 or percentile(differences, 0.975) < 0,
        })

    transition_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    transition_by_method_campaign: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in transitions:
        transition_by_method[row["method"]].append(row)
        transition_by_method_campaign[row["method"]][row["campaign"]].append(row)
    transition_rows: list[dict[str, Any]] = []
    transition_samples: dict[str, list[float]] = defaultdict(list)
    generator = random.Random(SEED + 1)
    for _ in range(ITERATIONS):
        sampled_campaigns = [generator.choice(campaign_names) for _ in campaign_names]
        for method in methods:
            sampled_rows = [row for campaign in sampled_campaigns for row in transition_by_method_campaign[method][campaign]]
            transition_samples[method].append(transition_f1(sampled_rows))
    for method in methods:
        point = transition_f1(transition_by_method[method])
        transition_rows.append({
            "method": method,
            "official_transition_f1": point,
            "campaign_cluster_bootstrap_low": percentile(transition_samples[method], 0.025),
            "campaign_cluster_bootstrap_high": percentile(transition_samples[method], 0.975),
            "top_level_campaigns": len(campaign_names),
        })

    for candidate, reference in ((CAMVO, FULL), (CAMVO, strongest_step_single), (FULL, strongest_step_single)):
        differences = [
            left - right
            for left, right in zip(transition_samples[candidate], transition_samples[reference], strict=True)
        ]
        point = transition_f1(transition_by_method[candidate]) - transition_f1(transition_by_method[reference])
        pair_rows.append({
            "endpoint": "official_transition_f1",
            "candidate": candidate,
            "reference": reference,
            "difference": point,
            "ci_low": percentile(differences, 0.025),
            "ci_high": percentile(differences, 0.975),
            "probability_candidate_better": sum(value > 0 for value in differences) / len(differences),
            "statistically_distinguishable_at_95pct": percentile(differences, 0.025) > 0 or percentile(differences, 0.975) < 0,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "official_step_uncertainty.csv", step_rows)
    write_csv(output_dir / "official_transition_uncertainty.csv", transition_rows)
    write_csv(output_dir / "timeline_uncertainty.csv", timeline_rows)
    write_csv(output_dir / "paired_reconstruction_differences.csv", pair_rows)
    write_csv(output_dir / "window_main_table.csv", window_main)
    write_csv(output_dir / "window_paired_differences.csv", window_pairs)
    write_csv(output_dir / "reconstruction_main_table.csv", reconstruction)
    write_csv(output_dir / "reconstruction_by_campaign.csv", campaigns)
    report = {
        "schema_version": 1,
        "state": "complete",
        "provider_calls": 0,
        "bootstrap_iterations": ITERATIONS,
        "bootstrap_seed": SEED,
        "top_level_campaigns": len(campaign_names),
        "strongest_single_official_step_recall": strongest_step_single,
        "claim_boundary": (
            "Wilson intervals treat official steps as finite evaluation units; campaign-stratified "
            "bootstrap is a sensitivity analysis, not evidence of 76 independent campaigns."
        ),
        "files": [
            "official_step_uncertainty.csv",
            "official_transition_uncertainty.csv",
            "timeline_uncertainty.csv",
            "paired_reconstruction_differences.csv",
            "window_main_table.csv",
            "window_paired_differences.csv",
            "reconstruction_main_table.csv",
            "reconstruction_by_campaign.csv",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
