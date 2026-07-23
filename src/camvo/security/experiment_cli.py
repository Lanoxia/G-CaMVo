"""CLI for the complete no-key CASIE/OpTC routing experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from camvo.security.optc_dataset import build_optc_label_graph_simulation_dataset
from camvo.security.simulated_experiments import (
    run_casie_six_strategy_experiment,
    run_optc_label_graph_experiment,
    run_optc_real_data_simulated_model_experiment,
    run_parameter_sweep,
    run_seed_stability,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seven routing policies with simulated models on real security data"
    )
    parser.add_argument(
        "--dataset",
        choices=("casie", "optc-label-graph", "optc-real"),
        default="optc-label-graph",
    )
    parser.add_argument("--casie-dir", type=Path, default=Path("data/raw/casie/data"))
    parser.add_argument(
        "--optc-labels",
        type=Path,
        default=Path("data/raw/optc-labels/labels.csv"),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("config/optc_scenarios.json"),
    )
    parser.add_argument("--attack-path", type=Path)
    parser.add_argument("--benign-path", type=Path)
    parser.add_argument("--max-items", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--delta", type=float, default=0.97)
    parser.add_argument("--graph-lambda", type=float, default=1.0)
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--sweep-json-out", type=Path)
    parser.add_argument("--sweep-csv-out", type=Path)
    parser.add_argument("--sweep-max-items", type=int, default=600)
    parser.add_argument("--stability-json-out", type=Path)
    parser.add_argument("--stability-csv-out", type=Path)
    parser.add_argument("--stability-seeds", default="11,17,23,29,31")
    parser.add_argument("--stability-delta", type=float, default=0.92)
    parser.add_argument("--stability-graph-lambda", type=float, default=1.0)
    return parser


def _write_method_csv(path: Path, report: dict[str, object]) -> None:
    rows = []
    methods = report["methods"]
    savings = report["cost_savings_vs_full"]
    for name, raw in methods.items():
        metrics = raw["metrics"]
        rows.append(
            {
                "method": name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "total_cost_usd": raw["total_cost_usd"],
                "cost_savings_vs_full": savings[name],
                "average_models": raw["average_models"],
                "escalation_rate": raw["escalation_rate"],
                "full_pool_rate": raw["full_pool_rate"],
                "average_parallel_latency_ms": raw["average_parallel_latency_ms"],
                "p95_parallel_latency_ms": raw["p95_parallel_latency_ms"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_csv(path: Path, points: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def _write_stability_csv(path: Path, report: dict[str, object]) -> None:
    rows = list(report["methods"].values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "max_items": args.max_items,
        "seed": args.seed,
        "confidence_threshold": args.delta,
        "graph_regularization": args.graph_lambda,
        "min_models": args.min_models,
        "warmup_rounds": args.warmup_rounds,
        "embedding_dim": args.embedding_dim,
    }
    if args.dataset == "casie":
        report = run_casie_six_strategy_experiment(args.casie_dir, **common)
    elif args.dataset == "optc-label-graph":
        report = run_optc_label_graph_experiment(
            args.optc_labels,
            args.scenario_manifest,
            **common,
        )
    else:
        if args.attack_path is None or args.benign_path is None:
            raise SystemExit("optc-real requires --attack-path and --benign-path")
        report = run_optc_real_data_simulated_model_experiment(
            args.attack_path,
            args.benign_path,
            args.optc_labels,
            args.scenario_manifest,
            **common,
        )
    payload = report.to_dict()
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    if args.csv_out:
        _write_method_csv(args.csv_out, payload)

    if (
        args.sweep_json_out
        or args.sweep_csv_out
        or args.stability_json_out
        or args.stability_csv_out
    ):
        if args.dataset == "casie":
            raise SystemExit("the current joint graph sweep is defined for the OpTC task")
        if args.dataset == "optc-real":
            raise SystemExit(
                "run the expensive optc-real sweep explicitly after the main raw-data audit"
            )
        sweep_dataset = build_optc_label_graph_simulation_dataset(
            args.optc_labels,
            args.scenario_manifest,
            max_positive_items=max(1, args.sweep_max_items // 2),
            seed=args.seed,
        )
        points = [
            asdict(point)
            for point in run_parameter_sweep(
                sweep_dataset,
                confidence_thresholds=(0.85, 0.92, 0.97),
                graph_regularizations=(0.0, 0.3, 1.0, 3.0),
                seed=args.seed,
                min_models=args.min_models,
                warmup_rounds=min(args.warmup_rounds, 20),
                embedding_dim=min(args.embedding_dim, 32),
            )
        ]
        sweep_payload = {
            "dataset": sweep_dataset.stats,
            "disclaimer": (
                "Real attack IDs/topology, synthetic benign controls and simulated model scores."
            ),
            "points": points,
        }
        if args.sweep_json_out:
            args.sweep_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.sweep_json_out.write_text(
                json.dumps(sweep_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.sweep_csv_out:
            _write_sweep_csv(args.sweep_csv_out, points)
        if args.stability_json_out or args.stability_csv_out:
            try:
                stability_seeds = tuple(
                    int(value.strip())
                    for value in args.stability_seeds.split(",")
                    if value.strip()
                )
            except ValueError as exc:
                raise SystemExit("--stability-seeds must be comma-separated integers") from exc
            stability = run_seed_stability(
                sweep_dataset,
                seeds=stability_seeds,
                confidence_threshold=args.stability_delta,
                graph_regularization=args.stability_graph_lambda,
                min_models=args.min_models,
                warmup_rounds=min(args.warmup_rounds, 20),
                embedding_dim=min(args.embedding_dim, 32),
            ).to_dict()
            stability_payload = {
                "dataset": sweep_dataset.stats,
                "confidence_threshold": args.stability_delta,
                "graph_regularization": args.stability_graph_lambda,
                "disclaimer": (
                    "Real attack IDs/topology, synthetic benign controls and simulated models."
                ),
                **stability,
            }
            if args.stability_json_out:
                args.stability_json_out.parent.mkdir(parents=True, exist_ok=True)
                args.stability_json_out.write_text(
                    json.dumps(stability_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if args.stability_csv_out:
                _write_stability_csv(args.stability_csv_out, stability_payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
