"""Evaluate the frozen causal graph-routing suite on the complete CASIE matrix.

The script never instantiates a provider.  It replays the 33,940 downloaded
responses, removes CASIE documents used during the earlier 500-item method
development run, and evaluates frozen SAGE/DCR/State-Switch settings with
whole-document outer folds.  Audited labels arrive only after the current
prediction and can affect future events in the same online replay.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from evaluate_dcr_casie_snapshot import analyze
from evaluate_safe_adaptive_graph import load_casie_panel
from camvo.security.statistics import paired_cluster_bootstrap_delta


ROOT = Path(__file__).resolve().parents[1]


METHOD_ORDER = (
    "single::dify/deepseek-v4-flash",
    "single::dify/deepseek-v4-pro",
    "single::dify/glm-5.2-fp8",
    "single::dify/minimax-m3",
    "best_train_single",
    "majority_vote",
    "calibration_weighted_vote",
    "document_persistence_best_single",
    "parent_persistence_best_single",
    "camvo_k1",
    "camvo_k2",
    "sage",
    "dcr_k1_frozen",
    "dcr_k2_reliability_only",
    "dcr_k2_diversity_no_graph",
    "dcr_k2_frozen",
    "state_switch_gcamvo_k2_frozen",
)


def _render(report: Mapping[str, Any]) -> str:
    lines = [
        "# Complete CASIE frozen-response causal routing suite",
        "",
        "## Protocol",
        "",
        f"- Complete source matrix: {report['matrix_response_files']:,} responses "
        f"for {report['source_matrix_items']:,} events;",
        f"- Leakage-controlled cohort: {report['eligible_complete_items']:,} events "
        f"from {report['documents']:,} whole documents;",
        f"- Excluded prior development material: {report['excluded_development_items']:,} "
        f"events from {report['excluded_development_documents']:,} documents;",
        f"- Outer evaluation: {report['outer_folds']} document-grouped folds;",
        "- Frozen hyperparameters transferred from the earlier Mordor/SAGE development;",
        "- Test feedback is post-prediction and past-only; provider calls: 0.",
        "",
        "## Main OOF table",
        "",
        "| Method | Accuracy | Macro-F1 | Macro-recall | Avg. models | Proxy cost USD | Graph/state rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = report["summaries"][method]
        metrics = row["metrics"]
        lines.append(
            f"| {method} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['macro_recall']:.4f} | {row['average_models']:.3f} | "
            f"{row['total_cost_usd']:.4f} | {100 * row['graph_applied_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## SAGE paired document-cluster bootstrap",
            "",
            "| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["sage_bootstrap"].items():
        lines.append(
            f"| {name} | {row['candidate_minus_reference']:+.5f} | "
            f"[{row['lower']:+.5f}, {row['upper']:+.5f}] | "
            f"{100 * row['probability_candidate_better']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## DCR k=2 paired document-cluster bootstrap",
            "",
            "| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["dcr_bootstrap"].items():
        lines.append(
            f"| {name} | {row['candidate_minus_reference']:+.5f} | "
            f"[{row['lower']:+.5f}, {row['upper']:+.5f}] | "
            f"{100 * row['probability_candidate_better']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## State-Switch paired document-cluster bootstrap",
            "",
            "| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["state_switch_bootstrap"].items():
        lines.append(
            f"| {name} | {row['candidate_minus_reference']:+.5f} | "
            f"[{row['lower']:+.5f}, {row['upper']:+.5f}] | "
            f"{100 * row['probability_candidate_better']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Feedback-rate sensitivity (Macro-F1)",
            "",
            "| Feedback rate | Persistence | DCR k=2 | State-Switch k=2 |",
            "|---:|---:|---:|---:|",
        ]
    )
    sensitivity = report["feedback_sensitivity"]
    for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
        key = f"rate_{rate:.2f}"
        lines.append(
            f"| {100 * rate:.0f}% | "
            f"{sensitivity['document_persistence'][key]['metrics']['macro_f1']:.4f} | "
            f"{sensitivity['dcr_k2'][key]['metrics']['macro_f1']:.4f} | "
            f"{sensitivity['state_switch_k2'][key]['metrics']['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Feedback-delay sensitivity (Macro-F1)",
            "",
            "| Delay (events) | Persistence | DCR k=2 | State-Switch k=2 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for delay in (0, 1, 5, 20):
        key = f"delay_{delay}"
        lines.append(
            f"| {delay} | "
            f"{sensitivity['document_persistence'][key]['metrics']['macro_f1']:.4f} | "
            f"{sensitivity['dcr_k2'][key]['metrics']['macro_f1']:.4f} | "
            f"{sensitivity['state_switch_k2'][key]['metrics']['macro_f1']:.4f} |"
        )
    lines.extend(["", "## Claim boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_records(report: Mapping[str, Any], path: Path) -> None:
    fields = (
        "method",
        "outer_fold",
        "item_id",
        "gold",
        "prediction",
        "cluster",
        "cost_usd",
        "models",
        "selected_models",
        "graph_applied",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in report["records"].items():
            for row in rows:
                writer.writerow(
                    {
                        "method": method,
                        **{field: row[field] for field in fields if field in row},
                        "selected_models": "|".join(row["selected_models"]),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "artifacts" / "adams_formal_casie_8485",
    )
    parser.add_argument(
        "--casie-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "casie" / "data",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=ROOT / "artifacts" / "adams_formal_casie_500" / "matrix_manifest.json",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "artifacts" / "full_casie_frozen_suite" / "RESULTS.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "artifacts" / "full_casie_frozen_suite" / "oof_records.csv",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "docs" / "FULL_CASIE_FROZEN_SUITE_2026-07-23.md",
    )
    args = parser.parse_args()

    all_panel = load_casie_panel(args.run_dir, args.casie_dir, None)
    clean_panel = load_casie_panel(
        args.run_dir, args.casie_dir, args.exclude_manifest
    )
    all_items = (
        all_panel.split.calibration + all_panel.split.validation + all_panel.split.test
    )
    clean_items = (
        clean_panel.split.calibration
        + clean_panel.split.validation
        + clean_panel.split.test
    )
    all_documents = {
        str(item.metadata["document_id"]) for item in all_items
    }
    clean_documents = {
        str(item.metadata["document_id"]) for item in clean_items
    }
    response_files = sum(1 for _ in (args.run_dir / "responses").glob("*/*.json"))
    report = analyze(
        clean_panel,
        outer_folds=args.outer_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        response_files=response_files,
    )
    report.update(
        {
            "status": "complete_casie_matrix_frozen_causal_routing_suite",
            "matrix_response_files": response_files,
            "source_matrix_items": len(all_items),
            "eligible_complete_items": len(clean_items),
            "documents": len(clean_documents),
            "excluded_development_items": len(all_items) - len(clean_items),
            "excluded_development_documents": len(all_documents - clean_documents),
            "hyperparameter_source": (
                "Frozen before this complete-matrix replay: DCR/state-switch settings "
                "transferred from Mordor and graph safety defaults from the earlier SAGE "
                "development study. The 23 CASIE development documents are excluded."
            ),
            "claim_boundary": (
                "Complete 33,940-cell real-model CASIE response matrix with zero new "
                "provider calls. OOF predictions use whole-document grouped folds and "
                "post-prediction past-only feedback. The 23 documents seen during earlier "
                "CASIE method development are excluded. CASIE supplies article sequence "
                "relations rather than host/process provenance, so independent OpTC "
                "confirmation is still required for SOC attack-chain claims."
            ),
        }
    )
    sage_rows = sorted(report["records"]["sage"], key=lambda row: row["item_id"])
    camvo_by_id = {
        row["item_id"]: row for row in report["records"]["camvo_k2"]
    }
    report["sage_bootstrap"] = {
        "vs_camvo_k2": paired_cluster_bootstrap_delta(
            [row["gold"] for row in sage_rows],
            [row["prediction"] for row in sage_rows],
            [camvo_by_id[row["item_id"]]["prediction"] for row in sage_rows],
            clean_panel.labels,
            [row["cluster"] for row in sage_rows],
            metric="macro_f1",
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1_500,
        ).to_dict()
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_records(report, args.csv_out)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_render(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "csv": str(args.csv_out),
                "markdown": str(args.markdown_out),
                "matrix_response_files": response_files,
                "eligible_items": len(clean_items),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
