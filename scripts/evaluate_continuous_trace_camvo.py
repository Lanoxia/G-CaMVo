"""Discover confidence-aware Continuous TRACE-CaMVo on frozen Mordor outputs.

This script performs zero provider calls.  Hyperparameters are chosen on the
fixed validation split and evaluated on test, but Mordor has already been used
during method development; the output is therefore explicitly exploratory and
must be externally confirmed on an untouched dataset such as CASIE.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from camvo.continuous_trace_router import (
    ContinuousTraceCaMVoRouter,
    ContinuousTraceConfig,
)
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.router import CaMVoRouter
from camvo.security.formal_experiment import _base_router_config
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.types import ModelPricing

from evaluate_causal_subset_trace import (
    _clients,
    _evaluate,
    _public,
    _rank,
    _typed_parents,
    _warm_start,
    replace_base_config,
)
from search_mordor_trace_algorithms import LABELS, _group_id, _load_offline_bundle
from camvo.causal_subset_router import StaticCausalTypedNeighborhood


ROOT = Path(__file__).resolve().parents[1]


def analyze(bundle_dir: Path, *, bootstrap_iterations: int = 2_000) -> dict[str, Any]:
    config, built, model_config, model_ids, matrix, complete, split, requested = (
        _load_offline_bundle(bundle_dir)
    )
    pricing = {
        model_id: ModelPricing(
            input_per_million=float(model_config[model_id]["input_usd_per_million"]),
            output_per_million=float(model_config[model_id]["output_usd_per_million"]),
        )
        for model_id in model_ids
    }
    base_config = _base_router_config(config)
    neighborhood = StaticCausalTypedNeighborhood(
        _typed_parents(built.correlations, {item.item_id for item in complete})
    )

    def router(trace: ContinuousTraceConfig, *, full_panel: bool) -> ContinuousTraceCaMVoRouter:
        routing_config = (
            replace_base_config(base_config, warmup_rounds=1_000_000)
            if full_panel
            else base_config
        )
        return ContinuousTraceCaMVoRouter(
            _clients(model_ids, pricing, matrix),
            HashingTextEmbedder(routing_config.embedding_dim),
            routing_config,
            trace,
            neighborhood,
        )

    def search(
        *, full_panel: bool, trim_each_side: int
    ) -> tuple[
        ContinuousTraceConfig,
        dict[str, Any],
        ContinuousTraceConfig,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        rows: list[tuple[ContinuousTraceConfig, dict[str, Any]]] = []
        for graph_strength in (0.0, 0.05, 0.1, 0.2, 0.3):
            for malicious_bias in (2.0, 2.5, 3.0, 3.5, 4.0):
                candidate = ContinuousTraceConfig(
                    graph_strength=graph_strength,
                    trim_each_side=trim_each_side,
                    label_logit_bias=(("malicious", malicious_bias),),
                )
                result = _evaluate(
                    _warm_start(router(candidate, full_panel=full_panel), split.calibration, matrix),
                    split.validation,
                )
                rows.append((candidate, result))
        selected_config, selected_result = max(rows, key=lambda row: _rank(row[1]))
        no_graph_config, no_graph_result = max(
            (row for row in rows if row[0].graph_strength == 0.0),
            key=lambda row: _rank(row[1]),
        )
        return (
            selected_config,
            selected_result,
            no_graph_config,
            no_graph_result,
            [
                {"config": asdict(candidate), "result": _public(result)}
                for candidate, result in rows
            ],
        )

    (
        efficient_config,
        efficient_validation,
        efficient_no_graph_config,
        efficient_no_graph_validation,
        efficient_grid,
    ) = search(
        full_panel=False, trim_each_side=0
    )
    (
        robust_config,
        robust_validation,
        robust_no_graph_config,
        robust_no_graph_validation,
        robust_grid,
    ) = search(
        full_panel=True, trim_each_side=1
    )
    baseline = _evaluate(
        _warm_start(
            CaMVoRouter(
                _clients(model_ids, pricing, matrix),
                HashingTextEmbedder(base_config.embedding_dim),
                base_config,
            ),
            split.calibration,
            matrix,
        ),
        split.test,
    )
    efficient = _evaluate(
        _warm_start(router(efficient_config, full_panel=False), split.calibration, matrix),
        split.test,
    )
    robust = _evaluate(
        _warm_start(router(robust_config, full_panel=True), split.calibration, matrix),
        split.test,
    )
    efficient_no_graph = _evaluate(
        _warm_start(
            router(efficient_no_graph_config, full_panel=False),
            split.calibration,
            matrix,
        ),
        split.test,
    )
    robust_no_graph = _evaluate(
        _warm_start(
            router(robust_no_graph_config, full_panel=True),
            split.calibration,
            matrix,
        ),
        split.test,
    )
    efficient_matched_no_graph_config = ContinuousTraceConfig(
        **{
            **asdict(efficient_config),
            "graph_strength": 0.0,
        }
    )
    robust_matched_no_graph_config = ContinuousTraceConfig(
        **{
            **asdict(robust_config),
            "graph_strength": 0.0,
        }
    )
    efficient_matched_no_graph = _evaluate(
        _warm_start(
            router(efficient_matched_no_graph_config, full_panel=False),
            split.calibration,
            matrix,
        ),
        split.test,
    )
    robust_matched_no_graph = _evaluate(
        _warm_start(
            router(robust_matched_no_graph_config, full_panel=True),
            split.calibration,
            matrix,
        ),
        split.test,
    )
    clusters_by_id = {item.item_id: _group_id(item) for item in split.test}

    def bootstrap(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return paired_cluster_bootstrap_delta(
            candidate["gold"],
            candidate["predictions"],
            baseline["predictions"],
            LABELS,
            [clusters_by_id[item_id] for item_id in candidate["item_ids"]],
            metric="macro_f1",
            iterations=bootstrap_iterations,
            seed=17,
        ).to_dict()

    return {
        "status": "exploratory_continuous_trace_camvo_discovery",
        "provider_calls": 0,
        "claim_boundary": (
            "Post-hoc Mordor method-development result, not a confirmatory frozen-test claim. "
            "The algorithm and hyperparameter ranges must be frozen before evaluation on an "
            "untouched external dataset such as CASIE. Calibration uses audited gold labels."
        ),
        "algorithm_contract": {
            "subset_selected_before_current_votes": True,
            "camvo_omega_mu_times_q_used": True,
            "model_only_consensus_drives_camvo_online_updates": True,
            "graph_fused_label_is_task_decision_head": True,
            "past_parents_only": True,
            "fixed_model_or_vendor_rule": False,
            "reported_confidence_required_by_task_contract": True,
        },
        "requested_items": requested,
        "complete_case_items": len(complete),
        "split": {
            "calibration": len(split.calibration),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "selected": {
            "efficient": {
                "config": asdict(efficient_config),
                "validation": _public(efficient_validation),
                "test": _public(efficient),
                "bootstrap_vs_camvo": bootstrap(efficient),
                "no_graph_ablation": {
                    "config": asdict(efficient_no_graph_config),
                    "validation": _public(efficient_no_graph_validation),
                    "test": _public(efficient_no_graph),
                },
                "matched_bias_no_graph_ablation": _public(
                    efficient_matched_no_graph
                ),
            },
            "robust_full_panel": {
                "config": asdict(robust_config),
                "validation": _public(robust_validation),
                "test": _public(robust),
                "bootstrap_vs_camvo": bootstrap(robust),
                "no_graph_ablation": {
                    "config": asdict(robust_no_graph_config),
                    "validation": _public(robust_no_graph_validation),
                    "test": _public(robust_no_graph),
                },
                "matched_bias_no_graph_ablation": _public(robust_matched_no_graph),
            },
        },
        "baseline": {"audited_warm_start_camvo": _public(baseline)},
        "validation_grids": {
            "efficient": efficient_grid,
            "robust_full_panel": robust_grid,
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    baseline = report["baseline"]["audited_warm_start_camvo"]
    rows = [("audited_warm_start_camvo", baseline)] + [
        (name, payload["test"]) for name, payload in report["selected"].items()
    ]
    lines = [
        "# Continuous TRACE-CaMVo Mordor discovery result",
        "",
        "> 这是方法开发结果，不是外部确认结果；零新增模型调用。",
        "",
        "| 方法 | Macro-F1 | 恶意召回 | 成本($) | 平均模型数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(
            f"| {name} | {row['metrics']['macro_f1']:.4f} | "
            f"{row['metrics']['per_class']['malicious']['recall']:.4f} | "
            f"{row['total_cost_usd']:.4f} | {row['average_models']:.2f} |"
        )
    lines.extend(["", "## 声明边界", "", str(report["claim_boundary"])])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "data" / "derived" / "mordor_offline_bundle",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()
    report = analyze(args.bundle_dir, bootstrap_iterations=args.bootstrap_iterations)
    json_path = args.bundle_dir / "CONTINUOUS_TRACE_DISCOVERY.json"
    md_path = args.bundle_dir / "CONTINUOUS_TRACE_DISCOVERY.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
