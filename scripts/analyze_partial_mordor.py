"""Generate a no-provider-call provisional analysis from the Mordor cache.

This script is intentionally descriptive.  It reports per-model metrics on
available cells and on the complete-case intersection, but it does not claim a
CaMVo/G-CaMVo comparison before the frozen response matrix is complete.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from camvo.security.metrics import classification_metrics
from camvo.security.mordor_dataset import build_mordor_cdb_binary_dataset


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _cache_rows(cache_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    created: dict[tuple[str, str], float] = {}
    for path in cache_dir.glob("*/*.json"):
        record = _read_json(path)
        key = record.get("key", {})
        response = record.get("response", {})
        model_id = str(key.get("model_id", ""))
        item_id = str(key.get("item_id", ""))
        label = str(response.get("label", ""))
        if not model_id or not item_id or not label:
            continue
        timestamp = float(record.get("created_at_unix", 0.0))
        identity = (model_id, item_id)
        if identity in created and timestamp <= created[identity]:
            continue
        created[identity] = timestamp
        rows.setdefault(item_id, {})[model_id] = response
    return rows


def _metric_dict(gold: list[str], predicted: list[str]) -> dict[str, object]:
    return classification_metrics(gold, predicted, ("benign", "malicious")).to_dict()


def analyze(run_dir: Path) -> dict[str, object]:
    config = _read_json(run_dir / "config.json")
    dataset = config["dataset"]
    requested = int(dataset["max_items"])
    built = build_mordor_cdb_binary_dataset(
        dataset["path"],
        dataset["flags_path"],
        max_positive_items=None if dataset.get("require_all_flags") else requested // 2,
        correlation_window_minutes=int(dataset.get("correlation_window_minutes", 30)),
        max_records_per_item=int(dataset.get("max_records_per_item", 6)),
        max_chars_per_record=int(dataset.get("max_chars_per_record", 1_200)),
    )
    items = {item.item_id: item for item in built.items}
    if len(items) != requested:
        raise RuntimeError(f"dataset reconstruction produced {len(items)}/{requested} items")

    model_config = {str(model["model_id"]): model for model in config["models"]}
    model_ids = tuple(model_config)
    rows = _cache_rows(Path(config["budget"]["cache_dir"]))
    available = {
        model_id: sorted(
            item_id for item_id, row in rows.items() if model_id in row and item_id in items
        )
        for model_id in model_ids
    }
    complete_items = sorted(
        item_id
        for item_id, row in rows.items()
        if item_id in items and all(model_id in row for model_id in model_ids)
    )

    per_model: dict[str, object] = {}
    for model_id in model_ids:
        ids = available[model_id]
        gold = [str(items[item_id].metadata["gold_label"]) for item_id in ids]
        predicted = [str(rows[item_id][model_id]["label"]) for item_id in ids]
        common_gold = [
            str(items[item_id].metadata["gold_label"]) for item_id in complete_items
        ]
        common_predicted = [
            str(rows[item_id][model_id]["label"]) for item_id in complete_items
        ]
        raw_cost = 0.0
        input_price = float(model_config[model_id]["input_usd_per_million"])
        output_price = float(model_config[model_id]["output_usd_per_million"])
        confidences: list[float] = []
        for item_id in ids:
            response = rows[item_id][model_id]
            raw_cost += (
                int(response.get("input_tokens", 0)) * input_price
                + int(response.get("output_tokens", 0)) * output_price
            ) / 1_000_000
            raw = response.get("raw", {})
            confidence = raw.get("confidence") if isinstance(raw, dict) else None
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
        per_model[model_id] = {
            "available_cells": len(ids),
            "coverage": len(ids) / requested,
            "missing_cells": requested - len(ids),
            "available_case_metrics": _metric_dict(gold, predicted),
            "complete_case_items": len(complete_items),
            "complete_case_metrics": (
                _metric_dict(common_gold, common_predicted) if complete_items else None
            ),
            "cached_proxy_cost_usd": raw_cost,
            "mean_confidence": mean(confidences) if confidences else None,
        }

    agreement_counts: Counter[int] = Counter()
    pairwise: dict[str, float] = {}
    for item_id in complete_items:
        agreement_counts[len({rows[item_id][model_id]["label"] for model_id in model_ids})] += 1
    for left, right in combinations(model_ids, 2):
        pairwise[f"{left}__{right}"] = (
            sum(rows[item_id][left]["label"] == rows[item_id][right]["label"] for item_id in complete_items)
            / len(complete_items)
            if complete_items
            else 0.0
        )

    budget = _read_json(run_dir / "budget.json")
    return {
        "status": "provisional_partial_matrix",
        "claim_boundary": (
            "Descriptive only. CaMVo/G-CaMVo comparisons are deferred until all methods can "
            "reuse the same complete frozen response matrix."
        ),
        "dataset_items": requested,
        "models": len(model_ids),
        "expected_cells": requested * len(model_ids),
        "cached_cells": sum(len(ids) for ids in available.values()),
        "complete_case_items": len(complete_items),
        "complete_case_fraction": len(complete_items) / requested,
        "per_model": per_model,
        "agreement": {
            "distinct_label_count_histogram": dict(sorted(agreement_counts.items())),
            "unanimous_fraction": (
                agreement_counts[1] / len(complete_items) if complete_items else 0.0
            ),
            "pairwise_agreement": pairwise,
        },
        "ledger": {
            key: budget.get(key)
            for key in (
                "provider_attempts",
                "completed_calls",
                "failed_calls",
                "spent_usd",
            )
        },
    }


def _markdown(report: dict[str, object]) -> str:
    per_model = report["per_model"]
    assert isinstance(per_model, dict)
    lines = [
        "# Mordor provisional cached-result analysis",
        "",
        "> **阶段性结果，不是最终实验结论。** 这里没有调用任何新 API。",
        "",
        f"- 数据集样本：{report['dataset_items']}",
        f"- 已缓存模型响应：{report['cached_cells']}/{report['expected_cells']}",
        f"- 四模型均齐全的可公平比较样本：{report['complete_case_items']} "
        f"({float(report['complete_case_fraction']):.1%})",
        "",
        "| Model | Coverage | Available Macro-F1 | Complete-case Macro-F1 | Complete-case Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, raw in per_model.items():
        assert isinstance(raw, dict)
        available_metrics = raw["available_case_metrics"]
        common_metrics = raw["complete_case_metrics"]
        assert isinstance(available_metrics, dict)
        common_f1 = "n/a" if not isinstance(common_metrics, dict) else f"{float(common_metrics['macro_f1']):.4f}"
        common_recall = "n/a" if not isinstance(common_metrics, dict) else f"{float(common_metrics['macro_recall']):.4f}"
        lines.append(
            f"| {model_id} | {float(raw['coverage']):.1%} | "
            f"{float(available_metrics['macro_f1']):.4f} | {common_f1} | {common_recall} |"
        )
    agreement = report["agreement"]
    assert isinstance(agreement, dict)
    lines.extend(
        [
            "",
            f"- 四模型一致率：{float(agreement['unanimous_fraction']):.1%}",
            "- Available-case 指标可能受缺失非随机影响；横向判断优先看 complete-case 列。",
            "- CaMVo、CCaMVo、G-CaMVo 和完整集成的正式比较必须等待冻结矩阵补齐。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "artifacts" / "adams_formal_mordor_1000",
    )
    args = parser.parse_args()
    report = analyze(args.run_dir)
    json_path = args.run_dir / "PARTIAL_ANALYSIS.json"
    markdown_path = args.run_dir / "PARTIAL_ANALYSIS.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
