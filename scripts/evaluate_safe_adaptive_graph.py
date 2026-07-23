"""Tune and evaluate Safe Adaptive G-CaMVo on frozen CASIE and Mordor caches.

No provider is instantiated and no network request is made.  One universal
configuration is selected by the mean validation Macro-F1 delta across both
datasets.  Test labels are opened only after that selection.  During each
partition the graph receives one-step-delayed gold feedback, modelling an
online SOC analyst/audit stream; only already processed nodes can be parents.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from camvo.adaptive_graph_router import AdaptiveGraphCaMVoRouter, AdaptiveGraphConfig
from camvo.causal_subset_router import StaticCausalTypedNeighborhood
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.router import CaMVoRouter
from camvo.security.casie import CASIE_LABELS, load_casie_event_items
from camvo.security.formal_experiment import FrozenResponseMatrixClient, _base_router_config
from camvo.security.metrics import classification_metrics
from camvo.security.splits import DatasetSplit, stratified_grouped_split
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OfflinePanel:
    name: str
    labels: tuple[str, ...]
    model_ids: tuple[str, ...]
    pricing: dict[str, ModelPricing]
    matrix: dict[str, dict[str, ModelResponse]]
    split: DatasetSplit
    base_config: Any
    neighborhood: StaticCausalTypedNeighborhood
    order_key: Callable[[AnnotationItem], tuple[Any, ...]]
    cluster_key: Callable[[AnnotationItem], str]
    source_items: int
    complete_items: int


def _response(raw: Mapping[str, Any]) -> ModelResponse:
    return ModelResponse(
        label=str(raw["label"]),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 1)),
        raw=raw.get("raw"),
    )


def _clients(panel: OfflinePanel) -> list[FrozenResponseMatrixClient]:
    return [
        FrozenResponseMatrixClient(
            model_id,
            panel.pricing[model_id],
            {
                item_id: row[model_id]
                for item_id, row in panel.matrix.items()
                if model_id in row
            },
        )
        for model_id in panel.model_ids
    ]


def _ordered(panel: OfflinePanel, items: Sequence[AnnotationItem]) -> tuple[AnnotationItem, ...]:
    return tuple(sorted(items, key=panel.order_key))


def _extract_snapshot(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError("unsafe path in CASIE snapshot archive")
        handle.extractall(destination)
    return destination


def _casie_cache(run_dir: Path) -> tuple[
    dict[str, dict[str, ModelResponse]], tuple[str, ...], dict[str, Any]
]:
    manifest = json.loads((run_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_ids = tuple(str(value) for value in manifest["model_ids"])
    rows: dict[str, dict[str, ModelResponse]] = defaultdict(dict)
    for path in sorted((run_dir / "responses").glob("*/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        item_id = str(raw["key"]["item_id"])
        model_id = str(raw["key"]["model_id"])
        if model_id in model_ids:
            rows[item_id][model_id] = _response(raw["response"])
    complete = {item_id: row for item_id, row in rows.items() if set(row) == set(model_ids)}
    return complete, model_ids, config


def _casie_parents(items: Sequence[AnnotationItem]) -> StaticCausalTypedNeighborhood:
    documents: dict[str, list[AnnotationItem]] = defaultdict(list)
    for item in items:
        documents[str(item.metadata["document_id"])].append(item)
    parents: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rows in documents.values():
        ordered = sorted(
            rows, key=lambda item: (int(item.metadata["start_offset"]), item.item_id)
        )
        for parent, child in zip(ordered, ordered[1:]):
            same_hopper = int(parent.metadata["hopper_index"]) == int(
                child.metadata["hopper_index"]
            )
            gap = max(
                0,
                int(child.metadata["start_offset"]) - int(parent.metadata["start_offset"]),
            )
            parents[child.item_id][parent.item_id] = {
                "weight": (1.0 if same_hopper else 0.20) * (2.718281828 ** (-gap / 800.0)),
                "relations": [
                    "same_hopper_sequence" if same_hopper else "document_sequence"
                ],
            }
    return StaticCausalTypedNeighborhood(dict(parents))


def load_casie_panel(
    run_dir: Path,
    casie_dir: Path,
    exclude_manifest: Path | None,
) -> OfflinePanel:
    matrix, model_ids, config = _casie_cache(run_dir)
    dataset = load_casie_event_items(casie_dir, strict=True)
    excluded_documents: set[str] = set()
    if exclude_manifest is not None:
        payload = json.loads(exclude_manifest.read_text(encoding="utf-8"))
        for item_id in payload.get("completed_item_ids", []):
            parts = str(item_id).split(":")
            if len(parts) >= 3:
                excluded_documents.add(parts[1])
    items = tuple(
        item
        for item in dataset.items
        if item.item_id in matrix
        and str(item.metadata["document_id"]) not in excluded_documents
    )
    allowed = {item.item_id for item in items}
    matrix = {item_id: matrix[item_id] for item_id in allowed}
    formal = dict(config.get("formal", {}))
    split = stratified_grouped_split(
        items,
        group_key=lambda item: str(item.metadata["document_id"]),
        calibration_fraction=float(formal.get("calibration_fraction", 0.20)),
        validation_fraction=float(formal.get("validation_fraction", 0.10)),
        seed=int(config.get("seed", 17)),
    )
    pricing = {
        str(row["model_id"]): ModelPricing(
            float(row["input_usd_per_million"]),
            float(row["output_usd_per_million"]),
        )
        for row in config["models"]
    }
    return OfflinePanel(
        name="CASIE",
        labels=CASIE_LABELS,
        model_ids=model_ids,
        pricing=pricing,
        matrix=matrix,
        split=split,
        base_config=_base_router_config(config),
        neighborhood=_casie_parents(items),
        order_key=lambda item: (
            str(item.metadata["document_id"]),
            int(item.metadata["start_offset"]),
            item.item_id,
        ),
        cluster_key=lambda item: str(item.metadata["document_id"]),
        source_items=len(dataset.items),
        complete_items=len(items),
    )


def load_mordor_panel(bundle_dir: Path) -> OfflinePanel:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    models = json.loads((bundle_dir / "models.json").read_text(encoding="utf-8"))
    model_ids = tuple(str(row["model_id"]) for row in models)
    pricing = {
        str(row["model_id"]): ModelPricing(
            float(row["input_usd_per_million"]),
            float(row["output_usd_per_million"]),
        )
        for row in models
    }
    raw_rows = [
        json.loads(line)
        for line in (bundle_dir / "items_with_responses.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    by_id = {str(row["item_id"]): row for row in raw_rows}
    items = tuple(
        AnnotationItem(
            item_id=str(row["item_id"]),
            text=str(row["text"]),
            labels=tuple(row["labels"]),
            metadata=dict(row["metadata"]),
        )
        for row in raw_rows
        if all(model_id in row["responses"] for model_id in model_ids)
    )
    complete_ids = {item.item_id for item in items}
    matrix = {
        item.item_id: {
            model_id: _response(by_id[item.item_id]["responses"][model_id])
            for model_id in model_ids
        }
        for item in items
    }
    split_ids = json.loads((bundle_dir / "splits.json").read_text(encoding="utf-8"))
    item_map = {item.item_id: item for item in items}

    def partition(name: str) -> tuple[AnnotationItem, ...]:
        return tuple(item_map[item_id] for item_id in split_ids[name] if item_id in complete_ids)

    parents: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in (bundle_dir / "directed_edges.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        edge = json.loads(line)
        parent = str(edge["parent_id"])
        child = str(edge["child_id"])
        if parent in complete_ids and child in complete_ids:
            parents[child][parent] = {
                "weight": float(edge["weight"]),
                "relations": [f"provenance_{value}" for value in edge["reasons"]],
            }
    split = DatasetSplit(
        calibration=partition("calibration"),
        validation=partition("validation"),
        test=partition("test"),
    )
    return OfflinePanel(
        name="Mordor",
        labels=("benign", "malicious"),
        model_ids=model_ids,
        pricing=pricing,
        matrix=matrix,
        split=split,
        base_config=_base_router_config({"router": {}}),
        neighborhood=StaticCausalTypedNeighborhood(dict(parents)),
        order_key=lambda item: (int(item.metadata.get("timestamp_ms", 0)), item.item_id),
        cluster_key=lambda item: (
            f"{item.metadata.get('hostname', 'unknown')}:"
            f"{int(item.metadata.get('timestamp_ms', 0)) // 1_800_000}"
        ),
        source_items=int(manifest["items"]),
        complete_items=len(items),
    )


def _warm(router: CaMVoRouter, panel: OfflinePanel) -> None:
    for item in _ordered(panel, panel.split.calibration):
        router.observe_complete_feedback(
            item,
            dict(panel.matrix[item.item_id]),
            str(item.metadata["gold_label"]),
        )
    reset = getattr(router, "reset_graph_history", None)
    if callable(reset):
        reset()


def _run_partition(
    router: CaMVoRouter,
    panel: OfflinePanel,
    items: Sequence[AnnotationItem],
    *,
    graph_feedback: bool,
) -> dict[str, Any]:
    gold: list[str] = []
    predictions: list[str] = []
    ids: list[str] = []
    clusters: list[str] = []
    costs: list[float] = []
    model_counts: list[int] = []
    graph_rows = 0
    fallback_reasons: Counter[str] = Counter()
    for item in _ordered(panel, items):
        result = router.route(item)
        label = str(item.metadata["gold_label"])
        gold.append(label)
        predictions.append(result.label)
        ids.append(item.item_id)
        clusters.append(panel.cluster_key(item))
        costs.append(float(result.actual_cost))
        model_counts.append(len(result.selected_models))
        graph_rows += int(result.graph_evidence_weight > 0)
        if result.routing_trace:
            reason = result.routing_trace[-1].get("fallback_reason")
            fallback_reasons[str(reason or "graph_applied")] += 1
        if graph_feedback:
            assert isinstance(router, AdaptiveGraphCaMVoRouter)
            router.observe_graph_feedback(item, label)
    metrics = classification_metrics(gold, predictions, panel.labels).to_dict()
    return {
        "metrics": metrics,
        "gold": gold,
        "predictions": predictions,
        "item_ids": ids,
        "clusters": clusters,
        "total_cost_usd": sum(costs),
        "average_models": sum(model_counts) / len(model_counts),
        "graph_applied_rows": graph_rows,
        "graph_applied_rate": graph_rows / len(items),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def _adaptive(panel: OfflinePanel, config: AdaptiveGraphConfig) -> AdaptiveGraphCaMVoRouter:
    return AdaptiveGraphCaMVoRouter(
        _clients(panel),
        HashingTextEmbedder(panel.base_config.embedding_dim),
        panel.base_config,
        config,
        panel.neighborhood,
    )


def _baseline(panel: OfflinePanel) -> CaMVoRouter:
    return CaMVoRouter(
        _clients(panel),
        HashingTextEmbedder(panel.base_config.embedding_dim),
        panel.base_config,
    )


def _validation_baseline(panel: OfflinePanel) -> dict[str, Any]:
    router = _baseline(panel)
    _warm(router, panel)
    return _run_partition(router, panel, panel.split.validation, graph_feedback=False)


def _validation_adaptive(panel: OfflinePanel, config: AdaptiveGraphConfig) -> dict[str, Any]:
    router = _adaptive(panel, config)
    _warm(router, panel)
    result = _run_partition(router, panel, panel.split.validation, graph_feedback=True)
    result["graph_diagnostics"] = router.graph_diagnostics()
    return result


def _candidate_configs() -> tuple[AdaptiveGraphConfig, ...]:
    rows: list[AdaptiveGraphConfig] = []
    for blend, threshold, protect, dynamic in itertools.product(
        (0.20, 0.35, 0.50),
        (0.45, 0.50),
        (0.20, 0.40),
        (False, True),
    ):
        rows.append(
            AdaptiveGraphConfig(
                gate_activation_threshold=threshold,
                max_graph_blend=blend,
                protect_base_margin=protect,
                dynamic_entity_edges=dynamic,
                min_transition_observations=6.0,
                min_gate_observations=8.0,
                gate_lower_z=0.5,
                max_js_divergence=0.35,
                minimum_graph_margin=0.03,
                require_entropy_reduction=True,
            )
        )
    return tuple(rows)


def _public(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"gold", "predictions", "item_ids", "clusters"}
    }


def analyze(
    casie_panel: OfflinePanel,
    mordor_panel: OfflinePanel,
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    panels = (casie_panel, mordor_panel)
    baselines = {panel.name: _validation_baseline(panel) for panel in panels}
    search: list[dict[str, Any]] = []
    for config in _candidate_configs():
        results = {
            panel.name: _validation_adaptive(panel, config) for panel in panels
        }
        deltas = {
            panel.name: (
                results[panel.name]["metrics"]["macro_f1"]
                - baselines[panel.name]["metrics"]["macro_f1"]
            )
            for panel in panels
        }
        negative_penalty = sum(max(0.0, -value) for value in deltas.values())
        objective = sum(deltas.values()) / len(deltas) - negative_penalty
        search.append(
            {
                "config": asdict(config),
                "validation_macro_f1_delta_vs_camvo": deltas,
                "objective": objective,
                "graph_applied_rate": {
                    panel.name: results[panel.name]["graph_applied_rate"] for panel in panels
                },
            }
        )
    selected_row = max(
        search,
        key=lambda row: (
            row["objective"],
            min(row["validation_macro_f1_delta_vs_camvo"].values()),
            -row["config"]["max_graph_blend"],
            not row["config"]["dynamic_entity_edges"],
        ),
    )
    selected = AdaptiveGraphConfig(**selected_row["config"])
    test_results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for panel in panels:
        base_router = _baseline(panel)
        _warm(base_router, panel)
        _run_partition(base_router, panel, panel.split.validation, graph_feedback=False)
        base_test = _run_partition(base_router, panel, panel.split.test, graph_feedback=False)

        graph_router = _adaptive(panel, selected)
        _warm(graph_router, panel)
        _run_partition(graph_router, panel, panel.split.validation, graph_feedback=True)
        graph_router.reset_graph_history()
        graph_test = _run_partition(
            graph_router, panel, panel.split.test, graph_feedback=True
        )
        graph_test["graph_diagnostics"] = graph_router.graph_diagnostics()
        comparison = paired_cluster_bootstrap_delta(
            graph_test["gold"],
            graph_test["predictions"],
            base_test["predictions"],
            panel.labels,
            graph_test["clusters"],
            metric="macro_f1",
            iterations=bootstrap_iterations,
            seed=29,
        ).to_dict()
        test_results[panel.name] = {
            "camvo": _public(base_test),
            "safe_adaptive_gcamvo": _public(graph_test),
        }
        comparisons[panel.name] = comparison

    return {
        "status": "offline_real_response_adaptive_graph_evaluation",
        "provider_calls": 0,
        "selection_protocol": {
            "universal_config_selected_on": "mean CASIE+Mordor validation Macro-F1 delta",
            "test_labels_used_for_selection": False,
            "online_feedback": "one-step delayed audited label; past nodes only",
            "negative_delta_penalty": True,
        },
        "datasets": {
            panel.name: {
                "source_items": panel.source_items,
                "complete_items": panel.complete_items,
                "split": {
                    "calibration": len(panel.split.calibration),
                    "validation": len(panel.split.validation),
                    "test": len(panel.split.test),
                },
                "models": list(panel.model_ids),
            }
            for panel in panels
        },
        "validation_camvo": {
            name: _public(result) for name, result in baselines.items()
        },
        "validation_search": sorted(search, key=lambda row: row["objective"], reverse=True),
        "selected_config": asdict(selected),
        "test": test_results,
        "paired_cluster_bootstrap": comparisons,
        "claim_boundary": (
            "All LLM responses are frozen real Adams outputs. CASIE is an in-progress "
            "complete-case snapshot and Mordor has 913/1000 complete rows. Hyperparameters "
            "are validation-selected; test feedback is strictly prequential and never affects "
            "an earlier prediction. This is an algorithm-development result, not a final paper claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casie-snapshot",
        type=Path,
        default=ROOT
        / "artifacts"
        / "casie_live_snapshot_20260723"
        / "CASIE_RESPONSES_24470_SNAPSHOT.tar.gz",
    )
    parser.add_argument(
        "--casie-dir", type=Path, default=ROOT / "data" / "raw" / "casie" / "data"
    )
    parser.add_argument(
        "--casie-exclude-manifest",
        type=Path,
        default=ROOT / "artifacts" / "adams_formal_casie_500" / "matrix_manifest.json",
    )
    parser.add_argument(
        "--mordor-bundle",
        type=Path,
        default=ROOT / "data" / "derived" / "mordor_offline_bundle",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "artifacts" / "safe_adaptive_graph" / "RESULTS.json",
    )
    args = parser.parse_args()
    if args.casie_snapshot.is_dir():
        casie_run = args.casie_snapshot
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="gcamvo-casie-")
        casie_run = _extract_snapshot(args.casie_snapshot, Path(temporary.name))
    try:
        report = analyze(
            load_casie_panel(casie_run, args.casie_dir, args.casie_exclude_manifest),
            load_mordor_panel(args.mordor_bundle),
            bootstrap_iterations=args.bootstrap_iterations,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
