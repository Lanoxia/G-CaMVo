"""Nested grouped cross-validation for DCR-G-CaMVo on frozen Mordor outputs.

The script performs no provider calls.  Whole host/time incident blocks are
assigned to outer folds.  DCR hyperparameters are selected only inside each
outer training partition, then evaluated on the untouched outer fold.  Every
graph decision is past-only and audited labels arrive after the corresponding
prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from camvo.adaptive_graph_router import AdaptiveGraphCaMVoRouter, AdaptiveGraphConfig
from camvo.aggregation import weighted_vote
from camvo.config import CaMVoConfig
from camvo.dcr_graph_router import DCRGraphCaMVoRouter, DCRGraphConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.router import CaMVoRouter
from camvo.security.metrics import classification_metrics
from camvo.security.statistics import paired_cluster_bootstrap_delta
from camvo.state_switch_router import (
    CausalStateSwitchGraphCaMVoRouter,
    StateSwitchConfig,
)
from camvo.types import AnnotationItem
from build_mordor_paper_tables import _cost
from evaluate_safe_adaptive_graph import OfflinePanel, _clients, _ordered, load_mordor_panel


ROOT = Path(__file__).resolve().parents[1]


def _switch_candidates() -> tuple[StateSwitchConfig, ...]:
    return tuple(
        StateSwitchConfig(
            pattern_prior_strength=prior,
            switch_threshold=threshold,
            information_cost_penalty=cost,
            minimum_pattern_support=support,
            subset_size=2,
        )
        for prior in (0.5, 2.0, 8.0, 32.0)
        for threshold in (0.12, 0.25, 0.50, 0.70)
        for cost in (0.01,)
        for support in (1, 4)
    )


def _stable(seed: int, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()[:8], "big"
    )


def _balanced_group_folds(
    items: Sequence[AnnotationItem],
    *,
    group_key: Callable[[AnnotationItem], str],
    folds: int,
    seed: int,
) -> list[tuple[AnnotationItem, ...]]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    groups: dict[str, list[AnnotationItem]] = defaultdict(list)
    labels: set[str] = set()
    for item in items:
        groups[group_key(item)].append(item)
        labels.add(str(item.metadata["gold_label"]))
    if len(groups) < folds:
        raise ValueError("number of groups is smaller than number of folds")
    label_order = tuple(sorted(labels))
    total = Counter(str(item.metadata["gold_label"]) for item in items)
    target_size = len(items) / folds
    target_label = {label: total[label] / folds for label in label_order}
    bins: list[list[AnnotationItem]] = [[] for _ in range(folds)]
    bin_labels = [Counter() for _ in range(folds)]
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            -max(Counter(str(i.metadata["gold_label"]) for i in groups[group]).values()),
            _stable(seed, group),
        ),
    )
    for group in ordered_groups:
        rows = groups[group]
        counts = Counter(str(item.metadata["gold_label"]) for item in rows)
        def need(fold: int) -> tuple[float, int]:
            size_need = (target_size - len(bins[fold])) / max(target_size, 1.0)
            label_need = sum(
                counts[label]
                * (target_label[label] - bin_labels[fold][label])
                / max(target_label[label], 1.0)
                for label in label_order
            ) / max(len(rows), 1)
            # Proportional deficits, rather than distance after insertion,
            # prevent a nearly full fold from attracting every small group.
            return 4.0 * size_need + label_need, -fold

        selected = max(range(folds), key=need)
        bins[selected].extend(rows)
        bin_labels[selected].update(counts)
    return [tuple(rows) for rows in bins]


def _causal_chunk_groups(
    items: Sequence[AnnotationItem],
    *,
    window_minutes: int = 5,
    max_events: int = 48,
) -> dict[str, str]:
    """Create bounded chronological blocks for stable development folds.

    Mordor contains several bursty host/windows with hundreds of timestamps.
    Leaving those buckets indivisible produced outer folds with a single giant
    group.  We retain host/time locality but split a burst sequentially; the
    original coarser incident cluster is still used for bootstrap uncertainty.
    """

    buckets: dict[str, list[AnnotationItem]] = defaultdict(list)
    width = window_minutes * 60_000
    for item in items:
        host = str(item.metadata.get("hostname", "unknown"))
        timestamp = int(item.metadata.get("timestamp_ms", 0))
        buckets[f"{host}:{timestamp // width}"].append(item)
    output: dict[str, str] = {}
    for bucket, rows in buckets.items():
        ordered = sorted(
            rows,
            key=lambda item: (int(item.metadata.get("timestamp_ms", 0)), item.item_id),
        )
        for index, item in enumerate(ordered):
            output[item.item_id] = f"{bucket}:chunk-{index // max_events}"
    return output


def _router(
    panel: OfflinePanel,
    *,
    kind: str,
    graph_config: AdaptiveGraphConfig,
    dcr_config: DCRGraphConfig | None = None,
    switch_config: StateSwitchConfig | None = None,
) -> CaMVoRouter:
    min_models = 1 if kind in {"camvo_k1", "dcr_k1"} else 2
    config: CaMVoConfig = replace(panel.base_config, min_models=min_models)
    embedder = HashingTextEmbedder(config.embedding_dim)
    clients = _clients(panel)
    if kind.startswith("camvo"):
        return CaMVoRouter(clients, embedder, config)
    if kind == "sage":
        return AdaptiveGraphCaMVoRouter(
            clients, embedder, config, graph_config, panel.neighborhood
        )
    if kind in {"dcr_k1", "dcr_k2"}:
        if dcr_config is None:
            raise ValueError("dcr_config is required")
        return DCRGraphCaMVoRouter(
            clients,
            embedder,
            config,
            graph_config,
            dcr_config,
            panel.neighborhood,
        )
    if kind == "state_switch":
        if switch_config is None:
            raise ValueError("switch_config is required")
        return CausalStateSwitchGraphCaMVoRouter(
            clients,
            embedder,
            config,
            graph_config,
            dcr_config or DCRGraphConfig(),
            switch_config,
            panel.neighborhood,
        )
    raise ValueError(f"unknown router kind: {kind}")


def _run_router(
    panel: OfflinePanel,
    train: Sequence[AnnotationItem],
    test: Sequence[AnnotationItem],
    *,
    kind: str,
    graph_config: AdaptiveGraphConfig,
    dcr_config: DCRGraphConfig | None = None,
    switch_config: StateSwitchConfig | None = None,
    feedback_rate: float = 1.0,
    feedback_delay: int = 0,
) -> list[dict[str, Any]]:
    if not 0 <= feedback_rate <= 1:
        raise ValueError("feedback_rate must be in [0, 1]")
    if feedback_delay < 0:
        raise ValueError("feedback_delay must be non-negative")
    router = _router(
        panel,
        kind=kind,
        graph_config=graph_config,
        dcr_config=dcr_config,
        switch_config=switch_config,
    )
    for item in _ordered(panel, train):
        router.observe_complete_feedback(
            item,
            dict(panel.matrix[item.item_id]),
            str(item.metadata["gold_label"]),
        )
    if isinstance(router, AdaptiveGraphCaMVoRouter):
        router.reset_graph_history()
    records = []
    feedback_queue: list[tuple[AnnotationItem, str, bool]] = []
    for item in _ordered(panel, test):
        result = router.route(item)
        gold = str(item.metadata["gold_label"])
        switch_trace = (
            result.routing_trace[-1]
            if isinstance(router, CausalStateSwitchGraphCaMVoRouter)
            else {}
        )
        records.append(
            {
                "item_id": item.item_id,
                "gold": gold,
                "prediction": result.label,
                "cluster": panel.cluster_key(item),
                "cost_usd": float(result.actual_cost),
                "models": len(result.selected_models),
                "selected_models": list(result.selected_models),
                "graph_applied": int(
                    result.graph_evidence_weight > 0
                    or switch_trace.get("previous_state") is not None
                ),
            }
        )
        if isinstance(router, AdaptiveGraphCaMVoRouter):
            audited = (
                _stable(991, item.item_id) / 2**64 < feedback_rate
            )
            feedback_queue.append((item, gold, audited))
            if len(feedback_queue) > feedback_delay:
                audited_item, audited_gold, should_audit = feedback_queue.pop(0)
                if should_audit:
                    router.observe_graph_feedback(audited_item, audited_gold)
    return records


def _static_records(
    panel: OfflinePanel,
    train: Sequence[AnnotationItem],
    test: Sequence[AnnotationItem],
    *,
    method: str,
) -> list[dict[str, Any]]:
    train_ordered = _ordered(panel, train)
    test_ordered = _ordered(panel, test)
    model_accuracy = {
        model_id: (
            1
            + sum(
                panel.matrix[item.item_id][model_id].label
                == str(item.metadata["gold_label"])
                for item in train_ordered
            )
        )
        / (len(train_ordered) + 2)
        for model_id in panel.model_ids
    }
    if method == "best_train_single":
        train_gold = [str(item.metadata["gold_label"]) for item in train_ordered]
        train_f1 = {
            model_id: classification_metrics(
                train_gold,
                [panel.matrix[item.item_id][model_id].label for item in train_ordered],
                panel.labels,
            ).macro_f1
            for model_id in panel.model_ids
        }
        selected = max(panel.model_ids, key=lambda value: (train_f1[value], value))
        weights = {selected: 1.0}
    elif method == "majority_vote":
        selected = None
        weights = {model_id: 1.0 for model_id in panel.model_ids}
    elif method == "calibration_weighted_vote":
        selected = None
        weights = model_accuracy
    elif method.startswith("single::"):
        selected = method.split("::", 1)[1]
        weights = {selected: 1.0}
    else:
        raise ValueError(method)
    records = []
    for item in test_ordered:
        subset = (selected,) if selected is not None else panel.model_ids
        responses = {
            model_id: panel.matrix[item.item_id][model_id].label for model_id in subset
        }
        prediction, _ties = weighted_vote(responses, weights, item.labels)
        records.append(
            {
                "item_id": item.item_id,
                "gold": str(item.metadata["gold_label"]),
                "prediction": prediction,
                "cluster": panel.cluster_key(item),
                "cost_usd": _cost(panel, item.item_id, subset),
                "models": len(subset),
                "selected_models": list(subset),
                "graph_applied": 0,
            }
        )
    return records


def _persistence_records(
    panel: OfflinePanel,
    train: Sequence[AnnotationItem],
    test: Sequence[AnnotationItem],
    *,
    mode: str,
    feedback_rate: float = 1.0,
    feedback_delay: int = 0,
) -> list[dict[str, Any]]:
    train_ordered = _ordered(panel, train)
    train_gold = [str(item.metadata["gold_label"]) for item in train_ordered]
    train_f1 = {
        model_id: classification_metrics(
            train_gold,
            [panel.matrix[item.item_id][model_id].label for item in train_ordered],
            panel.labels,
        ).macro_f1
        for model_id in panel.model_ids
    }
    fallback = max(panel.model_ids, key=lambda value: (train_f1[value], value))
    last_host: dict[str, str] = {}
    audited_nodes: dict[str, str] = {}
    queue: list[tuple[AnnotationItem, str, bool]] = []
    records = []
    for item in _ordered(panel, test):
        host = str(item.metadata.get("hostname", "unknown"))
        prediction: str | None = None
        if mode == "host":
            prediction = last_host.get(host)
        elif mode == "parent":
            totals: Counter[str] = Counter()
            for parent_id, raw in panel.neighborhood(item).items():
                if parent_id not in audited_nodes:
                    continue
                weight = float(raw.get("weight", 1.0)) if isinstance(raw, Mapping) else float(raw)
                totals[audited_nodes[parent_id]] += weight
            if totals:
                prediction = max(
                    item.labels,
                    key=lambda label: (totals[label], -item.labels.index(label)),
                )
        else:
            raise ValueError(mode)
        used_fallback = prediction is None
        if used_fallback:
            prediction = panel.matrix[item.item_id][fallback].label
        gold = str(item.metadata["gold_label"])
        records.append(
            {
                "item_id": item.item_id,
                "gold": gold,
                "prediction": prediction,
                "cluster": panel.cluster_key(item),
                "cost_usd": _cost(panel, item.item_id, (fallback,)) if used_fallback else 0.0,
                "models": int(used_fallback),
                "selected_models": [fallback] if used_fallback else [],
                "graph_applied": int(not used_fallback),
            }
        )
        queue.append(
            (
                item,
                gold,
                _stable(991, item.item_id) / 2**64 < feedback_rate,
            )
        )
        if len(queue) > feedback_delay:
            audited_item, audited_gold, should_audit = queue.pop(0)
            if should_audit:
                last_host[str(audited_item.metadata.get("hostname", "unknown"))] = audited_gold
                audited_nodes[
                    str(
                        audited_item.metadata.get(
                            "graph_node_id", audited_item.item_id
                        )
                    )
                ] = audited_gold
    return records


def _state_switch_records(
    panel: OfflinePanel,
    train: Sequence[AnnotationItem],
    test: Sequence[AnnotationItem],
    *,
    config: StateSwitchConfig,
    graph_config: AdaptiveGraphConfig,
    feedback_rate: float = 1.0,
    feedback_delay: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate the deployable router; no separate offline decision path."""

    return _run_router(
        panel,
        train,
        test,
        kind="state_switch",
        graph_config=graph_config,
        dcr_config=DCRGraphConfig(
            minimum_audited_rows=8,
            redundancy_penalty=1.0,
            correlation_discount=2.0,
            graph_prior_blend=0.8,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=16,
            max_graph_blend=1.0,
            latest_parent_per_relation=True,
        ),
        switch_config=config,
        feedback_rate=feedback_rate,
        feedback_delay=feedback_delay,
    )
def _candidate_configs() -> tuple[DCRGraphConfig, ...]:
    return (
        DCRGraphConfig(
            0.75, 1.0, 1.0, 0.30, 0.50, 1.00, 0.005, 0.02,
            pseudo_history_weight=0.0,
            dynamic_entity_edges=False,
        ),
        DCRGraphConfig(
            1.0, 1.0, 1.0, 0.30, 0.75, 1.50, 0.010, 0.03,
            pseudo_history_weight=0.10,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=8,
            max_graph_blend=0.35,
            protect_base_margin=0.40,
            max_js_divergence=0.50,
        ),
        DCRGraphConfig(
            1.0, 1.0, 0.75, 0.80, 1.00, 2.00, 0.015, 0.03,
            pseudo_history_weight=0.25,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=16,
            max_graph_blend=1.00,
            protect_base_margin=0.60,
            max_js_divergence=0.70,
            latest_parent_per_relation=True,
        ),
        DCRGraphConfig(
            1.0, 1.0, 0.50, 0.80, 0.75, 1.50, 0.005, 0.05,
            pseudo_history_weight=0.50,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=32,
            max_graph_blend=1.00,
            protect_base_margin=0.60,
            max_js_divergence=0.70,
            latest_parent_per_relation=True,
        ),
        DCRGraphConfig(
            1.0, 1.0, 1.0, 0.50, 1.00, 2.00, 0.010, 0.02,
            pseudo_history_weight=0.25,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=16,
            max_graph_blend=0.35,
            protect_base_margin=0.40,
            max_js_divergence=0.50,
        ),
        DCRGraphConfig(
            0.75, 1.0, 0.75, 0.80, 1.25, 2.50, 0.020, 0.02,
            pseudo_history_weight=0.50,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=32,
            max_graph_blend=1.00,
            protect_base_margin=0.60,
            max_js_divergence=0.70,
            latest_parent_per_relation=True,
        ),
        DCRGraphConfig(
            1.0, 1.0, 1.0, 0.00, 0.00, 0.00, 0.010, 0.03,
            pseudo_history_weight=0.25,
            dynamic_entity_edges=False,
        ),
        DCRGraphConfig(
            1.0, 1.0, 0.75, 0.00, 1.00, 2.00, 0.010, 0.03,
            pseudo_history_weight=0.50,
            dynamic_entity_edges=True,
            recent_nodes_per_entity=32,
            max_graph_blend=0.50,
            protect_base_margin=0.60,
            max_js_divergence=0.70,
        ),
    )


def _summary(records: Sequence[Mapping[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    metrics = classification_metrics(
        [str(row["gold"]) for row in records],
        [str(row["prediction"]) for row in records],
        labels,
    ).to_dict()
    return {
        "items": len(records),
        "metrics": metrics,
        "average_models": sum(float(row["models"]) for row in records) / len(records),
        "total_cost_usd": sum(float(row["cost_usd"]) for row in records),
        "cost_per_1000_usd": sum(float(row["cost_usd"]) for row in records)
        / len(records)
        * 1000,
        "graph_applied_rate": sum(int(row["graph_applied"]) for row in records)
        / len(records),
    }


def _inner_select(
    panel: OfflinePanel,
    outer_train: Sequence[AnnotationItem],
    graph_config: AdaptiveGraphConfig,
    *,
    group_key: Callable[[AnnotationItem], str],
    seed: int,
    inner_folds: int,
) -> tuple[DCRGraphConfig, list[dict[str, Any]]]:
    folds = _balanced_group_folds(
        outer_train,
        group_key=group_key,
        folds=inner_folds,
        seed=seed,
    )
    rows = []
    for config_index, config in enumerate(_candidate_configs()):
        condition_predictions: dict[str, list[dict[str, Any]]] = {
            "immediate": [],
            "sparse_25": [],
            "delay_5": [],
        }
        for fold_index, validation in enumerate(folds):
            validation_ids = {item.item_id for item in validation}
            train = tuple(
                item for item in outer_train if item.item_id not in validation_ids
            )
            for condition, rate, delay in (
                ("immediate", 1.0, 0),
                ("sparse_25", 0.25, 0),
                ("delay_5", 1.0, 5),
            ):
                fold_records = _run_router(
                    panel,
                    train,
                    validation,
                    kind="dcr_k2",
                    graph_config=graph_config,
                    dcr_config=config,
                    feedback_rate=rate,
                    feedback_delay=delay,
                )
                for record in fold_records:
                    record["inner_fold"] = fold_index
                condition_predictions[condition].extend(fold_records)
        condition_summaries = {
            name: _summary(predictions, panel.labels)
            for name, predictions in condition_predictions.items()
        }
        summary = condition_summaries["immediate"]
        robust_macro_f1 = sum(
            value["metrics"]["macro_f1"] for value in condition_summaries.values()
        ) / len(condition_summaries)
        robust_malicious_recall = sum(
            value["metrics"]["per_class"]["malicious"]["recall"]
            for value in condition_summaries.values()
        ) / len(condition_summaries)
        rows.append(
            {
                "config_index": config_index,
                "config": asdict(config),
                "macro_f1": summary["metrics"]["macro_f1"],
                "accuracy": summary["metrics"]["accuracy"],
                "malicious_recall": summary["metrics"]["per_class"]["malicious"][
                    "recall"
                ],
                "average_models": summary["average_models"],
                "total_cost_usd": summary["total_cost_usd"],
                "robust_macro_f1": robust_macro_f1,
                "robust_malicious_recall": robust_malicious_recall,
                "condition_macro_f1": {
                    name: value["metrics"]["macro_f1"]
                    for name, value in condition_summaries.items()
                },
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["robust_macro_f1"],
            row["robust_malicious_recall"],
            row["macro_f1"],
            row["accuracy"],
            -row["average_models"],
            -row["total_cost_usd"],
        ),
    )
    return DCRGraphConfig(**selected["config"]), rows


def _inner_select_switch(
    panel: OfflinePanel,
    outer_train: Sequence[AnnotationItem],
    *,
    graph_config: AdaptiveGraphConfig,
    group_key: Callable[[AnnotationItem], str],
    seed: int,
    inner_folds: int,
) -> tuple[StateSwitchConfig, list[dict[str, Any]]]:
    folds = _balanced_group_folds(
        outer_train,
        group_key=group_key,
        folds=inner_folds,
        seed=seed,
    )
    rows = []
    for config_index, config in enumerate(_switch_candidates()):
        condition_predictions: dict[str, list[dict[str, Any]]] = {
            "immediate": [],
            "sparse_25": [],
            "delay_5": [],
        }
        for validation in folds:
            validation_ids = {item.item_id for item in validation}
            train = tuple(
                item for item in outer_train if item.item_id not in validation_ids
            )
            for condition, rate, delay in (
                ("immediate", 1.0, 0),
                ("sparse_25", 0.25, 0),
                ("delay_5", 1.0, 5),
            ):
                condition_predictions[condition].extend(
                    _state_switch_records(
                        panel,
                        train,
                        validation,
                        config=config,
                        graph_config=graph_config,
                        feedback_rate=rate,
                        feedback_delay=delay,
                    )
                )
        condition_summaries = {
            name: _summary(predictions, panel.labels)
            for name, predictions in condition_predictions.items()
        }
        robust_macro_f1 = sum(
            value["metrics"]["macro_f1"]
            for value in condition_summaries.values()
        ) / len(condition_summaries)
        rows.append(
            {
                "config_index": config_index,
                "config": asdict(config),
                "robust_macro_f1": robust_macro_f1,
                "immediate_macro_f1": condition_summaries["immediate"]["metrics"][
                    "macro_f1"
                ],
                "condition_macro_f1": {
                    name: value["metrics"]["macro_f1"]
                    for name, value in condition_summaries.items()
                },
                "average_models": condition_summaries["immediate"]["average_models"],
                "total_cost_usd": condition_summaries["immediate"]["total_cost_usd"],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["robust_macro_f1"],
            row["immediate_macro_f1"],
            -row["total_cost_usd"],
        ),
    )
    return StateSwitchConfig(**selected["config"]), rows


def analyze(
    panel: OfflinePanel,
    graph_config: AdaptiveGraphConfig,
    *,
    outer_folds: int,
    inner_folds: int,
    bootstrap_iterations: int,
    seed: int,
    selection_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    all_items = tuple(
        sorted(
            panel.split.calibration + panel.split.validation + panel.split.test,
            key=panel.order_key,
        )
    )
    cv_groups = _causal_chunk_groups(all_items)
    cv_group_key = lambda item: cv_groups[item.item_id]
    folds = _balanced_group_folds(
        all_items,
        group_key=cv_group_key,
        folds=outer_folds,
        seed=seed,
    )
    methods = [
        *(f"single::{model_id}" for model_id in panel.model_ids),
        "best_train_single",
        "majority_vote",
        "calibration_weighted_vote",
        "host_persistence_best_single",
        "parent_persistence_best_single",
        "camvo_k1",
        "camvo_k2",
        "sage",
        "dcr_k1_full",
        "dcr_k2_reliability_only",
        "dcr_k2_diversity_no_graph",
        "dcr_k2_full",
        "state_switch_gcamvo_k2",
    ]
    records: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    sensitivity_records: dict[str, list[dict[str, Any]]] = {
        **{f"feedback_rate_{rate:.2f}": [] for rate in (0.0, 0.10, 0.25, 0.50, 1.0)},
        **{f"feedback_delay_{delay}": [] for delay in (0, 1, 5, 20)},
    }
    switch_sensitivity_records: dict[str, list[dict[str, Any]]] = {
        **{f"feedback_rate_{rate:.2f}": [] for rate in (0.0, 0.10, 0.25, 0.50, 1.0)},
        **{f"feedback_delay_{delay}": [] for delay in (0, 1, 5, 20)},
    }
    selections = []
    switch_selections = []
    fold_audit = []
    for fold_index, test in enumerate(folds):
        test_ids = {item.item_id for item in test}
        train = tuple(item for item in all_items if item.item_id not in test_ids)
        if selection_cache is None:
            selected, search = _inner_select(
                panel,
                train,
                graph_config,
                group_key=cv_group_key,
                seed=seed + 100 + fold_index,
                inner_folds=inner_folds,
            )
            selected_switch, switch_search = _inner_select_switch(
                panel,
                train,
                graph_config=graph_config,
                group_key=cv_group_key,
                seed=seed + 500 + fold_index,
                inner_folds=inner_folds,
            )
        else:
            cached_dcr = selection_cache["inner_selections"][fold_index]
            cached_switch = selection_cache["state_switch_inner_selections"][
                fold_index
            ]
            selected = DCRGraphConfig(**cached_dcr["selected_config"])
            search = list(cached_dcr["inner_search"])
            selected_switch = StateSwitchConfig(**cached_switch["selected_config"])
            switch_search = list(cached_switch["inner_search"])
        selections.append(
            {
                "outer_fold": fold_index,
                "selected_config": asdict(selected),
                "inner_search": search,
            }
        )
        switch_selections.append(
            {
                "outer_fold": fold_index,
                "selected_config": asdict(selected_switch),
                "inner_search": switch_search,
            }
        )
        fold_audit.append(
            {
                "outer_fold": fold_index,
                "train_items": len(train),
                "test_items": len(test),
                "test_groups": len({cv_group_key(item) for item in test}),
                "test_labels": dict(
                    Counter(str(item.metadata["gold_label"]) for item in test)
                ),
            }
        )
        for method in methods:
            if method.startswith("single::") or method in {
                "best_train_single",
                "majority_vote",
                "calibration_weighted_vote",
            }:
                rows = _static_records(panel, train, test, method=method)
            elif method in {
                "host_persistence_best_single",
                "parent_persistence_best_single",
            }:
                rows = _persistence_records(
                    panel,
                    train,
                    test,
                    mode="host" if method.startswith("host") else "parent",
                )
            elif method in {"camvo_k1", "camvo_k2", "sage"}:
                rows = _run_router(
                    panel,
                    train,
                    test,
                    kind=method,
                    graph_config=graph_config,
                )
            elif method == "state_switch_gcamvo_k2":
                rows = _state_switch_records(
                    panel,
                    train,
                    test,
                    config=selected_switch,
                    graph_config=graph_config,
                )
            else:
                if method == "dcr_k2_reliability_only":
                    dcr = replace(
                        selected,
                        graph_prior_blend=0.0,
                        redundancy_penalty=0.0,
                        correlation_discount=0.0,
                    )
                    graph = replace(
                        graph_config,
                        max_graph_blend=0.0,
                        gate_activation_threshold=1.0,
                    )
                elif method == "dcr_k2_diversity_no_graph":
                    dcr = replace(selected, graph_prior_blend=0.0)
                    graph = replace(
                        graph_config,
                        max_graph_blend=0.0,
                        gate_activation_threshold=1.0,
                    )
                else:
                    dcr = selected
                    graph = graph_config
                rows = _run_router(
                    panel,
                    train,
                    test,
                    kind="dcr_k1" if method == "dcr_k1_full" else "dcr_k2",
                    graph_config=graph,
                    dcr_config=dcr,
                )
            for row in rows:
                row["outer_fold"] = fold_index
            records[method].extend(rows)

        for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
            rows = _run_router(
                panel,
                train,
                test,
                kind="dcr_k2",
                graph_config=graph_config,
                dcr_config=selected,
                feedback_rate=rate,
            )
            sensitivity_records[f"feedback_rate_{rate:.2f}"].extend(rows)
        for delay in (0, 1, 5, 20):
            rows = _run_router(
                panel,
                train,
                test,
                kind="dcr_k2",
                graph_config=graph_config,
                dcr_config=selected,
                feedback_delay=delay,
            )
            sensitivity_records[f"feedback_delay_{delay}"].extend(rows)
        for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
            switch_sensitivity_records[f"feedback_rate_{rate:.2f}"].extend(
                _state_switch_records(
                    panel,
                    train,
                    test,
                    config=selected_switch,
                    graph_config=graph_config,
                    feedback_rate=rate,
                )
            )
        for delay in (0, 1, 5, 20):
            switch_sensitivity_records[f"feedback_delay_{delay}"].extend(
                _state_switch_records(
                    panel,
                    train,
                    test,
                    config=selected_switch,
                    graph_config=graph_config,
                    feedback_delay=delay,
                )
            )

    summaries = {method: _summary(rows, panel.labels) for method, rows in records.items()}
    comparisons = {}
    reference_map = {
        "vs_camvo_k1": "camvo_k1",
        "vs_camvo_k2": "camvo_k2",
        "vs_sage": "sage",
        "vs_best_train_single": "best_train_single",
        "vs_host_persistence": "host_persistence_best_single",
    }
    dcr_rows = sorted(records["dcr_k2_full"], key=lambda row: row["item_id"])
    for name, reference in reference_map.items():
        ref_by_id = {row["item_id"]: row for row in records[reference]}
        comparisons[name] = paired_cluster_bootstrap_delta(
            [row["gold"] for row in dcr_rows],
            [row["prediction"] for row in dcr_rows],
            [ref_by_id[row["item_id"]]["prediction"] for row in dcr_rows],
            panel.labels,
            [row["cluster"] for row in dcr_rows],
            metric="macro_f1",
            iterations=bootstrap_iterations,
            seed=seed + 900,
        ).to_dict()
    switch_comparisons = {}
    switch_reference_map = {
        "vs_host_persistence": "host_persistence_best_single",
        "vs_dcr_k2": "dcr_k2_full",
        "vs_camvo_k2": "camvo_k2",
        "vs_best_train_single": "best_train_single",
    }
    switch_rows = sorted(
        records["state_switch_gcamvo_k2"], key=lambda row: row["item_id"]
    )
    for name, reference in switch_reference_map.items():
        ref_by_id = {row["item_id"]: row for row in records[reference]}
        switch_comparisons[name] = paired_cluster_bootstrap_delta(
            [row["gold"] for row in switch_rows],
            [row["prediction"] for row in switch_rows],
            [ref_by_id[row["item_id"]]["prediction"] for row in switch_rows],
            panel.labels,
            [row["cluster"] for row in switch_rows],
            metric="macro_f1",
            iterations=bootstrap_iterations,
            seed=seed + 1200,
        ).to_dict()
    return {
        "status": "nested_group_cv_frozen_real_responses",
        "provider_calls": 0,
        "items": len(all_items),
        "models": list(panel.model_ids),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "inner_selection_source": (
            "fresh_nested_cv" if selection_cache is None else "cached_nested_cv_audit"
        ),
        "group_definition": (
            "hostname:5-minute causal block, sequentially capped at 48 events; "
            "bootstrap retains the coarser hostname:30-minute incident cluster"
        ),
        "fold_audit": fold_audit,
        "inner_selections": selections,
        "state_switch_inner_selections": switch_selections,
        "summaries": summaries,
        "paired_cluster_bootstrap": comparisons,
        "state_switch_paired_cluster_bootstrap": switch_comparisons,
        "feedback_sensitivity": {
            name: _summary(rows, panel.labels)
            for name, rows in sensitivity_records.items()
        },
        "state_switch_feedback_sensitivity": {
            name: _summary(rows, panel.labels)
            for name, rows in switch_sensitivity_records.items()
        },
        "records": records,
        "claim_boundary": (
            "Exploratory nested grouped cross-validation on one Mordor scenario. "
            "All LLM responses are frozen real Adams outputs and provider calls are zero. "
            "Outer-fold labels are unseen during inner selection; within an outer test stream, "
            "audited feedback arrives after each prediction and affects only future nodes. "
            "This repairs the prior single-split instability but remains method development, "
            "not external confirmation."
        ),
    }


def _render(report: Mapping[str, Any]) -> str:
    preferred = [
        "single::dify/deepseek-v4-flash",
        "best_train_single",
        "majority_vote",
        "calibration_weighted_vote",
        "host_persistence_best_single",
        "parent_persistence_best_single",
        "camvo_k1",
        "camvo_k2",
        "sage",
        "dcr_k1_full",
        "dcr_k2_reliability_only",
        "dcr_k2_diversity_no_graph",
        "dcr_k2_full",
        "state_switch_gcamvo_k2",
    ]
    lines = [
        "# Causal G-CaMVo：Mordor 嵌套分组交叉验证",
        "",
        "## 协议",
        "",
        f"- 完整冻结事件：{report['items']}；新增模型调用：0；",
        f"- 外层 {report['outer_folds']} 折、内层 {report['inner_folds']} 折；",
        f"- 内层选择来源：{report['inner_selection_source']}；",
        f"- 分组：{report['group_definition']}；",
        "- 外层测试标签不参与内层参数选择；测试流反馈在预测后到达，只影响未来节点。",
        "",
        "## 外层 OOF 主结果",
        "",
        "| 方法 | Accuracy | Macro-F1 | 恶意 P/R/F1 | 平均模型数 | 总成本 USD | 图介入率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in preferred:
        row = report["summaries"][method]
        metrics = row["metrics"]
        malicious = metrics["per_class"]["malicious"]
        lines.append(
            f"| {method} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{malicious['precision']:.4f}/{malicious['recall']:.4f}/{malicious['f1']:.4f} | "
            f"{row['average_models']:.3f} | {row['total_cost_usd']:.4f} | "
            f"{100*row['graph_applied_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## DCR-full 配对 cluster bootstrap",
            "",
            "| 对照 | Macro-F1 差值 | 95% CI | DCR 更优概率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["paired_cluster_bootstrap"].items():
        lines.append(
            f"| {name} | {row['candidate_minus_reference']:+.5f} | "
            f"[{row['lower']:+.5f}, {row['upper']:+.5f}] | "
            f"{100*row['probability_candidate_better']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## State-Switch G-CaMVo 配对 cluster bootstrap",
            "",
            "| 对照 | Macro-F1 差值 | 95% CI | State-Switch 更优概率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["state_switch_paired_cluster_bootstrap"].items():
        lines.append(
            f"| {name} | {row['candidate_minus_reference']:+.5f} | "
            f"[{row['lower']:+.5f}, {row['upper']:+.5f}] | "
            f"{100*row['probability_candidate_better']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 审计反馈稀疏度（DCR k=2）",
            "",
            "| 反馈率 | Accuracy | Macro-F1 | 恶意 Recall | 图介入率 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
        row = report["feedback_sensitivity"][f"feedback_rate_{rate:.2f}"]
        lines.append(
            f"| {100*rate:.0f}% | {row['metrics']['accuracy']:.4f} | "
            f"{row['metrics']['macro_f1']:.4f} | "
            f"{row['metrics']['per_class']['malicious']['recall']:.4f} | "
            f"{100*row['graph_applied_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 审计反馈延迟（DCR k=2，100% 最终到达）",
            "",
            "| 延迟事件数 | Accuracy | Macro-F1 | 恶意 Recall | 图介入率 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for delay in (0, 1, 5, 20):
        row = report["feedback_sensitivity"][f"feedback_delay_{delay}"]
        lines.append(
            f"| {delay} | {row['metrics']['accuracy']:.4f} | "
            f"{row['metrics']['macro_f1']:.4f} | "
            f"{row['metrics']['per_class']['malicious']['recall']:.4f} | "
            f"{100*row['graph_applied_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## State-Switch 反馈敏感度",
            "",
            "| 条件 | Accuracy | Macro-F1 | 恶意 Recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
        row = report["state_switch_feedback_sensitivity"][
            f"feedback_rate_{rate:.2f}"
        ]
        lines.append(
            f"| 反馈率 {100*rate:.0f}% | {row['metrics']['accuracy']:.4f} | "
            f"{row['metrics']['macro_f1']:.4f} | "
            f"{row['metrics']['per_class']['malicious']['recall']:.4f} |"
        )
    for delay in (1, 5, 20):
        row = report["state_switch_feedback_sensitivity"][f"feedback_delay_{delay}"]
        lines.append(
            f"| 延迟 {delay} | {row['metrics']['accuracy']:.4f} | "
            f"{row['metrics']['macro_f1']:.4f} | "
            f"{row['metrics']['per_class']['malicious']['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 外层折审计",
            "",
            "| Fold | Train | Test | Test groups | Benign | Malicious |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["fold_audit"]:
        lines.append(
            f"| {row['outer_fold']} | {row['train_items']} | {row['test_items']} | "
            f"{row['test_groups']} | {row['test_labels'].get('benign',0)} | "
            f"{row['test_labels'].get('malicious',0)} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            report["claim_boundary"],
            "",
            "本表的作用是修复旧 Mordor 单次划分的失衡并完成方法开发；最终主结论仍需"
            "在冻结后的未见 CASIE 文档或独立 OpTC 场景确认。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "data/derived/mordor_offline_bundle",
    )
    parser.add_argument(
        "--safe-results",
        type=Path,
        default=ROOT / "artifacts/safe_adaptive_graph/RESULTS.json",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--refresh-inner-search",
        action="store_true",
        help="recompute every inner-fold search instead of replaying the saved audit",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "artifacts/dcr_group_cv/RESULTS.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "artifacts/dcr_group_cv/oof_records.csv",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "docs/DCR_GCAMVO_GROUP_CV_RESULTS_2026-07-23.md",
    )
    args = parser.parse_args()
    selected = json.loads(args.safe_results.read_text(encoding="utf-8"))[
        "selected_config"
    ]
    selection_cache = None
    if args.json_out.exists() and not args.refresh_inner_search:
        previous = json.loads(args.json_out.read_text(encoding="utf-8"))
        if (
            previous.get("outer_folds") == args.outer_folds
            and previous.get("inner_folds") == args.inner_folds
            and len(previous.get("inner_selections", ())) == args.outer_folds
            and len(previous.get("state_switch_inner_selections", ()))
            == args.outer_folds
        ):
            selection_cache = previous
    report = analyze(
        load_mordor_panel(args.bundle_dir),
        AdaptiveGraphConfig(**selected),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        selection_cache=selection_cache,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
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
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_render(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "csv": str(args.csv_out),
                "markdown": str(args.markdown_out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
