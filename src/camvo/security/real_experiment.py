"""Config-driven real-provider experiment runner with hard cost protection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from camvo.budget import BudgetGuard
from camvo.ccamvo_router import CorrelatedCaMVoConfig
from camvo.config import CaMVoConfig
from camvo.credentials import load_api_keys
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import GraphCaMVoConfig
from camvo.llms.anthropic_messages import AnthropicMessagesLLMClient
from camvo.llms.cache import FileResponseCache
from camvo.llms.dify_workflow import DifyWorkflowLLMClient
from camvo.llms.guarded import CachedBudgetedLLMClient
from camvo.llms.openai_compatible import OpenAICompatibleLLMClient
from camvo.security.casie import build_casie_event_adjacency, load_casie_event_items
from camvo.security.experiment import ExperimentModelSpec, evaluate_security_strategies
from camvo.security.graph_diagnostics import graph_diagnostics
from camvo.security.mordor_dataset import build_mordor_cdb_binary_dataset
from camvo.security.optc_dataset import build_optc_real_binary_dataset
from camvo.security.simulated_experiments import optc_adjacency, stratified_sample
from camvo.security.sampling import graph_preserving_group_sample
from camvo.security.tasks import (
    CASIE_EVENT_CLASSIFICATION_TASK,
    MORDOR_BINARY_DETECTION_TASK,
    OPTC_BINARY_DETECTION_TASK,
)
from camvo.types import ModelPricing
from camvo.trace_router import TraceGraphCaMVoConfig


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration field {key!r} must be an object")
    return value


def load_real_experiment_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("real experiment config must use schema_version 1")
    _object(payload, "dataset")
    _object(payload, "router")
    _object(payload, "graph")
    _object(payload, "budget")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("models must contain at least two entries")
    if any(not isinstance(model, dict) for model in models):
        raise ValueError("every model configuration must be an object")
    model_ids = [str(model.get("model_id", "")).strip() for model in models]
    if any(not model_id for model_id in model_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("models must have non-empty, unique model_id values")
    return payload


def _dataset(
    config: dict[str, Any],
) -> tuple[list[Any], dict[str, dict[str, float]], dict[str, object], Any]:
    dataset_config = _object(config, "dataset")
    kind = str(dataset_config.get("kind", ""))
    seed = int(config.get("seed", 17))
    max_items = int(dataset_config.get("max_items", 100))
    if max_items <= 0:
        raise ValueError("dataset.max_items must be positive")
    if kind == "casie":
        loaded = load_casie_event_items(Path(dataset_config["path"]), strict=True)
        require_complete = bool(dataset_config.get("require_complete_corpus", False))
        if require_complete and max_items != len(loaded.items):
            raise ValueError(
                "complete CASIE run requires dataset.max_items to equal the loader total: "
                f"configured={max_items}, loaded={len(loaded.items)}"
            )
        items = graph_preserving_group_sample(
            loaded.items,
            min(max_items, len(loaded.items)),
            seed,
            group_key=lambda item: str(item.metadata["document_id"]),
            order_key=lambda item: int(item.metadata["start_offset"]),
        )
        if require_complete and len(items) != len(loaded.items):
            raise RuntimeError(
                f"complete CASIE sampling guard failed: {len(items)}/{len(loaded.items)}"
            )
        return (
            items,
            build_casie_event_adjacency(items),
            {
                "name": "CASIE",
                "task": "five-way marked event-subtype classification",
                "items": len(items),
                "source_documents": loaded.stats.files_scanned,
                "real_provider_run": True,
            },
            CASIE_EVENT_CLASSIFICATION_TASK,
        )
    if kind == "optc-real":
        built = build_optc_real_binary_dataset(
            dataset_config["attack_path"],
            dataset_config["benign_path"],
            dataset_config["labels_path"],
            dataset_config.get("scenario_manifest", "config/optc_scenarios.json"),
            scenario_id=str(
                dataset_config.get("scenario_id", "optc-day3-malicious-upgrade")
            ),
            utc_offset_minutes=int(dataset_config.get("utc_offset_minutes", -240)),
            padding_minutes=int(dataset_config.get("padding_minutes", 10)),
            max_positive_items=max(1, max_items // 2),
            negative_ratio=float(dataset_config.get("negative_ratio", 1.0)),
            seed=seed,
            correlation_window_minutes=int(
                dataset_config.get("correlation_window_minutes", 30)
            ),
        )
        return (
            list(built.items),
            optc_adjacency(built.correlations),
            {**built.stats, "real_provider_run": True},
            OPTC_BINARY_DETECTION_TASK,
        )
    if kind == "mordor-cdb-sample":
        require_all_flags = bool(dataset_config.get("require_all_flags", False))
        max_positive = None if require_all_flags else max(1, max_items // 2)
        built = build_mordor_cdb_binary_dataset(
            dataset_config["path"],
            dataset_config["flags_path"],
            max_positive_items=max_positive,
            seed=seed,
            correlation_window_minutes=int(
                dataset_config.get("correlation_window_minutes", 30)
            ),
            max_records_per_item=int(dataset_config.get("max_records_per_item", 6)),
            max_chars_per_record=int(dataset_config.get("max_chars_per_record", 1_200)),
        )
        if require_all_flags and max_items != len(built.items):
            raise ValueError(
                "complete Mordor flag run requires dataset.max_items to equal the balanced "
                f"loader total: configured={max_items}, loaded={len(built.items)}"
            )
        return (
            list(built.items),
            optc_adjacency(built.correlations),
            {**built.stats, "real_provider_run": True},
            MORDOR_BINARY_DETECTION_TASK,
        )
    raise ValueError(
        "dataset.kind must be 'casie', 'optc-real', or 'mordor-cdb-sample'"
    )


def build_guarded_provider_pool(
    config: dict[str, Any],
    task: Any,
) -> tuple[list[CachedBudgetedLLMClient], list[ExperimentModelSpec], BudgetGuard]:
    """Build one cache- and budget-protected provider pool from validated config."""

    budget_config = _object(config, "budget")
    budget = BudgetGuard(
        budget_config.get("ledger_path", "artifacts/provider_budget.json"),
        hard_limit_usd=float(budget_config.get("hard_limit_usd", 5.0)),
        max_provider_calls=int(budget_config.get("max_provider_calls", 500)),
    )
    cache = FileResponseCache(budget_config.get("cache_dir", "artifacts/responses"))
    models: list[CachedBudgetedLLMClient] = []
    model_specs: list[ExperimentModelSpec] = []
    for raw in config["models"]:
        provider = str(raw.get("provider", ""))
        if provider not in {"openai_compatible", "anthropic_messages", "dify_workflow"}:
            raise ValueError(
                "provider must be 'openai_compatible', 'anthropic_messages', or "
                "'dify_workflow'"
            )
        key_name = str(raw.get("api_key_env", ""))
        api_key = os.environ.get(key_name, "")
        if not api_key:
            raise ValueError(f"required API key environment variable is not configured: {key_name}")
        max_retries = int(raw.get("max_retries", 0))
        if max_retries != 0:
            raise ValueError(
                "budgeted experiments require max_retries=0; rerun failures so cache prevents "
                "duplicate successful charges"
            )
        pricing = ModelPricing(
            input_per_million=float(raw["input_usd_per_million"]),
            output_per_million=float(raw["output_usd_per_million"]),
        )
        if provider == "openai_compatible":
            delegate = OpenAICompatibleLLMClient(
                str(raw["model_id"]),
                pricing,
                provider_model=str(raw["provider_model"]),
                base_url=str(raw["base_url"]),
                api_key=api_key,
                task=task,
                timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
                max_retries=max_retries,
                max_output_tokens=int(raw.get("max_output_tokens", 128)),
                max_tokens_field=str(raw.get("max_tokens_field", "max_tokens")),
                temperature=(
                    None
                    if "temperature" in raw and raw["temperature"] is None
                    else float(raw.get("temperature", 0.0))
                ),
                response_format_json=bool(raw.get("response_format_json", True)),
                extra_body=dict(raw.get("extra_body", {})),
            )
        elif provider == "anthropic_messages":
            delegate = AnthropicMessagesLLMClient(
                str(raw["model_id"]),
                pricing,
                provider_model=str(raw["provider_model"]),
                base_url=str(raw.get("base_url", "https://api.anthropic.com/v1")),
                api_key=api_key,
                task=task,
                anthropic_version=str(raw.get("anthropic_version", "2023-06-01")),
                timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
                max_retries=max_retries,
                max_output_tokens=int(raw.get("max_output_tokens", 128)),
                temperature=(
                    None
                    if "temperature" in raw and raw["temperature"] is None
                    else float(raw.get("temperature", 0.0))
                ),
                extra_body=dict(raw.get("extra_body", {})),
            )
        else:
            platform_user_env = str(raw.get("platform_user_env", "")).strip()
            user_token_env = str(raw.get("user_token_env", "")).strip()
            if bool(platform_user_env) != bool(user_token_env):
                raise ValueError(
                    "Dify/Adams models must configure both platform_user_env and "
                    "user_token_env, or neither"
                )
            platform_user = os.environ.get(platform_user_env, "").strip()
            user_token = os.environ.get(user_token_env, "").strip()
            if platform_user_env and not platform_user:
                raise ValueError(
                    "required Adams platform-user environment variable is not configured: "
                    f"{platform_user_env}"
                )
            if user_token_env and not user_token:
                raise ValueError(
                    "required Adams user-token environment variable is not configured: "
                    f"{user_token_env}"
                )
            delegate = DifyWorkflowLLMClient(
                str(raw["model_id"]),
                pricing,
                workflow_model=str(raw["workflow_model"]),
                base_url=str(raw["base_url"]),
                api_key=api_key,
                task=task,
                timeout_seconds=float(raw.get("timeout_seconds", 100.0)),
                output_key=str(raw.get("output_key", "result")),
                user=str(raw.get("user", "g-camvo-experiment")),
                platform_user=platform_user,
                user_token=user_token,
            )
        models.append(
            CachedBudgetedLLMClient(
                delegate,
                cache,
                budget,
                prompt_version=delegate.prompt_version,
                max_output_tokens=int(raw.get("max_output_tokens", 128)),
                input_token_margin=float(budget_config.get("input_token_margin", 1.25)),
            )
        )
        model_specs.append(
            ExperimentModelSpec(
                model_id=str(raw["model_id"]),
                prior_quality=float(raw["prior_quality"]),
                latency_ms=float(raw.get("latency_ms", 1_000.0)),
            )
        )
    return models, model_specs, budget


def run_real_provider_experiment(config_path: str | Path) -> dict[str, object]:
    """Run all seven policies; cache ensures at most one paid call/model/item."""

    config = load_real_experiment_config(config_path)
    if config.get("pricing_verified") is not True:
        raise ValueError(
            "set pricing_verified=true only after replacing every example price with the current "
            "provider price"
        )
    key_file = Path(config.get("api_key_file", "config/api_keys.env"))
    load_api_keys(key_file)
    items, adjacency, dataset_metadata, task = _dataset(config)
    dataset_metadata = {
        **dataset_metadata,
        **graph_diagnostics(items, adjacency),
    }
    models, model_specs, budget = build_guarded_provider_pool(config, task)
    router_config = _object(config, "router")
    embedding_dim = int(router_config.get("embedding_dim", 64))
    camvo_config = CaMVoConfig(
        embedding_dim=embedding_dim,
        confidence_threshold=float(router_config.get("confidence_threshold", 0.97)),
        min_models=int(router_config.get("min_models", 2)),
        linucb_regularization=float(router_config.get("linucb_regularization", 1.0)),
        exploration_alpha=float(router_config.get("exploration_alpha", 0.2)),
        laplace_regularization=float(router_config.get("laplace_regularization", 1.0)),
        warmup_rounds=int(router_config.get("warmup_rounds", 20)),
        confidence_method=str(router_config.get("confidence_method", "exact")),
    )
    graph_config = _object(config, "graph")
    trace_config = dict(config.get("trace", {}))
    ccamvo_config = dict(config.get("ccamvo", {}))
    report = evaluate_security_strategies(
        items,
        models,
        model_specs,
        HashingTextEmbedder(embedding_dim),
        camvo_config,
        dataset_metadata=dataset_metadata,
        graph_neighborhood=adjacency,
        graph_config=GraphCaMVoConfig(
            regularization=float(graph_config.get("regularization", 1.0)),
            max_edge_weight=float(graph_config.get("max_edge_weight", 2.0)),
            max_total_neighbor_weight=float(
                graph_config.get("max_total_neighbor_weight", 8.0)
            ),
        ),
        ccamvo_config=CorrelatedCaMVoConfig(
            monte_carlo_samples=int(ccamvo_config.get("monte_carlo_samples", 4096)),
            seed=int(ccamvo_config.get("seed", config.get("seed", 17))),
        ),
        trace_config=TraceGraphCaMVoConfig(
            risk_tolerance=float(trace_config.get("risk_tolerance", 0.03)),
            critical_risk_tolerance=float(
                trace_config.get("critical_risk_tolerance", 0.01)
            ),
            symmetric_reliability_prior=float(
                trace_config.get("symmetric_reliability_prior", 0.68)
            ),
            reliability_prior_strength=float(
                trace_config.get("reliability_prior_strength", 6.0)
            ),
            min_reliability_observations=int(
                trace_config.get("min_reliability_observations", 8)
            ),
            graph_strength=float(trace_config.get("graph_strength", 0.8)),
            min_transition_observations=float(
                trace_config.get("min_transition_observations", 6.0)
            ),
            diversity_penalty=float(trace_config.get("diversity_penalty", 0.45)),
            abstain_when_risk_unmet=bool(
                trace_config.get("abstain_when_risk_unmet", True)
            ),
            fallback_model_id=trace_config.get("fallback_model_id"),
            fallback_after_models=int(trace_config.get("fallback_after_models", 2)),
            fallback_trigger_risk=float(
                trace_config.get("fallback_trigger_risk", 0.20)
            ),
        ),
        fixed_cheap_models=int(config.get("fixed_cheap_models", 3)),
        disclaimer=(
            "Dataset and provider responses are real. Provider priors, graph construction, and "
            "routing thresholds remain experimental; inspect the persistent budget ledger."
        ),
    )
    return {**report.to_dict(), "provider_budget": budget.snapshot().to_dict()}
