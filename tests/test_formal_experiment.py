import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from camvo.budget import BudgetGuard
from camvo.llms import CachedBudgetedLLMClient, CallableLLMClient, FileResponseCache
from camvo.security.experiment import ExperimentModelSpec
from camvo.security.formal_experiment import (
    collect_response_matrix,
    run_formal_provider_experiment,
)
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class FormalExperimentTests(unittest.TestCase):
    def test_response_matrix_is_complete_and_resumes_without_provider_calls(self) -> None:
        calls = {"a": 0, "b": 0}
        items = [
            AnnotationItem(
                item_id=f"item-{index}",
                text=f"security event {index}",
                labels=("safe", "unsafe"),
                metadata={"gold_label": "safe"},
            )
            for index in range(4)
        ]

        def client(model_id: str) -> CallableLLMClient:
            def predict(_item: AnnotationItem) -> ModelResponse:
                calls[model_id] += 1
                return ModelResponse("safe", 3, 1, raw={"confidence": 0.8})

            return CallableLLMClient(
                model_id,
                ModelPricing(0.1, 0.2),
                predict,
                token_counter=lambda _item: 3,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = BudgetGuard(root / "budget.json", hard_limit_usd=1.0, max_provider_calls=10)
            cache = FileResponseCache(root / "responses")
            models = [
                CachedBudgetedLLMClient(
                    client(model_id),
                    cache,
                    budget,
                    prompt_version="formal-v1",
                    max_output_tokens=1,
                    input_token_margin=1.0,
                )
                for model_id in ("a", "b")
            ]
            first, audit = collect_response_matrix(
                models, items, root / "matrix.json", workers=2
            )
            attempts = budget.snapshot().provider_attempts
            second, second_audit = collect_response_matrix(
                models, items, root / "matrix.json", workers=2
            )

        self.assertTrue(audit["complete"])
        self.assertTrue(second_audit["complete"])
        self.assertEqual(attempts, 8)
        self.assertEqual(calls, {"a": 4, "b": 4})
        self.assertEqual(set(first), {item.item_id for item in items})
        self.assertEqual(first["item-0"]["a"].raw["confidence"], 0.8)
        self.assertEqual(second["item-0"]["a"].raw["confidence"], 0.8)

    def test_matrix_collection_resumes_only_failed_cells(self) -> None:
        calls = {"a": 0, "b": 0}
        failed_once = False
        items = [
            AnnotationItem(
                item_id=f"item-{index}",
                text=f"event {index}",
                labels=("safe", "unsafe"),
                metadata={"gold_label": "safe"},
            )
            for index in range(4)
        ]

        def client(model_id: str) -> CallableLLMClient:
            def predict(item: AnnotationItem) -> ModelResponse:
                nonlocal failed_once
                calls[model_id] += 1
                if model_id == "b" and item.item_id == "item-0" and not failed_once:
                    failed_once = True
                    raise RuntimeError("transient")
                return ModelResponse("safe", 3, 1, raw={"confidence": 0.8})

            return CallableLLMClient(
                model_id,
                ModelPricing(0.1, 0.2),
                predict,
                token_counter=lambda _item: 3,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = BudgetGuard(root / "budget.json", hard_limit_usd=1.0, max_provider_calls=20)
            cache = FileResponseCache(root / "responses")
            models = [
                CachedBudgetedLLMClient(
                    client(model_id),
                    cache,
                    budget,
                    prompt_version="formal-v1",
                    max_output_tokens=1,
                    input_token_margin=1.0,
                )
                for model_id in ("a", "b")
            ]
            with self.assertRaisesRegex(RuntimeError, "transient"):
                collect_response_matrix(models, items, root / "matrix.json", workers=2)
            matrix, audit = collect_response_matrix(
                models, items, root / "matrix.json", workers=2
            )
            snapshot = budget.snapshot()

        self.assertTrue(audit["complete"])
        self.assertEqual(set(matrix), {item.item_id for item in items})
        self.assertEqual(snapshot.completed_calls, 8)
        self.assertEqual(snapshot.failed_calls, 1)
        self.assertEqual(snapshot.provider_attempts, 9)

    def test_broken_model_circuit_does_not_block_healthy_model(self) -> None:
        items = [
            AnnotationItem(
                item_id=f"item-{index}",
                text=f"event {index}",
                labels=("safe", "unsafe"),
                metadata={"gold_label": "safe"},
            )
            for index in range(6)
        ]
        calls = {"broken": 0, "healthy": 0}

        def client(model_id: str) -> CallableLLMClient:
            def predict(_item: AnnotationItem) -> ModelResponse:
                calls[model_id] += 1
                if model_id == "broken":
                    raise RuntimeError("branch unavailable")
                return ModelResponse("safe", 3, 1, raw={"confidence": 0.8})

            return CallableLLMClient(
                model_id,
                ModelPricing(0.1, 0.2),
                predict,
                token_counter=lambda _item: 3,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = BudgetGuard(
                root / "budget.json", hard_limit_usd=1.0, max_provider_calls=20
            )
            cache = FileResponseCache(root / "responses")
            models = [
                CachedBudgetedLLMClient(
                    client(model_id),
                    cache,
                    budget,
                    prompt_version="formal-v1",
                    max_output_tokens=1,
                    input_token_margin=1.0,
                )
                for model_id in ("broken", "healthy")
            ]
            with self.assertRaisesRegex(RuntimeError, "circuit_open_models=broken"):
                collect_response_matrix(
                    models,
                    items,
                    root / "matrix.json",
                    workers=2,
                    model_circuit_breaker_failures=3,
                )
            snapshot = budget.snapshot()

        self.assertEqual(calls["healthy"], len(items))
        self.assertLessEqual(calls["broken"], 4)
        self.assertEqual(snapshot.completed_calls, len(items))
        self.assertLessEqual(snapshot.failed_calls, 4)

    def test_end_to_end_protocol_freezes_validation_choice_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [
                AnnotationItem(
                    item_id=f"item-{index}",
                    text=f"document security evidence token-{index}",
                    labels=("safe", "unsafe"),
                    metadata={
                        "gold_label": "unsafe" if index % 3 == 0 else "safe",
                        "document_id": f"doc-{index}",
                        "graph_node_id": f"node-{index}",
                        "difficulty": "medium",
                    },
                )
                for index in range(60)
            ]
            adjacency = {
                f"node-{index}": {f"node-{index - 1}": 1.0}
                for index in range(1, len(items))
            }
            budget = BudgetGuard(
                root / "budget.json", hard_limit_usd=1.0, max_provider_calls=300
            )
            cache = FileResponseCache(root / "responses")
            guarded = []
            specs = []
            for model_index, model_id in enumerate(("m0", "m1", "m2", "m3")):
                def predictor(item: AnnotationItem, rank: int = model_index) -> ModelResponse:
                    gold = str(item.metadata["gold_label"])
                    index = int(item.item_id.split("-")[-1])
                    correct = (index + rank) % (rank + 3) != 0
                    label = gold if correct else ("unsafe" if gold == "safe" else "safe")
                    return ModelResponse(label, 5 + rank, 1, raw={"confidence": 0.75})

                delegate = CallableLLMClient(
                    model_id,
                    ModelPricing(0.1 * (model_index + 1), 0.2),
                    predictor,
                    token_counter=lambda _item: 8,
                )
                guarded.append(
                    CachedBudgetedLLMClient(
                        delegate,
                        cache,
                        budget,
                        prompt_version="formal-test-v1",
                        max_output_tokens=1,
                        input_token_margin=1.0,
                    )
                )
                specs.append(ExperimentModelSpec(model_id, 0.6 + 0.05 * model_index, 10.0))
            config = {
                "schema_version": 1,
                "seed": 19,
                "pricing_verified": True,
                "api_key_file": str(root / "keys.env"),
                "fixed_cheap_models": 2,
                "dataset": {"kind": "casie", "path": "unused", "max_items": 60},
                "router": {
                    "embedding_dim": 8,
                    "confidence_threshold": 0.85,
                    "min_models": 2,
                    "warmup_rounds": 0,
                },
                "graph": {"regularization": 0.5},
                "trace": {"min_transition_observations": 1.0},
                "ccamvo": {"monte_carlo_samples": 128, "seed": 19},
                "formal": {
                    "minimum_partition_items": 2,
                    "risk_grid": [0.2, 0.05],
                    "bootstrap_iterations": 20,
                    "matrix_manifest_path": str(root / "matrix.json"),
                    "prefetch_workers": 2,
                },
                "budget": {
                    "hard_limit_usd": 1.0,
                    "max_provider_calls": 300,
                    "cache_dir": str(root / "responses"),
                    "ledger_path": str(root / "budget.json"),
                },
                "models": [{"model_id": model_id} for model_id in ("m0", "m1", "m2", "m3")],
            }
            with (
                patch(
                    "camvo.security.formal_experiment.load_real_experiment_config",
                    return_value=config,
                ),
                patch("camvo.security.formal_experiment.load_api_keys"),
                patch(
                    "camvo.security.formal_experiment._dataset",
                    return_value=(items, adjacency, {"name": "unit"}, object()),
                ),
                patch(
                    "camvo.security.formal_experiment.build_guarded_provider_pool",
                    return_value=(guarded, specs, budget),
                ),
            ):
                report = run_formal_provider_experiment(root / "config.json")

        self.assertFalse(report["protocol"]["selection_uses_test_labels"])
        self.assertTrue(report["response_matrix"]["complete"])
        self.assertIn("ccamvo", report["test"]["methods"])
        self.assertIn("trace_gcamvo", report["test"]["methods"])
        self.assertEqual(report["provider_budget"]["completed_calls"], 240)


if __name__ == "__main__":
    unittest.main()
