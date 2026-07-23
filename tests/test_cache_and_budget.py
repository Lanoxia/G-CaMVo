import tempfile
import unittest
from pathlib import Path

from camvo.budget import BudgetGuard
from camvo.exceptions import BudgetExceededError, ResponseCacheError
from camvo.llms import CachedBudgetedLLMClient, CallableLLMClient, FileResponseCache
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


def _item(item_id: str = "item-1", text: str = "security event") -> AnnotationItem:
    return AnnotationItem(item_id=item_id, text=text, labels=("A", "B"))


class CacheAndBudgetTests(unittest.TestCase):
    def test_cache_hit_avoids_second_provider_call_and_cost(self) -> None:
        calls = 0

        def predictor(item: AnnotationItem) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse("A", input_tokens=4, output_tokens=1)

        delegate = CallableLLMClient(
            "provider/model",
            ModelPricing(input_per_million=0.5, output_per_million=2.0),
            predictor,
            token_counter=lambda _item: 4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = BudgetGuard(root / "ledger.json", hard_limit_usd=1.0, max_provider_calls=5)
            client = CachedBudgetedLLMClient(
                delegate,
                FileResponseCache(root / "responses"),
                guard,
                prompt_version="casie-v1",
                input_token_margin=1.0,
            )
            first = client.predict(_item())
            second = client.predict(_item())
            snapshot = guard.snapshot()

            self.assertEqual(first.label, second.label)
            self.assertEqual(calls, 1)
            self.assertEqual(snapshot.provider_attempts, 1)
            self.assertEqual(snapshot.completed_calls, 1)
            self.assertEqual(snapshot.cache_hits, 1)
            self.assertAlmostEqual(snapshot.spent_usd, 4e-6)

            reloaded = BudgetGuard(
                root / "ledger.json",
                hard_limit_usd=1.0,
                max_provider_calls=5,
            )
            self.assertEqual(reloaded.snapshot(), snapshot)

    def test_prompt_or_text_change_invalidates_cache(self) -> None:
        calls = 0

        def predictor(item: AnnotationItem) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse("A", input_tokens=2, output_tokens=1)

        delegate = CallableLLMClient(
            "provider/model",
            ModelPricing(input_per_million=0.1),
            predictor,
            token_counter=lambda _item: 2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = BudgetGuard(root / "ledger.json", hard_limit_usd=1.0, max_provider_calls=10)
            cache = FileResponseCache(root / "responses")
            v1 = CachedBudgetedLLMClient(
                delegate, cache, guard, prompt_version="v1", input_token_margin=1.0
            )
            v2 = CachedBudgetedLLMClient(
                delegate, cache, guard, prompt_version="v2", input_token_margin=1.0
            )
            v1.predict(_item())
            v1.predict(_item())
            v2.predict(_item())
            v2.predict(_item(text="changed security event"))

        self.assertEqual(calls, 3)

    def test_budget_blocks_before_provider_is_invoked(self) -> None:
        calls = 0

        def predictor(item: AnnotationItem) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse("A", input_tokens=2, output_tokens=1)

        delegate = CallableLLMClient(
            "expensive",
            ModelPricing(input_per_million=1_000_000.0),
            predictor,
            token_counter=lambda _item: 2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = BudgetGuard(root / "ledger.json", hard_limit_usd=0.5, max_provider_calls=5)
            client = CachedBudgetedLLMClient(
                delegate,
                FileResponseCache(root / "responses"),
                guard,
                prompt_version="v1",
                max_output_tokens=1,
                input_token_margin=1.0,
            )
            with self.assertRaises(BudgetExceededError):
                client.predict(_item())
            self.assertEqual(guard.snapshot().provider_attempts, 0)
        self.assertEqual(calls, 0)

    def test_call_limit_blocks_distinct_uncached_item(self) -> None:
        calls = 0

        def predictor(item: AnnotationItem) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse("A", input_tokens=1, output_tokens=1)

        delegate = CallableLLMClient(
            "provider/model",
            ModelPricing(input_per_million=0.1),
            predictor,
            token_counter=lambda _item: 1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = BudgetGuard(root / "ledger.json", hard_limit_usd=1.0, max_provider_calls=1)
            client = CachedBudgetedLLMClient(
                delegate,
                FileResponseCache(root / "responses"),
                guard,
                prompt_version="v1",
                input_token_margin=1.0,
            )
            client.predict(_item("item-1"))
            with self.assertRaises(BudgetExceededError):
                client.predict(_item("item-2"))
        self.assertEqual(calls, 1)

    def test_corrupt_cache_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = FileResponseCache(directory)
            item = _item()
            key = cache.key_for("model", item, "v1")
            path = cache.put(key, ModelResponse("A", 2, 1))
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ResponseCacheError):
                cache.get(key)

    def test_cache_preserves_confidence_but_drops_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = FileResponseCache(directory)
            item = _item()
            key = cache.key_for("model", item, "v1")
            cache.put(
                key,
                ModelResponse(
                    "A",
                    2,
                    1,
                    raw={
                        "confidence": 0.83,
                        "rationale": "normalized explanation",
                        "api_key": "must-not-persist",
                        "headers": {"Authorization": "must-not-persist"},
                    },
                ),
            )

            restored = cache.get(key)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.raw["confidence"], 0.83)
            self.assertEqual(restored.raw["rationale"], "normalized explanation")
            self.assertTrue(restored.raw["cache_hit"])
            self.assertNotIn("api_key", restored.raw)
            self.assertNotIn("headers", restored.raw)

    def test_schema_one_cache_remains_readable(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            cache = FileResponseCache(directory)
            item = _item()
            key = cache.key_for("model", item, "v1")
            path = cache.put(key, ModelResponse("A", 2, 1, raw={"confidence": 0.9}))
            record = json.loads(path.read_text(encoding="utf-8"))
            record["schema_version"] = 1
            record["response"].pop("raw")
            path.write_text(json.dumps(record), encoding="utf-8")

            restored = cache.get(key)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertTrue(restored.raw["cache_hit"])
            self.assertNotIn("confidence", restored.raw)

    def test_pending_reservation_survives_restart_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.json"
            guard = BudgetGuard(ledger, hard_limit_usd=1.0, max_provider_calls=5)
            reservation = guard.reserve(
                model_id="provider/model",
                item_id="item-1",
                estimated_max_cost_usd=0.4,
            )
            reloaded = BudgetGuard(ledger, hard_limit_usd=1.0, max_provider_calls=5)
            self.assertAlmostEqual(reloaded.snapshot().reserved_usd, 0.4)
            self.assertEqual(reloaded.snapshot().pending_reservations, 1)
            reloaded.release_stale_reservation(reservation.reservation_id)
            self.assertAlmostEqual(reloaded.snapshot().remaining_usd, 1.0)
            self.assertEqual(reloaded.snapshot().pending_reservations, 0)


if __name__ == "__main__":
    unittest.main()
