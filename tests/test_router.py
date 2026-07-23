import tempfile
import unittest
from pathlib import Path

from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.simulated import SimulatedLLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelPricing


def _build_router() -> CaMVoRouter:
    models = [
        SimulatedLLMClient(
            f"model-{index}",
            ModelPricing(0.1 + index),
            base_accuracy=0.70 + index * 0.05,
            seed=11,
        )
        for index in range(3)
    ]
    config = CaMVoConfig(
        embedding_dim=32,
        confidence_threshold=0.7,
        min_models=2,
        warmup_rounds=2,
    )
    return CaMVoRouter(models, HashingTextEmbedder(32), config)


class RouterTests(unittest.TestCase):
    def test_route_updates_and_checkpoint_round_trip(self) -> None:
        router = _build_router()
        items = [
            AnnotationItem(
                item_id=f"item-{index}",
                text=f"easy benchmark question {index}",
                labels=("A", "B"),
                metadata={"gold_label": "A", "difficulty_score": -0.5},
            )
            for index in range(5)
        ]
        results = router.route_many(items)
        self.assertEqual(router.round_index, 5)
        self.assertEqual(len(results[0].selected_models), 3)
        self.assertTrue(all(result.label in ("A", "B") for result in results))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            router.save_checkpoint(checkpoint)
            restored = _build_router()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.state_dict(), router.state_dict())


if __name__ == "__main__":
    unittest.main()

