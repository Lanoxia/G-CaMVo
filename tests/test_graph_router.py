import tempfile
import unittest
from pathlib import Path

from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import GraphCaMVoConfig, GraphCaMVoRouter, StaticGraphNeighborhood
from camvo.llms.simulated import SimulatedLLMClient
from camvo.types import AnnotationItem, ModelPricing


class GraphRouterTests(unittest.TestCase):
    @staticmethod
    def _router() -> GraphCaMVoRouter:
        models = [
            SimulatedLLMClient(
                f"m{index}",
                ModelPricing(input_per_million=0.1 * index),
                base_accuracy=0.75 + index / 20,
                seed=4,
            )
            for index in (1, 2)
        ]
        adjacency = {"e1": {"e2": 1.5}, "e2": {"e1": 1.5}}
        return GraphCaMVoRouter(
            models,
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.8,
                min_models=2,
                warmup_rounds=0,
            ),
            GraphCaMVoConfig(regularization=1.0),
            StaticGraphNeighborhood(adjacency),
        )

    @staticmethod
    def _item(event_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=f"optc:{event_id}",
            text=f"event evidence {event_id}",
            labels=("benign", "malicious"),
            metadata={"event_id": event_id, "gold_label": "malicious"},
        )

    def test_uses_only_previously_routed_neighbors(self) -> None:
        router = self._router()
        first = router.route(self._item("e1"))
        self.assertTrue(
            all(score.graph_neighbor_count == 0 for score in first.scores.values())
        )
        for history in router._graph_history.values():
            history["e1"] = 0.95

        second = router.route(self._item("e2"))
        for score in second.scores.values():
            self.assertEqual(score.graph_neighbor_count, 1)
            self.assertGreater(
                score.graph_regularized_lower_bound,
                score.smoothed_lower_bound,
            )

    def test_graph_history_survives_checkpoint(self) -> None:
        router = self._router()
        router.route(self._item("e1"))
        restored = self._router()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph-router.json"
            router.save_checkpoint(path)
            restored.load_checkpoint(path)

        self.assertEqual(restored.round_index, 1)
        self.assertEqual(restored._graph_history, router._graph_history)


if __name__ == "__main__":
    unittest.main()
