import tempfile
import unittest
from pathlib import Path

from camvo.config import CaMVoConfig
from camvo.continuous_trace_router import (
    ContinuousTraceCaMVoRouter,
    ContinuousTraceConfig,
)
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class ConfidenceClient(LLMClient):
    def __init__(self, model_id: str, labels: dict[str, str], confidence: float) -> None:
        super().__init__(model_id, ModelPricing(input_per_million=1.0))
        self.labels = labels
        self.confidence = confidence

    def estimate_cost(self, item: AnnotationItem) -> float:
        return 1.0

    def predict(self, item: AnnotationItem) -> ModelResponse:
        return ModelResponse(
            self.labels[item.item_id],
            10,
            1,
            raw={"confidence": self.confidence},
        )


class ContinuousTraceRouterTests(unittest.TestCase):
    @staticmethod
    def _item(item_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=item_id,
            text=f"event {item_id}",
            labels=("benign", "malicious"),
            metadata={"graph_node_id": item_id},
        )

    def _router(self) -> ContinuousTraceCaMVoRouter:
        labels = {
            "parent": "malicious",
            "child": "benign",
        }
        return ContinuousTraceCaMVoRouter(
            [
                ConfidenceClient("a", labels, 0.95),
                ConfidenceClient("b", labels, 0.90),
                ConfidenceClient("c", labels, 0.85),
            ],
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.5,
                min_models=3,
                warmup_rounds=0,
            ),
            ContinuousTraceConfig(
                graph_strength=0.3,
                label_logit_bias=(("malicious", 1.0),),
            ),
            lambda item: (
                {"parent": {"weight": 1.0}}
                if item.item_id == "child"
                else {}
            ),
        )

    def test_uses_only_already_processed_parent(self) -> None:
        router = self._router()
        first = router.route(self._item("parent"))
        second = router.route(self._item("child"))
        self.assertEqual(first.graph_evidence_weight, 0.0)
        self.assertEqual(second.graph_evidence_weight, 1.0)
        self.assertEqual(set(second.responses), set(second.selected_models))

    def test_model_consensus_not_graph_head_updates_competence(self) -> None:
        router = self._router()
        router.route(self._item("parent"))
        result = router.route(self._item("child"))
        self.assertEqual(result.label, "benign")
        self.assertTrue(all(state.agreements == 2 for state in router._states.values()))

    def test_checkpoint_round_trip_preserves_parent_posteriors(self) -> None:
        router = self._router()
        router.route(self._item("parent"))
        restored = self._router()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuous.json"
            router.save_checkpoint(path)
            restored.load_checkpoint(path)
        self.assertEqual(restored.state_dict(), router.state_dict())


if __name__ == "__main__":
    unittest.main()
