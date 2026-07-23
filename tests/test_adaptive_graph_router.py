import tempfile
import unittest
from pathlib import Path

from camvo.adaptive_graph_router import AdaptiveGraphCaMVoRouter, AdaptiveGraphConfig
from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class MatrixClient(LLMClient):
    def __init__(self, model_id: str, labels: dict[str, str]) -> None:
        super().__init__(model_id, ModelPricing(input_per_million=1.0))
        self.labels = labels

    def estimate_cost(self, item: AnnotationItem) -> float:
        return 1.0

    def predict(self, item: AnnotationItem) -> ModelResponse:
        return ModelResponse(self.labels[item.item_id], 10, 1)


class AdaptiveGraphRouterTests(unittest.TestCase):
    @staticmethod
    def _item(item_id: str, *, document: str = "d") -> AnnotationItem:
        return AnnotationItem(
            item_id=item_id,
            text=f"security event {item_id}",
            labels=("benign", "malicious"),
            metadata={
                "graph_node_id": item_id,
                "document_id": document,
                "hostname": "host-a",
            },
        )

    def _router(
        self,
        labels: dict[str, dict[str, str]],
        parents: dict[str, str],
        *,
        dynamic: bool = False,
    ):
        return AdaptiveGraphCaMVoRouter(
            [MatrixClient(model_id, rows) for model_id, rows in labels.items()],
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.5,
                min_models=3,
                warmup_rounds=0,
            ),
            AdaptiveGraphConfig(
                min_transition_observations=2,
                min_gate_observations=2,
                gate_lower_z=0.01,
                gate_activation_threshold=0.4,
                max_graph_blend=1.0,
                max_total_parent_weight=1.0,
                protect_base_margin=1.0,
                max_js_divergence=1.0,
                minimum_graph_margin=0.0,
                dynamic_entity_edges=dynamic,
            ),
            lambda item: (
                {parents[item.item_id]: {"weight": 1.0, "relations": ["process"]}}
                if item.item_id in parents
                else {}
            ),
        )

    def test_immature_graph_is_exact_camvo_fallback(self) -> None:
        labels = {
            "a": {"x": "benign"},
            "b": {"x": "benign"},
            "c": {"x": "malicious"},
        }
        router = self._router(labels, {})
        result = router.route(self._item("x"))
        self.assertEqual(result.label, "benign")
        self.assertEqual(result.graph_evidence_weight, 0.0)
        self.assertEqual(result.routing_trace[-1]["fallback_reason"], "no_mature_helpful_edge")

    def test_dynamic_entity_proposer_only_links_to_past_nodes(self) -> None:
        labels = {
            model_id: {"parent": "benign", "child": "benign"}
            for model_id in ("a", "b", "c")
        }
        router = self._router(labels, {}, dynamic=True)
        parent = self._item("parent", document="incident")
        child = self._item("child", document="incident")
        self.assertEqual(router._candidate_parents(parent), ())
        router.route(parent)
        router.observe_graph_feedback(parent, "benign")
        candidates = router._candidate_parents(child)
        self.assertEqual(candidates[0][0], "parent")
        self.assertIn("learned_host", candidates[0][2])

    def test_relation_can_activate_and_change_an_uncertain_vote(self) -> None:
        labels = {model_id: {} for model_id in ("a", "b", "c")}
        parents: dict[str, str] = {}
        router = self._router(labels, parents)
        for index in range(30):
            parent_id = f"p{index}"
            child_id = f"c{index}"
            parents[child_id] = parent_id
            for model_id in labels:
                labels[model_id][parent_id] = "malicious"
            labels["a"][child_id] = "benign"
            labels["b"][child_id] = "benign"
            labels["c"][child_id] = "malicious"
            parent = self._item(parent_id, document=f"d{index}")
            child = self._item(child_id, document=f"d{index}")
            router.observe_complete_feedback(
                parent,
                {
                    model_id: ModelResponse(rows[parent_id], 10, 1)
                    for model_id, rows in labels.items()
                },
                "malicious",
            )
            router.observe_complete_feedback(
                child,
                {
                    model_id: ModelResponse(rows[child_id], 10, 1)
                    for model_id, rows in labels.items()
                },
                "malicious",
            )
            # Keep the stream prior balanced so the relation is rewarded for
            # carrying information beyond class frequency, not merely for
            # predicting the majority class.
            for suffix in ("a", "b"):
                benign_id = f"n{index}{suffix}"
                for model_id in labels:
                    labels[model_id][benign_id] = "benign"
                benign = self._item(benign_id, document=f"negative-{index}-{suffix}")
                router.observe_complete_feedback(
                    benign,
                    {
                        model_id: ModelResponse(rows[benign_id], 10, 1)
                        for model_id, rows in labels.items()
                    },
                    "benign",
                )
        router.reset_graph_history()
        parents["test-child"] = "test-parent"
        for model_id in labels:
            labels[model_id]["test-parent"] = "malicious"
        labels["a"]["test-child"] = "benign"
        labels["b"]["test-child"] = "benign"
        labels["c"]["test-child"] = "malicious"
        parent = self._item("test-parent", document="test")
        child = self._item("test-child", document="test")
        router.route(parent)
        router.observe_graph_feedback(parent, "malicious")
        result = router.route(child)
        self.assertGreater(result.graph_evidence_weight, 0.0)
        self.assertEqual(result.routing_trace[-1]["graph_label"], "malicious")
        self.assertEqual(result.label, "malicious")

    def test_checkpoint_round_trip_preserves_learned_relation_state(self) -> None:
        labels = {model_id: {"x": "benign"} for model_id in ("a", "b", "c")}
        router = self._router(labels, {})
        item = self._item("x")
        router.route(item)
        router.observe_graph_feedback(item, "benign")
        restored = self._router(labels, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.json"
            router.save_checkpoint(path)
            restored.load_checkpoint(path)
        self.assertEqual(restored.state_dict(), router.state_dict())


if __name__ == "__main__":
    unittest.main()
