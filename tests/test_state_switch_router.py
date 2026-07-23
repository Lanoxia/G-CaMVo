import unittest

from camvo.adaptive_graph_router import AdaptiveGraphConfig
from camvo.config import CaMVoConfig
from camvo.dcr_graph_router import DCRGraphConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.base import LLMClient
from camvo.state_switch_router import (
    CausalStateSwitchGraphCaMVoRouter,
    StateSwitchConfig,
)
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class MatrixClient(LLMClient):
    def __init__(self, model_id: str, labels: dict[str, str]) -> None:
        super().__init__(model_id, ModelPricing(input_per_million=1.0))
        self.labels = labels

    def estimate_cost(self, item: AnnotationItem) -> float:
        return 1.0

    def predict(self, item: AnnotationItem) -> ModelResponse:
        return ModelResponse(self.labels[item.item_id], 10, 1)


class StateSwitchRouterTests(unittest.TestCase):
    @staticmethod
    def _item(item_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=item_id,
            text=f"ordered security event {item_id}",
            labels=("benign", "malicious"),
            metadata={"graph_node_id": item_id, "hostname": "host-a"},
        )

    def test_vote_pattern_can_switch_only_after_past_state_exists(self) -> None:
        matrix = {model_id: {} for model_id in ("a", "b", "c")}
        router = CausalStateSwitchGraphCaMVoRouter(
            [MatrixClient(model_id, rows) for model_id, rows in matrix.items()],
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.8,
                min_models=2,
                warmup_rounds=0,
            ),
            AdaptiveGraphConfig(dynamic_entity_edges=False),
            DCRGraphConfig(minimum_audited_rows=2),
            StateSwitchConfig(
                pattern_prior_strength=0.5,
                switch_threshold=0.60,
                minimum_pattern_support=1,
                subset_size=2,
            ),
            lambda _item: {},
        )
        for index in range(12):
            item_id = f"cal-{index}"
            gold = "benign" if index % 2 == 0 else "malicious"
            for model_id in matrix:
                matrix[model_id][item_id] = gold
            item = self._item(item_id)
            router.observe_complete_feedback(
                item,
                {
                    model_id: ModelResponse(rows[item_id], 10, 1)
                    for model_id, rows in matrix.items()
                },
                gold,
            )
        router.reset_graph_history()

        for model_id in matrix:
            matrix[model_id]["first"] = "benign"
            matrix[model_id]["second"] = "malicious"
        first = self._item("first")
        first_result = router.route(first)
        self.assertEqual(first_result.routing_trace[-1]["fallback"], "no_entity_state")
        router.observe_graph_feedback(first, "benign")

        second = self._item("second")
        second_result = router.route(second)
        self.assertEqual(len(second_result.selected_models), 2)
        self.assertEqual(second_result.label, "malicious")
        self.assertTrue(second_result.routing_trace[-1]["switched"])
        self.assertEqual(second_result.routing_trace[-1]["previous_state"], "benign")

    def test_reset_clears_stream_state_but_preserves_calibration(self) -> None:
        matrix = {model_id: {"x": "benign"} for model_id in ("a", "b")}
        router = CausalStateSwitchGraphCaMVoRouter(
            [MatrixClient(model_id, rows) for model_id, rows in matrix.items()],
            HashingTextEmbedder(8),
            CaMVoConfig(embedding_dim=8, min_models=2, warmup_rounds=0),
            AdaptiveGraphConfig(dynamic_entity_edges=False),
            DCRGraphConfig(minimum_audited_rows=0),
            StateSwitchConfig(subset_size=2),
            lambda _item: {},
        )
        item = self._item("x")
        router.route(item)
        router.observe_graph_feedback(item, "benign")
        self.assertEqual(router.state_switch_diagnostics()["stream_entities"], 1)
        router.reset_graph_history()
        diagnostics = router.state_switch_diagnostics()
        self.assertEqual(diagnostics["stream_entities"], 0)
        self.assertGreaterEqual(diagnostics["dcr"]["audited_rows"], 1)

    def test_state_transition_does_not_shadow_graph_relation_transition(self) -> None:
        matrix = {
            model_id: {"parent": "benign", "child": "malicious"}
            for model_id in ("a", "b")
        }
        router = CausalStateSwitchGraphCaMVoRouter(
            [MatrixClient(model_id, rows) for model_id, rows in matrix.items()],
            HashingTextEmbedder(8),
            CaMVoConfig(embedding_dim=8, min_models=2, warmup_rounds=0),
            AdaptiveGraphConfig(dynamic_entity_edges=False),
            DCRGraphConfig(minimum_audited_rows=0),
            StateSwitchConfig(subset_size=2),
            lambda item: (
                {"parent": {"weight": 1.0, "relations": ["process"]}}
                if item.item_id == "child"
                else {}
            ),
        )
        parent = self._item("parent")
        child = self._item("child")
        router.route(parent)
        router.observe_graph_feedback(parent, "benign")
        router.route(child)
        router.observe_graph_feedback(child, "malicious")
        relation_matrix = router._transition(("benign", "malicious"), "process")
        self.assertEqual(relation_matrix.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
