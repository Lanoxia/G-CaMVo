import unittest

from camvo.adaptive_graph_router import AdaptiveGraphConfig
from camvo.config import CaMVoConfig
from camvo.dcr_graph_router import DCRGraphCaMVoRouter, DCRGraphConfig
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


class DCRGraphRouterTests(unittest.TestCase):
    @staticmethod
    def _item(item_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=item_id,
            text=f"security telemetry observation identifier {item_id}",
            labels=("benign", "malicious"),
            metadata={"graph_node_id": item_id, "hostname": "host-a"},
        )

    def _router(self, matrix: dict[str, dict[str, str]], *, min_models: int = 2):
        return DCRGraphCaMVoRouter(
            [MatrixClient(model_id, rows) for model_id, rows in matrix.items()],
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.8,
                min_models=min_models,
                warmup_rounds=0,
            ),
            AdaptiveGraphConfig(dynamic_entity_edges=False),
            DCRGraphConfig(
                minimum_audited_rows=4,
                redundancy_penalty=2.0,
                cost_penalty=0.0,
                competence_bonus=0.0,
            ),
            lambda _item: {},
        )

    def test_redundant_pair_is_not_selected_when_complement_exists(self) -> None:
        matrix = {model_id: {} for model_id in ("a", "b", "c")}
        router = self._router(matrix)
        for index in range(20):
            item_id = f"cal-{index}"
            gold = "malicious" if index % 2 else "benign"
            shared_wrong = index in {0, 1, 2, 3}
            complement_wrong = index in {8, 9, 10, 11}
            matrix["a"][item_id] = (
                "malicious" if gold == "benign" else "benign"
            ) if shared_wrong else gold
            matrix["b"][item_id] = matrix["a"][item_id]
            matrix["c"][item_id] = (
                "malicious" if gold == "benign" else "benign"
            ) if complement_wrong else gold
            item = self._item(item_id)
            router.observe_complete_feedback(
                item,
                {
                    model_id: ModelResponse(rows[item_id], 10, 1)
                    for model_id, rows in matrix.items()
                },
                gold,
            )
        for model_id in matrix:
            matrix[model_id]["test"] = "benign"
        result = router.route(self._item("test"))
        self.assertEqual(len(result.selected_models), 2)
        self.assertIn("c", result.selected_models)
        self.assertNotEqual(set(result.selected_models), {"a", "b"})
        trace = result.routing_trace[-1]
        self.assertEqual(trace["component"], "dcr_subset")
        correlations = router.dcr_diagnostics()["error_correlations"][
            "benign|malicious"
        ]
        self.assertGreater(correlations["a|b"], correlations["a|c"])

    def test_audited_feedback_updates_confusion_and_pair_statistics(self) -> None:
        matrix = {
            "a": {"x": "benign"},
            "b": {"x": "malicious"},
        }
        router = self._router(matrix, min_models=1)
        item = self._item("x")
        result = router.route(item)
        router.observe_graph_feedback(item, "malicious")
        diagnostics = router.dcr_diagnostics()
        self.assertEqual(diagnostics["audited_rows"], 1)
        probabilities = diagnostics["confusion_probabilities"]["benign|malicious"]
        self.assertIn(result.selected_models[0], probabilities)
        self.assertEqual(diagnostics["graph"]["pending_feedback"], 0)

    def test_cold_start_falls_back_to_camvo_selection(self) -> None:
        matrix = {
            model_id: {"x": "benign"} for model_id in ("a", "b", "c")
        }
        router = self._router(matrix)
        result = router.route(self._item("x"))
        self.assertEqual(set(result.selected_models), {"a", "b", "c"})
        self.assertEqual(
            result.routing_trace[-1]["fallback"], "insufficient_audited_rows"
        )


if __name__ == "__main__":
    unittest.main()
