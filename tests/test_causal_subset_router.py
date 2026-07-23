import tempfile
import unittest
from pathlib import Path

from camvo.causal_subset_router import (
    CausalSubsetGraphCaMVoRouter,
    CausalSubsetGraphConfig,
    StaticCausalTypedNeighborhood,
)
from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class FixedClient(LLMClient):
    def __init__(self, model_id: str, labels: dict[str, str], cost: float) -> None:
        super().__init__(model_id, ModelPricing(input_per_million=cost))
        self.labels = labels
        self.calls: list[str] = []

    def estimate_cost(self, item: AnnotationItem) -> float:
        return self.pricing.input_per_million

    def predict(self, item: AnnotationItem) -> ModelResponse:
        self.calls.append(item.item_id)
        return ModelResponse(self.labels[item.item_id], 10, 1)


class CausalSubsetRouterTests(unittest.TestCase):
    def _router(self) -> CausalSubsetGraphCaMVoRouter:
        labels = {"p1": "malicious", "p2": "malicious", "c": "malicious"}
        models = [
            FixedClient("anonymous-a", labels, 1.0),
            FixedClient("anonymous-b", labels, 2.0),
            FixedClient("anonymous-c", labels, 3.0),
        ]
        parents = {
            "p2": {
                "p1": {"weight": 1.0, "relations": ["process"]},
            },
            "c": {
                "p1": {"weight": 1.0, "relations": ["process"]},
                "p2": {"weight": 1.0, "relations": ["process"]},
            }
        }
        return CausalSubsetGraphCaMVoRouter(
            models,
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                confidence_threshold=0.50,
                min_models=2,
                warmup_rounds=0,
            ),
            CausalSubsetGraphConfig(
                regularization=1.0,
                min_transition_observations=1.0,
                online_transition_weight=1.0,
            ),
            StaticCausalTypedNeighborhood(parents),
        )

    @staticmethod
    def _item(item_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=item_id,
            text=f"event {item_id}",
            labels=("benign", "malicious"),
            metadata={"graph_node_id": item_id},
        )

    def test_round_selects_subset_before_single_weighted_vote(self) -> None:
        router = self._router()
        result = router.route(self._item("p1"))
        self.assertGreaterEqual(len(result.selected_models), 2)
        self.assertEqual(set(result.responses), set(result.selected_models))
        self.assertEqual(result.label, "malicious")
        for model_id, client in router.models.items():
            expected = ["p1"] if model_id in result.selected_models else []
            self.assertEqual(client.calls, expected)

    def test_only_past_parent_rewards_regularize_current_lcb(self) -> None:
        router = self._router()
        responses = {
            model_id: ModelResponse("malicious", 10, 1) for model_id in router.models
        }
        router.observe_complete_feedback(self._item("p1"), responses, "malicious")
        router.observe_complete_feedback(self._item("p2"), responses, "malicious")
        context = router._context(self._item("c"))
        scores, _ = router._score_models(self._item("c"), context, router.round_index + 1)
        self.assertTrue(all(score.graph_neighbor_count == 2 for score in scores.values()))
        self.assertTrue(
            all(score.graph_regularized_lower_bound is not None for score in scores.values())
        )

    def test_reset_removes_cross_split_history_but_keeps_transitions(self) -> None:
        router = self._router()
        responses = {
            model_id: ModelResponse("malicious", 10, 1) for model_id in router.models
        }
        router.observe_complete_feedback(self._item("p1"), responses, "malicious")
        router.observe_complete_feedback(self._item("p2"), responses, "malicious")
        before = {
            model_id: {relation: matrix.copy() for relation, matrix in rows.items()}
            for model_id, rows in router._transitions.items()
        }
        router.reset_graph_history()
        self.assertTrue(all(not rows for rows in router._reward_history.values()))
        for model_id, rows in before.items():
            for relation, matrix in rows.items():
                self.assertTrue((router._transitions[model_id][relation] == matrix).all())

    def test_checkpoint_preserves_typed_graph_state(self) -> None:
        router = self._router()
        router.route(self._item("p1"))
        restored = self._router()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "causal-subset.json"
            router.save_checkpoint(path)
            restored.load_checkpoint(path)
        self.assertEqual(restored._reward_history, router._reward_history)

    def test_graph_prior_regularizes_but_does_not_replace_subset_vote(self) -> None:
        router = self._router()
        router.graph_config = CausalSubsetGraphConfig(
            regularization=0.0,
            label_vote_regularization=1000.0,
            min_transition_observations=1.0,
            online_transition_weight=0.0,
        )
        responses = {
            model_id: ModelResponse("malicious", 10, 1) for model_id in router.models
        }
        router.observe_complete_feedback(self._item("p1"), responses, "malicious")
        router.observe_complete_feedback(self._item("p2"), responses, "malicious")
        for client in router.models.values():
            client.labels["c"] = "benign"
        result = router.route(self._item("c"))
        self.assertGreaterEqual(len(result.selected_models), 2)
        self.assertTrue(all(label == "benign" for label in result.responses.values()))
        self.assertEqual(result.label, "malicious")

    def test_joint_oracle_still_selects_one_complete_subset_before_votes(self) -> None:
        router = self._router()
        router.graph_config = CausalSubsetGraphConfig(
            regularization=0.0,
            label_vote_regularization=1000.0,
            min_transition_observations=1.0,
            online_transition_weight=0.0,
            joint_graph_oracle=True,
        )
        responses = {
            model_id: ModelResponse("malicious", 10, 1) for model_id in router.models
        }
        router.observe_complete_feedback(self._item("p1"), responses, "malicious")
        router.observe_complete_feedback(self._item("p2"), responses, "malicious")
        for client in router.models.values():
            client.labels["c"] = "benign"
        result = router.route(self._item("c"))
        self.assertEqual(len(result.selected_models), 2)
        self.assertEqual(set(result.responses), set(result.selected_models))
        unselected = set(router.models) - set(result.selected_models)
        self.assertTrue(all("c" not in router.models[name].calls for name in unselected))


if __name__ == "__main__":
    unittest.main()
