import tempfile
import unittest
from pathlib import Path

from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import StaticGraphNeighborhood
from camvo.llms.base import LLMClient
from camvo.trace_router import (
    TraceGraphCaMVoConfig,
    TraceGraphCaMVoRouter,
    expected_information_gain,
    posterior_after_vote,
)
from camvo.types import AnnotationItem, ModelPricing, ModelResponse


class _PerfectModel(LLMClient):
    def predict(self, item: AnnotationItem) -> ModelResponse:
        return ModelResponse(
            label=str(item.metadata["gold_label"]),
            input_tokens=20,
            output_tokens=4,
            raw={"confidence": 0.95},
        )


class TraceRouterTests(unittest.TestCase):
    def test_posterior_update_and_information_gain_are_well_formed(self) -> None:
        import numpy as np

        prior = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)
        updated = posterior_after_vote(prior, 1, 0.8)
        self.assertAlmostEqual(float(updated.sum()), 1.0)
        self.assertGreater(updated[1], updated[0])
        self.assertGreater(expected_information_gain(prior, 0.8), 0.0)

    @staticmethod
    def _router() -> TraceGraphCaMVoRouter:
        models = [
            _PerfectModel(
                f"m{index}",
                ModelPricing(input_per_million=0.1 * index, output_per_million=0.2 * index),
            )
            for index in (1, 2, 3)
        ]
        adjacency = {
            "e1": {"e2": 1.0},
            "e2": {"e1": 1.0, "e3": 1.0},
            "e3": {"e2": 1.0},
        }
        return TraceGraphCaMVoRouter(
            models,
            HashingTextEmbedder(12),
            CaMVoConfig(
                embedding_dim=12,
                min_models=2,
                warmup_rounds=1,
                confidence_threshold=0.9,
            ),
            TraceGraphCaMVoConfig(
                risk_tolerance=0.16,
                min_transition_observations=0.1,
                transition_prior=0.1,
                min_graph_informativeness=0.0,
                min_confidence_bin_observations=5,
            ),
            StaticGraphNeighborhood(adjacency),
        )

    @staticmethod
    def _item(event_id: str) -> AnnotationItem:
        return AnnotationItem(
            item_id=f"item:{event_id}",
            text=f"event evidence {event_id}",
            labels=("a", "b", "c"),
            metadata={
                "event_id": event_id,
                "graph_node_id": event_id,
                "gold_label": "a",
            },
        )

    def test_escalates_sequentially_and_uses_only_mature_causal_graph_evidence(self) -> None:
        router = self._router()
        first = router.route(self._item("e1"))
        self.assertEqual(len(first.selected_models), 3)
        self.assertEqual(first.graph_evidence_weight, 0.0)

        second = router.route(self._item("e2"))
        self.assertEqual(len(second.selected_models), 2)
        self.assertEqual(second.graph_evidence_weight, 0.0)

        third = router.route(self._item("e3"))
        self.assertGreater(third.graph_evidence_weight, 0.0)
        self.assertLessEqual(len(third.selected_models), 2)
        self.assertFalse(third.abstained)

    def test_trace_state_survives_checkpoint(self) -> None:
        router = self._router()
        router.route(self._item("e1"))
        router.route(self._item("e2"))
        restored = self._router()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            router.save_checkpoint(path)
            restored.load_checkpoint(path)

        self.assertEqual(restored.round_index, router.round_index)
        expected = router.route(self._item("e3"))
        actual = restored.route(self._item("e3"))
        self.assertEqual(actual.selected_models, expected.selected_models)
        self.assertEqual(actual.label, expected.label)
        self.assertAlmostEqual(actual.subset_confidence, expected.subset_confidence)

    def test_complete_feedback_warm_starts_all_models_without_provider_calls(self) -> None:
        router = self._router()
        item = self._item("e1")
        responses = {
            model_id: ModelResponse("a", 20, 4, raw={"confidence": 0.9})
            for model_id in router.models
        }

        router.observe_complete_feedback(item, responses, "a")

        self.assertEqual(router.round_index, 1)
        self.assertTrue(all(state.observations == 1 for state in router._states.values()))
        self.assertTrue(
            all(tracker.observations == 1 for tracker in router._reliability.values())
        )

    def test_calibrated_fallback_is_prioritized_when_risk_remains_high(self) -> None:
        import numpy as np

        router = self._router()
        router.trace_config = TraceGraphCaMVoConfig(
            fallback_model_id="m3",
            fallback_after_models=1,
            fallback_trigger_risk=0.1,
        )
        model_id, _utility, _gain = router._next_model(
            np.asarray([1 / 3, 1 / 3, 1 / 3]),
            {"m2", "m3"},
            ["m1"],
            {"m1": 0.6, "m2": 0.7, "m3": 0.9},
            {"m1": 0.1, "m2": 0.01, "m3": 10.0},
        )
        self.assertEqual(model_id, "m3")


if __name__ == "__main__":
    unittest.main()
