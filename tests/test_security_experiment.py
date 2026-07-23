import unittest

from camvo.config import CaMVoConfig
from camvo.embeddings.hashing import HashingTextEmbedder
from camvo.graph_router import GraphCaMVoConfig
from camvo.security.experiment import evaluate_security_strategies
from camvo.security.model_pool import (
    OPTC_SPECIALTIES,
    attach_simulated_specialties,
    build_simulated_security_pool,
)
from camvo.security.optc_dataset import OptcBinaryDataset
from camvo.security.simulated_experiments import run_seed_stability
from camvo.types import AnnotationItem


class SecurityExperimentTests(unittest.TestCase):
    def test_evaluates_all_six_policies_on_identical_items(self) -> None:
        items = [
            AnnotationItem(
                item_id=f"item-{index}",
                text=f"security event {index}",
                labels=("benign", "malicious"),
                metadata={
                    "gold_label": "malicious" if index % 2 else "benign",
                    "difficulty": ("easy", "medium", "hard")[index % 3],
                    "difficulty_score": (index % 3 - 1) * 0.25,
                    "graph_node_id": f"node-{index}",
                },
            )
            for index in range(18)
        ]
        prepared = attach_simulated_specialties(items, OPTC_SPECIALTIES)
        models, specs = build_simulated_security_pool(seed=3)
        adjacency = {
            f"node-{index}": {f"node-{index - 1}": 1.0}
            for index in range(1, len(items))
        }
        report = evaluate_security_strategies(
            prepared,
            models,
            specs,
            HashingTextEmbedder(16),
            CaMVoConfig(
                embedding_dim=16,
                min_models=2,
                warmup_rounds=3,
                confidence_threshold=0.9,
            ),
            dataset_metadata={"name": "unit-test"},
            graph_neighborhood=adjacency,
            graph_config=GraphCaMVoConfig(regularization=0.5),
            fixed_cheap_models=3,
            disclaimer="simulated unit test",
        )

        self.assertEqual(
            set(report.methods),
            {
                "cheapest_single",
                "strongest_single",
                "fixed_cheap",
                "full_ensemble",
                "camvo",
                "gcamvo",
            },
        )
        self.assertTrue(all(summary.items == len(items) for summary in report.methods.values()))
        self.assertEqual(report.cost_savings_vs_full["full_ensemble"], 0.0)
        self.assertIsNotNone(report.macro_f1_delta_gcamvo_vs_camvo)

    def test_seed_stability_reports_mean_std_and_individual_runs(self) -> None:
        items = tuple(
            AnnotationItem(
                item_id=f"optc:{index}",
                text=f"process creation telemetry sample identifier token{index}",
                labels=("benign", "malicious"),
                metadata={
                    "gold_label": "malicious" if index % 2 else "benign",
                    "difficulty": "medium",
                    "difficulty_score": 0.1,
                    "graph_node_id": str(index),
                },
            )
            for index in range(20)
        )
        dataset = OptcBinaryDataset(
            items=items,
            correlations=(),
            stats={"name": "unit-test"},
            simulation_only=True,
        )
        report = run_seed_stability(
            dataset,
            seeds=(1, 2, 3),
            confidence_threshold=0.85,
            graph_regularization=0.0,
            warmup_rounds=2,
            embedding_dim=8,
        )

        self.assertEqual(report.seeds, (1, 2, 3))
        self.assertEqual(report.methods["camvo"].runs, 3)
        self.assertEqual(len(report.individual_runs), 3)
        self.assertAlmostEqual(report.graph_f1_gain_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
