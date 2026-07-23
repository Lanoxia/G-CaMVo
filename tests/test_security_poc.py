import unittest

from camvo.security.casie import CASIE_LABELS
from camvo.security.poc import evaluate_casie_items
from camvo.types import AnnotationItem


class SecurityPocTests(unittest.TestCase):
    def test_poc_compares_required_baselines(self) -> None:
        items = []
        for index in range(75):
            label = CASIE_LABELS[index % len(CASIE_LABELS)]
            items.append(
                AnnotationItem(
                    item_id=f"casie-test-{index}",
                    text=f"Security context [EVENT] marker {index} [/EVENT] label family {label}",
                    labels=CASIE_LABELS,
                    metadata={
                        "gold_label": label,
                        "difficulty_score": (-0.3, 0.1, 0.6)[index % 3],
                        "difficulty": ("easy", "medium", "hard")[index % 3],
                    },
                )
            )

        report = evaluate_casie_items(
            items,
            seed=3,
            confidence_threshold=0.75,
            min_models=2,
            warmup_rounds=5,
            embedding_dim=32,
        )
        self.assertEqual(
            set(report.methods),
            {
                "cheapest_single",
                "strongest_single",
                "fixed_cheap_three",
                "full_ensemble",
                "camvo",
            },
        )
        self.assertEqual(report.methods["full_ensemble"].average_models, 5.0)
        self.assertLessEqual(report.methods["camvo"].average_models, 5.0)
        self.assertEqual(report.dataset["items_evaluated"], len(items))
        self.assertIn("simulated", report.simulation_disclaimer.lower())


if __name__ == "__main__":
    unittest.main()
