import unittest

from camvo.simulation import run_simulation


class SimulationTests(unittest.TestCase):
    def test_offline_simulation_is_reproducible_and_saves_cost(self) -> None:
        report, _ = run_simulation(
            item_count=160,
            seed=5,
            confidence_threshold=0.85,
            min_models=3,
            warmup_rounds=15,
        )
        self.assertGreater(report.camvo.accuracy, 0.5)
        self.assertGreater(report.full_ensemble.accuracy, 0.5)
        self.assertGreater(report.cost_savings_fraction, 0.0)
        self.assertLess(report.camvo.average_models, report.full_ensemble.average_models)


if __name__ == "__main__":
    unittest.main()

