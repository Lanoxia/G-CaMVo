import unittest

import numpy as np

from camvo.algorithm.calibration import BetaMixtureCalibrator, laplace_smooth
from camvo.algorithm.linucb import LinUCBArm
from camvo.algorithm.oracle import ExhaustiveSubsetOracle, OracleCandidate
from camvo.algorithm.confidence import exact_majority_confidence


class AlgorithmComponentTests(unittest.TestCase):
    def test_linucb_learns_positive_agreement(self) -> None:
        arm = LinUCBArm(3, exploration_alpha=0.0)
        context = np.array([1.0, 0.0, 0.0])
        before = arm.score(context).prediction
        for _ in range(5):
            arm.update(context, 1.0)
        after = arm.score(context).prediction
        self.assertEqual(before, 0.0)
        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.0)

    def test_beta_calibrator_handles_cold_start_and_updates(self) -> None:
        calibrator = BetaMixtureCalibrator(min_samples_per_class=3)
        self.assertAlmostEqual(calibrator.posterior(0.4, 0.5), 0.4)
        for score in (0.7, 0.8, 0.9):
            calibrator.update(score, True)
        for score in (0.1, 0.2, 0.3):
            calibrator.update(score, False)
        posterior = calibrator.posterior(0.8, 0.5)
        self.assertTrue(0.0 < posterior < 1.0)
        self.assertGreater(posterior, 0.5)

    def test_laplace_smoothing_cold_start_is_half(self) -> None:
        self.assertEqual(laplace_smooth(0.9, 0, 1, 1.0), 0.5)

    def test_oracle_returns_global_cheapest_feasible_subset(self) -> None:
        oracle = ExhaustiveSubsetOracle(exact_majority_confidence)
        candidates = [
            OracleCandidate("cheap", 0.1, 0.90, 1.0),
            OracleCandidate("medium", 0.5, 0.85, 1.0),
            OracleCandidate("expensive", 2.0, 0.95, 1.0),
        ]
        selection = oracle.select(candidates, threshold=0.8, min_models=1)
        self.assertEqual(selection.model_ids, ("cheap",))
        self.assertTrue(selection.feasible)


if __name__ == "__main__":
    unittest.main()

