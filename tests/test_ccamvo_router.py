import unittest

import numpy as np

from camvo.ccamvo_router import (
    OnlineCorrelationEstimator,
    gaussian_copula_majority_confidence,
    nearest_psd_correlation,
)


class CorrelatedCaMVoTests(unittest.TestCase):
    def test_nearest_psd_correlation_is_valid(self) -> None:
        invalid = np.asarray([[1.0, 0.95, 0.95], [0.95, 1.0, -0.95], [0.95, -0.95, 1.0]])
        projected = nearest_psd_correlation(invalid)
        self.assertTrue(np.allclose(projected, projected.T))
        self.assertTrue(np.allclose(np.diag(projected), 1.0))
        self.assertGreaterEqual(float(np.linalg.eigvalsh(projected).min()), -1e-9)

    def test_gaussian_copula_confidence_is_deterministic(self) -> None:
        kwargs = dict(
            marginals=[0.75, 0.75, 0.75],
            weights=[1.0, 1.0, 1.0],
            correlation=np.eye(3),
            samples=20_000,
            seed=12,
        )
        first = gaussian_copula_majority_confidence(**kwargs)
        second = gaussian_copula_majority_confidence(**kwargs)
        self.assertEqual(first, second)
        # Exact independent probability of at least two correct votes is 0.84375.
        self.assertAlmostEqual(first, 0.84375, delta=0.012)

    def test_positive_correlation_reduces_ensemble_gain(self) -> None:
        independent = gaussian_copula_majority_confidence(
            [0.75] * 3, [1.0] * 3, np.eye(3), samples=30_000, seed=7
        )
        correlated = gaussian_copula_majority_confidence(
            [0.75] * 3,
            [1.0] * 3,
            np.full((3, 3), 0.8) + np.eye(3) * 0.2,
            samples=30_000,
            seed=7,
        )
        self.assertLess(correlated, independent)

    def test_online_estimator_tracks_model_order_and_bounds(self) -> None:
        estimator = OnlineCorrelationEstimator(["b", "a"])
        for rewards in (
            {"a": 1.0, "b": 1.0},
            {"a": 0.0, "b": 0.0},
            {"a": 1.0, "b": 1.0},
            {"a": 0.0, "b": 0.0},
        ):
            estimator.update(rewards)
        matrix = estimator.matrix_for(("a", "b"))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.all(np.abs(matrix) <= 1.0))
        self.assertGreater(matrix[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
