import unittest

from camvo.algorithm.confidence import beta_cdf_confidence, exact_majority_confidence
from camvo.mathutils import regularized_beta_cdf


class ConfidenceTests(unittest.TestCase):
    def test_lemma_2_1_three_equal_weight_models(self) -> None:
        confidence = exact_majority_confidence([0.8, 0.7, 0.6], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(confidence, 0.788, places=12)

    def test_beta_cdf_is_finite_probability(self) -> None:
        confidence = beta_cdf_confidence([0.8, 0.7, 0.6], [1.0, 1.0, 1.0])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_regularized_beta_cdf_symmetry(self) -> None:
        self.assertAlmostEqual(regularized_beta_cdf(0.5, 2.0, 2.0), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()

