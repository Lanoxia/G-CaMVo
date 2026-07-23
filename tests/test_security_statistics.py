import unittest

from camvo.security.statistics import paired_cluster_bootstrap_delta


class SecurityStatisticsTests(unittest.TestCase):
    def test_paired_cluster_bootstrap_detects_strictly_better_candidate(self) -> None:
        gold = ["a", "b"] * 20
        candidate = list(gold)
        reference = ["b", "a"] * 20
        clusters = [f"doc-{index // 2}" for index in range(40)]
        result = paired_cluster_bootstrap_delta(
            gold,
            candidate,
            reference,
            ("a", "b"),
            clusters,
            iterations=200,
            seed=3,
        )
        self.assertGreater(result.candidate_minus_reference, 0)
        self.assertGreater(result.lower, 0)
        self.assertEqual(result.probability_candidate_better, 1.0)


if __name__ == "__main__":
    unittest.main()
