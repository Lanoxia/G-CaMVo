import unittest

from camvo.algorithm.graph_regularization import WeightedGraphEdge, laplacian_smooth_scores


class GraphRegularizationTests(unittest.TestCase):
    def test_connected_scores_move_toward_each_other(self) -> None:
        result = laplacian_smooth_scores(
            {"early-alert": 0.95, "related-alert": 0.20, "isolated": 0.40},
            [WeightedGraphEdge("early-alert", "related-alert", 2.0)],
            regularization=1.5,
        )

        self.assertLess(result.smoothed_scores["early-alert"], 0.95)
        self.assertGreater(result.smoothed_scores["related-alert"], 0.20)
        self.assertAlmostEqual(result.smoothed_scores["isolated"], 0.40)
        self.assertLessEqual(result.objective_after, result.objective_before)

    def test_zero_regularization_is_identity(self) -> None:
        result = laplacian_smooth_scores(
            {"a": 0.1, "b": 0.9},
            [WeightedGraphEdge("a", "b")],
            regularization=0.0,
        )
        self.assertAlmostEqual(result.smoothed_scores["a"], 0.1)
        self.assertAlmostEqual(result.smoothed_scores["b"], 0.9)

    def test_sparse_cg_matches_direct_solver(self) -> None:
        scores = {f"n{index}": (index % 3) / 2 for index in range(40)}
        edges = [
            WeightedGraphEdge(f"n{index}", f"n{index + 1}", 0.5 + index / 100)
            for index in range(39)
        ]
        direct = laplacian_smooth_scores(scores, edges, regularization=0.8, solver="direct")
        sparse = laplacian_smooth_scores(scores, edges, regularization=0.8, solver="cg")

        self.assertEqual(sparse.solver, "cg")
        self.assertGreater(sparse.iterations, 0)
        for node_id in scores:
            self.assertAlmostEqual(
                sparse.smoothed_scores[node_id],
                direct.smoothed_scores[node_id],
                places=7,
            )

    def test_rejects_unknown_edge_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            laplacian_smooth_scores(
                {"a": 0.5},
                [WeightedGraphEdge("a", "missing")],
            )


if __name__ == "__main__":
    unittest.main()
