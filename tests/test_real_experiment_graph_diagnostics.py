import unittest

from camvo.security.graph_diagnostics import graph_diagnostics
from camvo.types import AnnotationItem


class RealExperimentGraphDiagnosticsTests(unittest.TestCase):
    def test_reports_isolates_components_and_undirected_edges(self) -> None:
        items = [
            AnnotationItem(f"n{index}", "event", ("A", "B"))
            for index in range(4)
        ]
        adjacency = {
            "n0": {"n1": 1.0},
            "n1": {"n0": 1.0, "n2": 0.5},
            "n2": {"n1": 0.5},
            "outside": {"n0": 1.0},
        }

        stats = graph_diagnostics(items, adjacency)

        self.assertEqual(stats["graph_nodes"], 4)
        self.assertEqual(stats["graph_edges"], 2)
        self.assertEqual(stats["graph_nonisolated_items"], 3)
        self.assertEqual(stats["graph_isolated_items"], 1)
        self.assertEqual(stats["graph_components"], 2)
        self.assertEqual(stats["graph_largest_component"], 3)
        self.assertAlmostEqual(stats["graph_density"], 2 / 6)


if __name__ == "__main__":
    unittest.main()
