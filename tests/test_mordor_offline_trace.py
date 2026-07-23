from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_mordor_trace_algorithms import (  # noqa: E402
    causal_parent_logit_score,
    directed_parent_adjacency,
)
from camvo.types import AnnotationItem  # noqa: E402


class MordorOfflineTraceTests(unittest.TestCase):
    def test_direction_comes_from_timestamp_not_hashed_id_order(self) -> None:
        early = AnnotationItem(
            "z-hash",
            "early event",
            ("benign", "malicious"),
            {"timestamp_ms": 10, "gold_label": "benign"},
        )
        late = AnnotationItem(
            "a-hash",
            "late event",
            ("benign", "malicious"),
            {"timestamp_ms": 20, "gold_label": "malicious"},
        )
        edge = SimpleNamespace(
            source_event_id="a-hash",
            target_event_id="z-hash",
            weight=0.8,
            reasons=("process",),
        )
        parents = directed_parent_adjacency(
            [edge], {early.item_id: early, late.item_id: late}
        )
        self.assertEqual(parents, {"a-hash": {"z-hash": 0.8}})

    def test_causal_score_reads_parent_but_not_future_child(self) -> None:
        item_ids = ["parent", "child", "future"]
        probabilities = np.asarray([[0.8], [0.2], [0.99]])
        partition = (
            item_ids,
            probabilities,
            np.zeros((3, 1)),
            np.asarray([1, 0, 1]),
        )
        parents = {"child": {"parent": 1.0}, "future": {"child": 1.0}}
        score, degree = causal_parent_logit_score(
            partition, parents, base_index=0, beta=0.25
        )
        changed = probabilities.copy()
        changed[2, 0] = 0.01
        changed_score, _ = causal_parent_logit_score(
            (item_ids, changed, partition[2], partition[3]),
            parents,
            base_index=0,
            beta=0.25,
        )
        self.assertAlmostEqual(score[1], changed_score[1])
        self.assertGreater(degree[1], 0)

    def test_zero_graph_strength_reduces_to_node_logit(self) -> None:
        probabilities = np.asarray([[0.2], [0.8]])
        partition = (
            ["a", "b"],
            probabilities,
            np.zeros((2, 1)),
            np.asarray([0, 1]),
        )
        score, _ = causal_parent_logit_score(
            partition, {"b": {"a": 1.0}}, base_index=0, beta=0.0
        )
        expected = np.log(probabilities[:, 0] / (1.0 - probabilities[:, 0]))
        np.testing.assert_allclose(score, expected)


if __name__ == "__main__":
    unittest.main()
