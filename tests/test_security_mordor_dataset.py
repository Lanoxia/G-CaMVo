from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from camvo.security.graph_diagnostics import graph_diagnostics
from camvo.security.mordor_dataset import build_mordor_cdb_binary_dataset
from camvo.security.simulated_experiments import optc_adjacency


class MordorDatasetTests(unittest.TestCase):
    def test_builds_balanced_exact_flag_task_and_process_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "sample.json"
            flags_path = root / "sample_flags.json"
            logs = [
                {
                    "TimeCreated": "2026-01-01T00:00:00Z",
                    "Hostname": "HOST-A",
                    "EventID": 1,
                    "ProcessGuid": "p1",
                    "CommandLine": "evil.exe",
                },
                {
                    "TimeCreated": "2026-01-01T00:00:01Z",
                    "Hostname": "HOST-A",
                    "EventID": 1,
                    "ParentProcessGuid": "p1",
                    "Image": "child.exe",
                },
                {
                    "TimeCreated": "2026-01-01T00:00:02Z",
                    "Hostname": "HOST-B",
                    "EventID": 2,
                    "Image": "routine.exe",
                },
            ]
            flags = {
                "tactic_ids": ["TA0002"],
                "chains": [
                    {"chain_idx": 0, "steps": [{"step_idx": 1, "tactics": ["TA0002"]}]}
                ],
                "flags": [
                    {
                        "value": "2026-01-01T00:00:00Z",
                        "chain_idx": 0,
                        "step_idx": 1,
                        "narrative_steps": [1],
                        "relevance": 2,
                    }
                ],
            }
            data_path.write_text(json.dumps({"logs": logs}), encoding="utf-8")
            flags_path.write_text(json.dumps(flags), encoding="utf-8")
            built = build_mordor_cdb_binary_dataset(data_path, flags_path, seed=3)

        self.assertEqual(len(built.items), 2)
        self.assertEqual(
            [item.metadata["gold_label"] for item in built.items],
            ["benign", "malicious"],
        )
        malicious = built.items[1]
        self.assertEqual(malicious.metadata["tactics"], ["TA0002"])
        self.assertNotIn("gold_label", malicious.text)
        self.assertFalse(built.stats["archive_membership_used_as_label"])
        self.assertEqual(len(built.correlations), 1)
        item_ids = {item.item_id for item in built.items}
        edge = built.correlations[0]
        self.assertIn(edge.source_event_id, item_ids)
        self.assertIn(edge.target_event_id, item_ids)
        diagnostics = graph_diagnostics(
            built.items,
            optc_adjacency(built.correlations),
        )
        self.assertEqual(diagnostics["graph_edges"], 1)
        self.assertEqual(diagnostics["graph_nonisolated_items"], 2)

    def test_rejects_unjoined_flag_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "sample.json"
            flags_path = root / "sample_flags.json"
            data_path.write_text(
                json.dumps(
                    {
                        "logs": [
                            {"TimeCreated": "2026-01-01T00:00:00Z", "EventID": 1}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            flags_path.write_text(
                json.dumps(
                    {
                        "flags": [
                            {
                                "value": "2026-01-02T00:00:00Z",
                                "chain_idx": 0,
                                "step_idx": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not join"):
                build_mordor_cdb_binary_dataset(data_path, flags_path)


if __name__ == "__main__":
    unittest.main()
