import json
import tempfile
import unittest
from pathlib import Path

from camvo.security.operational_metrics import optc_operational_metrics
from camvo.security.optc_readiness import build_optc_readiness_report
from camvo.types import AnnotationItem


def _record(event_id: str, timestamp: int, hostname: str = "SYSCLIENT0051") -> dict:
    return {
        "timestamp": timestamp,
        "id": event_id,
        "hostname": hostname,
        "objectID": f"object-{event_id}",
        "object": "PROCESS",
        "action": "CREATE",
        "actorID": f"actor-{event_id}",
        "properties": {},
    }


class OptcReadinessTests(unittest.TestCase):
    def test_blocks_before_provider_stage_when_raw_paths_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_optc_readiness_report(
                root / "attack",
                root / "benign",
                root / "labels.csv",
                "config/optc_scenarios.json",
                minimum_free_bytes=0,
            )

        self.assertFalse(report["ready"])
        self.assertEqual(report["provider_calls_made"], 0)
        self.assertEqual(report["status"], "blocked_external_data")
        self.assertGreaterEqual(len(report["blockers"]), 3)

    def test_accepts_schema_valid_bounded_attack_and_benign_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attack = root / "attack"
            benign = root / "benign"
            attack.mkdir()
            benign.mkdir()
            timestamp = 1_569_421_800_000
            (attack / "attack.jsonl").write_text(
                json.dumps(_record("attack-1", timestamp)) + "\n",
                encoding="utf-8",
            )
            (benign / "benign.jsonl").write_text(
                json.dumps(_record("benign-1", 1_560_000_000_000)) + "\n",
                encoding="utf-8",
            )
            labels = root / "labels.csv"
            labels.write_text(
                "hostname,id,objectID,actorID,timestamp,object,action\n"
                "SYSCLIENT0051,attack-1,object-attack-1,actor-attack-1,"
                "2019-09-25T10:30:00-04:00,PROCESS,CREATE\n",
                encoding="utf-8",
            )
            report = build_optc_readiness_report(
                attack,
                benign,
                labels,
                "config/optc_scenarios.json",
                sample_events=1,
                minimum_free_bytes=0,
            )

        self.assertTrue(report["ready"])
        self.assertEqual(report["provider_calls_made"], 0)
        self.assertEqual(report["attack"]["probe"]["events_loaded"], 1)
        self.assertEqual(report["benign"]["probe"]["events_loaded"], 1)
        self.assertGreater(report["labels"]["audit"]["labels_loaded"], 0)


class OperationalMetricsTests(unittest.TestCase):
    def _item(self, name: str, gold: str, timestamp: int) -> AnnotationItem:
        return AnnotationItem(
            item_id=name,
            text=name,
            labels=("benign", "malicious"),
            metadata={
                "gold_label": gold,
                "hostname": "host-a",
                "timestamp_ms": timestamp,
                "graph_node_id": name,
            },
        )

    def test_reports_host_delay_false_alerts_and_edge_proxy(self) -> None:
        items = (
            self._item("m1", "malicious", 1_000),
            self._item("m2", "malicious", 2_000),
            self._item("b1", "benign", 3_000),
            self._item("b2", "benign", 4_000),
        )
        report = optc_operational_metrics(
            items,
            ("malicious", "benign", "malicious", "benign"),
            (False, False, False, False),
            {"m1": {"m2": 1.0}, "m2": {"m1": 1.0}},
        )

        self.assertEqual(report["host_detection_rate"], 1.0)
        self.assertEqual(report["time_to_first_true_positive_seconds"]["median"], 0.0)
        self.assertEqual(report["false_alerts"], 1)
        self.assertEqual(report["observed_benign_host_hours"], 0.5)
        self.assertEqual(report["false_alerts_per_benign_host_hour"], 2.0)
        self.assertEqual(report["malicious_correlation_subgraph_proxy"]["gold_edges"], 1)
        self.assertEqual(report["malicious_correlation_subgraph_proxy"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()

