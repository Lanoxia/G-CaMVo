import gzip
import json
import tempfile
import unittest
from pathlib import Path

from camvo.security.optc import (
    OptcEvent,
    build_optc_event_correlations,
    build_optc_provenance_graph,
    load_optc_attack_labels,
    load_optc_events,
    load_optc_scenarios,
    sample_optc_events_by_hash,
)
from camvo.security.optc_dataset import build_optc_real_binary_dataset


def _record(
    event_id: str,
    timestamp: int,
    *,
    hostname: str = "Sysclient0051",
    actor_id: str = "process-parent",
    object_id: str = "process-child",
    object_type: str = "PROCESS",
    action: str = "CREATE",
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "id": event_id,
        "hostname": hostname,
        "objectID": object_id,
        "object": object_type,
        "action": action,
        "actorID": actor_id,
        "pid": 42,
        "ppid": -1,
        "tid": 7,
        "principal": "SYSTEMIA\\analyst",
        "properties": {"image_path": "C:\\Windows\\example.exe"},
    }


class OptcLoaderTests(unittest.TestCase):
    def test_event_accepts_official_offset_timestamp(self) -> None:
        event = OptcEvent.from_mapping(
            {
                "timestamp": "2019-09-25T09:04:41.981-04:00",
                "id": "event-iso",
                "hostname": "SysClient0051.systemia.com",
                "objectID": "object-iso",
                "object": "THREAD",
                "action": "CREATE",
                "actorID": "actor-iso",
            }
        )
        self.assertEqual(event.timestamp_ms, 1_569_416_681_981)
        self.assertEqual(event.hostname, "SYSCLIENT0051")

    def test_loads_positive_labels_and_normalizes_domain_suffix(self) -> None:
        csv_text = (
            "hostname,id,objectID,actorID,timestamp,object,action\n"
            "SysClient0051.systemia.com,e1,obj1,actor1,"
            "2019-09-25T10:29:42-04:00,PROCESS,CREATE\n"
            "SysClient0351.systemia.com,e2,obj2,actor2,"
            "2019-09-25T11:23:31-04:00,FLOW,START\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(csv_text, encoding="utf-8")
            index = load_optc_attack_labels(path, hostnames=["SYSCLIENT0051"], strict=True)

        self.assertEqual(len(index.labels), 1)
        self.assertEqual(index.labels[0].hostname, "SYSCLIENT0051")
        self.assertTrue(index.contains("e1"))
        self.assertFalse(index.contains("e2"))
        self.assertEqual(index.labels[0].to_minimal_event().event_id, "e1")

    def test_loads_utc_z_labels_consistently_with_python_310(self) -> None:
        csv_text = (
            "hostname,id,objectID,actorID,timestamp,object,action\n"
            "SYSCLIENT0051,e1,obj1,actor1,2019-09-25T14:29:42Z,PROCESS,CREATE\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(csv_text, encoding="utf-8")
            index = load_optc_attack_labels(path, strict=True)

        self.assertEqual(len(index.labels), 1)
        self.assertEqual(index.labels[0].timestamp_ms, 1_569_421_782_000)

    def test_loads_short_fractional_seconds_consistently_with_python_310(self) -> None:
        csv_text = (
            "hostname,id,objectID,actorID,timestamp,object,action\n"
            "SYSCLIENT0051,e1,obj1,actor1,"
            "2019-09-25T10:30:00.59-04:00,PROCESS,CREATE\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(csv_text, encoding="utf-8")
            index = load_optc_attack_labels(path, strict=True)

        self.assertEqual(len(index.labels), 1)
        self.assertEqual(index.labels[0].timestamp_ms, 1_569_421_800_590)

    def test_streams_jsonl_filters_and_deduplicates(self) -> None:
        records = [
            json.dumps(_record("e1", 1_569_405_000_000)),
            "not-json",
            json.dumps(_record("e1", 1_569_405_000_100)),
            json.dumps(_record("e2", 1_569_405_000_200, hostname="other-host")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("\n".join(records), encoding="utf-8")
            dataset = load_optc_events(directory, hostnames=["SYSCLIENT 0051"])

        self.assertEqual([event.event_id for event in dataset.events], ["e1"])
        self.assertEqual(dataset.events[0].hostname, "SYSCLIENT0051")
        self.assertEqual(dataset.events[0].ppid, None)
        self.assertEqual(dataset.stats.records_read, 4)
        self.assertEqual(dataset.stats.records_skipped, 1)
        self.assertEqual(dataset.stats.duplicates_skipped, 1)
        self.assertEqual(dataset.stats.records_filtered, 1)

    def test_reads_gzip_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump([_record("e1", 1000), _record("e2", 2000)], handle)
            dataset = load_optc_events(path)

        self.assertEqual(len(dataset.events), 2)
        self.assertEqual(dataset.stats.files_failed, 0)

    def test_bottom_k_sample_scans_full_stream_and_is_reproducible(self) -> None:
        records = [json.dumps(_record(f"e{index}", 1_000 + index)) for index in range(50)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("\n".join(records), encoding="utf-8")
            first = sample_optc_events_by_hash(path, sample_size=7, seed=5)
            second = sample_optc_events_by_hash(path, sample_size=7, seed=5)

        self.assertEqual(first.events, second.events)
        self.assertEqual(len(first.events), 7)
        self.assertEqual(first.stats.records_read, 50)
        self.assertEqual(first.stats.valid_candidates, 50)

    def test_builds_provenance_and_sparse_cross_role_correlation(self) -> None:
        first = OptcEvent.from_mapping(_record("e1", 1000, object_id="shared-process"))
        second = OptcEvent.from_mapping(
            _record("e2", 2000, actor_id="shared-process", object_id="target-file")
        )
        graph = build_optc_provenance_graph([first, second])
        neighborhood = graph.neighborhood(["event:e1"], hops=2)
        correlations = build_optc_event_correlations([first, second], max_time_gap_ms=10_000)

        self.assertIn("event:e2", neighborhood.nodes)
        self.assertEqual(len(correlations), 1)
        self.assertIn("actor", correlations[0].reasons)
        self.assertIn("object", correlations[0].reasons)
        item = first.to_annotation_item()
        self.assertEqual(item.metadata["event_node_id"], "event:e1")
        self.assertIn("untrusted evidence", item.text)

    def test_real_binary_builder_never_treats_unlabeled_attack_rows_as_benign(self) -> None:
        attack_records = [
            json.dumps(_record("labeled", 1_569_421_800_000)),
            json.dumps(_record("unlabeled", 1_569_421_801_000)),
        ]
        benign_records = [json.dumps(_record("benign-control", 1_560_000_000_000))]
        labels = (
            "hostname,id,objectID,actorID,timestamp,object,action\n"
            "SYSCLIENT0051,labeled,process-child,process-parent,"
            "2019-09-25T10:30:00-04:00,PROCESS,CREATE\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attack_path = root / "attack.jsonl"
            benign_path = root / "benign.jsonl"
            label_path = root / "labels.csv"
            attack_path.write_text("\n".join(attack_records), encoding="utf-8")
            benign_path.write_text("\n".join(benign_records), encoding="utf-8")
            label_path.write_text(labels, encoding="utf-8")
            dataset = build_optc_real_binary_dataset(
                attack_path,
                benign_path,
                label_path,
                "config/optc_scenarios.json",
                max_positive_items=10,
            )

        item_ids = {item.item_id for item in dataset.items}
        self.assertEqual(dataset.stats["positive_items"], 1)
        self.assertEqual(dataset.stats["negative_items"], 1)
        self.assertIn("optc:labeled", item_ids)
        self.assertIn("optc:benign-control", item_ids)
        self.assertNotIn("optc:unlabeled", item_ids)
        self.assertFalse(dataset.stats["unlabeled_attack_rows_treated_as_benign"])

    def test_loads_curated_day3_manifest_without_assuming_timezone(self) -> None:
        scenarios = load_optc_scenarios("config/optc_scenarios.json")
        scenario = scenarios["optc-day3-malicious-upgrade"]

        self.assertEqual(len(scenario.activities), 16)
        self.assertEqual(scenario.hosts, ("SYSCLIENT0051", "SYSCLIENT0351"))
        self.assertEqual(scenario.reference_utc_offset_minutes, -240)
        start, end = scenario.epoch_window(utc_offset_minutes=0, padding_minutes=5)
        self.assertLess(start, end)


if __name__ == "__main__":
    unittest.main()
