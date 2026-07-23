from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from camvo.security.mordor import audit_mordor_ndjson


class MordorAuditTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        path = root / "mordor.json"
        records = [
            {
                "@timestamp": "2026-01-01T00:00:00Z",
                "Hostname": "HOST-A",
                "Channel": "Sysmon",
                "EventID": 1,
                "ProcessGuid": "p1",
                "ParentProcessGuid": "p0",
                "tags": ["mordorDataset"],
            },
            {
                "@timestamp": "2026-01-01T00:00:01Z",
                "Hostname": "HOST-B",
                "Channel": "Security",
                "EventID": 5156,
                "SourceAddress": "10.0.0.1",
                "DestAddress": "10.0.0.2",
                "is_malicious": True,
            },
        ]
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        return path

    def test_audits_schema_graph_fields_and_label_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = audit_mordor_ndjson([self._fixture(Path(temp))])
        self.assertEqual(report["records_total"], 2)
        self.assertEqual(report["hosts"], {"HOST-A": 1, "HOST-B": 1})
        self.assertEqual(report["graph_evidence"]["records_with_process_links"], 1)
        self.assertEqual(report["graph_evidence"]["records_with_network_links"], 1)
        self.assertEqual(report["label_audit"]["records_with_explicit_label_fields"], 1)
        self.assertFalse(report["label_audit"]["archive_membership_is_event_ground_truth"])

    def test_lenient_mode_counts_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("not-json\n")
            report = audit_mordor_ndjson([path], strict=False)
        self.assertEqual(report["records_total"], 2)
        self.assertEqual(report["parse_errors"], 1)

    def test_strict_mode_rejects_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("not-json\n")
            with self.assertRaisesRegex(ValueError, "invalid Mordor record"):
                audit_mordor_ndjson([path], strict=True)


if __name__ == "__main__":
    unittest.main()

