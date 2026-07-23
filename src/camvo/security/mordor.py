"""Streaming audit utilities for OTRF Security Datasets / Mordor NDJSON logs.

Mordor compound archives contain real attack-emulation telemetry mixed with
background activity.  They are scenario-labelled, not necessarily labelled at
the individual-record level.  This module deliberately reports that distinction
instead of inferring malicious labels from membership in an attack archive.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


_EXPLICIT_LABEL_KEYS = {
    "attack_label",
    "is_malicious",
    "label",
    "malicious",
    "mitre_attack_id",
    "technique_id",
}
_PROCESS_LINK_KEYS = (
    "ProcessGuid",
    "ParentProcessGuid",
    "SourceProcessGUID",
    "TargetProcessGUID",
)
_NETWORK_LINK_KEYS = (
    "SourceAddress",
    "DestinationAddress",
    "DestAddress",
    "SourceIp",
    "DestinationIp",
)


def _first_text(record: dict[str, object], names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "<missing>"


def audit_mordor_ndjson(
    paths: Iterable[str | Path],
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Audit one or more Mordor newline-delimited JSON files without loading them all.

    ``strict=True`` raises on the first malformed or non-object row.  The report
    records only aggregate metadata and never treats archive membership as a
    malicious event label.
    """

    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("at least one Mordor NDJSON path is required")

    hosts: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    event_ids: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    files: list[dict[str, object]] = []
    records_total = 0
    parse_errors = 0
    blank_lines = 0
    explicit_label_records = 0
    process_link_records = 0
    network_link_records = 0
    earliest_timestamp: str | None = None
    latest_timestamp: str | None = None

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Mordor NDJSON file not found: {path}")
        file_records = 0
        file_errors = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    blank_lines += 1
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    parse_errors += 1
                    file_errors += 1
                    if strict:
                        raise ValueError(
                            f"invalid Mordor record at {path}:{line_number}: {exc}"
                        ) from exc
                    continue

                file_records += 1
                records_total += 1
                host = _first_text(record, ("Hostname", "Computer", "host"))
                channel = _first_text(record, ("Channel", "SourceName"))
                event_id = _first_text(record, ("EventID", "EventId", "event_id"))
                hosts[host] += 1
                channels[channel] += 1
                event_ids[event_id] += 1

                raw_tags = record.get("tags", [])
                if isinstance(raw_tags, list):
                    tags.update(str(value) for value in raw_tags)
                elif raw_tags is not None:
                    tags[str(raw_tags)] += 1

                normalized_keys = {key.casefold() for key in record}
                if normalized_keys.intersection(_EXPLICIT_LABEL_KEYS):
                    explicit_label_records += 1
                if sum(bool(record.get(key)) for key in _PROCESS_LINK_KEYS) >= 2:
                    process_link_records += 1
                if sum(bool(record.get(key)) for key in _NETWORK_LINK_KEYS) >= 2:
                    network_link_records += 1

                timestamp = _first_text(
                    record,
                    ("@timestamp", "TimeCreated", "UtcTime", "EventTime"),
                )
                if timestamp != "<missing>":
                    if earliest_timestamp is None or timestamp < earliest_timestamp:
                        earliest_timestamp = timestamp
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp

        files.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "records": file_records,
                "parse_errors": file_errors,
            }
        )

    return {
        "dataset": "OTRF Security Datasets / Mordor",
        "format": "newline-delimited JSON",
        "files": files,
        "records_total": records_total,
        "parse_errors": parse_errors,
        "blank_lines": blank_lines,
        "time_range": [earliest_timestamp, latest_timestamp],
        "hosts": dict(hosts.most_common()),
        "channels": dict(channels.most_common()),
        "event_ids": dict(event_ids.most_common()),
        "tags": dict(tags.most_common()),
        "graph_evidence": {
            "records_with_process_links": process_link_records,
            "records_with_network_links": network_link_records,
        },
        "label_audit": {
            "records_with_explicit_label_fields": explicit_label_records,
            "archive_membership_is_event_ground_truth": False,
            "warning": (
                "Mordor compound archives mix attack telemetry with background events; "
                "use timestamp/rule-derived ground truth or a separately labelled benchmark."
            ),
        },
    }

