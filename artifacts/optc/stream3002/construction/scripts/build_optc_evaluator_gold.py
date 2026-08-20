#!/usr/bin/env python3
"""Build physically separated evaluator-only OpTC activity gold.

Unlike E2/E3, this E4 stage is explicitly allowed to read the official PDF and
the frozen exact-event label file.  Its output must never be used by candidate
generation or prompt construction.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def find_repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "camvo" / "security" / "optc.py").is_file():
            return parent
    raise RuntimeError("cannot locate G-CaMVo repository root")


ROOT = find_repository_root()
CONSTRUCTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camvo.security.optc import _parse_offset_timestamp, normalize_hostname  # noqa: E402


STAMP_RE = re.compile(
    r"(?m)^\s*(09/(?:23|24|25)/19\s+\d{2}:\d{2}:\d{2})\s+--\s+"
)
HOST_RE = re.compile(r"(?i)\b(?:sysc(?:lient|linet)|sysclient)\s*0*(\d{1,4})\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-db",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/stream_stage.sqlite",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CONSTRUCTION / "config/optc_core_cases_v1.json",
    )
    parser.add_argument(
        "--ground-truth-pdf",
        type=Path,
        default=ROOT / "data/raw/optc-metadata/OpTCRedTeamGroundTruth.pdf",
    )
    parser.add_argument(
        "--exact-labels",
        type=Path,
        default=ROOT / "data/raw/optc-labels/labels.csv",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/evaluator/gold.sqlite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/evaluator/e4_gold_report.json",
    )
    parser.add_argument("--step-alignment-seconds", type=int, default=180)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pypdf is required; run this script with the project .venv") from exc
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def description_hosts(description: str) -> list[str]:
    return sorted({f"SYSCLIENT{int(match):04d}" for match in HOST_RE.findall(description)})


def parse_official_steps(text: str, offset_minutes: int) -> list[dict[str, Any]]:
    matches = list(STAMP_RE.finditer(text))
    output: list[dict[str, Any]] = []
    offset = timezone.utc if offset_minutes == 0 else timezone(timedelta(minutes=offset_minutes))
    section_starts = {
        day: text.index(f"Day {number} -")
        for number, day in ((1, "day1"), (2, "day2"), (3, "day3"))
    }
    ordered_section_starts = sorted(section_starts.values())
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # A day's final timestamp is followed by the next day's prose summary
        # before the next timestamp.  Stop at that section heading so future-day
        # narrative cannot be attached to the preceding official step.
        next_section = next(
            (position for position in ordered_section_starts if position > match.start()),
            None,
        )
        if next_section is not None:
            end = min(end, next_section)
        stamp = match.group(1)
        description = " ".join(text[match.end():end].replace("\x0c", " ").split())
        # Strip page footer text if a wrapped activity crosses a PDF page.
        description = description.split("The views and conclusions contained", 1)[0].strip()
        day = max(
            (item for item in section_starts.items() if item[1] < match.start()),
            key=lambda item: item[1],
        )[0]
        moment = datetime.strptime(stamp, "%m/%d/%y %H:%M:%S").replace(tzinfo=offset)
        output.append(
            {
                "step_id": f"{day}-official-{sum(1 for item in output if item['day'] == day) + 1:02d}",
                "day": day,
                "timestamp_ms": int(moment.timestamp() * 1000),
                "timestamp_local": stamp,
                "description": description,
                "hosts": description_hosts(description),
                "official_order": index + 1,
            }
        )
    return output


def load_scope(path: Path) -> tuple[dict[str, set[str]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hosts = {
        str(case["day"]): {normalize_hostname(str(host)) for host in case["core_hosts"]}
        for case in payload["cases"]
    }
    return hosts, int(payload["reference_utc_offset_minutes"])


def initialize(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        DROP TABLE IF EXISTS official_steps;
        DROP TABLE IF EXISTS positive_events;
        DROP TABLE IF EXISTS step_positive_events;
        DROP TABLE IF EXISTS attack_path_edges;
        DROP TABLE IF EXISTS metadata;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE official_steps(
          step_id TEXT PRIMARY KEY,
          day TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          timestamp_local TEXT NOT NULL,
          description TEXT NOT NULL,
          hosts_json TEXT NOT NULL,
          official_order INTEGER NOT NULL,
          in_download_scope INTEGER NOT NULL,
          observable INTEGER NOT NULL DEFAULT 0,
          aligned_positive_events INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE positive_events(
          event_id TEXT PRIMARY KEY,
          day TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          hostname TEXT NOT NULL,
          object_id TEXT NOT NULL,
          actor_id TEXT NOT NULL,
          object_type TEXT NOT NULL,
          action TEXT NOT NULL
        );
        CREATE TABLE step_positive_events(
          step_id TEXT NOT NULL,
          event_id TEXT NOT NULL,
          delta_ms INTEGER NOT NULL,
          PRIMARY KEY(step_id,event_id)
        );
        CREATE TABLE attack_path_edges(
          edge_id TEXT PRIMARY KEY,
          source_step_id TEXT NOT NULL,
          relation TEXT NOT NULL,
          target_step_id TEXT NOT NULL
        );
        """
    )
    return db


def main() -> int:
    args = parse_args()
    for path in (args.stage_db, args.cases, args.ground_truth_pdf, args.exact_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.step_alignment_seconds < 1:
        raise ValueError("--step-alignment-seconds must be positive")

    core_hosts, offset_minutes = load_scope(args.cases)
    steps = parse_official_steps(extract_pdf_text(args.ground_truth_pdf), offset_minutes)
    if len(steps) != 101:
        raise RuntimeError(f"expected 101 timestamped official steps, parsed {len(steps)}")
    output = initialize(args.output_db)
    for step in steps:
        named = set(step["hosts"])
        in_scope = not named or bool(named & core_hosts[step["day"]])
        output.execute(
            "INSERT INTO official_steps VALUES(?,?,?,?,?,?,?,?,0,0)",
            (
                step["step_id"], step["day"], step["timestamp_ms"],
                step["timestamp_local"], step["description"],
                json.dumps(step["hosts"], separators=(",", ":")),
                step["official_order"], int(in_scope),
            ),
        )

    stage = sqlite3.connect(args.stage_db)
    stage.row_factory = sqlite3.Row
    with args.exact_labels.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = str(row["id"]).strip()
            if stage.execute(
                "SELECT 1 FROM events WHERE corpus='attack' AND event_id=?", (event_id,)
            ).fetchone() is None:
                continue
            timestamp = _parse_offset_timestamp(row["timestamp"])
            timestamp_ms = int(timestamp.timestamp() * 1000)
            day = {
                "2019-09-23": "day1", "2019-09-24": "day2", "2019-09-25": "day3"
            }.get(timestamp.date().isoformat())
            if day is None:
                continue
            hostname = normalize_hostname(str(row["hostname"]))
            if hostname not in core_hosts[day]:
                continue
            output.execute(
                "INSERT OR IGNORE INTO positive_events VALUES(?,?,?,?,?,?,?,?)",
                (
                    event_id, day, timestamp_ms, hostname, str(row["objectID"]),
                    str(row["actorID"]), str(row["object"]).upper(), str(row["action"]).upper(),
                ),
            )
    output.commit()

    steps_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        steps_by_day[step["day"]].append(step)
    max_delta = args.step_alignment_seconds * 1000
    aligned = 0
    for event in output.execute("SELECT * FROM positive_events ORDER BY day,timestamp_ms,event_id"):
        candidates = steps_by_day[str(event["day"])]
        times = [int(item["timestamp_ms"]) for item in candidates]
        position = bisect.bisect_left(times, int(event["timestamp_ms"]))
        nearby = candidates[max(0, position - 2):position + 3]
        host = str(event["hostname"])
        compatible = [item for item in nearby if not item["hosts"] or host in item["hosts"]]
        if not compatible:
            continue
        selected = min(compatible, key=lambda item: abs(int(item["timestamp_ms"]) - int(event["timestamp_ms"])))
        delta = int(event["timestamp_ms"]) - int(selected["timestamp_ms"])
        if abs(delta) > max_delta:
            continue
        output.execute(
            "INSERT OR IGNORE INTO step_positive_events VALUES(?,?,?)",
            (selected["step_id"], event["event_id"], delta),
        )
        aligned += 1
    output.execute(
        """
        UPDATE official_steps
        SET aligned_positive_events=(
          SELECT COUNT(*) FROM step_positive_events s WHERE s.step_id=official_steps.step_id
        ), observable=CASE WHEN EXISTS(
          SELECT 1 FROM step_positive_events s WHERE s.step_id=official_steps.step_id
        ) THEN 1 ELSE 0 END
        """
    )

    for day, day_steps in steps_by_day.items():
        observable = [
            row for row in output.execute(
                "SELECT step_id,hosts_json FROM official_steps WHERE day=? AND observable=1 ORDER BY timestamp_ms",
                (day,),
            )
        ]
        for index, (left, right) in enumerate(zip(observable, observable[1:]), 1):
            left_hosts = set(json.loads(str(left["hosts_json"])))
            right_hosts = set(json.loads(str(right["hosts_json"])))
            relation = "precedes_same_host" if left_hosts & right_hosts else "precedes_activity"
            output.execute(
                "INSERT INTO attack_path_edges VALUES(?,?,?,?)",
                (f"{day}-path-{index:03d}", left["step_id"], relation, right["step_id"]),
            )

    hashes = {
        "ground_truth_pdf_sha256": sha256(args.ground_truth_pdf),
        "exact_labels_sha256": sha256(args.exact_labels),
        "cases_sha256": sha256(args.cases),
        "stage_signature": str(
            stage.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()[0]
        ),
    }
    for key, value in hashes.items():
        output.execute("INSERT INTO metadata VALUES(?,?)", (key, value))
    output.execute("INSERT INTO metadata VALUES('evaluator_only','true')")
    output.commit()

    step_counts = {
        str(row[0]): {"steps": int(row[1]), "in_scope": int(row[2]), "observable": int(row[3])}
        for row in output.execute(
            "SELECT day,COUNT(*),SUM(in_download_scope),SUM(observable) FROM official_steps GROUP BY day"
        )
    }
    type_counts = Counter(
        {str(row[0]): int(row[1]) for row in output.execute(
            "SELECT object_type,COUNT(*) FROM positive_events GROUP BY object_type"
        )}
    )
    report = {
        "schema_version": 1,
        "state": "complete",
        "evaluator_only": True,
        "official_steps": len(steps),
        "step_counts": step_counts,
        "exact_positive_events_in_frozen_stage": int(
            output.execute("SELECT COUNT(*) FROM positive_events").fetchone()[0]
        ),
        "positive_events_aligned_to_official_steps": int(
            output.execute("SELECT COUNT(*) FROM step_positive_events").fetchone()[0]
        ),
        "positive_event_object_types": dict(type_counts.most_common()),
        "attack_path_edges": int(output.execute("SELECT COUNT(*) FROM attack_path_edges").fetchone()[0]),
        "step_alignment_seconds": args.step_alignment_seconds,
        "hashes": hashes,
        "output_db": str(args.output_db.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    output.close()
    stage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
