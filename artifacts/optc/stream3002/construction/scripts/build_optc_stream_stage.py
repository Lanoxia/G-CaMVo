#!/usr/bin/env python3
"""Build the label-blind, time-ordered OpTC core event stage.

This stage reads only telemetry and the frozen acquisition/case manifests.
Ground-truth labels are intentionally not accepted as an argument.  Each raw
file is checkpointed independently so interrupted multi-gigabyte scans resume
without discarding completed work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


def find_repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "camvo" / "security" / "optc.py").is_file():
            return parent
    raise RuntimeError("cannot locate G-CaMVo repository root")


ROOT = find_repository_root()
CONSTRUCTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camvo.security.optc import OptcEvent, _stream_records, normalize_hostname  # noqa: E402


PROPERTY_ALLOWLIST = frozenset(
    {
        "acuity_level", "command_line", "dest_ip", "dest_port", "direction",
        "file_path", "image_path", "key", "l4protocol", "module_path", "path",
        "reg_key", "reg_value", "src_ip", "src_port", "target_path", "value",
    }
)
STAGE_SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CONSTRUCTION / "config/optc_core_acquisition_v1.tsv",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CONSTRUCTION / "config/optc_core_cases_v1.json",
    )
    parser.add_argument(
        "--stage-db",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/stream_stage.sqlite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/e2_parser_report.json",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def local_epoch_ms(text: str, offset_minutes: int) -> int:
    offset = timezone(timedelta(minutes=offset_minutes))
    return int(datetime.fromisoformat(text).replace(tzinfo=offset).timestamp() * 1000)


def load_scope(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    offset = int(payload["reference_utc_offset_minutes"])
    cases: dict[str, dict[str, Any]] = {}
    for item in payload["cases"]:
        cases[str(item["day"])] = {
            "case_id": str(item["case_id"]),
            "hosts": {normalize_hostname(value) for value in item["core_hosts"]},
            "start_ms": local_epoch_ms(str(item["start_local"]), offset),
            "end_ms": local_epoch_ms(str(item["end_local"]), offset),
        }
    controls = {
        str(bucket): normalize_hostname(str(host))
        for bucket, host in payload["benign_host_by_bucket"].items()
    }
    return cases, controls


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("empty acquisition manifest")
    return rows


def signature(manifest: Path, cases: Path, rows: Iterable[dict[str, str]]) -> str:
    inventory = []
    for row in rows:
        source = ROOT / row["local_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        inventory.append(
            {
                "path": row["local_path"],
                "expected": int(row["size_bytes"]),
                "actual": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            }
        )
    payload = {
        "schema": STAGE_SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        "inventory": inventory,
        "property_allowlist": sorted(PROPERTY_ALLOWLIST),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          event_id TEXT NOT NULL,
          corpus TEXT NOT NULL,
          day TEXT NOT NULL,
          case_id TEXT,
          timestamp_ms INTEGER NOT NULL,
          hostname TEXT NOT NULL,
          object_id TEXT NOT NULL,
          object_type TEXT NOT NULL,
          action TEXT NOT NULL,
          actor_id TEXT NOT NULL,
          pid INTEGER,
          ppid INTEGER,
          principal TEXT,
          properties_json TEXT NOT NULL,
          source_path TEXT NOT NULL,
          source_line INTEGER NOT NULL,
          PRIMARY KEY(corpus,event_id)
        );
        CREATE TABLE IF NOT EXISTS file_audit(
          source_path TEXT PRIMARY KEY,
          stats_json TEXT NOT NULL,
          complete INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bad_record_samples(
          source_path TEXT NOT NULL,
          source_line INTEGER NOT NULL,
          error TEXT NOT NULL,
          PRIMARY KEY(source_path, source_line)
        );
        """
    )
    return db


def safe_properties(event: OptcEvent) -> str:
    output: dict[str, object] = {}
    for key in sorted(PROPERTY_ALLOWLIST.intersection(event.properties)):
        value = event.properties[key]
        if value is None or isinstance(value, (bool, int, float)):
            output[key] = value
        elif isinstance(value, str):
            output[key] = value[:500]
        else:
            output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:500]
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insert_file(
    db: sqlite3.Connection,
    row: dict[str, str],
    cases: dict[str, dict[str, Any]],
    controls: dict[str, str],
    progress_every: int,
) -> dict[str, object]:
    source = (ROOT / row["local_path"]).resolve()
    source_key = str(source)
    prior = db.execute(
        "SELECT stats_json FROM file_audit WHERE source_path=? AND complete=1",
        (source_key,),
    ).fetchone()
    if prior is not None:
        return json.loads(str(prior["stats_json"]))

    db.execute("DELETE FROM events WHERE source_path=?", (source_key,))
    db.execute("DELETE FROM bad_record_samples WHERE source_path=?", (source_key,))
    db.execute("DELETE FROM file_audit WHERE source_path=?", (source_key,))
    db.commit()

    corpus = str(row["class"])
    day = str(row["day"])
    if corpus == "attack":
        scope = cases[day]
        allowed_hosts = scope["hosts"]
        start_ms, end_ms = int(scope["start_ms"]), int(scope["end_ms"])
        case_id = str(scope["case_id"])
    else:
        allowed_hosts = {controls[str(row["host_bucket"])]}
        matching_days = [
            candidate_day
            for candidate_day, scope in cases.items()
            if allowed_hosts <= scope["hosts"]
        ]
        if len(matching_days) != 1:
            raise RuntimeError(
                "benign control bucket must map to exactly one attack-day split: "
                f"bucket={row['host_bucket']} hosts={sorted(allowed_hosts)} "
                f"matching_days={matching_days}"
            )
        # The acquisition manifest uses class=benign/day=control to distinguish
        # the source collection.  For chronological evaluation, however, each
        # matched control host must inherit the unique attack-day split of its
        # corresponding core host.  Corpus remains benign, so this does not
        # expose labels or attack metadata to the prompt.
        day = matching_days[0]
        start_ms = end_ms = None
        case_id = None

    statement = """
      INSERT OR IGNORE INTO events(
        event_id,corpus,day,case_id,timestamp_ms,hostname,object_id,object_type,
        action,actor_id,pid,ppid,principal,properties_json,source_path,source_line
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    batch: list[tuple[object, ...]] = []
    read = invalid = filtered = duplicates = inserted = 0
    timestamp_reversals = 0
    previous_timestamp: int | None = None
    type_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for raw in _stream_records(source):
        read += 1
        if raw.error is not None:
            invalid += 1
            if invalid <= 100:
                db.execute(
                    "INSERT OR IGNORE INTO bad_record_samples VALUES(?,?,?)",
                    (source_key, raw.line, str(raw.error)[:1000]),
                )
            continue
        try:
            event = OptcEvent.from_mapping(
                raw.payload or {}, source_path=source_key, source_line=raw.line
            )
        except (TypeError, ValueError) as exc:
            invalid += 1
            if invalid <= 100:
                db.execute(
                    "INSERT OR IGNORE INTO bad_record_samples VALUES(?,?,?)",
                    (source_key, raw.line, str(exc)[:1000]),
                )
            continue
        if previous_timestamp is not None and event.timestamp_ms < previous_timestamp:
            timestamp_reversals += 1
        previous_timestamp = event.timestamp_ms
        if event.hostname not in allowed_hosts:
            filtered += 1
            continue
        if start_ms is not None and not (start_ms <= event.timestamp_ms <= end_ms):
            filtered += 1
            continue
        batch.append(
            (
                event.event_id, corpus, day, case_id, event.timestamp_ms,
                event.hostname, event.object_id, event.object_type, event.action,
                event.actor_id, event.pid, event.ppid, event.principal,
                safe_properties(event), source_key, event.source_line,
            )
        )
        type_counts[event.object_type] += 1
        action_counts[event.action] += 1
        if len(batch) >= 2000:
            before = db.total_changes
            db.executemany(statement, batch)
            inserted += db.total_changes - before
            duplicates += len(batch) - (db.total_changes - before)
            batch.clear()
            db.commit()
        if progress_every and read % progress_every == 0:
            print(f"stage file={source.name} read={read} inserted={inserted}", flush=True)
    if batch:
        before = db.total_changes
        db.executemany(statement, batch)
        inserted += db.total_changes - before
        duplicates += len(batch) - (db.total_changes - before)
        db.commit()

    stats: dict[str, object] = {
        "source_path": source_key,
        "records_read": read,
        "records_invalid": invalid,
        "records_filtered": filtered,
        "events_inserted": inserted,
        "duplicates_skipped": duplicates,
        "source_timestamp_reversals": timestamp_reversals,
        "object_types": dict(type_counts.most_common()),
        "actions": dict(action_counts.most_common()),
    }
    db.execute(
        "INSERT INTO file_audit VALUES(?,?,1)",
        (source_key, json.dumps(stats, sort_keys=True, separators=(",", ":"))),
    )
    db.commit()
    return stats


def main() -> int:
    args = parse_args()
    rows = manifest_rows(args.manifest)
    cases, controls = load_scope(args.cases)
    current_signature = signature(args.manifest, args.cases, rows)
    db = open_db(args.stage_db)
    old = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if args.rebuild or (old is not None and str(old["value"]) != current_signature):
        db.executescript("DELETE FROM events; DELETE FROM file_audit; DELETE FROM bad_record_samples; DELETE FROM metadata;")
        db.commit()
    db.execute("INSERT OR REPLACE INTO metadata VALUES('signature',?)", (current_signature,))
    db.commit()

    files = [insert_file(db, row, cases, controls, args.progress_every) for row in rows]
    db.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_events_order ON events(timestamp_ms,corpus,event_id);
        CREATE INDEX IF NOT EXISTS idx_events_actor_time ON events(actor_id,timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_events_object_time ON events(object_id,timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_events_host_time ON events(hostname,timestamp_ms);
        """
    )
    total = int(db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    cross_corpus_ids = int(db.execute(
        "SELECT COUNT(*) FROM (SELECT event_id FROM events GROUP BY event_id HAVING COUNT(DISTINCT corpus)>1)"
    ).fetchone()[0])
    ordered_reversals = 0
    previous: tuple[int, str, str] | None = None
    for timestamp_ms, corpus, event_id in db.execute(
        "SELECT timestamp_ms,corpus,event_id FROM events ORDER BY timestamp_ms,corpus,event_id"
    ):
        current = (int(timestamp_ms), str(corpus), str(event_id))
        if previous is not None and current < previous:
            ordered_reversals += 1
        previous = current
    report = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "state": "complete",
        "label_blind": True,
        "signature": current_signature,
        "stage_db": str(args.stage_db.resolve()),
        "events_indexed": total,
        "event_ids_present_in_both_corpora": cross_corpus_ids,
        "canonical_order_reversals": ordered_reversals,
        "future_edges_materialized": 0,
        "files": files,
        "totals": {
            key: sum(int(item[key]) for item in files)
            for key in (
                "records_read", "records_invalid", "records_filtered",
                "events_inserted", "duplicates_skipped", "source_timestamp_reversals",
            )
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    print(f"events_indexed={total} report={args.report}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
