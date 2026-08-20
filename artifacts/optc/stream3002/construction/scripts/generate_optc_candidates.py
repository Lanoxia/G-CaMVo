#!/usr/bin/env python3
"""Generate deterministic, label-blind online OpTC checkpoint candidates.

The generator consumes only the E2 telemetry stage.  It never accepts a gold
file or scenario label.  Events are replayed in canonical timestamp order and
every trigger is decided using state already observed in the same corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def find_repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "camvo" / "security" / "optc.py").is_file():
            return parent
    raise RuntimeError("cannot locate G-CaMVo repository root")


ROOT = find_repository_root()
REMOTE_PORTS = frozenset({22, 53, 80, 88, 135, 139, 389, 443, 445, 636, 3389, 5985, 5986})
SENSITIVE_PATH_MARKERS = (
    "\\windows\\system32\\",
    "\\windows\\syswow64\\",
    "\\programdata\\",
    "\\users\\public\\",
    "\\appdata\\roaming\\",
    "\\startup\\",
    "powershell",
    "cmd.exe",
    ".ps1",
    ".bat",
    ".vbs",
    ".dll",
    ".exe",
)
TRIGGER_PRIORITY = (
    "remote_execution_or_session",
    "cross_host_connection",
    "sensitive_object_change",
    "new_task_root",
    "active_task_periodic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-db",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/stream_stage.sqlite",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/candidates.sqlite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/processed/optc_3causalbench_v1/e3_candidate_report.json",
    )
    parser.add_argument("--component-cap", type=int, default=6)
    parser.add_argument("--host-hour-trigger-cap", type=int, default=8)
    parser.add_argument("--trigger-cooldown-seconds", type=int, default=60)
    parser.add_argument("--periodic-seconds", type=int, default=600)
    parser.add_argument("--periodic-min-events", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def safe_int(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def is_private_unicast(value: object) -> bool:
    try:
        address = ipaddress.ip_address(str(value))
    except ValueError:
        return False
    return address.is_private and not address.is_multicast and not address.is_loopback


def flow_is_cross_host_proxy(properties: dict[str, Any]) -> bool:
    src_ip = properties.get("src_ip")
    dest_ip = properties.get("dest_ip")
    src_port = safe_int(properties.get("src_port"))
    dest_port = safe_int(properties.get("dest_port"))
    if not (is_private_unicast(src_ip) and is_private_unicast(dest_ip)):
        return False
    return bool({port for port in (src_port, dest_port) if port in REMOTE_PORTS})


def is_sensitive_change(object_type: str, action: str, properties: dict[str, Any]) -> bool:
    if object_type == "REGISTRY" and action in {"ADD", "EDIT", "REMOVE"}:
        return True
    if object_type != "FILE" or action not in {"CREATE", "WRITE", "MODIFY", "RENAME", "DELETE"}:
        return False
    text = " ".join(
        str(properties.get(key, ""))
        for key in ("file_path", "target_path", "path", "image_path", "command_line")
    ).lower()
    acuity = safe_int(properties.get("acuity_level")) or 0
    return acuity >= 3 and any(marker in text for marker in SENSITIVE_PATH_MARKERS)


def event_rows(db: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    query = """
      SELECT corpus,day,case_id,event_id,timestamp_ms,hostname,object_id,
             object_type,action,actor_id,pid,ppid,principal,properties_json
      FROM events
      ORDER BY corpus,timestamp_ms,hostname,event_id
    """
    yield from db.execute(query)


def initialize_output(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        DROP TABLE IF EXISTS candidates;
        DROP TABLE IF EXISTS metadata;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE candidates(
          checkpoint_id TEXT PRIMARY KEY,
          anchor_event_id TEXT NOT NULL,
          corpus TEXT NOT NULL,
          day TEXT NOT NULL,
          case_id TEXT,
          timestamp_ms INTEGER NOT NULL,
          hostname TEXT NOT NULL,
          trigger_kind TEXT NOT NULL,
          component_id TEXT NOT NULL,
          source_group TEXT NOT NULL,
          object_type TEXT NOT NULL,
          action TEXT NOT NULL,
          actor_id TEXT NOT NULL,
          object_id TEXT NOT NULL
        );
        CREATE INDEX idx_candidates_order
          ON candidates(corpus,timestamp_ms,checkpoint_id);
        CREATE INDEX idx_candidates_component
          ON candidates(corpus,component_id,timestamp_ms);
        """
    )
    return db


def source_signature(stage_db: Path, source: sqlite3.Connection) -> str:
    stage_signature = source.execute(
        "SELECT value FROM metadata WHERE key='signature'"
    ).fetchone()
    payload = {
        "schema": 1,
        "stage_signature": None if stage_signature is None else str(stage_signature[0]),
        "stage_size": stage_db.stat().st_size,
        "trigger_priority": TRIGGER_PRIORITY,
        "remote_ports": sorted(REMOTE_PORTS),
        "sensitive_path_markers": SENSITIVE_PATH_MARKERS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    args = parse_args()
    for name in (
        "component_cap", "host_hour_trigger_cap", "trigger_cooldown_seconds",
        "periodic_seconds", "periodic_min_events",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.stage_db.is_file():
        raise FileNotFoundError(args.stage_db)

    source = sqlite3.connect(args.stage_db)
    source.row_factory = sqlite3.Row
    required = {"corpus", "day", "event_id", "timestamp_ms", "hostname", "properties_json"}
    columns = {str(row[1]) for row in source.execute("PRAGMA table_info(events)")}
    missing = required - columns
    if missing:
        raise RuntimeError(f"E2 stage is missing columns: {sorted(missing)}")
    output = initialize_output(args.output_db)
    signature = source_signature(args.stage_db, source)

    root_by_actor: dict[tuple[str, str], str] = {}
    root_event_count: Counter[tuple[str, str]] = Counter()
    root_last_periodic: dict[tuple[str, str], int] = {}
    root_last_trigger: dict[tuple[str, str, str], int] = {}
    component_counts: Counter[tuple[str, str]] = Counter()
    host_hour_counts: Counter[tuple[str, str, int, str]] = Counter()
    trigger_counts: Counter[str] = Counter()
    corpus_counts: Counter[str] = Counter()
    day_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    host_hours: set[tuple[str, str, int]] = set()
    components: set[tuple[str, str]] = set()
    rows_seen = 0

    insert = """
      INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    cooldown_ms = args.trigger_cooldown_seconds * 1000
    periodic_ms = args.periodic_seconds * 1000
    for event in event_rows(source):
        rows_seen += 1
        corpus = str(event["corpus"])
        day = str(event["day"])
        timestamp_ms = int(event["timestamp_ms"])
        hostname = str(event["hostname"])
        actor_id = str(event["actor_id"])
        object_id = str(event["object_id"])
        object_type = str(event["object_type"])
        action = str(event["action"])
        actor_key = (corpus, actor_id)
        actor_known = actor_key in root_by_actor
        root = root_by_actor.get(actor_key, actor_id)
        if object_type == "PROCESS" and action == "CREATE":
            root_by_actor[(corpus, object_id)] = root
        root_event_count[(corpus, root)] += 1
        properties = json.loads(str(event["properties_json"]))

        triggers: list[str] = []
        if object_type in {"THREAD", "SHELL", "USER_SESSION"} and action in {
            "REMOTE_CREATE", "COMMAND", "REMOTE", "LOGIN", "GRANT", "INTERACTIVE",
        }:
            triggers.append("remote_execution_or_session")
        if object_type == "FLOW" and action in {"START", "OPEN"} and flow_is_cross_host_proxy(properties):
            triggers.append("cross_host_connection")
        if is_sensitive_change(object_type, action, properties):
            triggers.append("sensitive_object_change")
        if object_type == "PROCESS" and action == "CREATE" and not actor_known:
            triggers.append("new_task_root")
        last_periodic = root_last_periodic.get((corpus, root))
        if (
            root_event_count[(corpus, root)] >= args.periodic_min_events
            and (last_periodic is None or timestamp_ms - last_periodic >= periodic_ms)
        ):
            triggers.append("active_task_periodic")

        if not triggers:
            continue
        trigger = min(triggers, key=TRIGGER_PRIORITY.index)
        if trigger == "active_task_periodic":
            root_last_periodic[(corpus, root)] = timestamp_ms
        component_id = stable_id("cmp", corpus, root)
        component_key = (corpus, component_id)
        if component_counts[component_key] >= args.component_cap:
            rejected["component_cap"] += 1
            continue
        last_trigger = root_last_trigger.get((corpus, root, trigger))
        if last_trigger is not None and timestamp_ms - last_trigger < cooldown_ms:
            rejected["trigger_cooldown"] += 1
            continue
        hour = timestamp_ms // 3_600_000
        host_hour_key = (corpus, hostname, hour, trigger)
        if host_hour_counts[host_hour_key] >= args.host_hour_trigger_cap:
            rejected["host_hour_trigger_cap"] += 1
            continue

        checkpoint_id = stable_id("cp", corpus, event["event_id"], trigger)
        source_group = stable_id("sg", corpus, day, hostname, hour)
        output.execute(
            insert,
            (
                checkpoint_id, str(event["event_id"]), corpus, day,
                None if event["case_id"] is None else str(event["case_id"]),
                timestamp_ms, hostname, trigger, component_id, source_group,
                object_type, action, actor_id, object_id,
            ),
        )
        component_counts[component_key] += 1
        host_hour_counts[host_hour_key] += 1
        root_last_trigger[(corpus, root, trigger)] = timestamp_ms
        trigger_counts[trigger] += 1
        corpus_counts[corpus] += 1
        day_counts[f"{corpus}:{day}"] += 1
        host_hours.add((corpus, hostname, hour))
        components.add(component_key)
        if rows_seen % 10_000 == 0:
            output.commit()
        if args.progress_every and rows_seen % args.progress_every == 0:
            print(
                f"candidate_scan rows={rows_seen} accepted={sum(trigger_counts.values())}",
                flush=True,
            )

    output.execute("INSERT INTO metadata VALUES('source_signature',?)", (signature,))
    output.execute("INSERT INTO metadata VALUES('label_blind','true')")
    output.commit()
    total = int(output.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "state": "complete",
        "label_blind": True,
        "source_signature": signature,
        "trigger_code_sha256": code_sha256,
        "parameters": {
            "component_cap": args.component_cap,
            "host_hour_trigger_cap": args.host_hour_trigger_cap,
            "trigger_cooldown_seconds": args.trigger_cooldown_seconds,
            "periodic_seconds": args.periodic_seconds,
            "periodic_min_events": args.periodic_min_events,
        },
        "events_scanned": rows_seen,
        "candidate_checkpoints": total,
        "host_hours": len(host_hours),
        "task_groups": len(components),
        "causal_components": len(components),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "corpus_day_counts": dict(sorted(day_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "output_db": str(args.output_db.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    source.close()
    output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
