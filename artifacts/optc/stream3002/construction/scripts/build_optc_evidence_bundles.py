#!/usr/bin/env python3
"""Build E5 past-only task-causal evidence bundles and leakage audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def find_repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "camvo" / "security" / "optc.py").is_file():
            return parent
    raise RuntimeError("cannot locate G-CaMVo repository root")


ROOT = find_repository_root()
CONSTRUCTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONSTRUCTION))

from prompt import user_prompt  # noqa: E402

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
ABSOLUTE_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:[T ][0-2]?\d:[0-5]\d(?::[0-5]\d(?:\.\d+)?)?)?\b"
)
FORBIDDEN_KEYS = frozenset(
    {
        "gold", "gold_label", "sample_kind", "scenario", "scenario_id", "case_id",
        "day", "split", "attack_name", "label_provenance", "source_path", "source_role",
        "calibration", "validation", "test",
    }
)
EDGE_PRIORITY = {
    "REMOTE_CREATE": 0, "COMMAND": 0, "REMOTE": 0, "LOGIN": 0,
    "CREATE": 1, "START": 1, "OPEN": 2, "WRITE": 2, "EDIT": 2,
    "ADD": 2, "MESSAGE": 3, "READ": 4, "MODIFY": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/processed/optc_3causalbench_v1"
    parser.add_argument("--stage-db", type=Path, default=base / "stream_stage.sqlite")
    parser.add_argument("--candidates-db", type=Path, default=base / "candidates.sqlite")
    parser.add_argument("--evaluator-gold", type=Path, default=base / "evaluator/gold.sqlite")
    parser.add_argument("--prompt-jsonl", type=Path, default=base / "prompt_visible/checkpoints.jsonl")
    parser.add_argument("--evaluator-map", type=Path, default=base / "evaluator/checkpoint_map.jsonl")
    parser.add_argument(
        "--router-feature-map",
        type=Path,
        default=base / "router/checkpoint_features.jsonl",
        help=(
            "Label-blind router sidecar. It contains only one-way hashes of "
            "entities observable by each checkpoint and never contains corpus, "
            "split, source-group, case, or gold fields."
        ),
    )
    parser.add_argument("--split-manifest", type=Path, default=base / "evaluator/splits.jsonl")
    parser.add_argument("--report", type=Path, default=base / "e5_bundle_audit.json")
    parser.add_argument("--lookback-seconds", type=int, default=900)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-query-events-per-hop", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--max-edges", type=int, default=40)
    parser.add_argument("--max-estimated-tokens", type=int, default=4000)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def event(db: sqlite3.Connection, corpus: str, event_id: str) -> sqlite3.Row:
    row = db.execute(
        "SELECT * FROM events WHERE corpus=? AND event_id=?", (corpus, event_id)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing anchor event {corpus}:{event_id}")
    return row


def related_events(
    db: sqlite3.Connection,
    anchor: sqlite3.Row,
    *,
    lookback_ms: int,
    hops: int,
    limit: int,
) -> dict[tuple[str, str], tuple[sqlite3.Row, int]]:
    corpus = str(anchor["corpus"])
    anchor_time = int(anchor["timestamp_ms"])
    lower = anchor_time - lookback_ms
    selected: dict[tuple[str, str], tuple[sqlite3.Row, int]] = {
        (corpus, str(anchor["event_id"])): (anchor, 0)
    }
    frontier = {str(anchor["actor_id"]), str(anchor["object_id"])}
    seen_entities = set(frontier)
    for distance in range(1, hops + 1):
        if not frontier:
            break
        values = sorted(frontier)
        placeholders = ",".join("?" for _ in values)
        query = f"""
          SELECT * FROM events
          WHERE corpus=? AND timestamp_ms BETWEEN ? AND ?
            AND (actor_id IN ({placeholders}) OR object_id IN ({placeholders}))
          ORDER BY timestamp_ms DESC,event_id DESC LIMIT ?
        """
        parameters: list[Any] = [corpus, lower, anchor_time, *values, *values, limit]
        new_frontier: set[str] = set()
        for row in db.execute(query, parameters):
            key = (corpus, str(row["event_id"]))
            selected.setdefault(key, (row, distance))
            for entity in (str(row["actor_id"]), str(row["object_id"])):
                if entity not in seen_entities:
                    new_frontier.add(entity)
                    seen_entities.add(entity)
        frontier = new_frontier
    return selected


def edge_rank(item: tuple[sqlite3.Row, int], anchor_time: int, anchor_id: str) -> tuple[Any, ...]:
    row, distance = item
    return (
        0 if str(row["event_id"]) == anchor_id else 1,
        distance,
        EDGE_PRIORITY.get(str(row["action"]), 5),
        abs(anchor_time - int(row["timestamp_ms"])),
        str(row["event_id"]),
    )


def safe_properties(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, ""):
            continue
        if isinstance(value, str):
            # Preserve causally useful command/file/registry context while
            # removing identifiers that can reveal the original capture and
            # absolute collection date.  Relative ordering is represented by
            # the bundle's replay position and edge offsets instead.
            value = UUID_RE.sub("<GUID>", value[:300])
            value = ABSOLUTE_DATE_RE.sub("<DATE>", value)
        output[str(key)] = value
    return output


def router_entity_hash(scope: str, kind: str, value: Any) -> str:
    """Return a stable, non-reversible identifier for an observable entity.

    ``scope`` is folded into the digest but never emitted. It represents an
    observable collection/tenant boundary, preventing synthetic attack and
    benign captures from being joined merely because they reuse a hostname or
    identifier. The output cannot reveal which semantic corpus a scope came
    from. Including the entity kind also avoids cross-type joins.
    """

    normalized = str(value or "").strip()
    if not normalized:
        return ""
    payload = f"optc-router-v1\x00{scope}\x00{kind}\x00{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_schedule(candidates: list[sqlite3.Row]) -> dict[str, int]:
    """Build a deterministic mixed replay without exposing capture dates.

    OpTC attack and benign-control telemetry were collected on different wall
    clock dates, so those absolute timestamps are not comparable as one live
    stream.  Within each (day, corpus) capture we preserve the original event
    order.  Captures assigned to the same benchmark day are then merged by
    normalized within-capture rank.  This is label-independent once the two
    source captures have been selected and prevents either capture from being
    replayed as one contiguous block.

    Returned values are opaque monotone replay positions (one second apart),
    not reconstructed real-world timestamps.
    """

    day_order = {"day1": 0, "day2": 1, "day3": 2}
    by_day_corpus: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in candidates:
        by_day_corpus[(str(row["day"]), str(row["corpus"]))].append(row)
    schedule: dict[str, int] = {}
    replay_sequence = 0
    for day in sorted({key[0] for key in by_day_corpus}, key=lambda value: day_order[value]):
        merged: list[tuple[float, str, int, str, sqlite3.Row]] = []
        for (_day, corpus), rows in sorted(by_day_corpus.items()):
            if _day != day:
                continue
            ordered_rows = sorted(
                rows,
                key=lambda row: (int(row["timestamp_ms"]), str(row["checkpoint_id"])),
            )
            denominator = max(1, len(ordered_rows) - 1)
            corpus_tie = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
            for rank, row in enumerate(ordered_rows):
                merged.append(
                    (rank / denominator, corpus_tie, rank, str(row["checkpoint_id"]), row)
                )
        merged.sort(key=lambda item: item[:4])
        for item in merged:
            replay_sequence += 1
            schedule[str(item[4]["checkpoint_id"])] = replay_sequence * 1000
    if len(schedule) != len(candidates):
        raise RuntimeError("replay schedule does not cover every candidate")
    return schedule


def match_benign_controls(
    evaluator_rows: list[dict[str, Any]],
    token_estimates: dict[str, int],
) -> dict[str, Any]:
    """Freeze a balanced evaluator set using observable pre-response fields.

    Exact-ID attack positives define the malicious side.  For every such
    checkpoint, one official benign-control checkpoint from the same benchmark
    day is selected without replacement.  Pairing minimizes a deterministic
    lexicographic distance over trigger, object type, action and bundle size.
    Unmatched benign checkpoints remain in the online stream as unscored
    context.  No provider output is available to this function.
    """

    report: dict[str, Any] = {
        "policy": "evaluator-only-daywise-one-to-one-observable-covariate-v1",
        "model_responses_used": False,
        "days": {},
    }
    for day in ("day1", "day2", "day3"):
        malicious = sorted(
            (
                row for row in evaluator_rows
                if row["day"] == day and row["gold_label"] == "malicious"
            ),
            key=lambda row: hashlib.sha256(str(row["checkpoint_id"]).encode()).hexdigest(),
        )
        available = {
            str(row["checkpoint_id"]): row
            for row in evaluator_rows
            if row["day"] == day and row["corpus"] == "benign"
        }
        selected: list[tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]] = []
        for positive in malicious:
            if not available:
                break
            positive_id = str(positive["checkpoint_id"])
            ranked: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
            for benign_id, benign in available.items():
                distance = (
                    int(benign["trigger_kind"] != positive["trigger_kind"]),
                    int(benign["object_type"] != positive["object_type"]),
                    int(benign["action"] != positive["action"]),
                    abs(token_estimates[benign_id] - token_estimates[positive_id]),
                    hashlib.sha256(f"{positive_id}\x00{benign_id}".encode()).hexdigest(),
                )
                ranked.append((distance, benign_id, benign))
            distance, benign_id, benign = min(ranked, key=lambda item: item[0])
            benign["gold_label"] = "benign"
            selected.append((positive, benign, distance))
            del available[benign_id]
        if len(selected) != len(malicious):
            raise RuntimeError(
                f"insufficient benign controls for {day}: malicious={len(malicious)} "
                f"matched={len(selected)}"
            )
        report["days"][day] = {
            "malicious": len(malicious),
            "matched_benign": len(selected),
            "unscored_benign_context": len(available),
            "exact_trigger_matches": sum(pair[2][0] == 0 for pair in selected),
            "exact_object_type_matches": sum(pair[2][1] == 0 for pair in selected),
            "exact_action_matches": sum(pair[2][2] == 0 for pair in selected),
            "median_absolute_token_difference": (
                sorted(int(pair[2][3]) for pair in selected)[len(selected) // 2]
                if selected else None
            ),
        }
    return report


def build_bundle(
    db: sqlite3.Connection,
    candidate: sqlite3.Row,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = event(db, str(candidate["corpus"]), str(candidate["anchor_event_id"]))
    anchor_time = int(anchor["timestamp_ms"])
    related = related_events(
        db,
        anchor,
        lookback_ms=args.lookback_seconds * 1000,
        hops=args.hops,
        limit=args.max_query_events_per_hop,
    )
    ordered = sorted(
        related.values(), key=lambda item: edge_rank(item, anchor_time, str(anchor["event_id"]))
    )

    kept: list[tuple[sqlite3.Row, int]] = []
    entities: set[str] = set()
    duplicate_keys: set[tuple[str, str, str, str, str]] = set()
    duplicates_merged = 0
    estimated_chars = 800
    for row, distance in ordered:
        edge_key = (
            str(row["hostname"]), str(row["actor_id"]), str(row["action"]),
            str(row["object_id"]), str(row["object_type"]),
        )
        if edge_key in duplicate_keys and str(row["event_id"]) != str(anchor["event_id"]):
            duplicates_merged += 1
            continue
        prospective = entities | {str(row["actor_id"]), str(row["object_id"])}
        if len(prospective) > args.max_nodes or len(kept) >= args.max_edges:
            continue
        edge_chars = len(
            json.dumps(
                {
                    "host": row["hostname"], "relation": row["action"],
                    "target_type": row["object_type"], "pid": row["pid"],
                    "ppid": row["ppid"], "principal": row["principal"],
                    "properties": safe_properties(str(row["properties_json"])),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ) + 180
        if estimated_chars + edge_chars > args.max_estimated_tokens * 4:
            continue
        duplicate_keys.add(edge_key)
        entities = prospective
        kept.append((row, distance))
        estimated_chars += edge_chars

    kept.sort(key=lambda item: (int(item[0]["timestamp_ms"]), str(item[0]["event_id"])))
    entity_ids = {
        raw: f"N{index:02d}" for index, raw in enumerate(sorted(entities), 1)
    }
    evidence = []
    anchor_evidence_id = ""
    evaluator_edges = []
    for index, (row, distance) in enumerate(kept, 1):
        evidence_id = f"E{index:02d}"
        if str(row["event_id"]) == str(anchor["event_id"]):
            anchor_evidence_id = evidence_id
        evidence.append(
            {
                "evidence_id": evidence_id,
                "time_offset_seconds": round((int(row["timestamp_ms"]) - anchor_time) / 1000, 3),
                "host": str(row["hostname"]),
                "source": entity_ids[str(row["actor_id"])],
                "relation": str(row["action"]),
                "target": entity_ids[str(row["object_id"])],
                "target_type": str(row["object_type"]),
                "pid": row["pid"],
                "ppid": row["ppid"],
                "principal": row["principal"] or "unknown",
                "properties": safe_properties(str(row["properties_json"])),
            }
        )
        evaluator_edges.append(
            {
                "evidence_id": evidence_id,
                "event_id": str(row["event_id"]),
                "actor_id": str(row["actor_id"]),
                "object_id": str(row["object_id"]),
                "hop_distance": distance,
            }
        )
    if not anchor_evidence_id:
        raise RuntimeError(f"anchor was pruned for {candidate['checkpoint_id']}")

    bundle = {
        "schema_version": 1,
        "checkpoint_id": str(candidate["checkpoint_id"]),
        # Replaced with an opaque, capture-date-free position by ``main``.
        "decision_time_ms": anchor_time,
        "anchor_evidence_id": anchor_evidence_id,
        "anchor": {
            "host": str(anchor["hostname"]),
            "object_type": str(anchor["object_type"]),
            "action": str(anchor["action"]),
        },
        "evidence_edges": evidence,
        "history_notice": (
            "Only telemetry observed at or before the decision time is included. "
            "The evidence may be incomplete because deterministic node and edge budgets apply."
        ),
    }
    audit = {
        "raw_related_events": len(related),
        "kept_edges": len(kept),
        "kept_nodes": len(entities),
        "duplicates_merged": duplicates_merged,
        "future_edges": sum(int(row["timestamp_ms"]) > anchor_time for row, _ in kept),
        "evaluator_edges": evaluator_edges,
    }
    return bundle, audit


def main() -> int:
    args = parse_args()
    for path in (args.stage_db, args.candidates_db, args.evaluator_gold):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_nodes < 2 or args.max_edges < 1 or args.hops < 1 or args.max_estimated_tokens < 256:
        raise ValueError("invalid graph budget")
    stage = sqlite3.connect(args.stage_db)
    stage.row_factory = sqlite3.Row
    candidates_db = sqlite3.connect(args.candidates_db)
    candidates_db.row_factory = sqlite3.Row
    gold = sqlite3.connect(args.evaluator_gold)
    gold.row_factory = sqlite3.Row
    positive_ids = {str(row[0]) for row in gold.execute("SELECT event_id FROM positive_events")}

    prompt_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    audit_totals: Counter[str] = Counter()
    sizes: list[int] = []
    token_estimates: dict[str, int] = {}
    split_names = {"day1": "calibration", "day2": "validation", "day3": "test"}
    candidate_rows = list(
        candidates_db.execute("SELECT * FROM candidates ORDER BY corpus,timestamp_ms,checkpoint_id")
    )
    schedule = replay_schedule(candidate_rows)
    for index, candidate in enumerate(candidate_rows, 1):
        bundle, audit = build_bundle(stage, candidate, args)
        checkpoint_id = str(candidate["checkpoint_id"])
        raw_anchor_timestamp_ms = int(bundle["decision_time_ms"])
        bundle["decision_time_ms"] = schedule[checkpoint_id]
        rendered = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        prompt_rows.append(bundle)
        token_estimate = (len(rendered) + 3) // 4
        sizes.append(token_estimate)
        token_estimates[checkpoint_id] = token_estimate
        audit_totals.update(
            {
                "raw_related_events": audit["raw_related_events"],
                "kept_edges": audit["kept_edges"],
                "kept_nodes": audit["kept_nodes"],
                "duplicates_merged": audit["duplicates_merged"],
                "future_edges": audit["future_edges"],
            }
        )
        corpus = str(candidate["corpus"])
        anchor_id = str(candidate["anchor_event_id"])
        gold_label = "malicious" if corpus == "attack" and anchor_id in positive_ids else "unknown"
        evaluator_rows.append(
            {
                "checkpoint_id": str(candidate["checkpoint_id"]),
                "anchor_event_id": anchor_id,
                "raw_anchor_timestamp_ms": raw_anchor_timestamp_ms,
                "corpus": corpus,
                "day": str(candidate["day"]),
                "case_id": candidate["case_id"],
                "source_group": str(candidate["source_group"]),
                "trigger_kind": str(candidate["trigger_kind"]),
                "object_type": str(candidate["object_type"]),
                "action": str(candidate["action"]),
                "gold_label": gold_label,
                "evidence_id_map": audit["evaluator_edges"],
            }
        )
        router_entities: set[str] = set()
        for edge in audit["evaluator_edges"]:
            for kind in ("event_id", "actor_id", "object_id"):
                digest = router_entity_hash(corpus, kind, edge.get(kind))
                if digest:
                    router_entities.add(digest)
        # ``build_bundle`` intentionally returns only prompt-safe anchor fields;
        # do not re-read or leak the raw stage row into the router sidecar.
        host_digest = router_entity_hash(
            corpus, "host", bundle["anchor"]["host"]
        )
        if host_digest:
            router_entities.add(host_digest)
        router_rows.append(
            {
                "schema_version": 1,
                "checkpoint_id": str(candidate["checkpoint_id"]),
                "decision_time_ms": int(bundle["decision_time_ms"]),
                "observable_entity_hashes": sorted(router_entities),
            }
        )
        split_rows.append(
            {
                "checkpoint_id": str(candidate["checkpoint_id"]),
                "split": split_names[str(candidate["day"])],
                "source_group": str(candidate["source_group"]),
            }
        )
        if args.progress_every and index % args.progress_every == 0:
            print(f"bundle_build completed={index}/{len(candidate_rows)}", flush=True)

    matching = match_benign_controls(evaluator_rows, token_estimates)

    atomic_jsonl(args.prompt_jsonl, prompt_rows)
    atomic_jsonl(args.evaluator_map, evaluator_rows)
    atomic_jsonl(args.router_feature_map, router_rows)
    atomic_jsonl(args.split_manifest, split_rows)

    prompt_text = args.prompt_jsonl.read_text(encoding="utf-8")
    # Audit the exact provider-visible rendering incrementally.  OpTC can
    # contain many checkpoints, so concatenating every prompt into one giant
    # string needlessly multiplies peak memory during the E5 gate.
    provider_prompt_absolute_date_hits = 0
    provider_prompt_checkpoint_id_key_hits = 0
    provider_prompt_decision_time_key_hits = 0
    for bundle in prompt_rows:
        rendered = user_prompt(bundle, variant="task_causal")
        provider_prompt_absolute_date_hits += len(ABSOLUTE_DATE_RE.findall(rendered))
        provider_prompt_checkpoint_id_key_hits += len(
            re.findall(r'"checkpoint_id"\s*:', rendered)
        )
        provider_prompt_decision_time_key_hits += len(
            re.findall(r'"decision_time_ms"\s*:', rendered)
        )
    forbidden_hits = {
        key: len(re.findall(rf'"{re.escape(key)}"\s*:', prompt_text, flags=re.IGNORECASE))
        for key in sorted(FORBIDDEN_KEYS)
    }
    forbidden_hits = {key: count for key, count in forbidden_hits.items() if count}
    uuid_hits = len(UUID_RE.findall(prompt_text))
    router_text = args.router_feature_map.read_text(encoding="utf-8")
    router_forbidden_hits = {
        key: len(re.findall(rf'"{re.escape(key)}"\s*:', router_text, flags=re.IGNORECASE))
        for key in sorted(FORBIDDEN_KEYS | {"corpus", "source_group", "case_id", "gold_label"})
    }
    router_forbidden_hits = {
        key: count for key, count in router_forbidden_hits.items() if count
    }
    router_uuid_hits = len(UUID_RE.findall(router_text))
    source_groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        source_groups_by_split[str(row["split"])].add(str(row["source_group"]))
    split_overlap = 0
    names = sorted(source_groups_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            split_overlap += len(source_groups_by_split[left] & source_groups_by_split[right])
    labels = Counter(str(row["gold_label"]) for row in evaluator_rows)
    replay_by_checkpoint = {
        str(row["checkpoint_id"]): int(row["decision_time_ms"]) for row in router_rows
    }
    corpus_by_checkpoint = {
        str(row["checkpoint_id"]): str(row["corpus"]) for row in evaluator_rows
    }
    split_by_checkpoint = {
        str(row["checkpoint_id"]): str(row["split"]) for row in split_rows
    }
    replay_mixing: dict[str, dict[str, Any]] = {}
    for split in sorted(set(split_by_checkpoint.values())):
        ids = sorted(
            (checkpoint_id for checkpoint_id, value in split_by_checkpoint.items() if value == split),
            key=lambda checkpoint_id: (replay_by_checkpoint[checkpoint_id], checkpoint_id),
        )
        corpora = [corpus_by_checkpoint[checkpoint_id] for checkpoint_id in ids]
        maximum_run = 0
        current_run = 0
        prior = None
        for corpus in corpora:
            current_run = current_run + 1 if corpus == prior else 1
            maximum_run = max(maximum_run, current_run)
            prior = corpus
        replay_mixing[split] = {
            "checkpoints": len(ids),
            "corpus_counts": dict(sorted(Counter(corpora).items())),
            "maximum_same_corpus_run": maximum_run,
        }
    sorted_sizes = sorted(sizes)
    report = {
        "schema_version": 1,
        "state": "complete",
        "past_only": audit_totals["future_edges"] == 0,
        "checkpoints": len(prompt_rows),
        "gold_label_counts": dict(sorted(labels.items())),
        "parameters": {
            "lookback_seconds": args.lookback_seconds,
            "hops": args.hops,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "max_estimated_tokens": args.max_estimated_tokens,
            "max_query_events_per_hop": args.max_query_events_per_hop,
            "replay_order": "normalized-within-capture-rank-merge-v1",
        },
        "audit_totals": dict(audit_totals),
        "token_estimate": {
            "min": min(sizes, default=0),
            "median": sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0,
            "max": max(sizes, default=0),
            "within_2000_4000": sum(2000 <= value <= 4000 for value in sizes),
        },
        "replay_mixing": replay_mixing,
        "evaluator_matching": matching,
        "leakage": {
            "future_edges": audit_totals["future_edges"],
            "forbidden_metadata_hits": forbidden_hits,
            "raw_uuid_hits": uuid_hits,
            "source_group_split_overlap": split_overlap,
            "router_forbidden_metadata_hits": router_forbidden_hits,
            "router_raw_uuid_hits": router_uuid_hits,
            "absolute_evidence_timestamp_fields": sum(
                "timestamp_ms" in edge for bundle in prompt_rows
                for edge in bundle.get("evidence_edges", [])
            ),
            "provider_prompt_absolute_date_hits": provider_prompt_absolute_date_hits,
            "provider_prompt_checkpoint_id_key_hits": provider_prompt_checkpoint_id_key_hits,
            "provider_prompt_decision_time_key_hits": provider_prompt_decision_time_key_hits,
        },
        "files": {
            "prompt_jsonl": str(args.prompt_jsonl.resolve()),
            "evaluator_map": str(args.evaluator_map.resolve()),
            "router_feature_map": str(args.router_feature_map.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    stage.close()
    candidates_db.close()
    gold.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
