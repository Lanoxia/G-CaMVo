#!/usr/bin/env python3
"""Score the completed OpTC replay as all-campaign attack reconstruction.

This evaluator performs no provider calls and reads no credentials. It joins
the frozen 3,002-checkpoint route trace to evaluator-only official evidence
after replay completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DAY_ORDER = {"day1": 0, "day2": 1, "day3": 2}


def args() -> argparse.Namespace:
    base = ROOT / "data/processed/optc_3causalbench_v1"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluator-map", type=Path, default=base / "evaluator/checkpoint_map.jsonl")
    p.add_argument("--gold-db", type=Path, default=base / "evaluator/gold.sqlite")
    p.add_argument("--stage-db", type=Path, default=base / "stream_stage.sqlite")
    p.add_argument("--responses", type=Path, default=ROOT / "repo/optc/runs/e6_prompt_freeze_v2")
    p.add_argument("--route-trace", type=Path, default=ROOT / "repo/optc/runs/e8_all_days_prequential_v1/route_trace.jsonl")
    p.add_argument("--route-report", type=Path, default=ROOT / "repo/optc/runs/e8_all_days_prequential_v1/report.json")
    p.add_argument("--protocol", type=Path, default=ROOT / "repo/optc/config/e14_all_campaign_reconstruction_protocol_v1.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "repo/optc/runs/e14_all_campaign_reconstruction_v1")
    p.add_argument("--alignment-mode", choices=("exact", "strong_entity_time"), default="exact")
    p.add_argument("--window-minutes", type=int, default=30)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def response_path(run: Path, model: str, checkpoint: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return run / "responses/task_causal" / safe / f"{checkpoint}.json"


def refs(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"evidence_edges", "evidence_ids"} and isinstance(child, list):
                found.update(map(str, child))
            else:
                found.update(refs(child))
    elif isinstance(value, list):
        for child in value:
            found.update(refs(child))
    return found


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def load_gold(path: Path) -> dict[str, Any]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    steps = {str(r["step_id"]): dict(r) for r in db.execute(
        "SELECT * FROM official_steps WHERE observable=1 ORDER BY day,timestamp_ms,step_id"
    )}
    event_steps: dict[str, set[str]] = defaultdict(set)
    for r in db.execute(
        "SELECT s.event_id,s.step_id FROM step_positive_events s "
        "JOIN official_steps o ON o.step_id=s.step_id WHERE o.observable=1"
    ):
        event_steps[str(r["event_id"])].add(str(r["step_id"]))
    positive = {str(r["event_id"]): dict(r) for r in db.execute("SELECT * FROM positive_events")}
    triples_by_step: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r in db.execute(
        "SELECT s.step_id,p.hostname,p.actor_id,p.object_id FROM step_positive_events s "
        "JOIN positive_events p ON p.event_id=s.event_id JOIN official_steps o ON o.step_id=s.step_id "
        "WHERE o.observable=1"
    ):
        triples_by_step[str(r["step_id"])].add((str(r["hostname"]), str(r["actor_id"]), str(r["object_id"])))
    transitions = {(str(r["source_step_id"]), str(r["target_step_id"])) for r in db.execute(
        "SELECT a.source_step_id,a.target_step_id FROM attack_path_edges a "
        "JOIN official_steps s ON s.step_id=a.source_step_id "
        "JOIN official_steps t ON t.step_id=a.target_step_id "
        "WHERE s.observable=1 AND t.observable=1"
    )}
    db.close()
    return {"steps": steps, "event_steps": event_steps, "positive": positive,
            "triples_by_step": triples_by_step, "transitions": transitions}


def strong_entity_time_alignment(
    stage_db: Path,
    evaluator: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Any],
    window_minutes: int,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Extend exact alignment with frozen strong identity-time case templates."""

    evidence_ids = {
        str(edge["event_id"])
        for row in evaluator.values()
        for edge in row.get("evidence_id_map", [])
    }
    db = sqlite3.connect(stage_db)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TEMP TABLE wanted(event_id TEXT PRIMARY KEY)")
    db.executemany("INSERT OR IGNORE INTO wanted VALUES(?)", ((event,) for event in evidence_ids))
    events = {str(r["event_id"]): dict(r) for r in db.execute(
        "SELECT e.event_id,e.day,e.timestamp_ms,e.hostname,e.actor_id,e.object_id "
        "FROM events e JOIN wanted w ON w.event_id=e.event_id"
    )}
    db.close()
    aligned: dict[str, set[str]] = defaultdict(set)
    for event, steps in gold["event_steps"].items():
        if event in evidence_ids:
            aligned[event].update(steps)
    radius = window_minutes * 60_000
    for event_id, event in events.items():
        triple = (str(event["hostname"]), str(event["actor_id"]), str(event["object_id"]))
        for step_id, step in gold["steps"].items():
            if str(step["day"]) != str(event["day"]):
                continue
            if abs(int(event["timestamp_ms"]) - int(step["timestamp_ms"])) > radius:
                continue
            if triple in gold["triples_by_step"].get(step_id, set()):
                aligned[event_id].add(step_id)
    reachable = set().union(*aligned.values()) if aligned else set()
    return dict(aligned), {
        "alignment_mode": "strong_entity_time",
        "window_minutes": window_minutes,
        "prompt_evidence_events_audited": len(events),
        "prompt_reachable_official_steps": len(reachable),
        "prompt_reachable_fraction": len(reachable) / len(gold["steps"]),
        "reachable_by_campaign": dict(Counter(str(gold["steps"][s]["day"]) for s in reachable)),
    }


def timeline_counts(detected: Mapping[str, int], steps: Mapping[str, Mapping[str, Any]], day: str) -> tuple[int, int, int]:
    ids = sorted((s for s in detected if steps[s]["day"] == day), key=lambda s: int(steps[s]["official_order"]))
    comparable = concordant = 0
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            observed = detected[left] - detected[right]
            official = int(steps[left]["official_order"]) - int(steps[right]["official_order"])
            if observed == 0 or official == 0:
                continue
            comparable += 1
            concordant += int((observed < 0) == (official < 0))
    total = sum(1 for i in range(sum(steps[s]["day"] == day for s in steps)) for _ in range(i + 1, sum(steps[s]["day"] == day for s in steps)))
    return concordant, comparable, total


def main() -> int:
    a = args()
    for p in (a.evaluator_map, a.gold_db, a.stage_db, a.route_trace, a.route_report, a.protocol):
        if not p.is_file():
            raise FileNotFoundError(p)
    protocol = json.loads(a.protocol.read_text())
    route_report = json.loads(a.route_report.read_text())
    contract = route_report.get("algorithm_contract", {})
    required_contract = (
        "all_3002_checkpoints_replayed_once_in_order",
        "current_subset_selected_before_current_responses",
        "evaluator_labels_not_used_by_router",
        "future_evidence_forbidden",
    )
    if route_report.get("state") != "complete" or not all(contract.get(k) is True for k in required_contract):
        raise RuntimeError("completed leakage-safe all-days replay is required")

    evaluator = {str(x["checkpoint_id"]): x for x in read_jsonl(a.evaluator_map)}
    routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(a.route_trace):
        routes[str(row["method"])].append(row)
    if any(len(rows) != 3002 for rows in routes.values()):
        raise RuntimeError("every method must cover all 3,002 checkpoints")
    gold = load_gold(a.gold_db)
    steps: dict[str, dict[str, Any]] = gold["steps"]
    if a.alignment_mode == "strong_entity_time":
        event_steps, alignment_audit = strong_entity_time_alignment(
            a.stage_db, evaluator, gold, a.window_minutes
        )
    else:
        event_steps = gold["event_steps"]
        reachable = set().union(*event_steps.values()) if event_steps else set()
        alignment_audit = {
            "alignment_mode": "exact", "window_minutes": None,
            "prompt_reachable_official_steps": len(reachable),
            "prompt_reachable_fraction": len(reachable) / len(gold["steps"]),
            "reachable_by_campaign": dict(Counter(str(gold["steps"][s]["day"]) for s in reachable)),
        }
    order = {s: (DAY_ORDER[str(v["day"])], int(v["official_order"])) for s, v in steps.items()}
    step_counts = Counter(str(x["day"]) for x in steps.values())
    transition_counts = Counter(str(steps[s]["day"]) for s, _ in gold["transitions"])

    rows_out: list[dict[str, Any]] = []
    campaign_out: list[dict[str, Any]] = []
    audit_out: list[dict[str, Any]] = []
    step_audit_out: list[dict[str, Any]] = []
    transition_audit_out: list[dict[str, Any]] = []
    for method, method_routes in sorted(routes.items()):
        detected: dict[str, int] = {}
        cited: set[str] = set()
        cited_positive: set[str] = set()
        predicted_transitions: set[tuple[str, str]] = set()
        selected_calls = 0
        total_cost = 0.0
        for route in sorted(method_routes, key=lambda x: (int(x["decision_time_ms"]), str(x["checkpoint_id"]))):
            cp = str(route["checkpoint_id"])
            ev = evaluator[cp]
            day = str(ev["day"])
            selected = list(map(str, route["selected_models"]))
            selected_calls += len(selected)
            total_cost += float(route["cost_usd"])
            edge_map = {str(x["evidence_id"]): x for x in ev.get("evidence_id_map", [])}
            route_events: set[str] = set()
            route_steps: set[str] = set()
            route_edges: set[tuple[str, str]] = set()
            for model in selected:
                cached = json.loads(response_path(a.responses, model, cp).read_text())
                if cached.get("state") != "succeeded" or cached.get("schema_errors"):
                    raise RuntimeError(f"invalid frozen response: {model}/{cp}")
                prediction = cached["prediction"]
                if prediction.get("verdict") != "malicious":
                    continue
                events = {str(edge_map[r]["event_id"]) for r in refs(prediction) if r in edge_map}
                route_events.update(events)
                route_steps.update(*(event_steps.get(event, set()) for event in events))
                for path_edge in prediction.get("attack_path_edges", []):
                    edge_events = {str(edge_map[r]["event_id"]) for r in refs(path_edge) if r in edge_map}
                    aligned = sorted(set().union(*(event_steps.get(e, set()) for e in edge_events)) if edge_events else set(), key=lambda s: order[s])
                    if len(aligned) >= 2 and steps[aligned[0]]["day"] == steps[aligned[-1]]["day"]:
                        route_edges.add((aligned[0], aligned[-1]))
            cited.update(route_events)
            cited_positive.update(e for e in route_events if e in gold["positive"])
            predicted_transitions.update(route_edges)
            valid_steps = route_steps if route.get("prediction") == "malicious" else set()
            for step in valid_steps:
                if steps[step]["day"] == day:
                    detected.setdefault(step, int(ev["raw_anchor_timestamp_ms"]))
            audit_out.append({
                "method": method, "checkpoint_id": cp, "day": day,
                "decision_time_ms": route["decision_time_ms"], "source_group": ev["source_group"],
                "selected_models": "|".join(selected), "route_prediction": route["prediction"],
                "aligned_official_steps": "|".join(sorted(valid_steps, key=lambda s: order[s])),
                "cited_event_count": len(route_events), "route_cost_usd": route["cost_usd"],
            })

        campaign_recalls = []
        total_concordant = total_comparable = total_pairs = 0
        for day in DAY_ORDER:
            ids = {s for s in detected if steps[s]["day"] == day}
            recall = len(ids) / step_counts[day]
            campaign_recalls.append(recall)
            concordant, comparable, total = timeline_counts(detected, steps, day)
            total_concordant += concordant; total_comparable += comparable; total_pairs += total
            campaign_out.append({
                "method": method, "campaign": day,
                "stream_checkpoints": sum(1 for r in method_routes if evaluator[str(r["checkpoint_id"])]["day"] == day),
                "observable_official_steps": step_counts[day], "detected_official_steps": len(ids),
                "official_step_recall": recall, "official_transitions": transition_counts[day],
                "timeline_concordant_pairs": concordant, "timeline_comparable_pairs": comparable,
                "timeline_total_official_pairs": total,
                "timeline_order_consistency": concordant / comparable if comparable else None,
                "timeline_pair_coverage": comparable / total if total else None,
                "coverage_adjusted_timeline": concordant / total if total else None,
            })

        gold_entities = set()
        cited_entities = set()
        for e, value in gold["positive"].items():
            entities = {f"host:{value['hostname']}", f"actor:{value['actor_id']}", f"object:{value['object_id']}"}
            gold_entities.update(entities)
            if e in cited_positive:
                cited_entities.update(entities)
        tr_p, tr_r, tr_f1 = prf({f"{a}->{b}" for a, b in predicted_transitions}, {f"{a}->{b}" for a, b in gold["transitions"]})
        reachable_steps = set().union(*event_steps.values()) if event_steps else set()
        for step_id, step in sorted(steps.items(), key=lambda item: order[item[0]]):
            step_audit_out.append({
                "method": method,
                "step_id": step_id,
                "campaign": step["day"],
                "official_order": step["official_order"],
                "reachable_under_alignment": step_id in reachable_steps,
                "detected": step_id in detected,
                "first_detection_timestamp_ms": detected.get(step_id),
            })
        for source_step, target_step in sorted(
            gold["transitions"] | predicted_transitions,
            key=lambda edge: (order.get(edge[0], (99, 99)), order.get(edge[1], (99, 99))),
        ):
            transition_audit_out.append({
                "method": method,
                "campaign": steps[source_step]["day"] if source_step in steps else "unknown",
                "source_step_id": source_step,
                "target_step_id": target_step,
                "gold_transition": (source_step, target_step) in gold["transitions"],
                "predicted_transition": (source_step, target_step) in predicted_transitions,
            })
        rows_out.append({
            "method": method, "stream_checkpoints": len(method_routes),
            "observable_official_steps": len(steps), "detected_official_steps": len(detected),
            "official_step_recall_micro": len(detected) / len(steps),
            "official_step_recall_macro_campaign": sum(campaign_recalls) / len(campaign_recalls),
            "official_transitions": len(gold["transitions"]), "reconstructed_official_transitions": len(predicted_transitions & gold["transitions"]),
            "official_transition_precision": tr_p, "official_transition_recall": tr_r, "official_transition_f1": tr_f1,
            "timeline_concordant_pairs": total_concordant, "timeline_comparable_pairs": total_comparable,
            "timeline_total_official_pairs": total_pairs,
            "timeline_order_consistency": total_concordant / total_comparable if total_comparable else None,
            "timeline_pair_coverage": total_comparable / total_pairs if total_pairs else None,
            "coverage_adjusted_timeline": total_concordant / total_pairs if total_pairs else None,
            "official_positive_evidence_precision": len(cited_positive) / len(cited) if cited else 0.0,
            "official_entity_coverage": len(cited_entities) / len(gold_entities) if gold_entities else 0.0,
            "average_models_per_checkpoint": selected_calls / len(method_routes), "total_cost_usd": total_cost,
        })

    write_csv(a.output_dir / "e14_all_campaign_main_table.csv", rows_out)
    write_csv(a.output_dir / "e14_all_campaign_by_campaign.csv", campaign_out)
    write_csv(a.output_dir / "e14_all_campaign_checkpoint_audit.csv", audit_out)
    write_csv(a.output_dir / "e14_step_detection_audit.csv", step_audit_out)
    write_csv(a.output_dir / "e14_transition_audit.csv", transition_audit_out)
    report = {
        "schema_version": 1, "state": "complete", "provider_calls": 0,
        "protocol": protocol, "protocol_sha256": sha256(a.protocol),
        "input_hashes": {"evaluator_map": sha256(a.evaluator_map), "gold_db": sha256(a.gold_db), "route_trace": sha256(a.route_trace)},
        "scale": {"campaigns": 3, "stream_checkpoints": 3002, "model_responses": 15010,
                  "observable_official_steps": len(steps), "official_transitions": len(gold["transitions"]),
                  "steps_by_campaign": dict(step_counts), "transitions_by_campaign": dict(transition_counts)},
        "alignment_audit": alignment_audit,
        "results": rows_out, "by_campaign": campaign_out,
    }
    write_json(a.output_dir / "report.json", report)
    print(json.dumps({"state": "complete", "provider_calls": 0, "methods": len(rows_out), "scale": report["scale"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
