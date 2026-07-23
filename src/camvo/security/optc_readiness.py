"""Pre-provider readiness checks for a bounded DARPA OpTC experiment.

The checks in this module never call an LLM endpoint.  They exist so large raw
telemetry can be rejected early when paths, labels, timestamps, or schemas are
not ready for an auditable experiment.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from camvo.security.optc import (
    load_optc_attack_labels,
    load_optc_events,
    load_optc_scenarios,
)


_SUPPORTED_SUFFIXES = (
    ".json",
    ".jsonl",
    ".ndjson",
    ".json.gz",
    ".jsonl.gz",
    ".ndjson.gz",
)


def _is_supported(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in _SUPPORTED_SUFFIXES)


def _path_inventory(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "supported_files": 0,
            "bytes": 0,
        }
    candidates = (path,) if path.is_file() else path.rglob("*")
    supported_files = 0
    total_bytes = 0
    for candidate in candidates:
        if not candidate.is_file() or not _is_supported(candidate):
            continue
        supported_files += 1
        total_bytes += candidate.stat().st_size
    return {
        "path": str(path),
        "exists": True,
        "supported_files": supported_files,
        "bytes": total_bytes,
    }


def build_optc_readiness_report(
    attack_path: str | Path,
    benign_path: str | Path,
    labels_path: str | Path,
    scenario_manifest: str | Path,
    *,
    scenario_id: str = "optc-day3-malicious-upgrade",
    utc_offset_minutes: int = -240,
    padding_minutes: int = 10,
    sample_events: int = 32,
    minimum_free_bytes: int = 20 * 1024**3,
) -> dict[str, Any]:
    """Return an auditable readiness report without making provider calls."""

    if sample_events <= 0:
        raise ValueError("sample_events must be positive")
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be non-negative")

    attack = Path(attack_path)
    benign = Path(benign_path)
    labels = Path(labels_path)
    manifest = Path(scenario_manifest)
    attack_inventory = _path_inventory(attack)
    benign_inventory = _path_inventory(benign)
    disk_root = attack.parent if attack.parent.exists() else Path.cwd()
    disk = shutil.disk_usage(disk_root)
    blockers: list[str] = []
    warnings: list[str] = [
        "UTC alignment is an explicit assumption and must be verified against the staged release.",
        "A single Day-3 scenario is a bounded PoC, not cross-scenario generalization evidence.",
    ]

    if not manifest.is_file():
        blockers.append(f"scenario manifest is missing: {manifest}")
    if not labels.is_file():
        blockers.append(f"positive-label CSV is missing: {labels}")
    for name, inventory in (("attack", attack_inventory), ("benign", benign_inventory)):
        if not bool(inventory["exists"]):
            blockers.append(f"{name} telemetry path is missing: {inventory['path']}")
        elif int(inventory["supported_files"]) == 0:
            blockers.append(f"{name} telemetry has no supported JSON/JSONL/gzip files")
    if disk.free < minimum_free_bytes:
        blockers.append(
            f"only {disk.free} free bytes are available; require at least {minimum_free_bytes}"
        )

    scenario_payload: dict[str, object] | None = None
    label_payload: dict[str, object] | None = None
    attack_probe: dict[str, object] | None = None
    benign_probe: dict[str, object] | None = None
    if manifest.is_file():
        try:
            scenarios = load_optc_scenarios(manifest)
            scenario = scenarios[scenario_id]
            start_ms, end_ms = scenario.epoch_window(
                utc_offset_minutes=utc_offset_minutes,
                padding_minutes=padding_minutes,
            )
            scenario_payload = {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "hosts": list(scenario.hosts),
                "activities": len(scenario.activities),
                "resolved_start_ms": start_ms,
                "resolved_end_ms": end_ms,
                "utc_offset_minutes": utc_offset_minutes,
                "padding_minutes": padding_minutes,
            }
        except (KeyError, OSError, TypeError, ValueError) as exc:
            blockers.append(f"scenario manifest cannot resolve {scenario_id!r}: {exc}")

    if labels.is_file() and scenario_payload is not None:
        try:
            label_index = load_optc_attack_labels(
                labels,
                hostnames=scenario_payload["hosts"],
                start_ms=int(scenario_payload["resolved_start_ms"]),
                end_ms=int(scenario_payload["resolved_end_ms"]),
                strict=False,
            )
            label_payload = asdict(label_index.stats)
            if not label_index.labels:
                blockers.append("no positive labels align with the selected hosts/time window")
            if label_index.stats.records_skipped:
                warnings.append(
                    "positive-label audit skipped "
                    f"{label_index.stats.records_skipped} malformed rows"
                )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"positive-label CSV cannot be audited: {exc}")

    if int(attack_inventory["supported_files"]) > 0 and scenario_payload is not None:
        try:
            probed = load_optc_events(
                attack,
                hostnames=scenario_payload["hosts"],
                start_ms=int(scenario_payload["resolved_start_ms"]),
                end_ms=int(scenario_payload["resolved_end_ms"]),
                max_events=sample_events,
                strict=False,
            )
            attack_probe = asdict(probed.stats)
            if not probed.events:
                blockers.append(
                    "attack telemetry probe found no events in the selected scenario window"
                )
            if probed.stats.files_failed or probed.stats.records_skipped:
                warnings.append(
                    "attack telemetry contains unreadable files or malformed rows; "
                    "inspect probe stats"
                )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"attack telemetry probe failed: {exc}")

    if int(benign_inventory["supported_files"]) > 0:
        try:
            probed = load_optc_events(
                benign,
                max_events=sample_events,
                strict=False,
            )
            benign_probe = asdict(probed.stats)
            if not probed.events:
                blockers.append("benign telemetry probe found no valid events")
            if probed.stats.files_failed or probed.stats.records_skipped:
                warnings.append(
                "benign telemetry contains unreadable files or malformed rows; "
                "inspect probe stats"
                )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"benign telemetry probe failed: {exc}")

    return {
        "schema_version": 1,
        "provider_calls_made": 0,
        "ready": not blockers,
        "status": "ready_for_dataset_build" if not blockers else "blocked_external_data",
        "blockers": blockers,
        "warnings": warnings,
        "disk": {
            "root": str(disk_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "minimum_required_free_bytes": minimum_free_bytes,
        },
        "attack": {**attack_inventory, "probe": attack_probe},
        "benign": {**benign_inventory, "probe": benign_probe},
        "labels": {"path": str(labels), "exists": labels.is_file(), "audit": label_payload},
        "scenario": scenario_payload,
    }
