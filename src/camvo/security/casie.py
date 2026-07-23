"""Robust CASIE event-mention classification adapter.

CASIE is an event-extraction corpus.  This first security PoC turns each gold
event mention into a five-way online classification item while retaining the
document context and provenance needed for richer extraction tasks later.
Gold labels are stored only in metadata for simulators/evaluators; the CaMVo
router does not read them.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from camvo.types import AnnotationItem

CASIE_LABELS: tuple[str, ...] = (
    "Databreach",
    "Phishing",
    "Ransom",
    "DiscoverVulnerability",
    "PatchVulnerability",
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Databreach": ("breach", "leak", "stolen", "compromise", "access"),
    "Phishing": ("phish", "email", "spoof", "link"),
    "Ransom": ("ransom", "encrypt", "payment", "bitcoin"),
    "DiscoverVulnerability": ("vulnerability", "flaw", "discover", "cve"),
    "PatchVulnerability": ("patch", "update", "fix", "release"),
}


@dataclass(frozen=True, slots=True)
class CasieLoadStats:
    files_scanned: int
    files_failed: int
    events_loaded: int
    events_skipped: int
    label_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CasieDataset:
    items: tuple[AnnotationItem, ...]
    stats: CasieLoadStats


def _difficulty_score(label: str, realis: str, nugget: str, argument_count: int) -> float:
    """Create an observable difficulty proxy used only by simulated LLMs."""

    score = {
        "Actual": -0.20,
        "Generic": 0.40,
        "Other": 0.55,
        "General": 0.30,
    }.get(realis, 0.35)
    lowered = nugget.casefold()
    if any(keyword in lowered for keyword in _KEYWORDS[label]):
        score -= 0.30
    if argument_count == 0:
        score += 0.15
    elif argument_count >= 3:
        score -= 0.10
    return max(-0.8, min(0.9, score))


def _difficulty_bucket(score: float) -> str:
    if score <= -0.10:
        return "easy"
    if score >= 0.45:
        return "hard"
    return "medium"


def _context_with_marker(
    content: str,
    *,
    title: str,
    start: int,
    end: int,
    context_chars: int,
) -> str:
    left = max(0, start - context_chars)
    right = min(len(content), end + context_chars)
    excerpt = (
        content[left:start]
        + " [EVENT] "
        + content[start:end]
        + " [/EVENT] "
        + content[end:right]
    )
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if left > 0:
        excerpt = "... " + excerpt
    if right < len(content):
        excerpt += " ..."
    return f"Title: {title.strip() or 'Untitled'}\nSecurity report context: {excerpt}"


def _resolve_nugget_span(content: str, nugget: dict[str, Any]) -> tuple[int, int]:
    """Resolve known CASIE offset drift without silently changing the label.

    A minority of upstream annotations are shifted by one or include terminal
    punctuation that is absent from ``content``. Prefer the declared span,
    then the closest exact occurrence, then a punctuation-trimmed occurrence.
    """

    declared_start = int(nugget["startOffset"])
    declared_end = int(nugget["endOffset"])
    annotated = str(nugget.get("text", ""))

    def normalized(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    if 0 <= declared_start < declared_end <= len(content):
        if not annotated or normalized(content[declared_start:declared_end]) == normalized(
            annotated
        ):
            return declared_start, declared_end

    candidates: list[tuple[int, int]] = []
    variants = [annotated]
    trimmed = annotated.rstrip(" \t\r\n.,!?;:")
    if trimmed and trimmed != annotated:
        variants.append(trimmed)
    for variant in variants:
        if not variant:
            continue
        position = content.find(variant)
        while position >= 0:
            candidates.append((position, position + len(variant)))
            position = content.find(variant, position + 1)
    if candidates:
        return min(candidates, key=lambda span: (abs(span[0] - declared_start), span[0]))

    clamped_start = max(0, min(declared_start, len(content) - 1))
    clamped_end = max(clamped_start + 1, min(declared_end, len(content)))
    source_value = normalized(content[clamped_start:clamped_end]).rstrip(".,!?;:")
    if source_value and source_value == normalized(annotated).rstrip(".,!?;:"):
        return clamped_start, clamped_end
    raise ValueError("nugget text cannot be aligned to source content")


def _event_item(
    document_id: str,
    content: str,
    info: dict[str, Any],
    hopper_index: int,
    event_index: int,
    event: dict[str, Any],
    source_path: Path,
    context_chars: int,
) -> AnnotationItem:
    label = str(event.get("subtype", ""))
    if label not in CASIE_LABELS:
        raise ValueError(f"unsupported or missing CASIE subtype: {label!r}")
    nugget = event.get("nugget")
    if not isinstance(nugget, dict):
        raise ValueError("event nugget is missing")
    start, end = _resolve_nugget_span(content, nugget)
    nugget_text = content[start:end]

    arguments = event.get("argument", [])
    if not isinstance(arguments, list):
        raise ValueError("event arguments must be a list")
    realis = str(event.get("realis", "Unknown"))
    difficulty = _difficulty_score(label, realis, nugget_text, len(arguments))
    event_id = str(event.get("index", f"H{hopper_index}E{event_index}"))
    text = _context_with_marker(
        content,
        title=str(info.get("title", "")),
        start=start,
        end=end,
        context_chars=context_chars,
    )
    item_id = f"casie:{document_id}:{event_id}:{hopper_index}:{event_index}"
    return AnnotationItem(
        item_id=item_id,
        text=text,
        labels=CASIE_LABELS,
        metadata={
            "dataset": "CASIE",
            "gold_label": label,
            "document_id": document_id,
            "event_id": event_id,
            "graph_node_id": item_id,
            "hopper_index": hopper_index,
            "event_index": event_index,
            "event_type": str(event.get("type", "")),
            "realis": realis,
            "nugget": nugget_text,
            "start_offset": start,
            "end_offset": end,
            "argument_count": len(arguments),
            "difficulty_score": difficulty,
            "difficulty": _difficulty_bucket(difficulty),
            "source_path": str(source_path),
        },
    )


def load_casie_event_items(
    data_dir: str | Path,
    *,
    context_chars: int = 320,
    strict: bool = False,
) -> CasieDataset:
    """Load all CASIE event mentions from ``data/annotation/*.json``.

    ``data_dir`` may point either to the CASIE ``data`` directory or directly
    to its ``annotation`` directory.  In tolerant mode malformed files/events
    are counted and skipped; strict mode raises immediately.
    """

    if context_chars < 32:
        raise ValueError("context_chars must be at least 32")
    root = Path(data_dir)
    annotation_dir = root if root.name == "annotation" else root / "annotation"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"CASIE annotation directory not found: {annotation_dir}")

    paths = sorted(annotation_dir.glob("*.json"), key=lambda path: path.stem)
    if not paths:
        raise FileNotFoundError(f"no CASIE JSON annotations found in {annotation_dir}")

    items: list[AnnotationItem] = []
    labels: Counter[str] = Counter()
    files_failed = 0
    events_skipped = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            content = payload["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content is missing or empty")
            info = payload.get("info", {})
            hoppers = payload.get("cyberevent", {}).get("hopper", [])
            if not isinstance(hoppers, list):
                raise ValueError("cyberevent.hopper must be a list")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            if strict:
                raise
            files_failed += 1
            continue

        for hopper_index, hopper in enumerate(hoppers):
            events = hopper.get("events", []) if isinstance(hopper, dict) else []
            for event_index, event in enumerate(events):
                try:
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    item = _event_item(
                        path.stem,
                        content,
                        info if isinstance(info, dict) else {},
                        hopper_index,
                        event_index,
                        event,
                        path,
                        context_chars,
                    )
                except (KeyError, TypeError, ValueError):
                    if strict:
                        raise
                    events_skipped += 1
                    continue
                items.append(item)
                labels[str(item.metadata["gold_label"])] += 1

    if not items:
        raise ValueError("CASIE loader did not produce any valid event items")
    stats = CasieLoadStats(
        files_scanned=len(paths),
        files_failed=files_failed,
        events_loaded=len(items),
        events_skipped=events_skipped,
        label_counts={label: labels[label] for label in CASIE_LABELS},
    )
    return CasieDataset(tuple(items), stats)


def build_casie_event_adjacency(
    items: Iterable[AnnotationItem],
    *,
    same_hopper_weight: float = 1.0,
    same_document_weight: float = 0.15,
    distance_scale_chars: float = 800.0,
) -> dict[str, dict[str, float]]:
    """Build a sparse graph without crossing document boundaries.

    Consecutive mentions in the same event hopper receive a strong edge;
    consecutive mentions elsewhere in the same article receive a weak edge.
    This graph is only a CASIE engineering baseline, not an OpTC provenance
    graph, and the distinction is preserved in experiment reports.
    """

    for name, value in {
        "same_hopper_weight": same_hopper_weight,
        "same_document_weight": same_document_weight,
        "distance_scale_chars": distance_scale_chars,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    documents: dict[str, list[AnnotationItem]] = defaultdict(list)
    hoppers: dict[tuple[str, int], list[AnnotationItem]] = defaultdict(list)
    for item in items:
        document_id = str(item.metadata.get("document_id", ""))
        if not document_id:
            raise ValueError("CASIE item is missing document_id")
        try:
            hopper_index = int(item.metadata["hopper_index"])
            int(item.metadata["start_offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("CASIE item lacks graph metadata") from exc
        documents[document_id].append(item)
        hoppers[(document_id, hopper_index)].append(item)

    adjacency: dict[str, dict[str, float]] = defaultdict(dict)

    def connect(left: AnnotationItem, right: AnnotationItem, base_weight: float) -> None:
        if left.item_id == right.item_id:
            return
        distance = abs(int(left.metadata["start_offset"]) - int(right.metadata["start_offset"]))
        weight = base_weight * math.exp(-distance / distance_scale_chars)
        adjacency[left.item_id][right.item_id] = (
            adjacency[left.item_id].get(right.item_id, 0.0) + weight
        )
        adjacency[right.item_id][left.item_id] = (
            adjacency[right.item_id].get(left.item_id, 0.0) + weight
        )

    for group in hoppers.values():
        ordered = sorted(group, key=lambda item: (int(item.metadata["start_offset"]), item.item_id))
        for left, right in zip(ordered, ordered[1:]):
            connect(left, right, same_hopper_weight)
    for group in documents.values():
        ordered = sorted(group, key=lambda item: (int(item.metadata["start_offset"]), item.item_id))
        for left, right in zip(ordered, ordered[1:]):
            if int(left.metadata["hopper_index"]) != int(right.metadata["hopper_index"]):
                connect(left, right, same_document_weight)
    return {node_id: dict(neighbors) for node_id, neighbors in adjacency.items()}
