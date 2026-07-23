"""Deterministic weighted label aggregation."""

from __future__ import annotations

import math


def weighted_vote(
    responses: dict[str, str],
    weights: dict[str, float],
    label_order: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Return the winning label and all labels tied at the maximum.

    Ties are resolved by ``label_order`` to keep checkpoints and tests fully
    reproducible.
    """

    if not responses:
        raise ValueError("at least one response is required")
    totals = {label: 0.0 for label in label_order}
    for model_id, label in responses.items():
        if label not in totals:
            raise ValueError(f"model {model_id!r} returned unknown label {label!r}")
        if model_id not in weights:
            raise ValueError(f"missing vote weight for model {model_id!r}")
        weight = weights[model_id]
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("vote weights must be finite and non-negative")
        totals[label] += weight

    maximum = max(totals.values())
    tied = tuple(label for label in label_order if math.isclose(totals[label], maximum))
    return tied[0], tied

