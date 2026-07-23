"""Exact cost-minimizing subset search for small and medium model pools."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OracleCandidate:
    model_id: str
    cost: float
    lower_bound: float
    vote_weight: float


@dataclass(frozen=True, slots=True)
class OracleSelection:
    model_ids: tuple[str, ...]
    confidence: float
    cost: float
    feasible: bool


class ExhaustiveSubsetOracle:
    """Find the global minimum-cost feasible subset by complete enumeration."""

    def __init__(
        self,
        confidence_function: Callable[[list[float], list[float]], float],
        *,
        max_models: int = 20,
    ) -> None:
        self._confidence_function = confidence_function
        self._max_models = max_models

    def select(
        self,
        candidates: list[OracleCandidate],
        *,
        threshold: float,
        min_models: int,
    ) -> OracleSelection:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if not candidates:
            raise ValueError("at least one candidate is required")
        if len(candidates) > self._max_models:
            raise ValueError(
                f"exhaustive search is limited to {self._max_models} models; "
                "provide a scalable Oracle implementation for larger pools"
            )
        if not 1 <= min_models <= len(candidates):
            raise ValueError("min_models must be between 1 and the pool size")
        if len({candidate.model_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate model ids must be unique")

        ordered = sorted(candidates, key=lambda candidate: candidate.model_id)
        feasible: list[OracleSelection] = []
        for subset_size in range(min_models, len(ordered) + 1):
            for subset in itertools.combinations(ordered, subset_size):
                confidence = self._confidence_function(
                    [candidate.lower_bound for candidate in subset],
                    [candidate.vote_weight for candidate in subset],
                )
                cost = sum(candidate.cost for candidate in subset)
                if confidence >= threshold:
                    feasible.append(
                        OracleSelection(
                            model_ids=tuple(candidate.model_id for candidate in subset),
                            confidence=confidence,
                            cost=cost,
                            feasible=True,
                        )
                    )

        if feasible:
            return min(
                feasible,
                key=lambda selection: (
                    selection.cost,
                    len(selection.model_ids),
                    selection.model_ids,
                ),
            )

        all_confidence = self._confidence_function(
            [candidate.lower_bound for candidate in ordered],
            [candidate.vote_weight for candidate in ordered],
        )
        return OracleSelection(
            model_ids=tuple(candidate.model_id for candidate in ordered),
            confidence=all_confidence,
            cost=sum(candidate.cost for candidate in ordered),
            feasible=False,
        )

