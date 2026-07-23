"""Laplacian confidence smoothing used by the G-CaMVo extension."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class WeightedGraphEdge:
    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("edge endpoints must not be empty")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("edge weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GraphRegularizationResult:
    raw_scores: dict[str, float]
    smoothed_scores: dict[str, float]
    objective_before: float
    objective_after: float
    solver: str
    iterations: int


def _objective(
    values: np.ndarray,
    raw: np.ndarray,
    fidelity: np.ndarray,
    edges: list[tuple[int, int, float]],
    regularization: float,
) -> float:
    value = float(np.dot(fidelity, np.square(values - raw)))
    value += regularization * sum(
        weight * float((values[left] - values[right]) ** 2)
        for left, right, weight in edges
    )
    return value


def laplacian_smooth_scores(
    raw_scores: Mapping[str, float],
    edges: Iterable[WeightedGraphEdge],
    *,
    regularization: float = 1.0,
    fidelity_weights: Mapping[str, float] | None = None,
    solver: str = "auto",
    direct_max_nodes: int = 2_000,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> GraphRegularizationResult:
    r"""Solve the graph-regularized least-squares objective exactly.

    The optimization is

    ``min_f sum_v a_v (f_v-y_v)^2 + lambda sum_(u,v) w_uv (f_u-f_v)^2``.

    Its unique solution is ``(A + lambda L)^-1 A y`` when every fidelity
    weight is positive. The implementation combines duplicate undirected
    edges and treats isolated nodes as identity, which is important for an
    online graph whose newest alert may not yet have neighbors.
    """

    if not raw_scores:
        raise ValueError("raw_scores must not be empty")
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    if solver not in {"auto", "direct", "cg"}:
        raise ValueError("solver must be 'auto', 'direct', or 'cg'")
    if direct_max_nodes <= 0 or max_iterations <= 0:
        raise ValueError("direct_max_nodes and max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    node_ids = tuple(sorted(raw_scores))
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    raw = np.asarray([float(raw_scores[node_id]) for node_id in node_ids], dtype=float)
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("raw scores must be finite probabilities in [0, 1]")

    if fidelity_weights is None:
        fidelity = np.ones(len(node_ids), dtype=float)
    else:
        if set(fidelity_weights) != set(node_ids):
            raise ValueError("fidelity_weights must have exactly the raw-score node IDs")
        fidelity = np.asarray([float(fidelity_weights[node_id]) for node_id in node_ids])
        if not np.all(np.isfinite(fidelity)) or np.any(fidelity <= 0):
            raise ValueError("fidelity weights must be finite and strictly positive")

    combined: dict[tuple[int, int], float] = {}
    for edge in edges:
        if edge.source not in index or edge.target not in index:
            raise ValueError("edge endpoint is absent from raw_scores")
        if edge.source == edge.target or edge.weight == 0:
            continue
        left, right = sorted((index[edge.source], index[edge.target]))
        combined[(left, right)] = combined.get((left, right), 0.0) + edge.weight
    indexed_edges = [
        (left, right, weight) for (left, right), weight in sorted(combined.items())
    ]

    target = fidelity * raw
    selected_solver = solver
    if selected_solver == "auto":
        selected_solver = "direct" if len(node_ids) <= direct_max_nodes else "cg"
    iterations = 1
    if selected_solver == "direct":
        laplacian = np.zeros((len(node_ids), len(node_ids)), dtype=float)
        for left, right, weight in indexed_edges:
            laplacian[left, left] += weight
            laplacian[right, right] += weight
            laplacian[left, right] -= weight
            laplacian[right, left] -= weight
        system = np.diag(fidelity) + regularization * laplacian
        try:
            smoothed = np.linalg.solve(system, target)
        except np.linalg.LinAlgError as exc:  # positive fidelity should make this SPD
            raise ValueError("graph regularization system is singular") from exc
    else:
        smoothed, iterations = _preconditioned_conjugate_gradient(
            raw,
            target,
            fidelity,
            indexed_edges,
            regularization=regularization,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    smoothed = np.clip(smoothed, 0.0, 1.0)
    before = _objective(raw, raw, fidelity, indexed_edges, regularization)
    after = _objective(smoothed, raw, fidelity, indexed_edges, regularization)
    objective_tolerance = max(1e-9, abs(before) * 1e-7)
    if after > before + objective_tolerance:
        raise RuntimeError("graph smoother increased its optimization objective")
    return GraphRegularizationResult(
        raw_scores={node_id: float(raw[index[node_id]]) for node_id in node_ids},
        smoothed_scores={node_id: float(smoothed[index[node_id]]) for node_id in node_ids},
        objective_before=before,
        objective_after=after,
        solver=selected_solver,
        iterations=iterations,
    )


def _preconditioned_conjugate_gradient(
    initial: np.ndarray,
    target: np.ndarray,
    fidelity: np.ndarray,
    edges: list[tuple[int, int, float]],
    *,
    regularization: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    """Solve ``(A + lambda L)x=b`` using only O(|V|+|E|) storage."""

    if edges:
        left = np.fromiter((edge[0] for edge in edges), dtype=np.int64)
        right = np.fromiter((edge[1] for edge in edges), dtype=np.int64)
        weight = np.fromiter((edge[2] for edge in edges), dtype=float)
    else:
        left = right = np.asarray([], dtype=np.int64)
        weight = np.asarray([], dtype=float)
    diagonal = fidelity.copy()
    if len(weight):
        np.add.at(diagonal, left, regularization * weight)
        np.add.at(diagonal, right, regularization * weight)

    def matrix_vector(value: np.ndarray) -> np.ndarray:
        output = fidelity * value
        if len(weight):
            contribution = regularization * weight * (value[left] - value[right])
            np.add.at(output, left, contribution)
            np.add.at(output, right, -contribution)
        return output

    estimate = initial.copy()
    residual = target - matrix_vector(estimate)
    target_norm = max(float(np.linalg.norm(target)), 1.0)
    if float(np.linalg.norm(residual)) <= tolerance * target_norm:
        return estimate, 0
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    residual_dot = float(np.dot(residual, preconditioned))
    for iteration in range(1, max_iterations + 1):
        product = matrix_vector(direction)
        denominator = float(np.dot(direction, product))
        if not math.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("conjugate-gradient solver lost positive definiteness")
        step = residual_dot / denominator
        estimate += step * direction
        residual -= step * product
        if float(np.linalg.norm(residual)) <= tolerance * target_norm:
            return estimate, iteration
        preconditioned = residual / diagonal
        next_residual_dot = float(np.dot(residual, preconditioned))
        direction = preconditioned + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot
    raise RuntimeError(
        f"conjugate-gradient solver did not converge within {max_iterations} iterations"
    )
