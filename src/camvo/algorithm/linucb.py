"""Numerically stable LinUCB-style per-model estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LinUCBScore:
    prediction: float
    uncertainty: float
    lower_confidence_score: float


class LinUCBArm:
    """Ridge-regression estimator updated with Sherman-Morrison."""

    def __init__(
        self,
        dimension: int,
        *,
        regularization: float = 1.0,
        exploration_alpha: float = 0.25,
        probability_epsilon: float = 1e-6,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        if exploration_alpha < 0:
            raise ValueError("exploration_alpha must be non-negative")
        if not 0 < probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be in (0, 0.5)")
        self.dimension = dimension
        self.regularization = regularization
        self.exploration_alpha = exploration_alpha
        self.probability_epsilon = probability_epsilon
        self.A = regularization * np.eye(dimension, dtype=np.float64)
        self.A_inv = np.eye(dimension, dtype=np.float64) / regularization
        self.b = np.zeros(dimension, dtype=np.float64)
        self.updates = 0

    def _validate_context(self, context: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = np.asarray(context, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"context shape {vector.shape} does not match ({self.dimension},)"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("context contains non-finite values")
        return vector

    def score(self, context: NDArray[np.float64]) -> LinUCBScore:
        vector = self._validate_context(context)
        theta_hat = self.A_inv @ self.b
        raw_prediction = float(vector @ theta_hat)
        prediction = float(np.clip(raw_prediction, 0.0, 1.0))
        variance_proxy = max(0.0, float(vector @ self.A_inv @ vector))
        uncertainty = self.exploration_alpha * float(np.sqrt(variance_proxy))
        lower = float(
            np.clip(
                raw_prediction - uncertainty,
                self.probability_epsilon,
                1.0 - self.probability_epsilon,
            )
        )
        return LinUCBScore(prediction, uncertainty, lower)

    def update(self, context: NDArray[np.float64], reward: float) -> None:
        vector = self._validate_context(context)
        if not np.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("reward must be a finite value in [0, 1]")

        projected = self.A_inv @ vector
        denominator = 1.0 + float(vector @ projected)
        if denominator <= 0 or not np.isfinite(denominator):
            raise FloatingPointError("invalid Sherman-Morrison denominator")

        self.A += np.outer(vector, vector)
        self.A_inv -= np.outer(projected, projected) / denominator
        self.A_inv = (self.A_inv + self.A_inv.T) / 2.0
        self.b += reward * vector
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "regularization": self.regularization,
            "exploration_alpha": self.exploration_alpha,
            "probability_epsilon": self.probability_epsilon,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["dimension"]) != self.dimension:
            raise ValueError("checkpoint LinUCB dimension mismatch")
        A = np.asarray(state["A"], dtype=np.float64)
        A_inv = np.asarray(state["A_inv"], dtype=np.float64)
        b = np.asarray(state["b"], dtype=np.float64)
        if A.shape != (self.dimension, self.dimension):
            raise ValueError("checkpoint A has an invalid shape")
        if A_inv.shape != A.shape or b.shape != (self.dimension,):
            raise ValueError("checkpoint LinUCB arrays have invalid shapes")
        if not all(np.all(np.isfinite(array)) for array in (A, A_inv, b)):
            raise ValueError("checkpoint LinUCB arrays contain non-finite values")
        self.A = A
        self.A_inv = A_inv
        self.b = b
        self.updates = int(state["updates"])

