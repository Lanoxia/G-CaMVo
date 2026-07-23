"""Beta-mixture score calibration and Laplace smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from camvo.mathutils import beta_log_pdf


@dataclass(slots=True)
class RunningMoments:
    """Stable online population moments using Welford's recurrence."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("moment value must be finite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    def state_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RunningMoments:
        return cls(count=int(state["count"]), mean=float(state["mean"]), m2=float(state["m2"]))


class BetaMixtureCalibrator:
    """Estimate P(agreement | score) with two conditional Beta densities."""

    def __init__(
        self,
        *,
        epsilon: float = 1e-6,
        min_samples_per_class: int = 3,
        max_concentration: float = 1e6,
    ) -> None:
        if not 0 < epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")
        if min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2")
        if max_concentration <= 0:
            raise ValueError("max_concentration must be positive")
        self.epsilon = epsilon
        self.min_samples_per_class = min_samples_per_class
        self.max_concentration = max_concentration
        self.match = RunningMoments()
        self.mismatch = RunningMoments()

    @property
    def ready(self) -> bool:
        return (
            self.match.count >= self.min_samples_per_class
            and self.mismatch.count >= self.min_samples_per_class
        )

    def update(self, score: float, matched: bool) -> None:
        clipped = float(np.clip(score, self.epsilon, 1.0 - self.epsilon))
        (self.match if matched else self.mismatch).update(clipped)

    def _shape_parameters(self, moments: RunningMoments) -> tuple[float, float]:
        mean = float(np.clip(moments.mean, self.epsilon, 1.0 - self.epsilon))
        variance = max(moments.variance, self.epsilon**2)
        concentration = mean * (1.0 - mean) / variance - 1.0
        concentration = float(np.clip(concentration, self.epsilon, self.max_concentration))
        alpha = max(self.epsilon, mean * concentration)
        beta = max(self.epsilon, (1.0 - mean) * concentration)
        return alpha, beta

    def posterior(self, score: float, prior_match_probability: float) -> float:
        q = float(np.clip(score, self.epsilon, 1.0 - self.epsilon))
        prior = float(
            np.clip(prior_match_probability, self.epsilon, 1.0 - self.epsilon)
        )
        if not self.ready:
            return q

        match_alpha, match_beta = self._shape_parameters(self.match)
        mismatch_alpha, mismatch_beta = self._shape_parameters(self.mismatch)
        log_match = np.log(prior) + beta_log_pdf(q, match_alpha, match_beta)
        log_mismatch = np.log1p(-prior) + beta_log_pdf(q, mismatch_alpha, mismatch_beta)
        log_denominator = np.logaddexp(log_match, log_mismatch)
        posterior = float(np.exp(log_match - log_denominator))
        if not np.isfinite(posterior):
            return q
        return float(np.clip(posterior, self.epsilon, 1.0 - self.epsilon))

    def state_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "min_samples_per_class": self.min_samples_per_class,
            "max_concentration": self.max_concentration,
            "match": self.match.state_dict(),
            "mismatch": self.mismatch.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.match = RunningMoments.from_state_dict(state["match"])
        self.mismatch = RunningMoments.from_state_dict(state["mismatch"])


def laplace_smooth(
    estimate: float,
    observations: int,
    round_index: int,
    regularization: float,
) -> float:
    """Paper Eq. (3), shrinking sparse estimates toward an uninformative 0.5."""

    if observations < 0 or round_index <= 0 or regularization < 0:
        raise ValueError("invalid smoothing arguments")
    estimate = float(np.clip(estimate, 0.0, 1.0))
    prior_mass = regularization * float(np.log(round_index + 1))
    denominator = observations + prior_mass
    if denominator == 0:
        return 0.5
    return float((estimate * observations + prior_mass * 0.5) / denominator)
