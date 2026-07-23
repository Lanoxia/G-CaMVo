"""Majority-vote confidence estimators from Lemma 2.1."""

from __future__ import annotations

import itertools

import numpy as np

from camvo.mathutils import regularized_beta_cdf


def _validated_arrays(
    lower_bounds: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(lower_bounds, dtype=np.float64)
    vote_weights = np.asarray(weights, dtype=np.float64)
    if probabilities.ndim != 1 or vote_weights.ndim != 1:
        raise ValueError("lower_bounds and weights must be one-dimensional")
    if probabilities.size == 0 or probabilities.shape != vote_weights.shape:
        raise ValueError("lower_bounds and weights must have the same non-zero length")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(vote_weights)):
        raise ValueError("confidence inputs must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("lower bounds must be in [0, 1]")
    if np.any(vote_weights < 0) or float(vote_weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return probabilities, vote_weights


def exact_majority_confidence(
    lower_bounds: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
) -> float:
    """Compute Lemma 2.1 exactly by enumerating correctness outcomes."""

    probabilities, vote_weights = _validated_arrays(lower_bounds, weights)
    half_weight = float(vote_weights.sum()) / 2.0
    confidence = 0.0
    for outcome in itertools.product((0, 1), repeat=probabilities.size):
        mask = np.asarray(outcome, dtype=bool)
        if float(vote_weights[mask].sum()) <= half_weight:
            continue
        joint_probability = float(
            np.prod(np.where(mask, probabilities, 1.0 - probabilities))
        )
        confidence += joint_probability
    return float(np.clip(confidence, 0.0, 1.0))


def beta_cdf_confidence(
    lower_bounds: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
    *,
    epsilon: float = 1e-9,
    normalize_weights: bool = False,
) -> float:
    """Paper's Beta-CDF approximation to Lemma 2.1.

    ``normalize_weights=False`` reproduces the paper formula. Normalization is
    available for experiments but changes the approximation's concentration.
    """

    probabilities, vote_weights = _validated_arrays(lower_bounds, weights)
    if normalize_weights:
        vote_weights = vote_weights / vote_weights.sum()
    total_weight = float(vote_weights.sum())
    correct_weight = float(vote_weights @ probabilities)
    alpha = max(epsilon, correct_weight)
    beta = max(epsilon, total_weight - correct_weight)
    confidence = 1.0 - regularized_beta_cdf(0.5, alpha, beta)
    if not np.isfinite(confidence):
        raise FloatingPointError("Beta-CDF confidence is not finite")
    return float(np.clip(confidence, 0.0, 1.0))
