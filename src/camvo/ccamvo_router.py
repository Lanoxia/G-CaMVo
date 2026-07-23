"""Correlation-aware CaMVo baseline from Appendix G of the NeurIPS paper.

CCaMVo keeps CaMVo's LinUCB, Bayesian calibration, vote weights, and online
updates.  It replaces only the independent-Bernoulli subset confidence with
the paper's Gaussian-copula Monte Carlo estimator (Algorithms 3 and 4).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from camvo.algorithm.oracle import OracleCandidate, OracleSelection
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.llms.base import LLMClient
from camvo.router import CaMVoRouter
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult


@dataclass(frozen=True, slots=True)
class CorrelatedCaMVoConfig:
    """Numerical controls for Appendix-G Gaussian-copula confidence."""

    monte_carlo_samples: int = 4_096
    seed: int = 17
    psd_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.monte_carlo_samples < 128:
            raise ValueError("monte_carlo_samples must be at least 128")
        if self.psd_epsilon <= 0 or not math.isfinite(self.psd_epsilon):
            raise ValueError("psd_epsilon must be finite and positive")


def nearest_psd_correlation(matrix: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Project a symmetric estimate to a positive-semidefinite correlation matrix."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not len(value):
        raise ValueError("correlation matrix must be non-empty and square")
    if not np.all(np.isfinite(value)):
        raise ValueError("correlation matrix must be finite")
    symmetric = (value + value.T) / 2.0
    np.fill_diagonal(symmetric, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.clip(eigenvalues, epsilon, None)) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(projected), epsilon, None))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) / 2.0
    np.fill_diagonal(projected, 1.0)
    return projected


def gaussian_copula_majority_confidence(
    marginals: list[float],
    weights: list[float],
    correlation: np.ndarray,
    *,
    samples: int,
    seed: int,
    psd_epsilon: float = 1e-8,
) -> float:
    """Estimate P(correct vote weight > half total weight), as in Algorithm 4."""

    probabilities = np.asarray(marginals, dtype=float)
    vote_weights = np.asarray(weights, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("marginals must be a non-empty vector")
    if vote_weights.shape != probabilities.shape:
        raise ValueError("weights must have the same shape as marginals")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("marginals must be in [0, 1]")
    if np.any(vote_weights <= 0) or not np.all(np.isfinite(vote_weights)):
        raise ValueError("weights must be finite and positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    covariance = nearest_psd_correlation(correlation, epsilon=psd_epsilon)
    if covariance.shape != (len(probabilities), len(probabilities)):
        raise ValueError("correlation dimension does not match marginals")
    normal = np.random.default_rng(seed).multivariate_normal(
        np.zeros(len(probabilities)), covariance, size=samples, check_valid="ignore"
    )
    # U=Phi(Z)<p is exactly equivalent to Z<Phi^-1(p).  Thresholding in
    # Gaussian space avoids a scipy dependency and is much faster than
    # applying a scalar CDF to every Monte Carlo draw.
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    thresholds = np.asarray([NormalDist().inv_cdf(float(value)) for value in clipped])
    correctness = normal < thresholds
    correct_weight = correctness @ vote_weights
    return float(np.mean(correct_weight > vote_weights.sum() / 2.0))


class OnlineCorrelationEstimator:
    """Paper Algorithm 3 with Welford-style marginal and pair updates."""

    def __init__(self, model_ids: list[str] | tuple[str, ...]) -> None:
        ordered = tuple(sorted(model_ids))
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("model_ids must be non-empty and unique")
        self.model_ids = ordered
        self._index = {model_id: index for index, model_id in enumerate(ordered)}
        size = len(ordered)
        self.counts = np.zeros(size, dtype=np.int64)
        self.means = np.zeros(size, dtype=float)
        self.m2 = np.zeros(size, dtype=float)
        self.pair_counts = np.zeros((size, size), dtype=np.int64)
        self.pair_deviation = np.zeros((size, size), dtype=float)
        self.correlation = np.eye(size, dtype=float)

    def update(self, rewards: Mapping[str, float]) -> None:
        selected = sorted(rewards)
        if any(model_id not in self._index for model_id in selected):
            raise ValueError("reward contains an unknown model")
        for model_id in selected:
            reward = float(rewards[model_id])
            if reward not in (0.0, 1.0):
                raise ValueError("CCaMVo correctness rewards must be binary")
            index = self._index[model_id]
            old_mean = self.means[index]
            self.counts[index] += 1
            delta = reward - old_mean
            self.means[index] = old_mean + delta / self.counts[index]
            self.m2[index] += (reward - self.means[index]) * (reward - old_mean)

        for left_pos, left_id in enumerate(selected):
            for right_id in selected[left_pos + 1 :]:
                left = self._index[left_id]
                right = self._index[right_id]
                left_delta = float(rewards[left_id]) - self.means[left]
                right_delta = float(rewards[right_id]) - self.means[right]
                self.pair_deviation[left, right] += left_delta * right_delta
                self.pair_deviation[right, left] = self.pair_deviation[left, right]
                self.pair_counts[left, right] += 1
                self.pair_counts[right, left] = self.pair_counts[left, right]
                pair_count = self.pair_counts[left, right]
                if pair_count <= 1 or self.counts[left] <= 1 or self.counts[right] <= 1:
                    continue
                left_std = math.sqrt(self.m2[left] / (self.counts[left] - 1))
                right_std = math.sqrt(self.m2[right] / (self.counts[right] - 1))
                if left_std <= 0 or right_std <= 0:
                    continue
                covariance = self.pair_deviation[left, right] / (pair_count - 1)
                rho = float(np.clip(covariance / (left_std * right_std), -1.0, 1.0))
                self.correlation[left, right] = rho
                self.correlation[right, left] = rho

    def matrix_for(self, model_ids: tuple[str, ...]) -> np.ndarray:
        indices = [self._index[model_id] for model_id in model_ids]
        return self.correlation[np.ix_(indices, indices)].copy()


class CorrelatedCaMVoRouter(CaMVoRouter):
    """Faithful CCaMVo comparison baseline with deterministic Monte Carlo."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
        correlated_config: CorrelatedCaMVoConfig | None = None,
    ) -> None:
        super().__init__(models, embedder, config)
        self.correlated_config = correlated_config or CorrelatedCaMVoConfig()
        self.correlation_estimator = OnlineCorrelationEstimator(tuple(self.models))

    def _subset_confidence(self, subset: tuple[OracleCandidate, ...], round_index: int) -> float:
        model_ids = tuple(candidate.model_id for candidate in subset)
        # Stable subset-specific seed makes exhaustive search reproducible while
        # still refreshing Monte Carlo draws across online rounds.
        subset_code = sum(
            (position + 1) * sum(model_id.encode("utf-8"))
            for position, model_id in enumerate(model_ids)
        )
        return gaussian_copula_majority_confidence(
            [candidate.lower_bound for candidate in subset],
            [candidate.vote_weight for candidate in subset],
            self.correlation_estimator.matrix_for(model_ids),
            samples=self.correlated_config.monte_carlo_samples,
            seed=self.correlated_config.seed + 1_000_003 * round_index + subset_code,
            psd_epsilon=self.correlated_config.psd_epsilon,
        )

    def _selection(self, scores: dict[str, ModelScore], warmup: bool) -> OracleSelection:
        ordered = tuple(
            OracleCandidate(
                model_id=model_id,
                cost=score.estimated_cost,
                lower_bound=self._selection_lower_bound(score),
                vote_weight=score.vote_weight,
            )
            for model_id, score in sorted(scores.items())
        )
        if warmup:
            confidence = self._subset_confidence(ordered, self.round_index + 1)
            return OracleSelection(
                model_ids=tuple(candidate.model_id for candidate in ordered),
                confidence=confidence,
                cost=sum(candidate.cost for candidate in ordered),
                feasible=confidence >= self.config.confidence_threshold,
            )
        feasible: list[OracleSelection] = []
        for subset_size in range(self.config.min_models, len(ordered) + 1):
            for subset in itertools.combinations(ordered, subset_size):
                confidence = self._subset_confidence(subset, self.round_index + 1)
                cost = sum(candidate.cost for candidate in subset)
                if confidence >= self.config.confidence_threshold:
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
                key=lambda value: (value.cost, len(value.model_ids), value.model_ids),
            )
        confidence = self._subset_confidence(ordered, self.round_index + 1)
        return OracleSelection(
            model_ids=tuple(candidate.model_id for candidate in ordered),
            confidence=confidence,
            cost=sum(candidate.cost for candidate in ordered),
            feasible=False,
        )

    def route(self, item: AnnotationItem) -> RoutingResult:
        result = super().route(item)
        if len(result.selected_models) > 1:
            self.correlation_estimator.update(
                {
                    model_id: float(result.responses[model_id] == result.label)
                    for model_id in result.selected_models
                }
            )
        return result

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        super().observe_complete_feedback(item, responses, gold_label)
        self.correlation_estimator.update(
            {
                model_id: float(response.label == gold_label)
                for model_id, response in responses.items()
            }
        )
