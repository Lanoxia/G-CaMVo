"""Validated configuration for the CaMVo router."""

from dataclasses import asdict, dataclass
from typing import Literal

from camvo.exceptions import ConfigurationError

ConfidenceMethod = Literal["exact", "beta_cdf"]


@dataclass(frozen=True, slots=True)
class CaMVoConfig:
    """Algorithm and engineering parameters.

    The explicit warm-up and numerical floors are engineering safeguards that
    are underspecified in the paper. Setting ``warmup_rounds=0`` follows the
    paper pseudocode more literally.
    """

    embedding_dim: int
    confidence_threshold: float = 0.95
    min_models: int = 3
    linucb_regularization: float = 1.0
    exploration_alpha: float = 0.25
    laplace_regularization: float = 1.0
    warmup_rounds: int = 10
    confidence_method: ConfidenceMethod = "exact"
    min_vote_weight: float = 1e-6
    probability_epsilon: float = 1e-6
    agreement_prior_successes: float = 1.0
    agreement_prior_failures: float = 1.0
    beta_min_samples_per_class: int = 3
    beta_max_concentration: float = 1e6
    max_exhaustive_models: int = 12
    update_single_model_rounds: bool = False
    allow_partial_responses: bool = False

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ConfigurationError("embedding_dim must be positive")
        if not 0 < self.confidence_threshold <= 1:
            raise ConfigurationError("confidence_threshold must be in (0, 1]")
        if self.min_models <= 0:
            raise ConfigurationError("min_models must be positive")
        if self.linucb_regularization <= 0:
            raise ConfigurationError("linucb_regularization must be positive")
        if self.exploration_alpha < 0:
            raise ConfigurationError("exploration_alpha must be non-negative")
        if self.laplace_regularization < 0:
            raise ConfigurationError("laplace_regularization must be non-negative")
        if self.warmup_rounds < 0:
            raise ConfigurationError("warmup_rounds must be non-negative")
        if self.confidence_method not in {"exact", "beta_cdf"}:
            raise ConfigurationError("unsupported confidence_method")
        if self.min_vote_weight <= 0:
            raise ConfigurationError("min_vote_weight must be positive")
        if not 0 < self.probability_epsilon < 0.5:
            raise ConfigurationError("probability_epsilon must be in (0, 0.5)")
        if self.agreement_prior_successes <= 0 or self.agreement_prior_failures <= 0:
            raise ConfigurationError("agreement priors must be positive")
        if self.beta_min_samples_per_class < 2:
            raise ConfigurationError("beta_min_samples_per_class must be at least 2")
        if self.beta_max_concentration <= 0:
            raise ConfigurationError("beta_max_concentration must be positive")
        if self.max_exhaustive_models <= 0:
            raise ConfigurationError("max_exhaustive_models must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
