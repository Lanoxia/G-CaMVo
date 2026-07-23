"""Domain-specific exceptions raised by CaMVo."""


class CaMVoError(Exception):
    """Base exception for the package."""


class ConfigurationError(CaMVoError, ValueError):
    """Raised when a configuration violates an algorithm invariant."""


class ModelInvocationError(CaMVoError):
    """Raised when a model cannot return a valid response."""


class InsufficientResponsesError(CaMVoError):
    """Raised when too few selected models return valid labels."""


class CheckpointError(CaMVoError):
    """Raised when a checkpoint is incompatible or malformed."""


class BudgetExceededError(CaMVoError):
    """Raised before a provider call would exceed an experiment limit."""


class BudgetLedgerError(CaMVoError):
    """Raised when persisted budget state is malformed or incompatible."""


class ResponseCacheError(CaMVoError):
    """Raised when a persisted provider response is corrupt or incompatible."""
