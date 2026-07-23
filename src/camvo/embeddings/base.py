"""Interface for converting input text into a fixed-size context vector."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """A replaceable text embedding backend."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the exact vector dimension."""

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float64]:
        """Return one finite, one-dimensional vector."""

