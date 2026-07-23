"""Deterministic dependency-light text embeddings for tests and simulation."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from numpy.typing import NDArray

from camvo.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


class HashingTextEmbedder(EmbeddingProvider):
    """Signed feature hashing with L2 normalization.

    This is not a semantic model. It is intentionally deterministic and fast,
    making it suitable for unit tests, offline demos, and CI environments.
    """

    def __init__(self, dimension: int = 128, *, lowercase: bool = True) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._lowercase = lowercase

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> NDArray[np.float64]:
        if not text.strip():
            raise ValueError("cannot embed empty text")
        normalized = text.lower() if self._lowercase else text
        tokens = _TOKEN_RE.findall(normalized)
        vector = np.zeros(self._dimension, dtype=np.float64)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("text produced an empty embedding")
        return vector / norm

