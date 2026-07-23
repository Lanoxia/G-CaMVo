"""Optional sentence-transformers adapter, loaded only when requested."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from camvo.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Adapter for the paper's ``all-MiniLM-L6-v2`` context encoder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional dependency with "
                "`pip install -e '.[sentence-transformers]'`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> NDArray[np.float64]:
        vector = np.asarray(
            self._model.encode(text, normalize_embeddings=True), dtype=np.float64
        )
        if vector.shape != (self._dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("embedding backend returned an invalid vector")
        return vector

