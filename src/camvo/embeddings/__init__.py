"""Embedding provider interfaces and built-in implementations."""

from camvo.embeddings.base import EmbeddingProvider
from camvo.embeddings.hashing import HashingTextEmbedder

__all__ = ["EmbeddingProvider", "HashingTextEmbedder"]

