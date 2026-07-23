"""Abstract interface for vocabulary serializers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..vocabulary import Vocabulary


class VocabularySerializer(ABC):
    """Convert vocabularies to/from a format and optionally files."""

    @abstractmethod
    def serialize(self, vocabulary: Vocabulary) -> str:
        """Convert a vocabulary to text."""
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, payload: str) -> Vocabulary:
        """Construct a vocabulary from serialized text."""
        raise NotImplementedError

    @abstractmethod
    def save(self, vocabulary: Vocabulary, path: Path) -> Path:
        """Serialize and write a vocabulary, returning the actual path."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> Vocabulary:
        """Read and deserialize a vocabulary file."""
        raise NotImplementedError


__all__ = ("VocabularySerializer",)
