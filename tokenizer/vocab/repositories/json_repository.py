"""
JSON repository implementation.
"""

from __future__ import annotations

from pathlib import Path

from ..serializers import JSONVocabularySerializer
from ..vocabulary import Vocabulary

from .base import VocabularyRepository


class JSONVocabularyRepository(
    VocabularyRepository,
):
    """
    Store vocabularies as JSON.
    """

    def __init__(
        self,
        serializer: JSONVocabularySerializer | None = None,
    ) -> None:
        self._serializer = serializer or JSONVocabularySerializer()

    @property
    def serializer(self) -> JSONVocabularySerializer:
        """Return the serializer used by this compatibility repository."""

        return self._serializer

    def save(
        self,
        vocabulary: Vocabulary,
        path: Path,
    ) -> None:
        """
        Save a vocabulary as JSON.
        """

        self._serializer.save(vocabulary, path)

    def load(
        self,
        path: Path,
    ) -> Vocabulary:
        """
        Load a vocabulary from JSON.
        """

        return self._serializer.load(path)
