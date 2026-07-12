"""
Vocabulary factory.
"""

from __future__ import annotations

from pathlib import Path

from .builders import VocabularyBuilder, DefaultVocabularyBuilder
from .metadata import VocabularyMetadata
from .repositories import VocabularyRepository, JSONVocabularyRepository
from .types import Corpus
from .vocabulary import Vocabulary


class VocabularyFactory:
    """
    High-level entry point for creating and loading vocabularies.
    """

    def __init__(
        self,
        *,
        builder: VocabularyBuilder | None = None,
        repository: VocabularyRepository | None = None,
    ) -> None:

        self._builder = (
            builder
            if builder is not None
            else DefaultVocabularyBuilder()
        )

        self._repository = (
            repository
            if repository is not None
            else JSONVocabularyRepository()
        )

    @property
    def builder(self) -> VocabularyBuilder:
        return self._builder

    @property
    def repository(self) -> VocabularyRepository:
        return self._repository

    def build(
        self,
        corpus: Corpus,
        *,
        metadata: VocabularyMetadata | None = None,
    ) -> Vocabulary:
        """
        Build a vocabulary from a corpus.
        """

        return self._builder.build(
            corpus,
            metadata=metadata,
        )

    def save(
        self,
        vocabulary: Vocabulary,
        path: Path,
    ) -> None:
        """
        Save a vocabulary.
        """

        self._repository.save(
            vocabulary,
            path,
        )

    def load(
        self,
        path: Path,
    ) -> Vocabulary:
        """
        Load a vocabulary.
        """

        return self._repository.load(path)