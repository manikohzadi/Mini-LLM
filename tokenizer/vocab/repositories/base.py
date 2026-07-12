"""
Abstract repository interface for vocabularies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..vocabulary import Vocabulary


class VocabularyRepository(ABC):
    """
    Abstract persistence interface for vocabularies.
    """

    @abstractmethod
    def save(
        self,
        vocabulary: Vocabulary,
        path: Path,
    ) -> None:
        """
        Persist a vocabulary.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: Path,
    ) -> Vocabulary:
        """
        Load a vocabulary.
        """
        raise NotImplementedError