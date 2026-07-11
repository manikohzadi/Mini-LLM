"""
Abstract base class for vocabulary builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..metadata import VocabularyMetadata
from ..vocabulary import Vocabulary
from ..types import Corpus


class VocabularyBuilder(ABC):
    """
    Abstract interface for all vocabulary builders.
    """

    @abstractmethod
    def build(
        self,
        corpus: Corpus,
        *,
        metadata: VocabularyMetadata | None = None,
    ) -> Vocabulary:
        """
        Build a Vocabulary from a tokenized corpus.
        """
        raise NotImplementedError