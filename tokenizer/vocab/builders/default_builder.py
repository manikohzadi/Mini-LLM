"""
Default implementation of the vocabulary builder.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

from ..constants import (
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_MIN_FREQUENCY,
)
from ..metadata import VocabularyMetadata
from ..types import (
    Corpus,
    Frequency,
    Token,
)
from ..vocabulary import Vocabulary

from .word import WordVocabularyBuilder


class DefaultVocabularyBuilder(WordVocabularyBuilder):
    """
    Standard vocabulary builder.

    Parameters
    ----------
    min_frequency:
        Ignore tokens occurring fewer than this number.

    max_vocabulary_size:
        Maximum number of normal vocabulary tokens.
        Special tokens are always preserved.
    """

    def __init__(
        self,
        *,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        max_vocabulary_size: int = DEFAULT_MAX_VOCAB_SIZE,
    ) -> None:

        super().__init__(
            min_frequency=min_frequency,
            max_normal_token_count=max_vocabulary_size,
        )

        self._max_vocabulary_size = max_vocabulary_size

    @property
    def max_vocabulary_size(self) -> int:
        """Return the compatibility name for the normal token limit."""

        return self._max_vocabulary_size

    def build(
        self,
        corpus: Corpus,
        *,
        metadata: VocabularyMetadata | None = None,
    ) -> Vocabulary:
        """
        Build a vocabulary from a tokenized corpus.
        """

        return super().build(
            corpus,
            metadata=metadata,
        )
    
    def _count_tokens(
        self,
        corpus: Corpus,
    ) -> Counter[Token]:
        """
        Count token frequencies.
        """

        return super()._count_tokens(corpus)
    
    def _select_tokens(
        self,
        counter: Counter[Token],
    ) -> Iterator[tuple[Token, Frequency]]:
        """
        Yield valid vocabulary tokens.
        """

        yield from super()._select_tokens(counter)
