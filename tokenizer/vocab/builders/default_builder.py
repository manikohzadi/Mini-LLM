"""
Default implementation of the vocabulary builder.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

from ..constants import (
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_MIN_FREQUENCY,
    SPECIAL_TOKENS,
)
from ..metadata import VocabularyMetadata
from ..types import (
    Corpus,
    Frequency,
    Token,
)
from ..vocabulary import Vocabulary

from .base import VocabularyBuilder


class DefaultVocabularyBuilder(VocabularyBuilder):
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

        self._min_frequency = min_frequency

        self._max_vocabulary_size = max_vocabulary_size

    def build(
        self,
        corpus: Corpus,
        *,
        metadata: VocabularyMetadata | None = None,
    ) -> Vocabulary:
        """
        Build a vocabulary from a tokenized corpus.
        """

        vocabulary = Vocabulary.empty(
            metadata=metadata,
        )

        counter = self._count_tokens(corpus)

        for token, frequency in self._select_tokens(counter):
            vocabulary.add_token(
                token,
                frequency=frequency,
            )

        vocabulary.freeze()

        return vocabulary
    
    def _count_tokens(
        self,
        corpus: Corpus,
    ) -> Counter[Token]:
        """
        Count token frequencies.
        """

        counter: Counter[Token] = Counter()

        for sentence in corpus:
            counter.update(sentence)

        return counter
    
    def _select_tokens(
        self,
        counter: Counter[Token],
    ) -> Iterator[tuple[Token, Frequency]]:
        """
        Yield valid vocabulary tokens.
        """

        if self._max_vocabulary_size <= 0:
            return

        added = 0

        for token, frequency in counter.most_common():

            if token in SPECIAL_TOKENS:
                continue

            if frequency < self._min_frequency:
                continue

            if added >= self._max_vocabulary_size:
                break

            yield token, frequency

            added += 1
