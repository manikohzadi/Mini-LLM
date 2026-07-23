"""Word-level vocabulary builder."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

from .._internal.validation import VocabularyValidator
from ..constants import (
    DEFAULT_MAX_NORMAL_TOKEN_COUNT,
    DEFAULT_MIN_FREQUENCY,
    SPECIAL_TOKENS,
)
from ..exceptions import EmptyCorpusError
from ..metadata import VocabularyMetadata
from ..types import Corpus, Frequency, Token
from ..vocabulary import Vocabulary
from .base import VocabularyBuilder


class WordVocabularyBuilder(VocabularyBuilder):
    """
    Build a frozen word vocabulary from a tokenized corpus.

    ``max_normal_token_count`` limits only normal corpus tokens. Reserved
    special tokens are always present and are not counted against this limit.
    """

    def __init__(
        self,
        *,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        max_normal_token_count: int = DEFAULT_MAX_NORMAL_TOKEN_COUNT,
        require_non_empty: bool = False,
    ) -> None:
        if isinstance(min_frequency, bool) or not isinstance(
            min_frequency,
            int,
        ):
            raise TypeError("min_frequency must be an integer.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

        if isinstance(max_normal_token_count, bool) or not isinstance(
            max_normal_token_count,
            int,
        ):
            raise TypeError("max_normal_token_count must be an integer.")
        if max_normal_token_count < 0:
            raise ValueError("max_normal_token_count cannot be negative.")

        if not isinstance(require_non_empty, bool):
            raise TypeError("require_non_empty must be a boolean.")

        self._min_frequency = min_frequency
        self._max_normal_token_count = max_normal_token_count
        self._require_non_empty = require_non_empty

    @property
    def min_frequency(self) -> int:
        return self._min_frequency

    @property
    def max_normal_token_count(self) -> int:
        return self._max_normal_token_count

    @property
    def require_non_empty(self) -> bool:
        """Return whether an empty corpus must be rejected."""

        return self._require_non_empty

    def build(
        self,
        corpus: Corpus,
        *,
        metadata: VocabularyMetadata | None = None,
    ) -> Vocabulary:
        """Count, filter, add, and freeze corpus tokens."""

        vocabulary = Vocabulary.empty(metadata=metadata)
        counter = self._count_tokens(corpus)

        if self._require_non_empty and not counter:
            raise EmptyCorpusError(
                "A non-empty corpus is required by this builder."
            )

        for token, frequency in self._select_tokens(counter):
            vocabulary.add_token(token, frequency=frequency)

        vocabulary.freeze()
        return vocabulary

    def _count_tokens(self, corpus: Corpus) -> Counter[Token]:
        """Count token frequencies across all sequences in ``corpus``."""

        counter: Counter[Token] = Counter()
        for sequence in corpus:
            for token in sequence:
                VocabularyValidator.validate_token(token)
                counter[token] += 1
        return counter

    def _select_tokens(
        self,
        counter: Counter[Token],
    ) -> Iterator[tuple[Token, Frequency]]:
        """Yield eligible tokens in descending frequency order."""

        added = 0
        for token, frequency in counter.most_common():
            if token in SPECIAL_TOKENS:
                continue
            if frequency < self._min_frequency:
                continue
            if added >= self._max_normal_token_count:
                break

            yield token, frequency
            added += 1


__all__ = ("WordVocabularyBuilder",)
