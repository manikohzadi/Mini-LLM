"""Statistics calculated from a vocabulary without mutating it."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import SPECIAL_TOKENS
from ..types import Frequency, Token
from ..vocabulary import Vocabulary


@dataclass(slots=True, frozen=True)
class VocabularyStatistics:
    """Immutable summary of vocabulary size and normal-token frequencies."""

    total_token_count: int
    special_token_count: int
    normal_token_count: int
    total_normal_token_frequency: int
    average_normal_token_frequency: float
    most_frequent_normal_token: Token | None
    least_frequent_normal_token: Token | None

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: Vocabulary,
    ) -> "VocabularyStatistics":
        """Calculate statistics from ``vocabulary``."""

        normal_frequencies: list[tuple[Token, Frequency]] = [
            (token, frequency)
            for token, frequency in vocabulary.frequencies.items()
            if token not in SPECIAL_TOKENS
        ]

        total_frequency = sum(
            frequency for _, frequency in normal_frequencies
        )
        normal_count = len(normal_frequencies)

        if normal_frequencies:
            most_frequent = max(
                normal_frequencies,
                key=lambda item: item[1],
            )[0]
            least_frequent = min(
                normal_frequencies,
                key=lambda item: item[1],
            )[0]
        else:
            most_frequent = None
            least_frequent = None

        return cls(
            total_token_count=len(vocabulary),
            special_token_count=len(SPECIAL_TOKENS),
            normal_token_count=normal_count,
            total_normal_token_frequency=total_frequency,
            average_normal_token_frequency=(
                total_frequency / normal_count
                if normal_count > 0
                else 0.0
            ),
            most_frequent_normal_token=most_frequent,
            least_frequent_normal_token=least_frequent,
        )


__all__ = ("VocabularyStatistics",)
