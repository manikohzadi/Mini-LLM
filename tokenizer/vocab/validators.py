"""
Vocabulary validators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._internal.validation import VocabularyValidator as _VocabularyValidator


class VocabularyValidator:
    """
    Collection of static validation methods.
    """

    @staticmethod
    def validate_token(token: str) -> None:
        """
        Validate a token.
        """

        _VocabularyValidator.validate_token(token)

    @staticmethod
    def validate_token_id(token_id: int) -> None:
        """
        Validate a token ID.
        """

        _VocabularyValidator.validate_token_id(token_id)

    @staticmethod
    def validate_special_tokens(
        token_to_id: dict[str, int],
    ) -> None:
        """
        Ensure every special token exists.
        """

        _VocabularyValidator.validate_special_tokens(token_to_id)

    @staticmethod
    def validate_frequency(frequency: object) -> None:
        """Validate the shared non-negative frequency representation."""

        _VocabularyValidator.validate_frequency(frequency)

    @staticmethod
    def validate_normal_frequency(frequency: object) -> None:
        """Require a strictly positive frequency for a normal token."""

        _VocabularyValidator.validate_normal_frequency(frequency)

    @staticmethod
    def validate_vocabulary_state(
        token_to_id: Mapping[Any, Any],
        id_to_token: Sequence[Any],
        frequencies: Mapping[Any, Any],
    ) -> None:
        """Validate complete vocabulary state through the central rules."""

        _VocabularyValidator.validate_vocabulary_state(
            token_to_id,
            id_to_token,
            frequencies,
        )


__all__ = ("VocabularyValidator",)
