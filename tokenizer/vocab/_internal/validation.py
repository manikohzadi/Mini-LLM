"""Private validation utilities for vocabulary data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..constants import SPECIAL_TOKEN_ID_PAIRS, SPECIAL_TOKENS
from ..exceptions import (
    InvalidFrequencyError,
    InvalidTokenError,
    InvalidTokenIDError,
    InvalidVocabularyError,
)


class VocabularyValidator:
    """Centralized validation methods used by public vocabulary objects."""

    @staticmethod
    def validate_token(token: object) -> None:
        """Validate a token value."""

        if not isinstance(token, str):
            raise InvalidTokenError("Token must be a string.")

        if token == "":
            raise InvalidTokenError("Token cannot be empty.")

    @staticmethod
    def validate_token_id(token_id: object) -> None:
        """Validate a token ID value."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise InvalidTokenIDError("Token ID must be an integer.")

        if token_id < 0:
            raise InvalidTokenIDError("Token ID cannot be negative.")

    @staticmethod
    def validate_frequency(frequency: object) -> None:
        """Validate the shared non-negative frequency representation."""

        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise InvalidFrequencyError("Frequency must be an integer.")

        if frequency < 0:
            raise InvalidFrequencyError("Frequency cannot be negative.")

    @classmethod
    def validate_normal_frequency(cls, frequency: object) -> None:
        """Require a strictly positive frequency for a normal token."""

        cls.validate_frequency(frequency)

        if frequency == 0:
            raise InvalidFrequencyError(
                "Normal token frequency must be greater than zero."
            )

    @classmethod
    def validate_special_tokens(
        cls,
        token_to_id: Mapping[Any, Any],
    ) -> None:
        """Ensure every reserved token exists at its exact reserved ID."""

        for token, expected_id in SPECIAL_TOKEN_ID_PAIRS:
            if token not in token_to_id:
                raise InvalidTokenError(f"Missing special token: {token}")

            actual_id = token_to_id[token]
            cls.validate_token_id(actual_id)
            if actual_id != expected_id:
                raise InvalidVocabularyError(
                    f"Special token {token!r} must have ID {expected_id}, "
                    f"not {actual_id}."
                )

    @classmethod
    def validate_vocabulary_state(
        cls,
        token_to_id: Mapping[Any, Any],
        id_to_token: Sequence[Any],
        frequencies: Mapping[Any, Any],
    ) -> None:
        """Validate consistency between all vocabulary lookup tables."""

        if isinstance(id_to_token, (str, bytes, bytearray)):
            raise InvalidVocabularyError(
                "id_to_token must be a sequence of token strings."
            )

        for token, token_id in token_to_id.items():
            cls.validate_token(token)
            cls.validate_token_id(token_id)

        cls.validate_special_tokens(token_to_id)

        if len(token_to_id) != len(id_to_token):
            raise InvalidVocabularyError(
                "token_to_id and id_to_token must have the same size."
            )

        if len(set(token_to_id.values())) != len(token_to_id):
            raise InvalidVocabularyError("Token IDs must be unique.")

        if set(token_to_id) != set(frequencies):
            raise InvalidVocabularyError(
                "Every vocabulary token must have exactly one frequency."
            )

        for expected_id, token in enumerate(id_to_token):
            cls.validate_token(token)
            actual_id = token_to_id.get(token)
            if actual_id != expected_id:
                raise InvalidVocabularyError(
                    "token_to_id and id_to_token are not bidirectionally "
                    f"consistent for token {token!r}."
                )

        for token, frequency in frequencies.items():
            cls.validate_token(token)

            if token in SPECIAL_TOKENS:
                cls.validate_frequency(frequency)
                if frequency != 0:
                    raise InvalidVocabularyError(
                        f"Special token {token!r} must have frequency zero."
                    )
                continue

            try:
                cls.validate_normal_frequency(frequency)
            except InvalidFrequencyError as exc:
                raise InvalidVocabularyError(
                    f"Normal token {token!r} must have a positive frequency."
                ) from exc


__all__ = ("VocabularyValidator",)
