"""
Vocabulary validators.
"""

from __future__ import annotations

from .constants import SPECIAL_TOKENS
from .exceptions import (
    InvalidTokenError,
    InvalidTokenIDError,
)


class VocabularyValidator:
    """
    Collection of static validation methods.
    """

    @staticmethod
    def validate_token(token: str) -> None:
        """
        Validate a token.
        """

        if not isinstance(token, str):
            raise InvalidTokenError(
                "Token must be a string."
            )

        if token == "":
            raise InvalidTokenError(
                "Token cannot be empty."
            )

    @staticmethod
    def validate_token_id(token_id: int) -> None:
        """
        Validate a token ID.
        """

        if not isinstance(token_id, int):
            raise InvalidTokenIDError(
                "Token ID must be an integer."
            )

        if token_id < 0:
            raise InvalidTokenIDError(
                "Token ID cannot be negative."
            )

    @staticmethod
    def validate_special_tokens(
        token_to_id: dict[str, int],
    ) -> None:
        """
        Ensure every special token exists.
        """

        for token in SPECIAL_TOKENS:

            if token not in token_to_id:

                raise InvalidTokenError(
                    f"Missing special token: {token}"
                )