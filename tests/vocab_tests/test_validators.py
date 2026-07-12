"""Tests for centralized vocabulary validation."""

from __future__ import annotations

import unittest
from typing import Any, cast

from tokenizer.vocab.constants import SPECIAL_TOKENS
from tokenizer.vocab.exceptions import (
    InvalidTokenError,
    InvalidTokenIDError,
)
from tokenizer.vocab.validators import VocabularyValidator


class VocabularyValidatorTests(unittest.TestCase):
    def test_validate_token_accepts_non_empty_string(self) -> None:
        VocabularyValidator.validate_token("سلام")

    def test_validate_token_rejects_non_string_and_empty_string(self) -> None:
        with self.assertRaisesRegex(
            InvalidTokenError,
            "Token must be a string",
        ):
            VocabularyValidator.validate_token(
                cast(Any, 123)
            )

        with self.assertRaisesRegex(
            InvalidTokenError,
            "Token cannot be empty",
        ):
            VocabularyValidator.validate_token("")

    def test_validate_token_id_accepts_non_negative_integer(self) -> None:
        VocabularyValidator.validate_token_id(0)
        VocabularyValidator.validate_token_id(42)

    def test_validate_token_id_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(
            InvalidTokenIDError,
            "Token ID must be an integer",
        ):
            VocabularyValidator.validate_token_id(
                cast(Any, "1")
            )

        with self.assertRaisesRegex(
            InvalidTokenIDError,
            "Token ID cannot be negative",
        ):
            VocabularyValidator.validate_token_id(-1)

    def test_validate_special_tokens_requires_every_reserved_token(self) -> None:
        complete_mapping = {
            token: token_id
            for token_id, token in enumerate(SPECIAL_TOKENS)
        }
        VocabularyValidator.validate_special_tokens(complete_mapping)

        del complete_mapping[SPECIAL_TOKENS[-1]]

        with self.assertRaisesRegex(
            InvalidTokenError,
            SPECIAL_TOKENS[-1],
        ):
            VocabularyValidator.validate_special_tokens(
                complete_mapping
            )


if __name__ == "__main__":
    unittest.main()
