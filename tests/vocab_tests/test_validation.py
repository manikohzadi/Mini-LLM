"""Tests for private centralized vocabulary validation."""

from __future__ import annotations

import unittest
from typing import Any, cast

from tokenizer.vocab._internal.validation import VocabularyValidator
from tokenizer.vocab.constants import SPECIAL_TOKEN_ID_PAIRS
from tokenizer.vocab.exceptions import (
    InvalidFrequencyError,
    InvalidTokenError,
    InvalidTokenIDError,
    InvalidVocabularyError,
)


class VocabularyValidatorTests(unittest.TestCase):
    def test_validate_token(self) -> None:
        VocabularyValidator.validate_token("سلام")

        with self.assertRaises(InvalidTokenError):
            VocabularyValidator.validate_token(cast(Any, 123))
        with self.assertRaises(InvalidTokenError):
            VocabularyValidator.validate_token("")

    def test_validate_token_id(self) -> None:
        VocabularyValidator.validate_token_id(0)
        VocabularyValidator.validate_token_id(42)

        with self.assertRaises(InvalidTokenIDError):
            VocabularyValidator.validate_token_id(cast(Any, "1"))
        with self.assertRaises(InvalidTokenIDError):
            VocabularyValidator.validate_token_id(-1)
        with self.assertRaises(InvalidTokenIDError):
            VocabularyValidator.validate_token_id(True)

    def test_validate_frequency(self) -> None:
        VocabularyValidator.validate_frequency(0)
        VocabularyValidator.validate_frequency(3)

        with self.assertRaises(InvalidFrequencyError):
            VocabularyValidator.validate_frequency(-1)
        with self.assertRaises(InvalidFrequencyError):
            VocabularyValidator.validate_frequency(True)

    def test_validate_normal_frequency(self) -> None:
        VocabularyValidator.validate_normal_frequency(1)
        VocabularyValidator.validate_normal_frequency(3)

        with self.assertRaises(InvalidFrequencyError):
            VocabularyValidator.validate_normal_frequency(0)
        with self.assertRaises(InvalidFrequencyError):
            VocabularyValidator.validate_normal_frequency(-1)
        with self.assertRaises(InvalidFrequencyError):
            VocabularyValidator.validate_normal_frequency(True)

    def test_validate_special_tokens_requires_exact_ids(self) -> None:
        complete_mapping = dict(SPECIAL_TOKEN_ID_PAIRS)
        VocabularyValidator.validate_special_tokens(complete_mapping)

        first_token, first_id = SPECIAL_TOKEN_ID_PAIRS[0]
        complete_mapping[first_token] = first_id + 1
        with self.assertRaises(InvalidVocabularyError):
            VocabularyValidator.validate_special_tokens(complete_mapping)

    def test_validate_state_rejects_zero_frequency_for_normal_token(self) -> None:
        token_to_id = dict(SPECIAL_TOKEN_ID_PAIRS)
        normal_token = "normal"
        token_to_id[normal_token] = len(token_to_id)
        id_to_token = [token for token, _ in SPECIAL_TOKEN_ID_PAIRS]
        id_to_token.append(normal_token)
        frequencies = {token: 0 for token in id_to_token}

        with self.assertRaises(InvalidVocabularyError):
            VocabularyValidator.validate_vocabulary_state(
                token_to_id,
                id_to_token,
                frequencies,
            )

    def test_validate_state_rejects_inconsistent_reverse_mapping(self) -> None:
        token_to_id = dict(SPECIAL_TOKEN_ID_PAIRS)
        id_to_token = [token for token, _ in SPECIAL_TOKEN_ID_PAIRS]
        frequencies = {token: 0 for token in id_to_token}
        id_to_token[0], id_to_token[1] = id_to_token[1], id_to_token[0]

        with self.assertRaises(InvalidVocabularyError):
            VocabularyValidator.validate_vocabulary_state(
                token_to_id,
                id_to_token,
                frequencies,
            )


if __name__ == "__main__":
    unittest.main()
