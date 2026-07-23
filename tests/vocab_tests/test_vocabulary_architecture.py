"""Tests for the core Vocabulary domain object."""

from __future__ import annotations

import unittest
from typing import Any, cast

from tokenizer.vocab.constants import (
    SPECIAL_TOKEN_ID_PAIRS,
    SPECIAL_TOKENS,
    UNK_ID,
)
from tokenizer.vocab.exceptions import (
    FrozenVocabularyError,
    InvalidFrequencyError,
    InvalidTokenError,
    InvalidTokenIDError,
    UnknownTokenError,
    UnknownTokenIDError,
)
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.types import Token, TokenID
from tokenizer.vocab.vocabulary import Vocabulary


class VocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = Vocabulary.empty()

    def test_empty_contains_reserved_tokens_with_exact_ids(self) -> None:
        self.assertEqual(tuple(self.vocabulary), SPECIAL_TOKENS)
        self.assertEqual(len(self.vocabulary), len(SPECIAL_TOKENS))
        self.assertTrue(self.vocabulary.is_empty)
        self.assertFalse(self.vocabulary)

        for token_value, expected_id in SPECIAL_TOKEN_ID_PAIRS:
            token = Token(token_value)
            self.assertEqual(
                self.vocabulary.token_to_id(token),
                expected_id,
            )
            self.assertEqual(
                self.vocabulary.id_to_token(TokenID(expected_id)),
                token,
            )
            self.assertEqual(self.vocabulary.frequency(token), 0)

    def test_add_lookup_duplicate_and_frequency(self) -> None:
        token = Token("سلام")
        token_id = self.vocabulary.add_token(token, frequency=3)

        self.assertEqual(token_id, len(SPECIAL_TOKENS))
        self.assertEqual(self.vocabulary.token_to_id(token), token_id)
        self.assertEqual(self.vocabulary.id_to_token(token_id), token)
        self.assertEqual(self.vocabulary.frequency(token), 3)
        self.assertEqual(self.vocabulary.add_token(token), token_id)
        self.assertEqual(
            self.vocabulary.add_token(token, frequency=3),
            token_id,
        )
        self.assertEqual(self.vocabulary.frequency(token), 3)

        with self.assertRaises(InvalidFrequencyError):
            self.vocabulary.add_token(token, frequency=4)

    def test_new_normal_token_requires_positive_frequency(self) -> None:
        with self.assertRaises(InvalidFrequencyError):
            self.vocabulary.add_token(Token("missing-frequency"))

        with self.assertRaises(InvalidFrequencyError):
            self.vocabulary.add_token(Token("zero-frequency"), frequency=0)

    def test_extend_requires_frequency_for_every_token(self) -> None:
        self.vocabulary.extend(
            {
                Token("الف"): 2,
                Token("ب"): 1,
            }
        )
        self.vocabulary.extend([(Token("پ"), 3)])

        self.assertEqual(self.vocabulary.frequency(Token("الف")), 2)
        self.assertEqual(self.vocabulary.frequency(Token("ب")), 1)
        self.assertEqual(self.vocabulary.frequency(Token("پ")), 3)

    def test_public_operations_use_centralized_validation(self) -> None:
        with self.assertRaises(InvalidTokenError):
            self.vocabulary.add_token(Token(""))

        with self.assertRaises(InvalidTokenError):
            self.vocabulary.add_token(cast(Any, 123))

        with self.assertRaises(InvalidTokenError):
            self.vocabulary.token_to_id(cast(Any, None))

        with self.assertRaises(InvalidTokenIDError):
            self.vocabulary.id_to_token(TokenID(-1))

        with self.assertRaises(InvalidTokenIDError):
            self.vocabulary.id_to_token(cast(Any, "0"))

        with self.assertRaises(InvalidFrequencyError):
            self.vocabulary.add_token(Token("x"), frequency=-1)

    def test_valid_but_unknown_values_raise_lookup_errors(self) -> None:
        with self.assertRaises(UnknownTokenError):
            self.vocabulary.token_to_id(Token("missing"))

        with self.assertRaises(UnknownTokenIDError):
            self.vocabulary.id_to_token(TokenID(999))

    def test_encode_and_decode_map_unknown_tokens_to_unk(self) -> None:
        known = Token("known")
        known_id = self.vocabulary.add_token(known, frequency=1)
        encoded = self.vocabulary.encode([known, Token("missing")])

        self.assertEqual(encoded, [known_id, UNK_ID])
        self.assertEqual(
            self.vocabulary.decode(encoded),
            [known, self.vocabulary.unk_token],
        )

    def test_encode_rejects_unknown_replacement_id(self) -> None:
        with self.assertRaises(UnknownTokenIDError):
            self.vocabulary.encode(
                [Token("missing")],
                unknown_token_id=TokenID(999),
            )

    def test_freeze_prevents_mutation(self) -> None:
        self.vocabulary.freeze()

        with self.assertRaises(InvalidTokenError):
            self.vocabulary.add_token(Token(""))

        with self.assertRaises(FrozenVocabularyError):
            self.vocabulary.add_token(Token("new"))

        with self.assertRaises(FrozenVocabularyError):
            self.vocabulary.clear()

    def test_mapping_views_are_read_only(self) -> None:
        with self.assertRaises(TypeError):
            self.vocabulary.token_to_id_map[Token("x")] = TokenID(10)  # type: ignore[index]

        with self.assertRaises(TypeError):
            self.vocabulary.frequencies[Token("x")] = 1  # type: ignore[index]

    def test_copy_is_independent(self) -> None:
        copied = self.vocabulary.copy()
        copied.add_token(Token("copy-only"), frequency=1)

        self.assertNotIn(Token("copy-only"), self.vocabulary)
        self.assertIn(Token("copy-only"), copied)

    def test_clear_keeps_only_special_tokens_with_exact_ids(self) -> None:
        self.vocabulary.add_token(Token("temporary"), frequency=1)
        self.vocabulary.clear()

        self.assertEqual(tuple(self.vocabulary), SPECIAL_TOKENS)
        self.assertTrue(self.vocabulary.is_empty)
        for token_value, expected_id in SPECIAL_TOKEN_ID_PAIRS:
            self.assertEqual(
                self.vocabulary.token_to_id(Token(token_value)),
                expected_id,
            )

    def test_metadata_is_preserved(self) -> None:
        metadata = VocabularyMetadata(
            vocabulary_name="persian-test",
            created_at="2026-01-01T00:00:00+00:00",
        )
        vocabulary = Vocabulary.empty(metadata=metadata)
        self.assertEqual(vocabulary.metadata, metadata)


if __name__ == "__main__":
    unittest.main()
