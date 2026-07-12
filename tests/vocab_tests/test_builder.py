"""Tests for the default vocabulary builder."""

from __future__ import annotations

import unittest

from tokenizer.vocab.builders import DefaultVocabularyBuilder
from tokenizer.vocab.constants import PAD_TOKEN, SPECIAL_TOKENS
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.types import Corpus, Token


class DefaultVocabularyBuilderTests(unittest.TestCase):
    def test_build_counts_filters_orders_and_freezes_tokens(self) -> None:
        corpus: Corpus = [
            [Token("الف"), Token("ب"), Token("الف")],
            [Token("ج"), Token("ب"), Token("الف")],
        ]
        builder = DefaultVocabularyBuilder(min_frequency=2)

        vocabulary = builder.build(corpus)

        self.assertEqual(
            vocabulary.tokens()[len(SPECIAL_TOKENS):],
            (Token("الف"), Token("ب")),
        )
        self.assertEqual(vocabulary.frequency(Token("الف")), 3)
        self.assertEqual(vocabulary.frequency(Token("ب")), 2)
        self.assertNotIn(Token("ج"), vocabulary)
        self.assertTrue(vocabulary.frozen)

    def test_max_size_limits_normal_tokens_only(self) -> None:
        corpus: Corpus = [[
            Token(PAD_TOKEN),
            Token(PAD_TOKEN),
            Token("first"),
            Token("first"),
            Token("second"),
        ]]
        builder = DefaultVocabularyBuilder(
            max_vocabulary_size=2,
        )

        vocabulary = builder.build(corpus)

        self.assertEqual(vocabulary.normal_token_count, 2)
        self.assertIn(Token("first"), vocabulary)
        self.assertIn(Token("second"), vocabulary)
        self.assertEqual(vocabulary.frequency(Token(PAD_TOKEN)), 0)

    def test_zero_max_size_builds_special_only_vocabulary(self) -> None:
        builder = DefaultVocabularyBuilder(
            max_vocabulary_size=0,
        )

        vocabulary = builder.build([[Token("ignored")]])

        self.assertTrue(vocabulary.is_empty)
        self.assertTrue(vocabulary.frozen)

    def test_build_preserves_supplied_metadata(self) -> None:
        metadata = VocabularyMetadata(
            vocabulary_name="custom",
            created_at="2026-01-01T00:00:00+00:00",
        )

        vocabulary = DefaultVocabularyBuilder().build(
            [],
            metadata=metadata,
        )

        self.assertEqual(vocabulary.metadata, metadata)


if __name__ == "__main__":
    unittest.main()
