"""Tests for vocabulary statistics."""

from __future__ import annotations

import unittest

from tokenizer.vocab.constants import SPECIAL_TOKENS
from tokenizer.vocab.statistics import VocabularyStatistics
from tokenizer.vocab.types import Token
from tokenizer.vocab.vocabulary import Vocabulary


class VocabularyStatisticsTests(unittest.TestCase):
    def test_empty_vocabulary_statistics(self) -> None:
        statistics = VocabularyStatistics.from_vocabulary(
            Vocabulary.empty()
        )

        self.assertEqual(
            statistics.total_token_count,
            len(SPECIAL_TOKENS),
        )
        self.assertEqual(statistics.special_token_count, 4)
        self.assertEqual(statistics.normal_token_count, 0)
        self.assertEqual(statistics.total_normal_token_frequency, 0)
        self.assertEqual(statistics.average_normal_token_frequency, 0.0)
        self.assertIsNone(statistics.most_frequent_normal_token)
        self.assertIsNone(statistics.least_frequent_normal_token)

    def test_statistics_use_only_normal_tokens(self) -> None:
        vocabulary = Vocabulary.empty()
        vocabulary.add_token(Token("الف"), frequency=5)
        vocabulary.add_token(Token("ب"), frequency=1)

        statistics = VocabularyStatistics.from_vocabulary(vocabulary)

        self.assertEqual(statistics.normal_token_count, 2)
        self.assertEqual(statistics.total_normal_token_frequency, 6)
        self.assertEqual(statistics.average_normal_token_frequency, 3.0)
        self.assertEqual(
            statistics.most_frequent_normal_token,
            Token("الف"),
        )
        self.assertEqual(
            statistics.least_frequent_normal_token,
            Token("ب"),
        )

    def test_total_count_includes_special_tokens(self) -> None:
        vocabulary = Vocabulary.empty()
        vocabulary.add_token(Token("normal"), frequency=1)
        statistics = VocabularyStatistics.from_vocabulary(vocabulary)

        self.assertEqual(
            statistics.total_token_count,
            len(SPECIAL_TOKENS) + 1,
        )


if __name__ == "__main__":
    unittest.main()
