"""Tests for the high-level VocabularyFactory API."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from tokenizer.vocab.builders import (
    DefaultVocabularyBuilder,
    VocabularyBuilder,
)
from tokenizer.vocab.factory import VocabularyFactory
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.repositories import (
    JSONVocabularyRepository,
    VocabularyRepository,
)
from tokenizer.vocab.types import Corpus, Token
from tokenizer.vocab.vocabulary import Vocabulary


class VocabularyFactoryTests(unittest.TestCase):
    def test_defaults_use_standard_builder_and_repository(self) -> None:
        factory = VocabularyFactory()

        self.assertIsInstance(
            factory.builder,
            DefaultVocabularyBuilder,
        )
        self.assertIsInstance(
            factory.repository,
            JSONVocabularyRepository,
        )

    def test_injected_dependencies_receive_exact_calls(self) -> None:
        builder = Mock(spec=VocabularyBuilder)
        repository = Mock(spec=VocabularyRepository)
        factory = VocabularyFactory(
            builder=builder,
            repository=repository,
        )
        corpus: Corpus = [[Token("test")]]
        metadata = VocabularyMetadata(
            created_at="2026-01-01T00:00:00+00:00"
        )
        vocabulary = Vocabulary.empty(metadata=metadata)
        path = Path("vocabulary.json")
        builder.build.return_value = vocabulary
        repository.load.return_value = vocabulary

        self.assertIs(
            factory.build(corpus, metadata=metadata),
            vocabulary,
        )
        factory.save(vocabulary, path)
        self.assertIs(factory.load(path), vocabulary)

        builder.build.assert_called_once_with(
            corpus,
            metadata=metadata,
        )
        repository.save.assert_called_once_with(vocabulary, path)
        repository.load.assert_called_once_with(path)

    def test_end_to_end_build_save_load(self) -> None:
        factory = VocabularyFactory()
        corpus: Corpus = [[Token("سلام"), Token("سلام")]]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            vocabulary = factory.build(corpus)

            factory.save(vocabulary, path)
            restored = factory.load(path)

        self.assertEqual(restored, vocabulary)
        self.assertEqual(restored.frequency(Token("سلام")), 2)


if __name__ == "__main__":
    unittest.main()
