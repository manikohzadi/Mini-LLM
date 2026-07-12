"""Tests for JSON vocabulary persistence."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizer.vocab.exceptions import (
    InvalidVocabularyFormatError,
    VocabularyFileNotFoundError,
)
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.repositories import JSONVocabularyRepository
from tokenizer.vocab.types import Token
from tokenizer.vocab.vocabulary import Vocabulary


class JSONVocabularyRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = JSONVocabularyRepository()

    @staticmethod
    def _vocabulary() -> Vocabulary:
        metadata = VocabularyMetadata(
            vocabulary_name="repository-test",
            created_at="2026-01-01T00:00:00+00:00",
        )
        vocabulary = Vocabulary.empty(metadata=metadata)
        vocabulary.add_token(Token("فارسی"), frequency=5)
        vocabulary.freeze()
        return vocabulary

    def test_save_and_load_round_trip_in_nested_directory(self) -> None:
        vocabulary = self._vocabulary()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "vocabulary.json"

            self.repository.save(vocabulary, path)
            restored = self.repository.load(path)

            self.assertTrue(path.is_file())
            self.assertIn("فارسی", path.read_text(encoding="utf-8"))
            self.assertEqual(restored, vocabulary)

    def test_load_missing_file_raises_domain_error(self) -> None:
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.json"

            with self.assertRaises(VocabularyFileNotFoundError):
                self.repository.load(missing_path)

    def test_load_rejects_invalid_json_and_non_object_root(self) -> None:
        with TemporaryDirectory() as directory:
            invalid_json = Path(directory) / "invalid.json"
            invalid_json.write_text("{broken", encoding="utf-8")

            with self.assertRaises(InvalidVocabularyFormatError):
                self.repository.load(invalid_json)

            non_object = Path(directory) / "array.json"
            non_object.write_text("[]", encoding="utf-8")

            with self.assertRaises(InvalidVocabularyFormatError):
                self.repository.load(non_object)

    def test_load_rejects_missing_special_tokens(self) -> None:
        state = self._vocabulary().to_dict()
        del state["token_to_id"]["<PAD>"]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-state.json"
            path.write_text(
                json.dumps(state),
                encoding="utf-8",
            )

            with self.assertRaises(InvalidVocabularyFormatError):
                self.repository.load(path)


if __name__ == "__main__":
    unittest.main()
