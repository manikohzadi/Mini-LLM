"""Tests for JSON vocabulary serialization."""

from __future__ import annotations

import json
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizer.vocab.constants import (
    VOCAB_FILE_ENCODING,
    VOCAB_FORMAT_VERSION,
)
from tokenizer.vocab.exceptions import (
    InvalidVocabularyFormatError,
    UnsupportedFormatError,
    UnsupportedVocabularyFormatVersionError,
    VocabularyFileNotFoundError,
)
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.serializers import JSONVocabularySerializer
from tokenizer.vocab.types import Token
from tokenizer.vocab.vocabulary import Vocabulary


class JSONVocabularySerializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serializer = JSONVocabularySerializer()

    @staticmethod
    def _vocabulary() -> Vocabulary:
        metadata = VocabularyMetadata(
            vocabulary_name="serializer-test",
            created_at="2026-01-01T00:00:00+00:00",
        )
        vocabulary = Vocabulary.empty(metadata=metadata)
        vocabulary.add_token(Token("فارسی"), frequency=5)
        vocabulary.freeze()
        return vocabulary

    def test_serialize_deserialize_round_trip(self) -> None:
        vocabulary = self._vocabulary()
        payload = self.serializer.serialize(vocabulary)
        restored = self.serializer.deserialize(payload)

        self.assertIn("فارسی", payload)
        self.assertEqual(restored, vocabulary)
        self.assertIsNot(restored, vocabulary)

    def test_save_appends_json_extension_and_load_resolves_it(self) -> None:
        vocabulary = self._vocabulary()

        with TemporaryDirectory() as directory:
            extensionless_path = Path(directory) / "nested" / "vocabulary"
            saved_path = self.serializer.save(
                vocabulary,
                extensionless_path,
            )
            restored = self.serializer.load(extensionless_path)

            self.assertEqual(saved_path.suffix, ".json")
            self.assertTrue(saved_path.is_file())
            self.assertIn(
                "فارسی".encode(VOCAB_FILE_ENCODING),
                saved_path.read_bytes(),
            )
            self.assertEqual(restored, vocabulary)

    def test_wrong_file_extension_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.txt"
            with self.assertRaises(UnsupportedFormatError):
                self.serializer.save(self._vocabulary(), path)
            with self.assertRaises(UnsupportedFormatError):
                self.serializer.load(path)

    def test_load_missing_file_raises_domain_error(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaises(VocabularyFileNotFoundError):
                self.serializer.load(path)

    def test_deserialize_rejects_invalid_json_and_non_object_root(self) -> None:
        with self.assertRaises(InvalidVocabularyFormatError):
            self.serializer.deserialize("{broken")

        with self.assertRaises(InvalidVocabularyFormatError):
            self.serializer.deserialize("[]")

    def test_deserialize_rejects_missing_or_wrong_special_token_id(self) -> None:
        state = json.loads(
            self.serializer.serialize(self._vocabulary())
        )
        del state["token_to_id"]["<PAD>"]

        with self.assertRaises(InvalidVocabularyFormatError):
            self.serializer.deserialize(json.dumps(state))

        state = json.loads(
            self.serializer.serialize(self._vocabulary())
        )
        state["token_to_id"]["<PAD>"] = 10

        with self.assertRaises(InvalidVocabularyFormatError):
            self.serializer.deserialize(json.dumps(state))

    def test_format_version_is_required_and_enforced(self) -> None:
        state = json.loads(
            self.serializer.serialize(self._vocabulary())
        )
        self.assertEqual(
            state["metadata"]["format_version"],
            VOCAB_FORMAT_VERSION,
        )

        state["metadata"]["format_version"] = (
            VOCAB_FORMAT_VERSION + 1
        )
        with self.assertRaises(
            UnsupportedVocabularyFormatVersionError
        ):
            self.serializer.deserialize(json.dumps(state))

        del state["metadata"]["format_version"]
        with self.assertRaises(InvalidVocabularyFormatError):
            self.serializer.deserialize(json.dumps(state))

    def test_metadata_format_version_is_not_configurable(self) -> None:
        parameters = signature(VocabularyMetadata).parameters

        self.assertNotIn("format_version", parameters)
        self.assertEqual(
            VocabularyMetadata().format_version,
            VOCAB_FORMAT_VERSION,
        )

        with self.assertRaises(TypeError):
            VocabularyMetadata(  # type: ignore[call-arg]
                format_version=VOCAB_FORMAT_VERSION + 1,
            )


if __name__ == "__main__":
    unittest.main()
