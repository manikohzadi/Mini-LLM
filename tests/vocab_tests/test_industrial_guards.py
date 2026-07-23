"""Industrial guard tests for compatibility and persisted vocabulary safety."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Any, cast

from tokenizer.build_vocab import build_vocab
from tokenizer.vocab import (
    DefaultVocabularyBuilder,
    JSONVocabularyRepository,
    JSONVocabularySerializer,
    Vocabulary,
    VocabularyFactory,
    VocabularyStatistics,
    WordVocabularyBuilder,
)
from tokenizer.vocab.constants import (
    DEFAULT_ENCODING,
    VOCAB_FILE_ENCODING,
    VOCAB_FORMAT_VERSION,
)
from tokenizer.vocab.exceptions import (
    DeserializationError,
    EmptyCorpusError,
    InvalidMetadataError,
    InvalidTokenError,
    InvalidVocabularyFormatError,
    LoadVocabularyError,
    LookupError,
    SaveVocabularyError,
    SerializationError,
    UnknownTokenError,
    VocabularyLookupError,
)
from tokenizer.vocab import exceptions
from tokenizer.vocab.metadata import VocabularyMetadata
from tokenizer.vocab.types import Token


class IndustrialVocabularyGuardTests(unittest.TestCase):
    @staticmethod
    def _serialized_state() -> dict[str, Any]:
        serializer = JSONVocabularySerializer()
        return cast(
            dict[str, Any],
            json.loads(serializer.serialize(Vocabulary.empty())),
        )

    def test_exception_hierarchy_preserves_old_handlers(self) -> None:
        self.assertTrue(issubclass(UnknownTokenError, LookupError))
        self.assertTrue(
            issubclass(UnknownTokenError, VocabularyLookupError)
        )
        self.assertTrue(
            issubclass(SerializationError, SaveVocabularyError)
        )
        self.assertTrue(
            issubclass(DeserializationError, LoadVocabularyError)
        )

    def test_every_public_exception_has_a_docstring(self) -> None:
        for name in exceptions.__all__:
            exception_type = getattr(exceptions, name)
            if inspect.isclass(exception_type):
                self.assertIsNotNone(
                    inspect.getdoc(exception_type),
                    msg=f"Missing docstring for {name}",
                )

    def test_metadata_rejects_invalid_runtime_values(self) -> None:
        with self.assertRaises(InvalidMetadataError):
            VocabularyMetadata(vocabulary_name="")

        with self.assertRaises(TypeError):
            VocabularyMetadata(  # type: ignore[call-arg]
                format_version=cast(Any, True),
            )

        with self.assertRaises(InvalidMetadataError):
            VocabularyMetadata(created_at="2026-01-01T00:00:00")

    def test_builder_does_not_own_serialization_settings(self) -> None:
        parameters = inspect.signature(WordVocabularyBuilder).parameters

        self.assertNotIn("encoding", parameters)
        self.assertNotIn("format_version", parameters)

    def test_builder_can_require_a_non_empty_corpus(self) -> None:
        builder = WordVocabularyBuilder(require_non_empty=True)

        with self.assertRaises(EmptyCorpusError):
            builder.build([])

    def test_builder_validates_every_corpus_token(self) -> None:
        builder = WordVocabularyBuilder(min_frequency=2)
        corpus = [[Token("valid"), cast(Any, 123)]]

        with self.assertRaises(InvalidTokenError):
            builder.build(cast(Any, corpus))

    def test_deserializer_rejects_boolean_ids_and_frequencies(self) -> None:
        serializer = JSONVocabularySerializer()
        state = self._serialized_state()
        state["token_to_id"]["<UNK>"] = True

        with self.assertRaises(InvalidVocabularyFormatError):
            serializer.deserialize(json.dumps(state))

        state = self._serialized_state()
        state["frequencies"]["<PAD>"] = True

        with self.assertRaises(InvalidVocabularyFormatError):
            serializer.deserialize(json.dumps(state))

    def test_deserializer_rejects_nonzero_special_frequency(self) -> None:
        serializer = JSONVocabularySerializer()
        state = self._serialized_state()
        state["frequencies"]["<PAD>"] = 1

        with self.assertRaises(InvalidVocabularyFormatError):
            serializer.deserialize(json.dumps(state))

    def test_deserializer_rejects_zero_normal_frequency(self) -> None:
        serializer = JSONVocabularySerializer()
        vocabulary = Vocabulary.empty()
        vocabulary.add_token(Token("normal"), frequency=1)
        state = cast(
            dict[str, Any],
            json.loads(serializer.serialize(vocabulary)),
        )
        state["frequencies"]["normal"] = 0

        with self.assertRaises(InvalidVocabularyFormatError):
            serializer.deserialize(json.dumps(state))

    def test_atomic_save_leaves_no_temporary_file(self) -> None:
        serializer = JSONVocabularySerializer()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "vocabulary"
            saved_path = serializer.save(Vocabulary.empty(), path)
            temporary_files = list(saved_path.parent.glob("*.tmp"))

        self.assertEqual(temporary_files, [])

    def test_high_level_build_entry_point_is_operational(self) -> None:
        metadata = VocabularyMetadata(
            vocabulary_name="integration-test",
            created_at="2026-01-01T00:00:00+00:00",
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary"
            vocabulary = build_vocab(
                [["سلام", "سلام", "دنیا"]],
                min_frequency=2,
                metadata=metadata,
                output_path=path,
            )
            restored = JSONVocabularySerializer().load(path)

        self.assertEqual(restored, vocabulary)
        self.assertEqual(vocabulary.metadata, metadata)
        self.assertIn(Token("سلام"), vocabulary)
        self.assertNotIn(Token("دنیا"), vocabulary)

    def test_old_and_new_public_entry_points_are_importable(self) -> None:
        self.assertIsInstance(DefaultVocabularyBuilder(), WordVocabularyBuilder)
        self.assertIsInstance(
            JSONVocabularyRepository().serializer,
            JSONVocabularySerializer,
        )
        self.assertIsInstance(VocabularyFactory(), VocabularyFactory)
        self.assertEqual(
            VocabularyStatistics.from_vocabulary(
                Vocabulary.empty()
            ).normal_token_count,
            0,
        )
        self.assertEqual(VOCAB_FORMAT_VERSION, 1)
        self.assertEqual(VOCAB_FILE_ENCODING, "utf-8")
        self.assertEqual(DEFAULT_ENCODING, VOCAB_FILE_ENCODING)


if __name__ == "__main__":
    unittest.main()
