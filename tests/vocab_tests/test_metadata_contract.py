"""Tests for metadata ownership of persisted format information."""

from __future__ import annotations

import unittest
from typing import Any, cast

from tokenizer.vocab.constants import VOCAB_FORMAT_VERSION
from tokenizer.vocab.exceptions import (
    InvalidMetadataError,
    UnsupportedVocabularyFormatVersionError,
)
from tokenizer.vocab.metadata import VocabularyMetadata


class VocabularyMetadataContractTests(unittest.TestCase):
    def test_to_dict_always_emits_current_format_version(self) -> None:
        metadata = VocabularyMetadata()

        self.assertEqual(
            metadata.to_dict()["format_version"],
            VOCAB_FORMAT_VERSION,
        )

    def test_from_dict_accepts_current_format_version(self) -> None:
        source = VocabularyMetadata().to_dict()

        restored = VocabularyMetadata.from_dict(source)

        self.assertEqual(restored, VocabularyMetadata.from_dict(source))
        self.assertEqual(restored.format_version, VOCAB_FORMAT_VERSION)

    def test_from_dict_requires_format_version(self) -> None:
        source = cast(dict[str, Any], VocabularyMetadata().to_dict())
        del source["format_version"]

        with self.assertRaises(InvalidMetadataError):
            VocabularyMetadata.from_dict(cast(Any, source))

    def test_from_dict_rejects_other_format_versions(self) -> None:
        source = cast(dict[str, Any], VocabularyMetadata().to_dict())
        source["format_version"] = VOCAB_FORMAT_VERSION + 1

        with self.assertRaises(
            UnsupportedVocabularyFormatVersionError
        ):
            VocabularyMetadata.from_dict(cast(Any, source))

    def test_from_dict_rejects_boolean_format_version(self) -> None:
        source = cast(dict[str, Any], VocabularyMetadata().to_dict())
        source["format_version"] = True

        with self.assertRaises(
            UnsupportedVocabularyFormatVersionError
        ):
            VocabularyMetadata.from_dict(cast(Any, source))


if __name__ == "__main__":
    unittest.main()
