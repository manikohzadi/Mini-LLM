"""
Metadata definitions for the vocabulary subsystem.

This module contains immutable metadata associated with a vocabulary.

Notes
-----
- Metadata does not store vocabulary contents.
- Metadata is immutable.
- Metadata is serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .constants import VOCAB_FORMAT_VERSION
from .exceptions import (
    InvalidMetadataError,
    UnsupportedVocabularyFormatVersionError,
)
from .types import MetadataDict


@dataclass(slots=True, frozen=True)
class VocabularyMetadata:
    """
    Immutable metadata describing a vocabulary.
    """

    vocabulary_name: str = "default"

    language: str = "fa"

    tokenizer_version: str = "1.0.0"

    format_version: int = field(
        default=VOCAB_FORMAT_VERSION,
        init=False,
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    description: str = ""

    author: str = ""

    license: str = ""

    def __post_init__(self) -> None:
        """Validate metadata before it becomes part of a vocabulary."""

        required_text_fields = {
            "vocabulary_name": self.vocabulary_name,
            "language": self.language,
            "tokenizer_version": self.tokenizer_version,
            "created_at": self.created_at,
        }
        for field_name, value in required_text_fields.items():
            if not isinstance(value, str) or value == "":
                raise InvalidMetadataError(
                    f"{field_name} must be a non-empty string."
                )

        optional_text_fields = {
            "description": self.description,
            "author": self.author,
            "license": self.license,
        }
        for field_name, value in optional_text_fields.items():
            if not isinstance(value, str):
                raise InvalidMetadataError(
                    f"{field_name} must be a string."
                )

        try:
            created_at = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise InvalidMetadataError(
                "created_at must be a valid ISO 8601 datetime."
            ) from exc

        if created_at.tzinfo is None:
            raise InvalidMetadataError(
                "created_at must include a timezone offset."
            )

    def to_dict(self) -> MetadataDict:
        """
        Convert metadata into a serializable dictionary.
        """

        return {
            "vocabulary_name": self.vocabulary_name,
            "language": self.language,
            "tokenizer_version": self.tokenizer_version,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "description": self.description,
            "author": self.author,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: MetadataDict) -> "VocabularyMetadata":
        """
        Construct metadata from a dictionary.
        """

        if "format_version" not in data:
            raise InvalidMetadataError(
                "format_version is required in persisted metadata."
            )

        found_version = data["format_version"]
        if (
            isinstance(found_version, bool)
            or not isinstance(found_version, int)
            or found_version != VOCAB_FORMAT_VERSION
        ):
            raise UnsupportedVocabularyFormatVersionError(
                found_version,
                VOCAB_FORMAT_VERSION,
            )

        return cls(
            vocabulary_name=data.get("vocabulary_name", "default"),
            language=data.get("language", "fa"),
            tokenizer_version=data.get("tokenizer_version", "1.0.0"),
            created_at=data.get(
                "created_at",
                datetime.now(UTC).isoformat(),
            ),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
        )


__all__ = (
    "VocabularyMetadata",
)