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
from .types import MetadataDict


@dataclass(slots=True, frozen=True)
class VocabularyMetadata:
    """
    Immutable metadata describing a vocabulary.
    """

    vocabulary_name: str = "default"

    language: str = "fa"

    tokenizer_version: str = "1.0.0"

    format_version: int = VOCAB_FORMAT_VERSION

    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    description: str = ""

    author: str = ""

    license: str = ""

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

        return cls(
            vocabulary_name=data.get("vocabulary_name", "default"),
            language=data.get("language", "fa"),
            tokenizer_version=data.get("tokenizer_version", "1.0.0"),
            format_version=data.get(
                "format_version",
                VOCAB_FORMAT_VERSION,
            ),
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