"""
Custom exceptions used by the vocabulary subsystem.

The purpose of this module is to provide a well-structured exception
hierarchy for all vocabulary-related operations.

Notes
-----
- All custom exceptions inherit from ``VocabularyError``.
- Built-in exceptions (TypeError, ValueError, FileNotFoundError, ...)
  are still used where appropriate.
"""

from __future__ import annotations


class VocabularyError(Exception):
    """
    Base class for all vocabulary-related exceptions.
    """


# ============================================================================
# Validation
# ============================================================================


class ValidationError(VocabularyError):
    """
    Raised when validation of an object fails.
    """


class InvalidTokenError(ValidationError):
    """
    Raised when a token is invalid.
    """


class InvalidTokenIDError(ValidationError):
    """
    Raised when a token ID is invalid.
    """


class InvalidVocabularyError(ValidationError):
    """
    Raised when a vocabulary object is invalid.
    """


class InvalidMetadataError(ValidationError):
    """
    Raised when metadata is invalid.
    """


# ============================================================================
# Lookup
# ============================================================================


class LookupError(VocabularyError):
    """
    Base class for lookup-related exceptions.
    """


class UnknownTokenError(LookupError):
    """
    Raised when a token does not exist.
    """

    __slots__ = ("token",)

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"Unknown token: {token!r}")


class UnknownTokenIDError(LookupError):
    """
    Raised when a token ID does not exist.
    """

    __slots__ = ("token_id",)

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        super().__init__(f"Unknown token id: {token_id}")


# ============================================================================
# Builder
# ============================================================================


class BuilderError(VocabularyError):
    """
    Base class for builder-related exceptions.
    """


class EmptyCorpusError(BuilderError):
    """
    Raised when attempting to build a vocabulary from an empty corpus.
    """


class DuplicateTokenError(BuilderError):
    """
    Raised when duplicate reserved tokens are detected.
    """


# ============================================================================
# Repository
# ============================================================================


class RepositoryError(VocabularyError):
    """
    Base class for repository-related exceptions.
    """


class SaveVocabularyError(RepositoryError):
    """
    Raised when saving a vocabulary fails.
    """


class LoadVocabularyError(RepositoryError):
    """
    Raised when loading a vocabulary fails.
    """


class VocabularyFileNotFoundError(
    LoadVocabularyError,
    FileNotFoundError,
):
    """
    Raised when a vocabulary file does not exist.
    """

    __slots__ = ("path",)

    def __init__(self, path: object) -> None:
        self.path = path
        super().__init__(f"Vocabulary file not found: {path}")


class InvalidVocabularyFormatError(LoadVocabularyError):
    """
    Raised when persisted vocabulary data has an invalid format.
    """


# ============================================================================
# Serializer
# ============================================================================


class SerializerError(VocabularyError):
    """
    Base class for serializer-related exceptions.
    """


class SerializationError(SerializerError):
    """
    Raised when serialization fails.
    """


class DeserializationError(SerializerError):
    """
    Raised when deserialization fails.
    """


class UnsupportedFormatError(SerializerError):
    """
    Raised when a serialization format is unsupported.
    """


# ============================================================================
# State
# ============================================================================


class FrozenVocabularyError(VocabularyError):
    """
    Raised when attempting to modify a frozen vocabulary.
    """


__all__ = (
    "VocabularyError",
    "ValidationError",
    "InvalidTokenError",
    "InvalidTokenIDError",
    "InvalidVocabularyError",
    "InvalidMetadataError",
    "LookupError",
    "UnknownTokenError",
    "UnknownTokenIDError",
    "BuilderError",
    "EmptyCorpusError",
    "DuplicateTokenError",
    "RepositoryError",
    "SaveVocabularyError",
    "LoadVocabularyError",
    "VocabularyFileNotFoundError",
    "InvalidVocabularyFormatError",
    "SerializerError",
    "SerializationError",
    "DeserializationError",
    "UnsupportedFormatError",
    "FrozenVocabularyError",
)
