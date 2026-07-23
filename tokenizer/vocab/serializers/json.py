"""JSON serializer for vocabularies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, cast

from .._internal.helpers import ensure_file_extension
from ..constants import (
    VOCAB_FILE_ENCODING,
    JSON_EXTENSION,
    JSON_INDENT,
    VOCAB_FORMAT_VERSION,
)
from ..exceptions import (
    DeserializationError,
    InvalidVocabularyFormatError,
    SerializationError,
    UnsupportedVocabularyFormatVersionError,
    ValidationError,
    VocabularyFileNotFoundError,
)
from ..types import VocabularyState
from ..vocabulary import Vocabulary
from .base import VocabularySerializer


class JSONVocabularySerializer(VocabularySerializer):
    """Serialize vocabularies as UTF-8 JSON files."""

    def serialize(self, vocabulary: Vocabulary) -> str:
        """Convert ``vocabulary`` to formatted JSON text."""

        if not isinstance(vocabulary, Vocabulary):
            raise SerializationError(
                "Only Vocabulary instances can be serialized."
            )

        self._validate_supported_version(
            vocabulary.metadata.format_version
        )

        try:
            return json.dumps(
                vocabulary.to_dict(),
                ensure_ascii=False,
                indent=JSON_INDENT,
            )
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                "Vocabulary could not be encoded as JSON."
            ) from exc

    def deserialize(self, payload: str) -> Vocabulary:
        """Construct a validated vocabulary from JSON text."""

        if not isinstance(payload, str):
            raise InvalidVocabularyFormatError(
                "Vocabulary payload must be text."
            )

        try:
            raw_data: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidVocabularyFormatError(
                "Vocabulary text contains invalid JSON."
            ) from exc

        data = self._validate_state_shape(raw_data)
        found_version = data["metadata"]["format_version"]
        self._validate_supported_version(found_version)

        try:
            return Vocabulary.from_dict(data)
        except UnsupportedVocabularyFormatVersionError:
            raise
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise InvalidVocabularyFormatError(
                "Vocabulary JSON has an invalid structure."
            ) from exc

    def save(self, vocabulary: Vocabulary, path: Path) -> Path:
        """Write JSON, appending ``.json`` when no extension is supplied."""

        resolved_path = ensure_file_extension(path, JSON_EXTENSION)
        payload = self.serialize(vocabulary)
        temporary_path: Path | None = None

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = mkstemp(
                prefix=f".{resolved_path.name}.",
                suffix=".tmp",
                dir=resolved_path.parent,
                text=True,
            )
            temporary_path = Path(temporary_name)

            with os.fdopen(
                file_descriptor,
                "w",
                encoding=VOCAB_FILE_ENCODING,
                newline="\n",
            ) as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())

            temporary_path.replace(resolved_path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise SerializationError(
                f"Could not save vocabulary to {resolved_path}."
            ) from exc

        return resolved_path

    def load(self, path: Path) -> Vocabulary:
        """Read a JSON vocabulary, resolving a missing ``.json`` suffix."""

        resolved_path = ensure_file_extension(path, JSON_EXTENSION)
        if not resolved_path.is_file():
            raise VocabularyFileNotFoundError(resolved_path)

        try:
            payload = resolved_path.read_text(encoding=VOCAB_FILE_ENCODING)
        except UnicodeError as exc:
            raise InvalidVocabularyFormatError(
                f"Vocabulary file is not valid {VOCAB_FILE_ENCODING} text."
            ) from exc
        except OSError as exc:
            raise DeserializationError(
                f"Could not read vocabulary from {resolved_path}."
            ) from exc

        return self.deserialize(payload)

    @classmethod
    def _validate_state_shape(cls, raw_data: object) -> VocabularyState:
        """Validate the outer JSON schema before constructing domain types."""

        if not isinstance(raw_data, dict):
            raise InvalidVocabularyFormatError(
                "Vocabulary JSON root must be an object."
            )

        required_keys = {
            "metadata",
            "token_to_id",
            "id_to_token",
            "frequencies",
            "frozen",
        }
        missing_keys = required_keys.difference(raw_data)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise InvalidVocabularyFormatError(
                f"Vocabulary JSON is missing required fields: {missing}."
            )

        metadata = raw_data["metadata"]
        token_to_id = raw_data["token_to_id"]
        id_to_token = raw_data["id_to_token"]
        frequencies = raw_data["frequencies"]
        frozen = raw_data["frozen"]

        if not isinstance(metadata, dict):
            raise InvalidVocabularyFormatError(
                "metadata must be a JSON object."
            )
        if "format_version" not in metadata:
            raise InvalidVocabularyFormatError(
                "metadata.format_version is required."
            )
        if not isinstance(token_to_id, dict):
            raise InvalidVocabularyFormatError(
                "token_to_id must be a JSON object."
            )
        if not isinstance(id_to_token, list):
            raise InvalidVocabularyFormatError(
                "id_to_token must be a JSON array."
            )
        if not isinstance(frequencies, dict):
            raise InvalidVocabularyFormatError(
                "frequencies must be a JSON object."
            )
        if not isinstance(frozen, bool):
            raise InvalidVocabularyFormatError(
                "frozen must be a boolean."
            )

        return cast(VocabularyState, raw_data)

    @staticmethod
    def _validate_supported_version(version: object) -> None:
        """Reject missing, malformed, older, or newer format versions."""

        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != VOCAB_FORMAT_VERSION
        ):
            raise UnsupportedVocabularyFormatVersionError(
                version,
                VOCAB_FORMAT_VERSION,
            )


__all__ = ("JSONVocabularySerializer",)
