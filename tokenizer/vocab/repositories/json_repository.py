"""
JSON repository implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from ..constants import DEFAULT_ENCODING, JSON_INDENT
from ..exceptions import (
    InvalidVocabularyFormatError,
    ValidationError,
    VocabularyFileNotFoundError,
)
from ..types import VocabularyState
from ..vocabulary import Vocabulary

from .base import VocabularyRepository


class JSONVocabularyRepository(
    VocabularyRepository,
):
    """
    Store vocabularies as JSON.
    """

    def save(
        self,
        vocabulary: Vocabulary,
        path: Path,
    ) -> None:
        """
        Save a vocabulary as JSON.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding=DEFAULT_ENCODING,
        ) as file:

            json.dump(
                vocabulary.to_dict(),
                file,
                ensure_ascii=False,
                indent=JSON_INDENT,
            )

    def load(
        self,
        path: Path,
    ) -> Vocabulary:
        """
        Load a vocabulary from JSON.
        """

        if not path.exists():
            raise VocabularyFileNotFoundError(path)

        try:
            with path.open(
                "r",
                encoding=DEFAULT_ENCODING,
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise InvalidVocabularyFormatError(
                "Vocabulary file contains invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise InvalidVocabularyFormatError(
                "Vocabulary JSON must be an object."
            )

        try:
            return Vocabulary.from_dict(
                cast(VocabularyState, data)
            )
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
