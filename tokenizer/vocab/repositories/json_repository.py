"""
JSON repository implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..exceptions import (
    InvalidVocabularyFormatError,
    VocabularyFileNotFoundError,
)
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
            encoding="utf-8",
        ) as file:

            json.dump(
                vocabulary.to_dict(),
                file,
                ensure_ascii=False,
                indent=4,
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

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            raise InvalidVocabularyFormatError(
                "Vocabulary JSON must be an object."
            )

        return Vocabulary.from_dict(data)
