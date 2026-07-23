"""Private helper functions for vocabulary construction and file paths."""

from __future__ import annotations

from pathlib import Path

from ..constants import SPECIAL_TOKEN_ID_PAIRS
from ..exceptions import (
    DuplicateTokenError,
    InvalidVocabularyError,
    UnsupportedFormatError,
)
from ..types import (
    BuildResult,
    IDToToken,
    Token,
    TokenFrequencies,
    TokenID,
    TokenToID,
)


def create_special_token_state() -> BuildResult:
    """Create consistent lookup tables containing only reserved tokens."""

    token_to_id: TokenToID = {}
    id_to_token: IDToToken = []
    frequencies: TokenFrequencies = {}
    seen_ids: set[int] = set()

    for token_value, token_id_value in SPECIAL_TOKEN_ID_PAIRS:
        if token_value in token_to_id:
            raise DuplicateTokenError(
                f"Duplicate special token: {token_value!r}."
            )

        if token_id_value in seen_ids:
            raise InvalidVocabularyError(
                f"Duplicate special token ID: {token_id_value}."
            )

        if token_id_value != len(id_to_token):
            raise InvalidVocabularyError(
                "Special token IDs must be contiguous and start at zero."
            )

        token = Token(token_value)
        token_id = TokenID(token_id_value)
        token_to_id[token] = token_id
        id_to_token.append(token)
        frequencies[token] = 0
        seen_ids.add(token_id_value)

    return BuildResult(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        frequencies=frequencies,
    )


def ensure_file_extension(path: Path, extension: str) -> Path:
    """Append a missing extension and reject a conflicting extension."""

    if not extension.startswith("."):
        raise ValueError("File extension must start with '.'.")

    if path.suffix == "":
        return path.with_suffix(extension)

    if path.suffix.lower() != extension.lower():
        raise UnsupportedFormatError(
            f"Expected a {extension} file, got {path.suffix or '<none>'}."
        )

    return path


__all__ = (
    "create_special_token_state",
    "ensure_file_extension",
)
