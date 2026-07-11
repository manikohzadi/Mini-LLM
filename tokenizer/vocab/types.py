"""
Common type aliases used throughout the vocabulary subsystem.

This module centralizes frequently used type definitions to improve
readability and maintainability.

Notes
-----
These are purely type aliases.
No runtime logic should exist in this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypedDict, NewType
from dataclasses import dataclass

# ============================================================================
# Basic Types
# ============================================================================

Token = NewType(
    "Token",
    str,
)

TokenID = NewType(
    "TokenID",
    int,
)

type Frequency = int

# ============================================================================
# Collections
# ============================================================================

type TokenSequence = Sequence[Token]

type MutableTokenSequence = list[Token]

type Corpus = Sequence[TokenSequence]

type TokenIterable = Iterable[Token]

# ============================================================================
# Lookup Tables
# ============================================================================

type TokenToID = dict[Token, TokenID]

type IDToToken = list[Token]

type TokenFrequencies = dict[Token, Frequency]

# ============================================================================
# Serialization
# ============================================================================

type JSONPrimitive = (
    str
    | int
    | float
    | bool
    | None
)

type JSONValue = (
    JSONPrimitive
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)

type JSONDict = dict[str, JSONValue]

class MetadataDict(TypedDict):

    vocabulary_name: str

    language: str

    tokenizer_version: str

    format_version: int

    created_at: str

    description: str

    author: str

    license: str

class VocabularyState(TypedDict):

    metadata: MetadataDict

    token_to_id: dict[str, int]

    id_to_token: list[str]

    frequencies: dict[str, int]

    frozen: bool

# ============================================================================
# Builder
# ============================================================================

@dataclass(slots=True)
class BuildResult:

    token_to_id: TokenToID

    id_to_token: IDToToken

    frequencies: TokenFrequencies

# ============================================================================
# Public API
# ============================================================================

__all__ = (
    "Token",
    "TokenID",
    "Frequency",
    "TokenSequence",
    "MutableTokenSequence",
    "Corpus",
    "TokenIterable",
    "TokenToID",
    "IDToToken",
    "TokenFrequencies",
    "JSONDict",
    "MetadataDict",
    "VocabularyState",
    "BuildResult",
)