"""
Global constants used by the vocabulary subsystem.

This module contains only immutable constants shared across the
entire vocabulary package.

Notes
-----
- No runtime logic should exist in this module.
- No object creation other than constant tuples.
- No dependency on other project modules.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# Library
# ============================================================================

DEFAULT_ENCODING: Final[str] = "utf-8"

VOCAB_FORMAT_VERSION: Final[int] = 1

# ============================================================================
# Special Tokens
# ============================================================================

PAD_TOKEN: Final[str] = "<PAD>"

UNK_TOKEN: Final[str] = "<UNK>"

BOS_TOKEN: Final[str] = "<BOS>"

EOS_TOKEN: Final[str] = "<EOS>"

SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

SPECIAL_TOKEN_COUNT: Final[int] = len(SPECIAL_TOKENS)

# ============================================================================
# Reserved IDs
# ============================================================================

PAD_ID: Final[int] = 0

UNK_ID: Final[int] = 1

BOS_ID: Final[int] = 2

EOS_ID: Final[int] = 3

# ============================================================================
# Builder Defaults
# ============================================================================

DEFAULT_MIN_FREQUENCY: Final[int] = 1

DEFAULT_MAX_VOCAB_SIZE: Final[int] = 100_000

# ============================================================================
# Serialization
# ============================================================================

JSON_INDENT: Final[int] = 4

JSON_EXTENSION: Final[str] = ".json"

# ============================================================================
# Public API
# ============================================================================

__all__ = (
    "DEFAULT_ENCODING",
    "VOCAB_FORMAT_VERSION",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKEN_COUNT",
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
    "DEFAULT_MIN_FREQUENCY",
    "DEFAULT_MAX_VOCAB_SIZE",
    "JSON_INDENT",
    "JSON_EXTENSION",
)