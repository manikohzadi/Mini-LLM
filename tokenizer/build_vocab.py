"""Convenience entry point for building and saving the project vocabulary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .vocab.builders import WordVocabularyBuilder
from .vocab.constants import (
    DEFAULT_MAX_NORMAL_TOKEN_COUNT,
    DEFAULT_MIN_FREQUENCY,
)
from .vocab.metadata import VocabularyMetadata
from .vocab.serializers import (
    JSONVocabularySerializer,
    VocabularySerializer,
)
from .vocab.types import Corpus, Token
from .vocab.vocabulary import Vocabulary


def build_vocab(
    all_tokens: Sequence[Sequence[str]],
    *,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_normal_token_count: int = DEFAULT_MAX_NORMAL_TOKEN_COUNT,
    metadata: VocabularyMetadata | None = None,
    output_path: Path = Path("vocab.json"),
    serializer: VocabularySerializer | None = None,
) -> Vocabulary:
    """Build a frozen vocabulary and save it as versioned UTF-8 JSON."""

    corpus: Corpus = [
        [Token(token) for token in sequence]
        for sequence in all_tokens
    ]
    builder = WordVocabularyBuilder(
        min_frequency=min_frequency,
        max_normal_token_count=max_normal_token_count,
    )
    vocabulary = builder.build(corpus, metadata=metadata)
    active_serializer = serializer or JSONVocabularySerializer()
    active_serializer.save(vocabulary, output_path)
    return vocabulary


__all__ = ("build_vocab",)
